#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import importlib

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from PDFParametrization import PDFParametrization
from TemplateBase import TemplateBase
import awkward as ak


class FeatureTemplate(TemplateBase):
    """
    Template for a *binned feature* from a sample loader, with PDF reweighting.

    Updates vs previous version:
      - Generator_scalePDF is Q (NOT Q^2): no sqrt.
      - No LHAPDF calls in the constructor (xfxQ can be slow).
      - Denominator f0(id1,x1,Q)*f0(id2,x2,Q) is computed lazily on first request for a variation template.
      - Adds max_events: still loads all shards, then truncates arrays to first max_events.
      - Supports derived features via a runtime registry: register[name] = expression string.
    """

    def __init__(
        self,
        pdf: PDFParametrization,
        sample: str,
        feature: str,
        bin_edges,
        module_samples: str = "data.samples_RunII",
        name: str = "",
        max_events: int | None = None,
        selection=None,
        required_branches=[],
        use_abs=False,
        register=None,
    ):
        super().__init__(name=name or f"{sample}:{feature}")

        self.pdf = pdf
        self.sample = sample
        self.feature = feature
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.max_events = None if (max_events is None or max_events <= 0) else int(max_events)

        self.register = register or {}
        self.is_derived = feature in self.register
        self.expr = self.register.get(feature, None)

        # ---------------- resolve loader ----------------
        samples_mod = importlib.import_module(module_samples)
        if not hasattr(samples_mod, sample):
            raise RuntimeError(f"Loader/view '{sample}' not found in module {module_samples}.")
        self.L = getattr(samples_mod, sample).clone()

        # Ask loader for either:
        #  - the concrete feature (direct), or
        #  - the required inputs needed to evaluate a derived feature (derived).
        if self.is_derived:
            if len(required_branches) == 0:
                raise RuntimeError(f"Derived feature '{feature}' requested but required_branches is empty.")
            self.L.setFeatures(list(required_branches))
        else:
            self.L.setFeatures([feature])

        if selection is not None:
            print(f"Applying selection: {selection}")
            self.L.addSelection(selection, required_branches=required_branches)

        fn = getattr(self.L, "feature_names", None)
        if fn is None:
            raise RuntimeError("Loader has no feature_names attribute.")

        # Resolve the (single) feature index from feature_names (only for direct features)
        self.feature_index = None
        if not self.is_derived:
            if feature not in fn:
                raise RuntimeError(f"Feature '{feature}' not found in loader.feature_names={fn}")
            self.feature_index = fn.index(feature)

        # Resolve observer indices
        obs = getattr(self.L, "observer_names", None)
        if obs is None:
            raise RuntimeError("Loader has no observer_names attribute, cannot locate Generator_* columns.")

        def idx(name_):
            if name_ not in obs:
                raise RuntimeError(f"Observer '{name_}' not found. Available: {obs}")
            return obs.index(name_)

        ix1 = idx("Generator_x1")
        ix2 = idx("Generator_x2")
        iid1 = idx("Generator_id1")
        iid2 = idx("Generator_id2")
        iQ  = idx("Generator_scalePDF")  # Q

        # ---------------- read all shards ----------------
        F_list, O_list, W_list = [], [], []
        n_shards = len(self.L)

        for ish in range(n_shards):
            F, O, W = self.L.materialize(ish, "fow")
            F_list.append(np.asarray(F))
            O_list.append(np.asarray(O))
            W_list.append(np.asarray(W))

        F_all = np.concatenate(F_list, axis=0).astype(float)
        O_all = np.concatenate(O_list, axis=0)
        W_all = np.concatenate(W_list, axis=0).astype(float)

        if self.max_events is not None:
            F_all = F_all[: self.max_events]
            O_all = O_all[: self.max_events]
            W_all = W_all[: self.max_events]

        self.F = F_all
        self.O = O_all
        self.W = W_all

        # Observers
        self.x1 = np.asarray(self.O[:, ix1], dtype=float)
        self.x2 = np.asarray(self.O[:, ix2], dtype=float)
        self.id1 = np.asarray(self.O[:, iid1], dtype=int)
        self.id2 = np.asarray(self.O[:, iid2], dtype=int)
        self.Q  = np.asarray(self.O[:, iQ], dtype=float)

        # Build env for derived expressions (only if needed)
        env = {}
        if self.is_derived:
            fn_list  = list(fn)
            obs_list = list(obs)

            for k in required_branches:
                if k in fn_list:
                    env[k] = ak.Array(np.asarray(self.F[:, fn_list.index(k)], dtype=float))
                elif k in obs_list:
                    env[k] = ak.Array(np.asarray(self.O[:, obs_list.index(k)], dtype=float))
                else:
                    raise RuntimeError(f"Required branch '{k}' not found in feature_names or observer_names.")

        safe = {
            "np": np,
            "ak": ak,
            "abs": abs,
            "log": np.log,
            "sqrt": np.sqrt,
            "exp": np.exp,
            "where": np.where,
            "arcsinh": np.arcsinh,
            "sinh": np.sinh,
            "cosh": np.cosh,
        }

        # Feature values (direct or derived)
        if not self.is_derived:
            self.values = np.asarray(self.F[:, self.feature_index], dtype=float)
        else:
            self.values = np.asarray(ak.to_numpy(eval(self.expr, {"__builtins__": {}}, {**safe, **env})), dtype=float)

        if use_abs:
            self.values = abs(self.values)

        # Lazily computed denominator for reweighting (central PDF product)
        self._denom = None

        # Central template (just W)
        self._central_template = np.histogram(
            self.values,
            bins=self.bin_edges,
            weights=self.W,
        )[0].astype(float)

        print("Constructor done.")

    @property
    def n_members(self) -> int:
        return self.pdf.n_members

    def _ensure_denom(self):
        if self._denom is not None:
            return

        pdf0 = self.pdf.pdfs[0]
        d1 = np.array(pdf0.xfxQ(self.x1, self.Q), dtype=object)
        d2 = np.array(pdf0.xfxQ(self.x2, self.Q), dtype=object)

        f0_1 = np.array([d[int(pid)] for d, pid in zip(d1, self.id1)], dtype=float)
        f0_2 = np.array([d[int(pid)] for d, pid in zip(d2, self.id2)], dtype=float)

        self._denom = f0_1 * f0_2

    def get_template(self, member: int) -> np.ndarray:
        if member == 0:
            return self._central_template.copy()

        self._ensure_denom()

        pdfm = self.pdf.pdfs[member]
        d1 = np.array(pdfm.xfxQ(self.x1, self.Q), dtype=object)
        d2 = np.array(pdfm.xfxQ(self.x2, self.Q), dtype=object)

        f1 = np.array([d[int(pid)] for d, pid in zip(d1, self.id1)], dtype=float)
        f2 = np.array([d[int(pid)] for d, pid in zip(d2, self.id2)], dtype=float)

        ratio = (f1 * f2) / self._denom
        w = self.W * ratio

        h = np.histogram(
            self.values,
            bins=self.bin_edges,
            weights=w,
        )[0].astype(float)

        return h

    def get_bin_edges(self) -> np.ndarray:
        return self.bin_edges

    def get_bin_centers(self) -> np.ndarray:
        e = self.bin_edges
        return 0.5 * (e[:-1] + e[1:])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Make feature templates with PDF reweighting (central + symmhessian).")
    parser.add_argument("--pdf-set", default="NNPDF31_nnlo_hessian_pdfas", help="LHAPDF set name")
    parser.add_argument("--sample", default="TTLep_pow_2018", help="Sample loader name from data.samples_RunII")
    parser.add_argument("--feature", default="tr_ttbar_mass", help="Feature name passed to loader.setFeatures([...])")
    parser.add_argument("--xmin", type=float, default=200.0, help="Histogram xmin")
    parser.add_argument("--xmax", type=float, default=5000.0, help="Histogram xmax")
    parser.add_argument("--nbins", type=int, default=60, help="Number of histogram bins")
    parser.add_argument("--include-alphas", action="store_true", help="Include alpha_s members if present")
    parser.add_argument("--max_events", type=int, default=None, help="Load all shards but keep only first N events (debug)")
    args = parser.parse_args()

    pdf = PDFParametrization(args.pdf_set, include_alphas_members=args.include_alphas)
    bins = np.linspace(args.xmin, args.xmax, args.nbins + 1)

    T = FeatureTemplate(
        pdf=pdf,
        sample=args.sample,
        feature=args.feature,
        bin_edges=bins,
        module_samples="data.samples_RunII",
        name=f"{args.sample}:{args.feature}",
        max_events=args.max_events,
    )

    print(f"[info] n_members = {T.n_members} (including central)")

    t0 = T.get_template(0)
    t1 = T.get_template(1)
    print("[info] template shapes:", t0.shape, t1.shape)
    print("[info] central sum:", float(np.sum(t0)))
    print("[info] var1   sum:", float(np.sum(t1)))

