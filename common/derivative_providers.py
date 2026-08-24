"""Derivative providers for PDF and EFT coefficients.

Shared by the BIT modification/closure plots (``plot/bit/*_modification_plot.py``) and
the BIT calibration checks (``ML/Calibration/*_calibration.py``). A provider evaluates
a truth (weighted) derivative matrix for a BIT job from loader-materialized arrays, and
exposes the combination/observer bookkeeping needed to do so.

A provider exposes:
  - ``combinations``       : list of canonical derivative tuples, native column order,
                             including the nominal ``()``.
  - ``parameters``         : list of coefficient / operator names.
  - ``required_observers`` : observer branch names the provider needs.
  - ``truth_weight_matrix(G, w, observer_names)`` : returns an ``(N, M)`` matrix of
                             truth weights aligned to ``combinations`` (column 0 ==
                             nominal weight).
  - ``expansion_point``    : dict, the point the coefficients in ``parameters`` are
                             expanded around ({} for PDF, which expands around zero;
                             the generation point r for EFT, see EFTWeightInterface).
"""

from __future__ import annotations

import numpy as np

from pdf.PDFParametrization import PDFParametrization
from eft.EFTWeightInterface import EFTWeightInterface


def canonical_combination(comb) -> tuple:
    """Canonical, hashable key for a derivative combination (sorted if length>=2)."""
    comb = tuple(comb)
    if len(comb) <= 1:
        return comb
    return tuple(sorted(comb))


class PDFDerivativeProvider:
    """PDF derivative provider: evaluates the parametrization from generator info."""

    GEN_OBSERVERS = [
        "Generator_x1",
        "Generator_x2",
        "Generator_id1",
        "Generator_id2",
        "Generator_scalePDF",
    ]

    def __init__(self, pdf_cfg):
        self._pdf = PDFParametrization(
            n=pdf_cfg.get("pdf_n"),
            typ=pdf_cfg.get("pdf_type"),
            basis=pdf_cfg.get("pdf_basis"),
            rescale_pod_amplitudes=pdf_cfg.get("rescale_pod_amplitudes", True),
        )
        self.combinations = [canonical_combination(c) for c in self._pdf.combinations]
        self.parameters = [c[0] for c in self.combinations if len(c) == 1]
        self.required_observers = list(self.GEN_OBSERVERS)
        self.expansion_point = {}

    def truth_weight_matrix(self, G, w, observer_names):
        idx = {name: i for i, name in enumerate(observer_names)}
        missing = [name for name in self.GEN_OBSERVERS if name not in idx]
        if missing:
            raise RuntimeError(f"observer_names missing generator branches: {missing}")

        Q = G[:, idx["Generator_scalePDF"]].astype(np.float32, copy=False)
        x1 = G[:, idx["Generator_x1"]].astype(np.float32, copy=False)
        x2 = G[:, idx["Generator_x2"]].astype(np.float32, copy=False)
        id1 = G[:, idx["Generator_id1"]].astype(np.int32, copy=False)
        id2 = G[:, idx["Generator_id2"]].astype(np.int32, copy=False)

        deriv = self._pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q).astype(np.float32, copy=False)
        return deriv * w.reshape(-1, 1)


class EFTDerivativeProvider:
    """EFT derivative provider: combines precomputed derivative branches."""

    def __init__(self, parameters):
        self._eft = EFTWeightInterface(parameters)
        self.parameters = list(parameters)
        self.combinations = [canonical_combination(c) for c in self._eft.combinations]
        self.required_observers = list(self._eft.required_observers)
        self.expansion_point = self._eft.reference_point

    def truth_weight_matrix(self, G, w, observer_names):
        return self._eft.make_weight_matrix(G, observer_names, w)


def build_derivative_provider(job):
    """Build the PDF or EFT derivative provider for a BIT job, dispatching on its config block."""
    if job.get("pdf"):
        return PDFDerivativeProvider(job["pdf"])
    if job.get("eft"):
        return EFTDerivativeProvider(job["eft"].get("parameters", []))

    RuntimeWarning(f"Job '{job['id']}' has neither a 'pdf' nor an 'eft' block. Returning None.")
