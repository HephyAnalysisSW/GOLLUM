#!/usr/bin/env python3
"""
fit_toys_global_eq238.py

Global fit on toys built from cached Asimov events, using an Eq.(2.38)-style objective:

  -1/2 u(D|theta) ≈  - sum_{nominal sim} w0 * T(x;c,nu)
                     + sum_{toy draws}    w_toy * log1p( T(x;c,nu) )

So:
      u = 2*sum(w0*T_full) - 2*sum(w_toy*log1p(T_toy)) + penalty(theta)

This script supports two toy formats in the .npz:

  (A) old index-only toys:
      toy0000_<rid>_indices : (Ndraw,) int
      -> weights are assumed +1 for each draw.

  (B) signed-weight toys (for NLO negative weights / signed bootstrap):
      toy0000_<rid>_indices : (Ndraw,) int
      toy0000_<rid>_weights : (Ndraw,) float (can be +/-)
      -> duplicates are compressed by summing signed weights per unique index.

CLI is kept compatible with your original script:
  python3 user/kaan/fit_toys_global_eq238.py configs/unbinned_merged.yaml \
      /path/to/toys_*.npz --max-toys 20 --print-every 1

Optional:
  --no-syst   freeze all nuisances at 0 (faster)
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import os
import re
import time
import argparse
from typing import Dict, Any, Tuple

import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL


# -----------------------------------------------------------------------------
# Toy I/O: allow both index-only and (indices, weights) toys
# -----------------------------------------------------------------------------
_TOY_IDX_RE = re.compile(r"^toy(\d{4})_(.*)_indices$")
_TOY_WGT_RE = re.compile(r"^toy(\d{4})_(.*)_weights$")


def load_toys_npz(npz_path: str) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns:
      toys[itoy][rid] = (indices, weights)
    where
      indices: (Ndraw,) int
      weights: (Ndraw,) float (signed allowed). If absent, filled with +1.
    """
    z = np.load(npz_path, allow_pickle=False)
    # temporary structure: toys[itoy][rid] = {"idx":..., "w":...}
    tmp: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for key in z.files:
        m = _TOY_IDX_RE.match(key)
        if not m:
            continue
        itoy = int(m.group(1))
        rid = m.group(2)
        tmp.setdefault(itoy, {}).setdefault(rid, {})["idx"] = np.asarray(z[key], dtype=np.int64)

    for key in z.files:
        m = _TOY_WGT_RE.match(key)
        if not m:
            continue
        itoy = int(m.group(1))
        rid = m.group(2)
        tmp.setdefault(itoy, {}).setdefault(rid, {})["w"] = np.asarray(z[key], dtype=np.float64)

    if not tmp:
        raise RuntimeError(f"No toy keys matched '{_TOY_IDX_RE.pattern}' in {npz_path}")

    toys: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for itoy, by_rid in tmp.items():
        toys[itoy] = {}
        for rid, d in by_rid.items():
            if "idx" not in d:
                continue
            idx = d["idx"]
            w = d.get("w", None)
            if w is None:
                w = np.ones(idx.shape[0], dtype=np.float64)
            if w.shape[0] != idx.shape[0]:
                raise RuntimeError(
                    f"[toys] itoy={itoy} rid={rid}: indices and weights have different lengths "
                    f"({idx.shape[0]} vs {w.shape[0]})"
                )
            toys[itoy][rid] = (idx, w)

    return toys


# -----------------------------------------------------------------------------
# HDF5 -> numpy preloading (as in your original script)
# -----------------------------------------------------------------------------
class H5NumpyCache:
    def __init__(self):
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}

    def get(self, n2ll: N2LL, rid: str, cid: str, dset: str) -> np.ndarray:
        key = (rid, cid, dset)
        if key in self._cache:
            return self._cache[key]
        f = n2ll._h5[(rid, cid)]
        arr = f[dset][()]  # force numpy array
        self._cache[key] = arr
        return arr


def preload_full_arrays(n2ll: N2LL, region_ids: list[str]) -> tuple[dict, dict]:
    """
    Returns:
      full_by_region[rid][cid][dset] = numpy array (full length)
      w0_full[rid] = numpy array (full length), taken from first cid
    """
    cache = H5NumpyCache()
    full_by_region: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    w0_full: dict[str, np.ndarray] = {}

    t0 = time.time()
    total_bytes = 0

    for rid in region_ids:
        cids = n2ll._class_ids_by_region.get(rid, [])
        if not cids:
            continue

        full_by_region[rid] = {}

        # w0: take from first class (same convention as Likelihood.py)
        first_cid = cids[0]
        w0 = cache.get(n2ll, rid, first_cid, "w0")
        w0_full[rid] = w0
        total_bytes += w0.nbytes

        for cid in cids:
            comp: dict[str, np.ndarray] = {}
            g = cache.get(n2ll, rid, cid, "g")
            R = cache.get(n2ll, rid, cid, "R")
            comp["g"] = g
            comp["R"] = R
            total_bytes += g.nbytes + R.nbytes

            meta = n2ll._meta[(rid, cid)]
            for gm in meta.get("delta_groups", []):
                dset_name = gm.get("dset", f"Delta::{gm['id']}")
                D = cache.get(n2ll, rid, cid, dset_name)
                comp[dset_name] = D
                total_bytes += D.nbytes

            full_by_region[rid][cid] = comp

    dt = time.time() - t0
    print(f"[preload] done in {dt:.2f}s, total arrays ~{total_bytes/1024**3:.3f} GB")
    return full_by_region, w0_full


# -----------------------------------------------------------------------------
# Build toy slices once per toy: compress duplicates + sum signed weights
# -----------------------------------------------------------------------------
def build_toy_slices_for_region(
    rid: str,
    indices: np.ndarray,
    weights: np.ndarray,
    full_region: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """
    Compress duplicates:
      unique_idx, summed_weights

    Returns:
      toy_by_class[cid][dset] sliced to unique_idx
      w_toy: summed signed weights per unique index (float)
    """
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    toy_by_class: dict[str, dict[str, np.ndarray]] = {}
    if indices.size == 0:
        return toy_by_class, np.empty(0, dtype=np.float64)

    # Unique indices + inverse map
    unique_idx, inv = np.unique(indices, return_inverse=True)

    # Sum signed weights per unique index
    w_toy = np.zeros(unique_idx.shape[0], dtype=np.float64)
    np.add.at(w_toy, inv, weights)

    # Drop exact cancellations
    keep = (w_toy != 0.0)
    unique_idx = unique_idx[keep]
    w_toy = w_toy[keep]

    if unique_idx.size == 0:
        return toy_by_class, np.empty(0, dtype=np.float64)

    for cid, comp_full in full_region.items():
        comp_slice: dict[str, np.ndarray] = {}
        for dset_name, arr in comp_full.items():
            if arr.ndim == 1:
                comp_slice[dset_name] = arr[unique_idx]
            else:
                comp_slice[dset_name] = arr[unique_idx, :]
        toy_by_class[cid] = comp_slice

    return toy_by_class, w_toy


# -----------------------------------------------------------------------------
# Compute T from cached components (mirrors Likelihood.py logic)
# -----------------------------------------------------------------------------
def compute_T_from_comps(
    n2ll: N2LL,
    rid: str,
    hypothesis,
    comps_by_class: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Build T(x;c,nu) from cached arrays:
      T += g * ( (R@cA)*exp(expo+ln_bias) + (exp(expo+ln_bias)-1) )
    """
    if not comps_by_class:
        return np.empty(0, dtype=np.float64)

    # length determined from first available class
    any_cid = next(iter(comps_by_class.keys()))
    N = np.asarray(comps_by_class[any_cid]["g"]).shape[0]
    T = np.zeros(N, dtype=np.float64)

    # assemble cA, nuA
    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis)      # keyed by cid (string)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis)       # keyed by cid (string)

    nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, "parameters", []) if not p.isPOI}

    # lnN biases per class id (string)
    ln_bias: dict[str, float] = {}
    for cid in n2ll._class_ids_by_region.get(rid, []):
        ln_bias[cid] = sum(
            log1p_alpha * nu_vals.get(pname, 0.0)
            for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), [])
        )

    for cid, comp in comps_by_class.items():
        g = np.asarray(comp["g"], dtype=np.float64)
        R = np.asarray(comp["R"], dtype=np.float64)

        cA = cA_per_class[cid]
        c_dot_R = R @ cA  # (N,)

        expo = np.zeros(N, dtype=np.float64)
        group_list = nuA_per_group.get(cid, [])
        for gm, nuA in group_list:
            dset = gm.get("dset", f"Delta::{gm['id']}")
            dA = np.asarray(comp[dset], dtype=np.float64)
            expo += dA @ nuA

        exp_expo = np.exp(expo + ln_bias.get(cid, 0.0))
        T += g * (c_dot_R * exp_expo + (exp_expo - 1.0))

    return T


def log1p_safe(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # hard fail if any invalid, so Minuit doesn't wander into log domain errors
    if np.any(T <= -1.0 + eps):
        raise FloatingPointError("Encountered T <= -1 (log1p invalid).")
    return np.log1p(T)


# -----------------------------------------------------------------------------
# Minuit global fit for one toy, minimizing u(D|theta)
# -----------------------------------------------------------------------------
def free_params(hyp):
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def fit_one_toy(
    n2ll: N2LL,
    hyp_template,
    region_ids: list[str],
    full_by_region: dict,
    w0_full: dict,
    toy_slices: dict[str, dict[str, dict[str, np.ndarray]]],
    toy_weights: dict[str, np.ndarray],
    *,
    step: float = 0.1,
    print_level: int = 0,
    strategy: int = 1,
    tol: float | None = None,
) -> tuple[float, dict[str, float]]:
    hyp = hyp_template.clone()

    pars = free_params(hyp)
    if not pars:
        raise RuntimeError("No free parameters to fit.")

    names = [p.name for p in pars]
    x0 = [float(p.val) for p in pars]

    def fcn(*x):
        for i, p in enumerate(pars):
            p.val = float(x[i])

        sum_w0T = 0.0
        sum_log = 0.0

        try:
            for rid in region_ids:
                # full term
                comps_full = full_by_region[rid]
                T_full = compute_T_from_comps(n2ll, rid, hyp, comps_full)
                w0 = w0_full[rid]
                sum_w0T += float(np.sum(w0 * T_full, dtype=np.float64))

                # toy term (signed weights allowed)
                comps_toy = toy_slices.get(rid, {})
                w_toy = toy_weights.get(rid, np.empty(0, dtype=np.float64))
                if w_toy.size == 0:
                    continue

                T_toy = compute_T_from_comps(n2ll, rid, hyp, comps_toy)
                sum_log += float(np.sum(w_toy * log1p_safe(T_toy), dtype=np.float64))

        except FloatingPointError:
            return 1e100
        except Exception:
            return 1e100

        u = 2.0 * sum_w0T - 2.0 * sum_log + float(hyp.penalty())
        if not np.isfinite(u):
            return 1e100
        return u

    m = Minuit(fcn, *x0, name=names)
    m.errordef = 1.0
    m.print_level = int(print_level)
    m.strategy = int(strategy)
    if tol is not None:
        m.tol = float(tol)

    for i in range(len(names)):
        m.errors[i] = float(step)

    m.migrad()

    for i, p in enumerate(pars):
        p.val = float(m.values[i])

    c_hats = {p.name: float(p.val) for p in hyp.parameters if p.isPOI and p.name.startswith("c")}
    return float(m.fval), c_hats


# -----------------------------------------------------------------------------
# Main (CLI kept compatible)
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Global fit on toys using Eq.(2.38)-style objective.")
    ap.add_argument("config", help="YAML config")
    ap.add_argument("toys_npz", help="toys_*.npz from toy generation (indices-only or indices+weights)")
    ap.add_argument("--version", default=None, help="Cache version string")
    ap.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache (slow)")
    ap.add_argument("--max-toys", type=int, default=None)
    ap.add_argument("--toy-number", type=int, default=None)
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--minuit-step", type=float, default=0.1)
    ap.add_argument("--minuit-print-level", type=int, default=0)
    ap.add_argument("--minuit-strategy", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--minuit-tol", type=float, default=None)
    ap.add_argument("--no-syst", action="store_true", help="Freeze all nuisances at 0 (faster)")
    ap.add_argument("--out", default=None, help="Output npz")
    args = ap.parse_args()

    # load cfg + surrogates
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    Likelihood.cfg = cfg
    like_info = load_likelihood(cfg)
    hyp0 = build_hypothesis_from_likelihood(like_info, name="SR")

    if args.no_syst:
        for p in hyp0.parameters:
            if not p.isPOI:
                p.val = 0.0
                p.isFrozen = True
        print("[opts] --no-syst: all nuisances frozen at 0")

    # setup N2LL (cache access + helper methods)
    base = os.path.splitext(os.path.basename(args.config))[0]
    version = args.version or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)

    n2ll = N2LL(
        likelihood=like_info,
        module_samples="data.samples",
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=args.overwrite_cache,
    )
    n2ll.build_cache()
    n2ll.prepare_runtime()

    region_ids = [R["id"] for R in n2ll.regions]
    print("[regions]", region_ids)

    # load toys
    toys = load_toys_npz(args.toys_npz)
    toy_ids = sorted(toys.keys())
    if args.max_toys is not None:
        toy_ids = toy_ids[:args.max_toys]
    elif args.toy_number is not None:
        toy_ids = toy_ids[args.toy_number:args.toy_number+1]
    print(f"[toys] file={args.toys_npz}  n={len(toy_ids)}")

    # preload full arrays into RAM
    full_by_region, w0_full = preload_full_arrays(n2ll, region_ids)

    # outputs
    fvals = []
    chat_list = []
    c_names = sorted([p.name for p in hyp0.parameters if p.isPOI and p.name.startswith("c")])

    t0 = time.time()
    for k, itoy in enumerate(toy_ids, start=1):
        toy_slices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        toy_w: dict[str, np.ndarray] = {}
        n_draws = 0

        for rid in region_ids:
            idx, w = toys[itoy].get(rid, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
            n_draws += int(idx.size)

            comp_toy, w_toy = build_toy_slices_for_region(
                rid, idx, w, full_by_region[rid]
            )
            toy_slices[rid] = comp_toy
            toy_w[rid] = w_toy

        # fit
        fval, c_hats = fit_one_toy(
            n2ll, hyp0, region_ids,
            full_by_region, w0_full,
            toy_slices, toy_w,
            step=args.minuit_step,
            print_level=args.minuit_print_level,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
        )

        fvals.append(fval)
        chat_list.append([c_hats.get(nm, np.nan) for nm in c_names])

        if args.print_every > 0 and (k % args.print_every == 0):
            dt = time.time() - t0
            rate = k / dt if dt > 0 else float("inf")
            print(f"[toy {itoy:04d}] k={k:4d}/{len(toy_ids)}  Ndraw={n_draws:7d}  u_min={fval:.6f}  ({rate:.2f} toys/s)")
            print("  c_hat:", {nm: c_hats.get(nm, np.nan) for nm in c_names})

    fvals = np.asarray(fvals, dtype=np.float64)
    chat = np.asarray(chat_list, dtype=np.float64)

    out = args.out or (os.path.splitext(args.toys_npz)[0] + "__eq238_globalfit.npz")
    np.savez(
        out,
        config=np.asarray([args.config]),
        toys_npz=np.asarray([args.toys_npz]),
        toy_ids=np.asarray(toy_ids, dtype=np.int64),
        region_ids=np.asarray(region_ids),
        c_names=np.asarray(c_names),
        u_min=fvals,
        c_hat=chat,
    )
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
