#!/usr/bin/env python3
"""
fit_toys_global_eq238.py

Global fit on index-only toys using Eq. (2.38):

  -1/2 u(D|theta) ≈  - sum_{nominal sim} w0 * T(x;theta)
                    + sum_{toy events}   log1p( T(x;theta) )

So:
      u = 2*sum(w0*T_full) - 2*sum(w_toy*log1p(T_toy)) + penalty(theta)

This is what you want for toys sampled from lambda_i = w0_i*(1 + T_i(theta_inj)).

Usage:
  python3 user/kaan/fit_toys_global_eq238.py configs/unbinned_merged.yaml \
    /path/to/toys_c1_0.2_mode-poisson_N100.npz \
    --max-toys 20 --print-every 1

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
from typing import Dict, Any

import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL


# ------------------------------
# Toy loading: toy0000_<rid>_indices
# ------------------------------
_TOY_KEY_RE = re.compile(r"^toy(\d{4})_(.*)_indices$")


def load_toys_indices_npz(npz_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    z = np.load(npz_path, allow_pickle=False)
    toys: Dict[int, Dict[str, np.ndarray]] = {}
    for key in z.files:
        m = _TOY_KEY_RE.match(key)
        if not m:
            continue
        itoy = int(m.group(1))
        rid = m.group(2)
        toys.setdefault(itoy, {})[rid] = np.asarray(z[key], dtype=np.int64)
    if not toys:
        raise RuntimeError(f"No toy keys matched '{_TOY_KEY_RE.pattern}' in {npz_path}")
    return toys


# ------------------------------
# Numpy cache: f[dset][()] once
# ------------------------------
class H5NumpyCache:
    def __init__(self):
        self._cache: dict[tuple[str, int, str], np.ndarray] = {}

    def get(self, n2ll: N2LL, rid: str, cid: int, dset: str) -> np.ndarray:
        key = (rid, cid, dset)
        if key in self._cache:
            return self._cache[key]
        f = n2ll._h5[(rid, cid)]
        arr = f[dset][()]  # <-- forces numpy array (what you asked)
        self._cache[key] = arr
        return arr


def preload_full_arrays(n2ll: N2LL, region_ids: list[str]) -> tuple[dict, dict]:
    """
    Returns:
      full_by_region[rid][cid][dset] = numpy array (full length)
      w0_full[rid] = numpy array (full length), taken from first cid
    """
    cache = H5NumpyCache()
    full_by_region: dict[str, dict[int, dict[str, np.ndarray]]] = {}
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


# ------------------------------
# Build toy slices once per toy
# ------------------------------
def build_toy_slices_for_region(
    n2ll: N2LL,
    rid: str,
    indices: np.ndarray,
    full_region: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    """
    Compress duplicates:
      unique_idx, counts
    Return:
      toy_by_class[cid][dset] sliced to unique_idx
      w_toy = counts (float)
    """
    indices = np.asarray(indices, dtype=np.int64)
    toy_by_class: dict[int, dict[str, np.ndarray]] = {}

    if indices.size == 0:
        return toy_by_class, np.empty(0, dtype=np.float64)

    unique_idx, counts = np.unique(indices, return_counts=True)
    w_toy = counts.astype(np.float64)

    for cid, comp_full in full_region.items():
        comp_slice: dict[str, np.ndarray] = {}
        for dset_name, arr in comp_full.items():
            if arr.ndim == 1:
                comp_slice[dset_name] = arr[unique_idx]
            else:
                comp_slice[dset_name] = arr[unique_idx, :]
        toy_by_class[cid] = comp_slice

    return toy_by_class, w_toy


# ------------------------------
# Compute T from (g,R,Delta) comps (same formula as Likelihood.py)
# ------------------------------
def compute_T_from_comps(n2ll: N2LL, rid: str, hypothesis, comps_by_class: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """
    Reproduce the T construction from Likelihood.py observation mode:
      T += g * ( (R@cA)*exp(expo+ln_bias) + (exp(expo+ln_bias)-1) )
    """
    # length N determined from first available class
    any_cid = next(iter(comps_by_class.keys()))
    g0 = comps_by_class[any_cid]["g"]
    N = g0.shape[0]
    T = np.zeros(N, dtype=np.float64)

    # assemble cA, nuA
    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis)
    nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, "parameters", []) if not p.isPOI}

    # lnN biases per class
    ln_bias = {
        cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                 for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, str(cid)), []))
        for cid in comps_by_class.keys()
    }
    # NOTE: in Likelihood.py, class ids are strings in many places.
    # Here cids in n2ll._class_ids_by_region are strings; our dict uses ints if your ids are ints.
    # Safer: build ln_bias by iterating over n2ll's class ids directly:
    ln_bias = {}
    for cid_str in n2ll._class_ids_by_region.get(rid, []):
        # map cid_str to our key type
        try:
            cid_key = int(cid_str) if isinstance(next(iter(comps_by_class.keys())), int) else cid_str
        except Exception:
            cid_key = cid_str
        ln_bias[cid_key] = sum(
            log1p_alpha * nu_vals.get(pname, 0.0)
            for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid_str), [])
        )

    for cid_key, comp in comps_by_class.items():
        g = np.asarray(comp["g"], dtype=np.float64)
        R = np.asarray(comp["R"], dtype=np.float64)
        cA = cA_per_class[str(cid_key)] if str(cid_key) in cA_per_class else cA_per_class[cid_key]
        c_dot_R = R @ cA  # (N,)

        expo = np.zeros(N, dtype=np.float64)
        # nuA_per_group keyed by class id string in Likelihood.py
        group_list = nuA_per_group.get(str(cid_key), nuA_per_group.get(cid_key, []))
        for gm, nuA in group_list:
            dset = gm.get("dset", f"Delta::{gm['id']}")
            dA = np.asarray(comp[dset], dtype=np.float64)
            expo += dA @ nuA

        exp_expo = np.exp(expo + ln_bias[cid_key])
        T += g * (c_dot_R * exp_expo + (exp_expo - 1.0))

    return T


def log1p_safe(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # hard fail if any invalid, so Minuit doesn't wander into log domain errors
    if np.any(T <= -1.0 + eps):
        raise FloatingPointError("Encountered T <= -1 (log1p invalid).")
    return np.log1p(T)


# ------------------------------
# Minuit global fit for one toy, minimizing u(D|theta)
# ------------------------------
def free_params(hyp):
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def fit_one_toy(
    n2ll: N2LL,
    hyp_template,
    region_ids: list[str],
    full_by_region: dict,
    w0_full: dict,
    toy_slices: dict[str, dict[int, dict[str, np.ndarray]]],
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
        # set params
        for i, p in enumerate(pars):
            p.val = float(x[i])

        # compute Eq.(2.38) objective: u = 2*sum(w0*T_full) - 2*sum(w_toy*log1p(T_toy)) + penalty
        sum_w0T = 0.0
        sum_log = 0.0

        try:
            for rid in region_ids:
                # full term
                comps_full = full_by_region[rid]
                T_full = compute_T_from_comps(n2ll, rid, hyp, comps_full)
                w0 = w0_full[rid]
                sum_w0T += float(np.sum(w0 * T_full, dtype=np.float64))

                # toy term
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

    # push best-fit back
    for i, p in enumerate(pars):
        p.val = float(m.values[i])

    # snapshot c* hats
    c_hats = {p.name: float(p.val) for p in hyp.parameters if p.isPOI and p.name.startswith("c")}

    return float(m.fval), c_hats


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Global fit on toys using Eq.(2.38) objective.")
    ap.add_argument("config", help="YAML config")
    ap.add_argument("toys_npz", help="toys_*.npz from generate_toys_v3.py")
    ap.add_argument("--version", default=None, help="Cache version string")
    ap.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache (slow)")
    ap.add_argument("--max-toys", type=int, default=None)
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

    # setup N2LL (only for helper methods + cache access)
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
    toys = load_toys_indices_npz(args.toys_npz)
    toy_ids = sorted(toys.keys())
    if args.max_toys is not None:
        toy_ids = toy_ids[:args.max_toys]
    print(f"[toys] file={args.toys_npz}  n={len(toy_ids)}")

    # preload all arrays into RAM
    full_by_region, w0_full = preload_full_arrays(n2ll, region_ids)

    # outputs
    fvals = []
    chat_list = []
    c_names = sorted([p.name for p in hyp0.parameters if p.isPOI and p.name.startswith("c")])

    t0 = time.time()
    for k, itoy in enumerate(toy_ids, start=1):
        # build per-toy slices once
        toy_slices: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        toy_w: dict[str, np.ndarray] = {}
        n_ev = 0

        for rid in region_ids:
            idx = toys[itoy].get(rid, np.empty(0, dtype=np.int64))
            n_ev += int(idx.size)
            comp_toy, w_toy = build_toy_slices_for_region(n2ll, rid, idx, full_by_region[rid])
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
            print(f"[toy {itoy:04d}] k={k:4d}/{len(toy_ids)}  N={n_ev:7d}  u_min={fval:.6f}  ({rate:.2f} toys/s)")
            print("  c_hat:", {nm: c_hats.get(nm, np.nan) for nm in c_names})

    fvals = np.asarray(fvals, dtype=np.float64)
    chat = np.asarray(chat_list, dtype=np.float64)

    # save
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
