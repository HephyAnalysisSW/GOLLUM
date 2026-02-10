#!/usr/bin/env python3
import sys
from pathlib import Path

# Make project root importable when running from user/kaan/
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import os
import re
import time
import argparse
from typing import Dict, Tuple, Any

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
# HDF5 -> NumPy cache
# Uses f[dset][()] once, then fast NumPy indexing
# ------------------------------
class H5NumpyCache:
    def __init__(self):
        self._cache: dict[tuple[str, int, str], np.ndarray] = {}

    def get(self, n2ll: N2LL, rid: str, cid: int, dset: str) -> np.ndarray:
        key = (rid, cid, dset)
        if key in self._cache:
            return self._cache[key]
        f = n2ll._h5[(rid, cid)]
        arr = f[dset][()]  # <-- forces a NumPy array (your preferred pattern)
        self._cache[key] = arr
        return arr

    def preload_all(self, n2ll: N2LL, region_ids: list[str]) -> None:
        t0 = time.time()
        total_bytes = 0
        for rid in region_ids:
            for cid in n2ll._class_ids_by_region[rid]:
                # Always needed:
                for dset in ("g", "R"):
                    a = self.get(n2ll, rid, cid, dset)
                    total_bytes += a.nbytes

                # Needed for profiling nuisances:
                meta = n2ll._meta[(rid, cid)]
                for gm in meta.get("delta_groups", []):
                    dset_name = gm.get("dset", f"Delta::{gm['id']}")
                    a = self.get(n2ll, rid, cid, dset_name)
                    total_bytes += a.nbytes

        dt = time.time() - t0
        print(f"[preload] Loaded cache arrays into RAM in {dt:.2f}s, total ~{total_bytes/1024**3:.3f} GB")


# ------------------------------
# Build observation blocks from indices (compressed -> unique_idx + multiplicities)
# ------------------------------
def build_by_class_slices_compressed(
    n2ll: N2LL,
    cache: H5NumpyCache,
    rid: str,
    indices: np.ndarray
) -> Tuple[dict, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    by_class: Dict[int, dict] = {}

    if indices.size == 0:
        # empty region toy
        w = np.empty(0, dtype=np.float64)
        for cid in n2ll._class_ids_by_region[rid]:
            # use dataset shapes cheaply
            f = n2ll._h5[(rid, cid)]
            nA = f["R"].shape[1]
            comp = {"g": np.empty(0, np.float64), "R": np.empty((0, nA), np.float64)}
            meta = n2ll._meta[(rid, cid)]
            for gm in meta.get("delta_groups", []):
                dset_name = gm.get("dset", f"Delta::{gm['id']}")
                nB = f[dset_name].shape[1]
                comp[dset_name] = np.empty((0, nB), np.float64)
            by_class[cid] = comp
        return by_class, w

    unique_idx, counts = np.unique(indices, return_counts=True)
    w = counts.astype(np.float64)

    for cid in n2ll._class_ids_by_region[rid]:
        comp: Dict[str, np.ndarray] = {}

        g_all = cache.get(n2ll, rid, cid, "g")
        R_all = cache.get(n2ll, rid, cid, "R")
        comp["g"] = g_all[unique_idx]
        comp["R"] = R_all[unique_idx, :]

        meta = n2ll._meta[(rid, cid)]
        for gm in meta.get("delta_groups", []):
            dset_name = gm.get("dset", f"Delta::{gm['id']}")
            D_all = cache.get(n2ll, rid, cid, dset_name)
            comp[dset_name] = D_all[unique_idx, :]

        by_class[cid] = comp

    return by_class, w


def set_observation_from_toy(n2ll: N2LL, toy_blocks: Dict[str, Dict[str, Any]]) -> None:
    # disable Asimov
    n2ll._asimov_hypothesis_set = False
    n2ll._asimov_active = False
    n2ll._asimov_hyp = None
    n2ll._asimov_T.clear()
    n2ll._binned_asimov_lambda.clear()

    # set observation mode
    n2ll._obs_unbinned = {}
    n2ll._obs_binned = {}
    n2ll._observation_set = True

    for rid, block in toy_blocks.items():
        n2ll._obs_unbinned[rid] = {
            "by_class": block["by_class"],
            "w": np.asarray(block["w"], dtype=np.float64),
        }


# ------------------------------
# Minuit: global fit of ALL free parameters
# ------------------------------
def free_params(hyp):
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def minimize_global(n2ll: N2LL, hyp, *, step=0.1, print_level=0, strategy=1, tol=None) -> float:
    pars = free_params(hyp)
    if not pars:
        return float(n2ll(hyp))

    names = [p.name for p in pars]
    x0 = [float(p.val) for p in pars]

    def fcn(*x):
        for i, p in enumerate(pars):
            p.val = float(x[i])
        v = float(n2ll(hyp))
        # avoid NaN thrashing
        if not np.isfinite(v):
            return 1e100
        return v

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

    return float(m.fval)


def c_poi_names(hyp) -> list[str]:
    return sorted([p.name for p in hyp.parameters if getattr(p, "isPOI", False) and p.name.startswith("c")])


def snapshot(hyp, names: list[str]) -> list[float]:
    vals = []
    for n in names:
        for p in hyp.parameters:
            if p.name == n:
                vals.append(float(p.val))
                break
        else:
            vals.append(np.nan)
    return vals


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Global fit on toy datasets (indices only).")
    ap.add_argument("config", help="YAML config")
    ap.add_argument("toys_npz", help="toys_*.npz from generate_toys_v3.py")
    ap.add_argument("--version", default=None, help="Cache version name")
    ap.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache (slow)")
    ap.add_argument("--max-toys", type=int, default=None, help="Evaluate only first N toys")
    ap.add_argument("--print-every", type=int, default=1, help="Print every N toys")
    ap.add_argument("--preload", action="store_true", help="Preload g/R/Delta arrays into RAM (recommended)")
    ap.add_argument("--minuit-step", type=float, default=0.1)
    ap.add_argument("--minuit-print-level", type=int, default=0)
    ap.add_argument("--minuit-strategy", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--minuit-tol", type=float, default=None)
    ap.add_argument("--out", default=None, help="Output npz (default next to toys file)")
    args = ap.parse_args()

    # load config + surrogates
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    Likelihood.cfg = cfg
    like_info = load_likelihood(cfg)
    hyp0 = build_hypothesis_from_likelihood(like_info, name="SR")

    # N2LL setup
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

    # POI list
    c_names = c_poi_names(hyp0)
    print("[POIs c*]", c_names)

    # cache
    cache = H5NumpyCache()
    if args.preload:
        cache.preload_all(n2ll, region_ids)

    # outputs
    nll_list = []
    c_hat_mat = []

    t0 = time.time()
    for k, itoy in enumerate(toy_ids, start=1):
        # build toy observation blocks
        toy_blocks = {}
        n_ev = 0
        for rid in region_ids:
            idx = toys[itoy].get(rid, np.empty(0, dtype=np.int64))
            n_ev += int(idx.size)
            by_class, w = build_by_class_slices_compressed(n2ll, cache, rid, idx)
            toy_blocks[rid] = {"by_class": by_class, "w": w}

        set_observation_from_toy(n2ll, toy_blocks)

        # global fit from a fresh clone (starting values nominal each time)
        hyp = hyp0.clone()
        nll_min = minimize_global(
            n2ll, hyp,
            step=args.minuit_step,
            print_level=args.minuit_print_level,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
        )
        c_hat = snapshot(hyp, c_names)

        nll_list.append(nll_min)
        c_hat_mat.append(c_hat)

        if args.print_every > 0 and (k % args.print_every == 0):
            dt = time.time() - t0
            rate = k / dt if dt > 0 else float("inf")
            print(f"[toy {itoy:04d}] k={k:4d}/{len(toy_ids)}  N={n_ev:7d}  N2LLmin={nll_min:.6f}  ({rate:.2f} toys/s)")
            print("  c_hat:", {nm: v for nm, v in zip(c_names, c_hat)})

    nll_arr = np.asarray(nll_list, dtype=np.float64)
    c_hat_arr = np.asarray(c_hat_mat, dtype=np.float64)

    print("\n[summary]")
    print(f"  toys: {len(toy_ids)}")
    print(f"  N2LLmin: mean={nll_arr.mean():.6g} std={nll_arr.std(ddof=1) if nll_arr.size>1 else 0.0:.6g}")
    for j, nm in enumerate(c_names):
        col = c_hat_arr[:, j]
        print(f"  {nm}: mean={np.nanmean(col):.6g} std={np.nanstd(col, ddof=1) if col.size>1 else 0.0:.6g}")

    # save
    out = args.out
    if out is None:
        out = os.path.splitext(args.toys_npz)[0] + "__globalfit.npz"

    np.savez(
        out,
        config=np.asarray([args.config]),
        toys_npz=np.asarray([args.toys_npz]),
        toy_ids=np.asarray(toy_ids, dtype=np.int64),
        region_ids=np.asarray(region_ids),
        c_names=np.asarray(c_names),
        n2ll_min=nll_arr,
        c_hat=c_hat_arr,
    )
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
