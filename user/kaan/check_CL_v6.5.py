#!/usr/bin/env python3
"""
check_CL_v6.5.py

Global fit on toys using the N2LL methods added to Likelihood.py:

  n2ll.preload_unbinned_numpy()
  n2ll.setToyFromIndicesAndWeights(toy_idx_by_region, toy_w_by_region)
  n2ll.n2ll_toy(hypothesis)   # (or whatever you named it)

Toy NPZ formats:
  toy0000_<rid>_indices : (Ndraw,) int
  toy0000_<rid>_weights : (Ndraw,) float  (signed allowed)
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import os, re, time, argparse
import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL

_TOY_IDX_RE = re.compile(r"^toy(\d{4})_(.*)_indices$")
_TOY_WGT_RE = re.compile(r"^toy(\d{4})_(.*)_weights$")


def load_toys_npz(npz_path):
    """
    returns toys[itoy][rid] = (idx, w)
      idx: np.ndarray[int] shape (Ndraw,)
      w  : np.ndarray[float] shape (Ndraw,)  (must exist)
    """
    z = np.load(npz_path, allow_pickle=False)
    tmp = {}  # tmp[itoy][rid]["idx"/"w"] = array

    for key in z.files:
        m = _TOY_IDX_RE.match(key)
        if m:
            itoy = int(m.group(1)); rid = m.group(2)
            tmp.setdefault(itoy, {}).setdefault(rid, {})["idx"] = np.asarray(z[key], dtype=np.int64)

    for key in z.files:
        m = _TOY_WGT_RE.match(key)
        if m:
            itoy = int(m.group(1)); rid = m.group(2)
            tmp.setdefault(itoy, {}).setdefault(rid, {})["w"] = np.asarray(z[key], dtype=np.float64)

    if not tmp:
        raise RuntimeError(f"No toy keys matched {_TOY_IDX_RE.pattern} in {npz_path}")

    toys = {}  # toys[itoy][rid] = (idx, w)
    for itoy, by_rid in tmp.items():
        toys[itoy] = {}
        for rid, d in by_rid.items():
            if "idx" not in d or "w" not in d:
                raise RuntimeError(f"[toys] itoy={itoy} rid={rid}: both indices and weights must exist in npz")
            idx = d["idx"]; w = d["w"]
            if idx.shape[0] != w.shape[0]:
                raise RuntimeError(f"[toys] itoy={itoy} rid={rid}: len(idx)={idx.size} != len(w)={w.size}")
            toys[itoy][rid] = (idx, w)

    return toys


def free_params(hyp):
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def fit_one_toy(n2ll, hyp_template, toy_idx_by_region, toy_w_by_region, step=0.1, print_level=0, strategy=1, tol=None):
    """
    returns (n2ll_min, c_hat_dict)
    """
    hyp = hyp_template.clone()

    pars = free_params(hyp)
    if not pars:
        raise RuntimeError("No free parameters to fit.")

    names = [p.name for p in pars]
    x0 = [float(p.val) for p in pars]

    # set toy once, then FCN only updates parameters and calls Likelihood.py evaluator
    n2ll.setToyFromIndicesAndWeights(toy_idx_by_region, toy_w_by_region)

    def fcn(*x):
        for i, p in enumerate(pars):
            p.val = float(x[i])
        # use the method you added in Likelihood.py
        return float(n2ll.n2ll_toy(hyp))

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

    c_hat = {p.name: float(p.val) for p in hyp.parameters if p.isPOI and p.name.startswith("c")}
    return float(m.fval), c_hat


def main():
    ap = argparse.ArgumentParser(description="Global fit on toys using Likelihood.py cached evaluator.")
    ap.add_argument("config", help="YAML config")
    ap.add_argument("toys_npz", help="toys_*.npz (must contain indices+weights)")
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

    toys = load_toys_npz(args.toys_npz)
    toy_ids = sorted(toys.keys())
    if args.max_toys is not None:
        toy_ids = toy_ids[:args.max_toys]
    elif args.toy_number is not None:
        toy_ids = toy_ids[args.toy_number:args.toy_number + 1]
    print(f"[toys] file={args.toys_npz}  n={len(toy_ids)}")

    # preload H5 arrays into RAM once (this is the whole point)
    n2ll.preload_unbinned_numpy()

    c_names = sorted([p.name for p in hyp0.parameters if p.isPOI and p.name.startswith("c")])
    fvals = []
    chat_list = []

    t0 = time.time()
    for k, itoy in enumerate(toy_ids, start=1):
        toy_idx_by_region = {}
        toy_w_by_region = {}
        n_draws = 0

        for rid in region_ids:
            idx, w = toys[itoy].get(rid, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
            toy_idx_by_region[rid] = idx
            toy_w_by_region[rid] = w
            n_draws += int(idx.size)

        fval, c_hat = fit_one_toy(
            n2ll, hyp0,
            toy_idx_by_region, toy_w_by_region,
            step=args.minuit_step,
            print_level=args.minuit_print_level,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
        )

        fvals.append(fval)
        chat_list.append([c_hat.get(nm, np.nan) for nm in c_names])

        if args.print_every > 0 and (k % args.print_every == 0):
            dt = time.time() - t0
            rate = k / dt if dt > 0 else float("inf")
            print(f"[toy {itoy:04d}] k={k:4d}/{len(toy_ids)}  Ndraw={n_draws:7d}  n2ll_min={fval:.6f}  ({rate:.2f} toys/s)")
            print("  c_hat:", {nm: c_hat.get(nm, np.nan) for nm in c_names})

    out = args.out or (os.path.splitext(args.toys_npz)[0] + "_globalfit.npz")
    np.savez(
        out,
        config=np.asarray([args.config]),
        toys_npz=np.asarray([args.toys_npz]),
        toy_ids=np.asarray(toy_ids, dtype=np.int64),
        region_ids=np.asarray(region_ids),
        c_names=np.asarray(c_names),
        n2ll_min=np.asarray(fvals, dtype=np.float64),
        c_hat=np.asarray(chat_list, dtype=np.float64),
    )
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
