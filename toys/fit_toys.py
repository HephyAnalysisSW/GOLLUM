#!/usr/bin/env python3
"""
fit_toys.npy

Fit on toys using the N2LL methods added to Likelihood.py:

  n2ll.preload_unbinned_numpy()
  n2ll.setToyFromIndicesAndWeights(toy_idx_by_region, toy_w_by_region)
  n2ll.n2ll_toy(hypothesis)

Toy NPZ formats:
  toy0000_<rid>_indices : (Ndraw,) int
  toy0000_<rid>_weights : (Ndraw,) float  (signed)

Warning:
For some reason minuit uses all the cores, but the performance does not increase.
export OMP_NUM_THREADS=1 if you are using the login nodes.


Usage:
export OMP_NUM_THREADS=1
python3 -u user/kaan/fit_toys.py configs/unbinned/unbinned_2016APV.yaml /scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_shape_2_1.0_N100.npz --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_2016APV.json --toy-number 0 --print-every 1 --minuit-print-level 2

"""

import sys
import importlib
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import os, re, time, argparse
import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list
import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL
from fit.Modeling import Rotated

_TOY_IDX_RE = re.compile(r"^toy(\d{4})_(.*)_indices$")
_TOY_WGT_RE = re.compile(r"^toy(\d{4})_(.*)_weights$")


def list_toy_ids_npz(npz_path):
    """
    Read only archive metadata and return available toy ids.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        toy_ids = sorted(
            {
                int(m.group(1))
                for key in z.files
                for m in [_TOY_IDX_RE.match(key)]
                if m
            }
        )
    if not toy_ids:
        raise RuntimeError(f"No toy keys matched {_TOY_IDX_RE.pattern} in {npz_path}")
    return toy_ids


def load_toys_npz(npz_path, toy_number):
    """
    Load exactly one toy from the NPZ file.

    Returns:
      toys[rid] = (idx, w)
    """
    if toy_number is None:
        raise ValueError("load_toys_npz requires a specific toy_number")

    toy_number = int(toy_number)
    with np.load(npz_path, allow_pickle=False) as z:
        tmp = {}  # tmp[rid]["idx"/"w"] = array

        for key in z.files:
            m = _TOY_IDX_RE.match(key)
            if not m or int(m.group(1)) != toy_number:
                continue
            rid = m.group(2)
            tmp.setdefault(rid, {})["idx"] = np.asarray(z[key], dtype=np.int64)

        for key in z.files:
            m = _TOY_WGT_RE.match(key)
            if not m or int(m.group(1)) != toy_number:
                continue
            rid = m.group(2)
            tmp.setdefault(rid, {})["w"] = np.asarray(z[key], dtype=np.float64)

    if not tmp:
        raise RuntimeError(f"[toys] toy_number={toy_number}: no matching toy found in {npz_path}")

    toys = {}  # toys[rid] = (idx, w)
    for rid, d in tmp.items():
        if "idx" not in d or "w" not in d:
            raise RuntimeError(f"[toys] toy_number={toy_number} rid={rid}: both indices and weights must exist in npz")
        idx = d["idx"]
        w = d["w"]
        if idx.shape[0] != w.shape[0]:
            raise RuntimeError(f"[toys] toy_number={toy_number} rid={rid}: len(idx)={idx.size} != len(w)={w.size}")
        toys[rid] = (idx, w)

    return toys


def free_params(hyp):
    '''
    If POIs are not frozen, they are free parameters.
    If nuisances are not frozen and not ignored, they are free parameters.
    '''
    if isinstance(hyp, Rotated):
        return (
            [p for p in hyp.POIs if not p.isFrozen] +
            [p for p in hyp.nuisances if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]
        )
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]



def fit_one_toy(n2ll, hypothesis, toy_idx_by_region, toy_w_by_region,
                step=0.1, print_level=0, strategy=1, tol=None):
    '''
    Get name and value of the free parameters.
    POIs + nuisances are free.
    
    '''
    pars0 = free_params(hypothesis)
    if not pars0:
        raise RuntimeError("No free parameters to fit.")

    names = [p.name for p in pars0]
    x0    = [float(p.val) for p in pars0]

    n2ll.setToyFromIndicesAndWeights(toy_idx_by_region, toy_w_by_region)

    def fcn(*x):
        pdict = {names[i]: float(x[i]) for i in range(len(names))}
        h_eval = hypothesis.cloneModify(**pdict)
        return float(n2ll.n2ll_toy(h_eval))

    m = Minuit(fcn, *x0, name=names)
    m.errordef = 1.0
    m.print_level = int(print_level)
    m.strategy = int(strategy)
    if tol is not None:
        m.tol = float(tol)

    for i in range(len(names)):
        m.errors[i] = float(step)

    t0 = time.time()
    m.migrad()
    t1 = time.time()
    print('Migrad time: ', t1- t0)

    best = {names[i]: float(m.values[i]) for i in range(len(names))}
    h_best = hypothesis.cloneModify(**best)
    h_best.print()

    if isinstance(h_best, Rotated):
        c_hat = {p.name: float(p.val) for p in h_best.POIs}
    else:
        c_hat = {p.name: float(p.val) for p in h_best.parameters if p.isPOI}

    t1 = time.time()
    print('Total fit time: ', t1- t0)
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
    ap.add_argument("--minuit-step", type=float, default=1.0)
    ap.add_argument("--minuit-print-level", type=int, default=0)
    ap.add_argument("--minuit-strategy", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--minuit-tol", type=float, default=None)
    ap.add_argument("--no-syst", action="store_true", help="Freeze all nuisances at 0 (faster)")
    ap.add_argument("--out", default=None, help="Output npz")
    ap.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")
    args = ap.parse_args()

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False)
    # if fails here copy /groups/hephy/cms/robert.schoefbeck/SBIPDF/models/<config_name> to your directory.

    Likelihood.cfg = cfg
    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    rotated = bool(args.rotate)
    if rotated:
        print('[INFO]: Rotation json is passed. Using the rotated POIs...')
    hyp = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

    if args.no_syst:
        for p in hyp.nuisances:   # not hyp.parameters
            p.val = 0.0
            p.isFrozen = True

    POI_names = sorted([p.name for p in (hyp.POIs if isinstance(hyp, Rotated) else hyp.parameters) if p.isPOI])

    base = os.path.splitext(os.path.basename(args.config))[0]
    version = args.version or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)

    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])

    factory = samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )

    n2ll = Likelihood.N2LL(
        likelihood=like_info,
        factory=factory,
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=args.overwrite_cache,
    )
    n2ll.build_cache()
    n2ll.prepare_runtime()

    region_ids = [R["id"] for R in n2ll.regions]
    print("[regions]", region_ids)

    if args.toy_number is not None:
        toy_ids = [int(args.toy_number)]
    else:
        toy_ids = list_toy_ids_npz(args.toys_npz)
        if args.max_toys is not None:
            toy_ids = toy_ids[:args.max_toys]
    print(f"[toys] file={args.toys_npz}  n={len(toy_ids)}")

    # preload H5 arrays into RAM once (this is the whole point)
    n2ll.preload_unbinned_numpy()

    POI_names = sorted([p.name for p in hyp.parameters if p.isPOI])
    fvals = []
    chat_list = []

    t0 = time.time()
    for k, itoy in enumerate(toy_ids, start=1):
        toy = load_toys_npz(args.toys_npz, toy_number=itoy)
        toy_idx_by_region = {}
        toy_w_by_region = {}
        n_draws = 0

        for rid in region_ids:
            idx, w = toy.get(rid, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
            toy_idx_by_region[rid] = idx
            toy_w_by_region[rid] = w
            n_draws += int(idx.size)

        fval, c_hat = fit_one_toy(
            n2ll, hyp,
            toy_idx_by_region, toy_w_by_region,
            step=args.minuit_step,
            print_level=args.minuit_print_level,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
        )

        fvals.append(fval)
        chat_list.append([c_hat.get(nm, np.nan) for nm in POI_names])

        if args.print_every > 0 and (k % args.print_every == 0):
            dt = time.time() - t0
            rate = k / dt if dt > 0 else float("inf")
            print(f"[toy {itoy:04d}] k={k:4d}/{len(toy_ids)}  Ndraw={n_draws:7d}  n2ll_min={fval:.6f}  ({rate:.2f} toys/s)")
            print("  c_hat:", {nm: c_hat.get(nm, np.nan) for nm in POI_names})

    out = args.out or (os.path.splitext(args.toys_npz)[0] + f"_{args.toy_number}" + "_fit.npz")
    np.savez(
        out,
        config=np.asarray([args.config]),
        toys_npz=np.asarray([args.toys_npz]),
        toy_ids=np.asarray(toy_ids, dtype=np.int64),
        region_ids=np.asarray(region_ids),
        POI_names=np.asarray(POI_names),
        n2ll_min=np.asarray(fvals, dtype=np.float64),
        c_hat=np.asarray(chat_list, dtype=np.float64),
    )
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
