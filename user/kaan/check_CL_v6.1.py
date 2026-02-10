#!/usr/bin/env python3
"""
check_PLR_multiPOI.py

Profile likelihood-ratio test statistic for one tested POI (e.g. c1=0.2),
while the global fit returns best-fit values for ALL POIs whose names start with 'c'
(e.g. c0_hat..c5_hat).

Data sources:
- Asimov mode (expected "observation") OR one specific toy as observation
- toy ensembles stored as indices (generate_toys_v3.py output): toys_*.npz
  keys: toy0000_<rid>_indices, toy0001_<rid>_indices, ...

Core statistic (one-dimensional profile LR for the tested POI):
- Global fit:     (ĉ, ν̂) = argmin_{all c*, all nuis} N2LL(c, ν)
- Conditional fit:          argmin_{all c* except poi, all nuis} N2LL(poi=mu_test, other c*, ν)
- q = N2LL_cond - N2LL_global
Optional one-sided convention (upper-limit style): if poi_hat > mu_test -> q = 0

Important performance note:
- HDF5 point selection like f["g"][unique_idx] is very slow.
- We read full datasets once into NumPy via f["g"][()] and then index NumPy:
    arr = f["g"][()]     # NumPy
    arr[unique_idx]      # fast
- We cache these arrays per (rid,cid,dset_name), so each dataset is loaded only once.
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
from typing import Dict, Tuple, Optional, Any

import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
import common.user as common_user

import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL


# -----------------------------------------------------------------------------
# Toy loading (indices only)
# -----------------------------------------------------------------------------
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
        raise RuntimeError(
            f"No toy keys matched '{_TOY_KEY_RE.pattern}' in {npz_path}.\n"
            f"Example keys: {z.files[:20]}{' ...' if len(z.files) > 20 else ''}"
        )
    return toys


# -----------------------------------------------------------------------------
# N2LL mode switching
# -----------------------------------------------------------------------------
def set_observation_from_toy(n2ll: N2LL, toy_blocks: Dict[str, Dict[str, Any]]) -> None:
    # Disable Asimov mode
    n2ll._asimov_hypothesis_set = False
    n2ll._asimov_active = False
    n2ll._asimov_hyp = None
    n2ll._asimov_T.clear()
    n2ll._binned_asimov_lambda.clear()

    # Set observation mode
    n2ll._obs_unbinned = {}
    n2ll._obs_binned = {}
    n2ll._observation_set = True

    for rid, block in toy_blocks.items():
        n2ll._obs_unbinned[rid] = {
            "by_class": block["by_class"],
            "w": np.asarray(block["w"], dtype=np.float64),
        }


def set_asimov_mode(n2ll: N2LL, hyp_asimov=None) -> None:
    n2ll._observation_set = False
    n2ll._obs_unbinned = {}
    n2ll._obs_binned = {}
    n2ll.setAsimov(hyp_asimov)


# -----------------------------------------------------------------------------
# Helper: identify POIs starting with 'c' and get/set parameter objects
# -----------------------------------------------------------------------------
def get_param(hyp, name: str):
    for p in hyp.parameters:
        if p.name == name:
            return p
    raise KeyError(f"Parameter '{name}' not found in hypothesis.")


def poi_names_starting_with_c(hyp) -> list[str]:
    # In your Likelihood.py: POIs are names starting with 'c' (build_hypothesis_from_likelihood)
    # but we'll enforce both isPOI and name prefix for safety.
    out = [p.name for p in hyp.parameters if getattr(p, "isPOI", False) and p.name.startswith("c")]
    return sorted(out)


def snapshot_pois(hyp, poi_names: list[str]) -> dict[str, float]:
    return {n: float(get_param(hyp, n).val) for n in poi_names}


# -----------------------------------------------------------------------------
# HDF5 -> NumPy cache using the pattern you requested: f[dset][()]
# -----------------------------------------------------------------------------
class H5NumpyCache:
    """
    Cache for (rid,cid,dset_name) -> numpy array loaded with f[dset_name][()]
    to avoid slow h5py point selection inside the toy loop.
    """
    def __init__(self):
        self._cache: dict[tuple[str, int, str], np.ndarray] = {}

    def get(self, n2ll: N2LL, rid: str, cid: int, dset_name: str) -> np.ndarray:
        key = (rid, cid, dset_name)
        arr = self._cache.get(key)
        if arr is not None:
            return arr
        f = n2ll._h5[(rid, cid)]
        # IMPORTANT: [()] forces read as numpy array
        arr = f[dset_name][()]
        self._cache[key] = arr
        return arr

    def preload_all(self, n2ll: N2LL, region_ids: list[str], *, include_delta: bool = True) -> None:
        t0 = time.time()
        total_bytes = 0
        for rid in region_ids:
            for cid in n2ll._class_ids_by_region[rid]:
                # always g and R
                for name in ("g", "R"):
                    a = self.get(n2ll, rid, cid, name)
                    total_bytes += a.nbytes

                if include_delta:
                    meta = n2ll._meta[(rid, cid)]
                    for gm in meta.get("delta_groups", []):
                        dset_name = gm.get("dset", f"Delta::{gm['id']}")
                        a = self.get(n2ll, rid, cid, dset_name)
                        total_bytes += a.nbytes

        dt = time.time() - t0
        print(f"[preload] cached datasets in {dt:.2f}s, total ~{total_bytes/1024**3:.3f} GB")


# -----------------------------------------------------------------------------
# Build observation blocks from indices (compressed -> unique_idx + multiplicity weights)
# Uses cached NumPy arrays via f[dset][()][unique_idx]
# -----------------------------------------------------------------------------
def build_by_class_slices_for_indices_compressed(
    n2ll: N2LL,
    cache: H5NumpyCache,
    rid: str,
    indices: np.ndarray
) -> Tuple[dict, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    by_class: Dict[int, dict] = {}

    # empty toy for this region
    if indices.size == 0:
        for cid in n2ll._class_ids_by_region[rid]:
            # use dataset shapes without forcing full read (cheap)
            f = n2ll._h5[(rid, cid)]
            nA = f["R"].shape[1]
            comp = {
                "g": np.empty(0, dtype=np.float64),
                "R": np.empty((0, nA), dtype=np.float64),
            }
            meta = n2ll._meta[(rid, cid)]
            for gm in meta.get("delta_groups", []):
                dset_name = gm.get("dset", f"Delta::{gm['id']}")
                nB = f[dset_name].shape[1]
                comp[dset_name] = np.empty((0, nB), dtype=np.float64)
            by_class[cid] = comp
        w = np.empty(0, dtype=np.float64)
        return by_class, w

    # compress duplicates -> use multiplicity weights
    unique_idx, counts = np.unique(indices, return_counts=True)
    w = counts.astype(np.float64)

    for cid in n2ll._class_ids_by_region[rid]:
        comp: Dict[str, np.ndarray] = {}

        # Your requested pattern: f["g"][()][unique_idx] etc.
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


# -----------------------------------------------------------------------------
# Minuit minimization: float all non-frozen parameters
# -----------------------------------------------------------------------------
def _free_params(hyp) -> list:
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def minimize_n2ll(
    n2ll: N2LL,
    hyp,
    *,
    step: float = 0.1,
    print_level: int = 0,
    strategy: int = 1,
    tol: Optional[float] = None,
) -> float:
    free = _free_params(hyp)
    if not free:
        v = float(n2ll(hyp))
        return v

    names = [p.name for p in free]
    x0 = [float(p.val) for p in free]

    def fcn(*x):
        for i, p in enumerate(free):
            p.val = float(x[i])
        v = float(n2ll(hyp))
        if not np.isfinite(v):
            # prevent NaN thrashing
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

    m.migrad()   # no HESSE here (faster for toy loops)

    # push best-fit back
    for i, p in enumerate(free):
        p.val = float(m.values[i])

    return float(m.fval)


# -----------------------------------------------------------------------------
# Profile LR statistic for tested POI (but return all c*_hats from the global fit)
# -----------------------------------------------------------------------------
def qpoi_for_current_dataset(
    n2ll: N2LL,
    hyp_template,
    *,
    tested_poi: str,
    tested_value: float,
    step: float = 0.1,
    strategy: int = 1,
    tol: Optional[float] = None,
    print_level: int = 0,
    one_sided: bool = True,
) -> Tuple[float, dict[str, float], float, float, dict[str, float]]:
    """
    Returns:
      q,
      c_hats_global (dict),
      n2ll_global,
      n2ll_cond,
      c_hats_cond (dict)   (best-fit for all c* under tested_poi fixed)
    """
    poi_names = poi_names_starting_with_c(hyp_template)

    # 1) Global fit: float all POIs (c*) and nuisances
    hyp_hat = hyp_template.clone()
    # ensure tested poi is free in global
    get_param(hyp_hat, tested_poi).isFrozen = False
    n2ll_hat = minimize_n2ll(
        n2ll, hyp_hat,
        step=step, print_level=print_level, strategy=strategy, tol=tol
    )
    c_hats = snapshot_pois(hyp_hat, poi_names)
    poi_hat_value = float(get_param(hyp_hat, tested_poi).val)

    # 2) Conditional fit: fix tested_poi to tested_value; float all other c* and nuisances
    hyp_cond = hyp_template.clone()
    get_param(hyp_cond, tested_poi).val = float(tested_value)
    get_param(hyp_cond, tested_poi).isFrozen = True
    n2ll_cond = minimize_n2ll(
        n2ll, hyp_cond,
        step=step, print_level=print_level, strategy=strategy, tol=tol
    )
    c_hats_cond = snapshot_pois(hyp_cond, poi_names)

    # q = -2 ln lambda
    q = float(n2ll_cond - n2ll_hat)
    if not np.isfinite(q) or q < 0:
        q = 0.0

    # Optional one-sided convention (upper-limit style)
    if one_sided and (poi_hat_value > tested_value):
        q = 0.0

    return q, c_hats, n2ll_hat, n2ll_cond, c_hats_cond


# -----------------------------------------------------------------------------
# Obs parsing
# -----------------------------------------------------------------------------
def parse_obs(obs: str) -> Tuple[str, Optional[int]]:
    if obs == "asimov":
        return "asimov", None
    if obs.startswith("toy:"):
        return "toy", int(obs.split(":", 1)[1])
    raise ValueError("Invalid --obs. Use 'asimov' or 'toy:<id>'.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Profile LR q for one tested POI; report all c*_hats from global fit.")
    ap.add_argument("config", help="Path to YAML config.")
    ap.add_argument("--version", default=None, help="Cache version string (must match cache).")
    ap.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache (slow).")

    ap.add_argument("--tested-poi", default="c1", help="Which POI to test (default c1).")
    ap.add_argument("--tested-value", type=float, required=True, help="Test value for the tested POI (e.g. 0.2).")
    ap.add_argument("--one-sided", action="store_true", help="Use one-sided convention: if poi_hat>tested -> q=0.")
    ap.add_argument("--two-sided", action="store_true", help="Disable one-sided convention.")

    ap.add_argument("--b-toys", default=None, help="Optional: toys_*.npz (b-only ensemble).")
    ap.add_argument("--sb-toys", default=None, help="Optional: toys_*.npz (s+b ensemble).")
    ap.add_argument("--obs", default="asimov", help="Observed dataset: 'asimov' or 'toy:<id>' (from b-toys).")

    ap.add_argument("--max-toys", type=int, default=None, help="Cap number of toys per ensemble.")
    ap.add_argument("--print-every", type=int, default=1, help="Print every N toys (default 1).")

    ap.add_argument("--minuit-step", type=float, default=0.1)
    ap.add_argument("--minuit-strategy", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--minuit-tol", type=float, default=None)
    ap.add_argument("--minuit-print-level", type=int, default=0)

    ap.add_argument("--preload", action="store_true", help="Preload g/R/Delta into NumPy cache at start.")
    ap.add_argument("--out", default=None, help="Output npz path.")
    args = ap.parse_args()

    if args.two_sided and args.one_sided:
        raise RuntimeError("Choose only one of --one-sided or --two-sided.")
    one_sided = True if args.one_sided else (False if args.two_sided else True)

    # --- load cfg + surrogates ---
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    Likelihood.cfg = cfg
    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    # sanity: tested poi exists
    get_param(hyp, args.tested_poi)
    poi_names = poi_names_starting_with_c(hyp)
    print("[POIs starting with 'c']:", poi_names)

    # --- setup N2LL ---
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
    print("\n[N2LL] Regions:", region_ids)
    print(f"[test] tested_poi={args.tested_poi}  tested_value={args.tested_value}  one_sided={one_sided}")

    # --- HDF5->NumPy cache ---
    cache = H5NumpyCache()
    if args.preload:
        cache.preload_all(n2ll, region_ids, include_delta=True)

    # --- load toys (optional) ---
    toys_b = load_toys_indices_npz(args.b_toys) if args.b_toys else None
    toys_sb = load_toys_indices_npz(args.sb_toys) if args.sb_toys else None

    def capped_ids(toys: Dict[int, Dict[str, np.ndarray]]) -> list[int]:
        ids = sorted(toys.keys())
        if args.max_toys is not None:
            ids = ids[:args.max_toys]
        return ids

    # --- q_obs ---
    obs_kind, obs_id = parse_obs(args.obs)
    if obs_kind == "asimov":
        set_asimov_mode(n2ll, hyp_asimov=None)
        q_obs, c_hat_obs, ll_hat_obs, ll_cond_obs, c_hat_cond_obs = qpoi_for_current_dataset(
            n2ll, hyp,
            tested_poi=args.tested_poi,
            tested_value=args.tested_value,
            step=args.minuit_step,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
            print_level=args.minuit_print_level,
            one_sided=one_sided,
        )
        print(f"\n[obs=asimov] q_obs={q_obs:.6f}  ll_hat={ll_hat_obs:.6f}  ll_cond={ll_cond_obs:.6f}")
        print("  c*_hat (global):", c_hat_obs)
        print("  c*_hat (cond)  :", c_hat_cond_obs)
    else:
        if toys_b is None:
            raise RuntimeError("--obs toy:<id> requires --b-toys.")
        if obs_id not in toys_b:
            raise RuntimeError(f"--obs toy:{obs_id} requested but not found in b-toys.")

        toy_blocks = {}
        for rid in region_ids:
            idx = toys_b[obs_id].get(rid, np.empty(0, dtype=np.int64))
            by_class, w = build_by_class_slices_for_indices_compressed(n2ll, cache, rid, idx)
            toy_blocks[rid] = {"by_class": by_class, "w": w}

        set_observation_from_toy(n2ll, toy_blocks)
        q_obs, c_hat_obs, ll_hat_obs, ll_cond_obs, c_hat_cond_obs = qpoi_for_current_dataset(
            n2ll, hyp,
            tested_poi=args.tested_poi,
            tested_value=args.tested_value,
            step=args.minuit_step,
            strategy=args.minuit_strategy,
            tol=args.minuit_tol,
            print_level=args.minuit_print_level,
            one_sided=one_sided,
        )
        print(f"\n[obs=toy:{obs_id}] q_obs={q_obs:.6f}  ll_hat={ll_hat_obs:.6f}  ll_cond={ll_cond_obs:.6f}")
        print("  c*_hat (global):", c_hat_obs)
        print("  c*_hat (cond)  :", c_hat_cond_obs)

    # --- evaluate ensembles (optional) ---
    def eval_ensemble(toys: Dict[int, Dict[str, np.ndarray]], label: str) -> Dict[str, np.ndarray]:
        ids = capped_ids(toys)

        q_list = []
        llhat_list = []
        llcond_list = []
        nev_list = []
        c_hat_list = []        # list of dicts
        c_hat_cond_list = []   # list of dicts

        t0 = time.time()
        for k, itoy in enumerate(ids, start=1):
            toy_blocks = {}
            n_ev = 0

            for rid in region_ids:
                idx = toys[itoy].get(rid, np.empty(0, dtype=np.int64))
                n_ev += int(idx.size)
                by_class, w = build_by_class_slices_for_indices_compressed(n2ll, cache, rid, idx)
                toy_blocks[rid] = {"by_class": by_class, "w": w}

            set_observation_from_toy(n2ll, toy_blocks)

            q, c_hat, ll_hat, ll_cond, c_hat_cond = qpoi_for_current_dataset(
                n2ll, hyp,
                tested_poi=args.tested_poi,
                tested_value=args.tested_value,
                step=args.minuit_step,
                strategy=args.minuit_strategy,
                tol=args.minuit_tol,
                print_level=0,
                one_sided=one_sided,
            )

            q_list.append(q)
            llhat_list.append(ll_hat)
            llcond_list.append(ll_cond)
            nev_list.append(n_ev)
            c_hat_list.append(c_hat)
            c_hat_cond_list.append(c_hat_cond)

            if args.print_every > 0 and (k % args.print_every == 0):
                dt = time.time() - t0
                rate = k / dt if dt > 0 else float("inf")
                print(f"[{label} toy {itoy:04d}] k={k:4d}/{len(ids)} N={n_ev:7d} q={q: .6f} ({rate:.2f} toys/s)")
                print(f"  c*_hat: {c_hat}")

        # pack c-hats into arrays with fixed order
        c_arr = np.zeros((len(ids), len(poi_names)), dtype=np.float64)
        c_arr_cond = np.zeros_like(c_arr)
        for i, d in enumerate(c_hat_list):
            for j, nm in enumerate(poi_names):
                c_arr[i, j] = d.get(nm, np.nan)
        for i, d in enumerate(c_hat_cond_list):
            for j, nm in enumerate(poi_names):
                c_arr_cond[i, j] = d.get(nm, np.nan)

        return {
            "toy_ids": np.asarray(ids, dtype=np.int64),
            "n_events": np.asarray(nev_list, dtype=np.int64),
            "q": np.asarray(q_list, dtype=np.float64),
            "ll_hat": np.asarray(llhat_list, dtype=np.float64),
            "ll_cond": np.asarray(llcond_list, dtype=np.float64),
            "c_hat": c_arr,
            "c_hat_cond": c_arr_cond,
            "c_names": np.asarray(poi_names),
        }

    results = {
        "config": np.asarray([args.config]),
        "tested_poi": np.asarray([args.tested_poi]),
        "tested_value": np.asarray([args.tested_value], dtype=np.float64),
        "one_sided": np.asarray([int(one_sided)], dtype=np.int64),
        "q_obs": np.asarray([q_obs], dtype=np.float64),
        "ll_hat_obs": np.asarray([ll_hat_obs], dtype=np.float64),
        "ll_cond_obs": np.asarray([ll_cond_obs], dtype=np.float64),
        "c_names": np.asarray(poi_names),
        "c_hat_obs": np.asarray([c_hat_obs.get(nm, np.nan) for nm in poi_names], dtype=np.float64),
        "c_hat_cond_obs": np.asarray([c_hat_cond_obs.get(nm, np.nan) for nm in poi_names], dtype=np.float64),
    }

    if toys_b is not None:
        print(f"\n[ensemble] b-only: {args.b_toys}")
        res_b = eval_ensemble(toys_b, "b")
        results.update({f"b_{k}": v for k, v in res_b.items()})

    if toys_sb is not None:
        print(f"\n[ensemble] s+b: {args.sb_toys}")
        res_sb = eval_ensemble(toys_sb, "sb")
        results.update({f"sb_{k}": v for k, v in res_sb.items()})

    # save
    out = args.out
    if out is None:
        out_dir = os.path.join(common_user.output_directory, "toys")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"PLR_multiPOI_{args.tested_poi}_{args.tested_value}.npz")

    np.savez(out, **results)
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
