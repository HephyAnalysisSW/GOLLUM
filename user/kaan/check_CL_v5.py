#!/usr/bin/env python3
"""
check_CLs_v6.py

Proper CLs evaluation using:
  - toy datasets stored as indices (generate_toys_v3.py output: toys_*.npz)
  - N2LL in OBSERVATION mode
  - profile likelihood ratio test statistic q_mu (two fits per toy)

You must provide:
  --b-toys  : toys generated under mu=0
  --sb-toys : toys generated under mu=mu_test
and the tested point:
  --poi c1
  --mu  0.2

By default, q_obs is computed on the *background-only Asimov* (expected CLs).
You can switch to q_obs from a specific b-toy via --obs toy:<id>.

Example:
  python3 user/kaan/check_CLs_v6.py configs/unbinned_merged.yaml \
    --b-toys  /.../toys_nominal_mode-poisson_N1000.npz \
    --sb-toys /.../toys_c1_0.2_mode-poisson_N1000.npz \
    --poi c1 --mu 0.2 --alpha 0.05
"""

import sys
from pathlib import Path

# Ensure project root is importable when running from user/kaan/
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import os
import re
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
from iminuit import Minuit

import common.yaml_loader as yaml_loader
import common.user as common_user

import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL


# -----------------------------------------------------------------------------
# Toy loading (indices only)
# Keys: toy0000_<rid>_indices
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
# Observation-mode reconstruction from indices
# (compressed: read only UNIQUE indices, set weights = multiplicities)
# -----------------------------------------------------------------------------
def build_by_class_slices_for_indices_compressed(n2ll: N2LL, rid: str, indices: np.ndarray) -> Tuple[dict, np.ndarray]:
    """
    Returns:
      by_class dict with arrays of length K = n_unique(indices)
      w : float weights array of length K (multiplicity counts)
    """
    indices = np.asarray(indices, dtype=np.int64)
    by_class = {}

    # empty toy for this region
    if indices.size == 0:
        # create empty structures with correct second dims
        for cid in n2ll._class_ids_by_region[rid]:
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

    # compress duplicates
    unique_idx, counts = np.unique(indices, return_counts=True)
    w = counts.astype(np.float64)

    for cid in n2ll._class_ids_by_region[rid]:
        f = n2ll._h5[(rid, cid)]
        meta = n2ll._meta[(rid, cid)]
        comp = {}

        comp["g"] = f["g"][()][unique_idx]
        comp["R"] = f["R"][()][unique_idx, :]
        
        for gm in meta.get("delta_groups", []):
            dset_name = gm.get("dset", f"Delta::{gm['id']}")
            comp[dset_name] = f[dset_name][()][unique_idx, :]

        by_class[cid] = comp

    return by_class, w


def set_observation_from_toy(n2ll: N2LL, toy_blocks: Dict[str, Dict[str, object]]) -> None:
    """
    toy_blocks[rid] = {"by_class": ..., "w": ...}
    """
    # Ensure we are NOT in Asimov mode
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
    """
    Switch to Asimov mode cleanly (disables observation mode).
    hyp_asimov=None -> null Asimov (no bias)
    hyp_asimov=hyp -> off-nominal Asimov (bias)
    """
    n2ll._observation_set = False
    n2ll._obs_unbinned = {}
    n2ll._obs_binned = {}
    n2ll.setAsimov(hyp_asimov)


# -----------------------------------------------------------------------------
# Minimization helpers (silent Minuit)
# -----------------------------------------------------------------------------
def _free_params(hyp) -> list:
    return [p for p in hyp.parameters if (not p.isFrozen) and (not getattr(p, "isIgnored", False))]


def minimize_n2ll(n2ll: N2LL, hyp, *, step=0.1, limits: Optional[dict] = None, verbosity=2) -> float:
    """
    Minimizes n2ll(hyp) over free parameters (hyp.parameters where isFrozen=False).
    Updates hyp parameter values in-place to best fit.
    Returns minimum value.
    """
    
    free = _free_params(hyp)
    if not free:
        return float(n2ll(hyp))

    names = [p.name for p in free]
    x0 = [float(p.val) for p in free]

    def fcn(*x):
        for i, p in enumerate(free):
            p.val = float(x[i])
        return float(n2ll(hyp))

    m = Minuit(fcn, *x0, name=names)
    m.errordef = 1.0
    m.print_level = verbosity

    # step sizes
    for i in range(len(names)):
        m.errors[i] = float(step)

    # parameter limits (e.g. POI >= 0)
    if limits:
        for n, lim in limits.items():
            if n in m.parameters:
                m.limits[n] = lim

    m.migrad()
    # m.hesse()        # omit for speedup.
    if verbosity >= 1:
        print(m.fmin)      # convergence summary
        print(m.params)    # parameters, values, errors, limits, fixed/free

    # push best-fit back
    for i, p in enumerate(free):
        p.val = float(m.values[i])

    return float(m.fval)


def get_param(hyp, name: str):
    for p in hyp.parameters:
        if p.name == name:
            return p
    raise KeyError(f"Parameter '{name}' not found in hypothesis.")


def qmu_for_current_dataset(n2ll: N2LL, hyp_template, *, poi: str, mu_test: float,
                           poi_lower: Optional[float] = 0.0, step=0.1, verbosity=2) -> Tuple[float, float, float, float]:
    """
    Compute one-sided q_mu for the dataset currently loaded in n2ll (either observation or Asimov mode).

    Returns:
      (q_mu, mu_hat, n2ll_hat, n2ll_mu)
    """
    # --- global fit: mu and nuisances float ---
    hyp_hat = hyp_template.clone()
    # ensure POI not frozen
    get_param(hyp_hat, poi).isFrozen = False
    n2ll_hat = minimize_n2ll(
        n2ll, hyp_hat,
        step=step,
        limits={poi: (poi_lower, None)} if poi_lower is not None else None,
        verbosity=verbosity
    )
    mu_hat = float(get_param(hyp_hat, poi).val)

    # --- conditional fit: mu fixed to mu_test, nuisances float ---
    hyp_mu = hyp_template.cloneModify(**{poi: float(mu_test)})
    get_param(hyp_mu, poi).isFrozen = True
    n2ll_mu = minimize_n2ll(n2ll, hyp_mu, step=step, verbosity=verbosity)

    # -2ln lambda
    q = float(n2ll_mu - n2ll_hat)
    if q < 0:
        q = 0.0

    # one-sided for upper limits
    if mu_hat > mu_test:
        q = 0.0

    return q, mu_hat, n2ll_hat, n2ll_mu


# -----------------------------------------------------------------------------
# CLs core
# -----------------------------------------------------------------------------
def tail_prob_ge(arr: np.ndarray, threshold: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr >= threshold))


def parse_obs(obs: str) -> Tuple[str, Optional[int]]:
    """
    obs formats:
      "asimov"      -> background-only Asimov as observed
      "toy:<id>"    -> use a specific b-only toy as observed, e.g. toy:7
    """
    if obs == "asimov":
        return "asimov", None
    if obs.startswith("toy:"):
        return "toy", int(obs.split(":", 1)[1])
    raise ValueError("Invalid --obs. Use 'asimov' or 'toy:<id>'.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Proper CLs from index-only toys using N2LL observation mode.")
    ap.add_argument("config", help="Path to YAML config.")
    ap.add_argument("--version", default=None, help="Cache version string (must match cache).")
    ap.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache (slow). Usually keep OFF.")
    ap.add_argument("--b-toys", required=True, help="toys_*.npz generated under mu=0 (b-only).")
    ap.add_argument("--sb-toys", required=True, help="toys_*.npz generated under mu=mu_test (s+b).")
    ap.add_argument("--poi", default="c1", help="Name of the POI (mu). Default: c1")
    ap.add_argument("--mu", type=float, required=True, help="Tested mu value (must match sb toy generation).")
    ap.add_argument("--alpha", type=float, default=0.05, help="CLs threshold (default 0.05).")
    ap.add_argument("--obs", default="asimov", help="Observed dataset choice: 'asimov' or 'toy:<id>'")
    ap.add_argument("--max-toys", type=int, default=None, help="Optional cap on number of toys used per ensemble.")
    ap.add_argument("--poi-lower", type=float, default=0.0, help="Lower bound for POI in global fit (default 0).")
    ap.add_argument("--minuit-step", type=float, default=0.1, help="Minuit step size for all floating params.")
    ap.add_argument("--out", default=None, help="Output npz path for results.")
    args = ap.parse_args()

    # --- load cfg + surrogates ---
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    # Make cfg visible to Likelihood.N2LL.build_cache()
    Likelihood.cfg = cfg

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    # sanity: poi exists
    get_param(hyp, args.poi)

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

    # --- load toys ---
    toys_b = load_toys_indices_npz(args.b_toys)
    toys_sb = load_toys_indices_npz(args.sb_toys)

    toy_ids_b = sorted(toys_b.keys())
    toy_ids_sb = sorted(toys_sb.keys())
    if args.max_toys is not None:
        toy_ids_b = toy_ids_b[:args.max_toys]
        toy_ids_sb = toy_ids_sb[:args.max_toys]

    print(f"\n[toys] b-only file : {args.b_toys}  (toys: {len(toy_ids_b)})")
    print(f"[toys] s+b   file : {args.sb_toys} (toys: {len(toy_ids_sb)})")
    print(f"[test] POI={args.poi}  mu_test={args.mu}")

    # # --- q_obs ---
    # obs_kind, obs_id = parse_obs(args.obs)
    # if obs_kind == "asimov":
    #     # expected: observed = background-only Asimov (null)
    #     set_asimov_mode(n2ll, hyp_asimov=None)
    #     q_obs, mu_hat_obs, ll_hat_obs, ll_mu_obs = qmu_for_current_dataset(
    #         n2ll, hyp, poi=args.poi, mu_test=args.mu,
    #         poi_lower=args.poi_lower, step=args.minuit_step
    #     )
    #     print(f"\n[obs=asimov] q_obs={q_obs:.6f}  mu_hat={mu_hat_obs:.6f}  ll_hat={ll_hat_obs:.6f}  ll_mu={ll_mu_obs:.6f}")
    # else:
    #     # observed = a specific b-toy
    #     if obs_id not in toys_b:
    #         raise RuntimeError(f"--obs toy:{obs_id} requested, but toy id {obs_id} not found in b-toys file.")
    #     toy_blocks = {}
    #     for rid in region_ids:
    #         idx = toys_b[obs_id].get(rid, np.empty(0, dtype=np.int64))
    #         by_class, w = build_by_class_slices_for_indices_compressed(n2ll, rid, idx)
    #         toy_blocks[rid] = {"by_class": by_class, "w": w}
    #     set_observation_from_toy(n2ll, toy_blocks)
    #     q_obs, mu_hat_obs, ll_hat_obs, ll_mu_obs = qmu_for_current_dataset(
    #         n2ll, hyp, poi=args.poi, mu_test=args.mu,
    #         poi_lower=args.poi_lower, step=args.minuit_step
    #     )
    #     print(f"\n[obs=toy:{obs_id}] q_obs={q_obs:.6f}  mu_hat={mu_hat_obs:.6f}  ll_hat={ll_hat_obs:.6f}  ll_mu={ll_mu_obs:.6f}")

    # --- evaluate ensembles ---
    def eval_qs(toys: Dict[int, Dict[str, np.ndarray]], toy_ids: list[int], label: str) -> np.ndarray:
        qs = []
        for itoy in toy_ids:
            toy_blocks = {}
            for rid in region_ids:
                idx = toys[itoy].get(rid, np.empty(0, dtype=np.int64))
                by_class, w = build_by_class_slices_for_indices_compressed(n2ll, rid, idx)
                toy_blocks[rid] = {"by_class": by_class, "w": w}

            set_observation_from_toy(n2ll, toy_blocks)
            q, mu_hat, ll_hat, ll_mu = qmu_for_current_dataset(
                n2ll, hyp, poi=args.poi, mu_test=args.mu,
                poi_lower=args.poi_lower, step=args.minuit_step,
                verbosity=0
            )
            qs.append(q)

            
            n_ev = sum(len(toys[itoy].get(rid, [])) for rid in region_ids)
            print(f"[{label} toy {itoy:04d}] N={n_ev:6d}  q={q: .6f}  mu_hat={mu_hat: .6f}")

        return np.asarray(qs, dtype=np.float64)

    # print("\n[Compute q_mu] b-only toys ...")
    # q_b = eval_qs(toys_b, toy_ids_b, "b")
    print("\n[Compute q_mu] s+b toys ...")
    q_sb = eval_qs(toys_sb, toy_ids_sb, "sb")

    # --- CLs ---
    CLsb = tail_prob_ge(q_sb, q_obs)
    CLb = tail_prob_ge(q_b, q_obs)
    CLs = float(CLsb / CLb) if (CLb is not None and CLb > 0) else float("inf")

    print("\n[CLs results]")
    print(f"  q_obs     = {q_obs:.6f}")
    print(f"  CL_s+b    = {CLsb:.6g}   (P(q>=q_obs | s+b))")
    print(f"  CL_b      = {CLb:.6g}    (P(q>=q_obs | b))")
    print(f"  CL_s      = {CLs:.6g}    (CL_s+b / CL_b)")
    print(f"  exclude?  = {CLs < args.alpha}   (alpha={args.alpha})")

    # --- save ---
    out = args.out
    if out is None:
        out = os.path.splitext(os.path.basename(args.sb_toys))[0] + f"__CLs_poi-{args.poi}_mu-{args.mu}.npz"
        out = os.path.join(os.path.dirname(args.sb_toys), out)

    np.savez(
        out,
        config=np.array([args.config]),
        b_toys=np.array([args.b_toys]),
        sb_toys=np.array([args.sb_toys]),
        poi=np.array([args.poi]),
        mu_test=np.array([args.mu], dtype=np.float64),
        alpha=np.array([args.alpha], dtype=np.float64),
        obs=np.array([args.obs]),
        q_obs=np.array([q_obs], dtype=np.float64),
        q_b=q_b,
        q_sb=q_sb,
        CLsb=np.array([CLsb], dtype=np.float64),
        CLb=np.array([CLb], dtype=np.float64),
        CLs=np.array([CLs], dtype=np.float64),
    )
    print(f"\n[out] wrote {out}")


if __name__ == "__main__":
    main()
