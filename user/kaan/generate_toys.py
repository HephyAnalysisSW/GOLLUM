#!/usr/bin/env python
"""
Generate unbinned toy datasets from N2LL Asimov cache.

Supports signed-weight toy sampling (for NLO negative weights).
- weights are ±1 and Poisson means are Poisson(sum λ±)

Usage examples:
    python3 user/kaan/generate_toys.py configs/unbinned/unbinned_2016APV.yaml --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_2016APV.json --n-toys 1000 --shape_2 1.0
    python3 user/kaan/generate_toys.py configs/unbinned/unbinned_2016APV.yaml --n-toys 1000 --shape_2 1.0 --c1 0.5 --c2 1.0 --nu_pu 0.2
"""

import argparse
import os
import numpy as np

import common.yaml_loader as yaml_loader
import common.user as user

import fit.Likelihood as Likelihood
from fit.Modeling import Rotated


# ----------------------------------------------------------------------
# Helpers to compute λ_i(θ) and sample toys
# ----------------------------------------------------------------------
def compute_lambda_unbinned_for_region(n2ll: Likelihood.N2LL, hypothesis, rid: str) -> np.ndarray:
    """
    Compute event-wise signed rates λ_i(θ) for a single unbinned region using cached Asimov events:

        dσ(x; c,ν)/dx = dσ(x; 0,0)/dx * (1 + T(x; c,ν))
        w0_i = dσ(x_i; 0,0)/dx * (integrated luminosity)
        λ_i(θ) = w0_i * (1 + T_i(θ))

    Returns
    -------
    lam : np.ndarray of shape (N_region,)
        May contain negative entries when NLO weights are present (through w0_i).
    """
    class_ids = n2ll._class_ids_by_region.get(rid, [])
    if not class_ids:
        raise RuntimeError(f"[toys] Region '{rid}' has no classes.")

    N = n2ll._N_region.get(rid, 0)
    if N == 0:
        raise RuntimeError(f"[toys] Region '{rid}' has N = 0 events.")

    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis._base)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis._base)

    # nuisance values
    nu_vals = {}
    for p in getattr(hypothesis._base, "parameters", []):
        if not p.isPOI:
            nu_vals[p.name] = float(p.val)

    # lnN biases per class
    ln_bias = {}
    for cid in class_ids:
        s = 0.0
        for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), []):
            s += log1p_alpha * nu_vals.get(pname, 0.0)
        ln_bias[cid] = s

    lam = np.empty(N, dtype=np.float64)
    chunk = n2ll.eval_chunk_size
    first_cid = class_ids[0]  # convention: w0 stored per region, read from first class

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        T_chunk = n2ll._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
        w0_chunk = n2ll._h5[(rid, first_cid)]["w0"][start:stop]
        lam[start:stop] = w0_chunk * (1.0 + T_chunk)

    return lam


def sample_toy_indices_from_lambda_signed(lam, rng):
    lam = np.asarray(lam, np.float64)
    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())
    if (tot_pos + tot_neg) <= 0.0:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    N_pos = int(rng.poisson(tot_pos)) if tot_pos > 0 else 0
    N_neg = int(rng.poisson(tot_neg)) if tot_neg > 0 else 0
    if (N_pos + N_neg) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    idxs, ws = [], []
    if N_pos:
        idxs.append(rng.choice(lam.size, size=N_pos, replace=True, p=lam_pos / tot_pos))
        ws.append(np.ones(N_pos, np.float64))
    if N_neg:
        idxs.append(rng.choice(lam.size, size=N_neg, replace=True, p=lam_neg / tot_neg))
        ws.append(-np.ones(N_neg, np.float64))

    idx = np.concatenate(idxs).astype(np.int64, copy=False)
    w   = np.concatenate(ws).astype(np.float64, copy=False)
    perm = rng.permutation(idx.size) # OPTIONAL: mix neg/pos weighted events. 
    return idx[perm], w[perm]



# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate unbinned toys from N2LL Asimov cache.")
    ap.add_argument("config", help="Path to global YAML config.")
    ap.add_argument("--version", default=None, help="Analysis version (used in cache dir name).")
    ap.add_argument("--overwrite-cache", action="store_true", help="Force rebuild of the N2LL cache.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for toy generation.")
    ap.add_argument("--n-toys", type=int, default=1, help="Number of toy datasets to generate.")
    ap.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")

    # parse known + unknown args
    args, unknown = ap.parse_known_args()

    # Parse dynamic parameter modifications: --c1 1.0 --nu_pu 0.2 ...
    dynamic_kwargs = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if not key.startswith("--"):
            raise RuntimeError("We expect POIs and nuisances with -- prefix.")
        name = key[2:]
        if (i + 1) >= len(unknown):
            raise RuntimeError(f"Missing value for parameter {key}")
        dynamic_kwargs[name] = unknown[i + 1]
        i += 2

    print("\n[dynamic params] Parsed:", dynamic_kwargs)


    print(f"[toys] n_toys={args.n_toys}")

    rng = np.random.default_rng(args.seed)

    # Load config + surrogates
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False)

    Likelihood.cfg = cfg
    like_info = Likelihood.load_likelihood(cfg)
    hyp = Likelihood.build_hypothesis_from_likelihood(like_info, name="SR")

    rotated = bool(args.rotate)
    if rotated:
        print('[INFO]: Rotation json is passed. Using the rotated POIs...')
    hyp = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

    print("\n[Hypothesis] Initial parameters:")
    hyp.print()

    # Apply dynamic parameter modifications
    valid_param_names = [p.name for p in hyp.parameters]
    print('Valid parameter names: ')
    print(valid_param_names)
    kwargs = {}
    for k, v in dynamic_kwargs.items():
        if k in valid_param_names:
            kwargs[k] = float(v)
        else:
            raise ValueError(
                f"Unknown model parameter: {k}\n"
                f"Available parameters:\n{valid_param_names}"
            )

    hyp_test = hyp.cloneModify(**kwargs)
    print("\n[Test hypothesis] Modified parameters:")
    hyp_test.print()

    # Set up N2LL
    base = os.path.splitext(os.path.basename(args.config))[0]
    version = args.version or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)

    n2ll = Likelihood.N2LL(
        likelihood=like_info,
        module_samples=cfg["defaults"]["module_samples"],
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=args.overwrite_cache,
    )
    n2ll.build_cache()
    n2ll.prepare_runtime()

    region_ids = [R["id"] for R in n2ll.regions]
    print("\n[toys] Unbinned regions:", region_ids)

    # Store all toys for this hypothesis
    store = {}

    for rid in region_ids:
        lam = compute_lambda_unbinned_for_region(n2ll, hyp_test, rid)

        # optional helpful prints
        lam_pos = np.clip(lam, 0.0, None)
        lam_neg = np.clip(-lam, 0.0, None)
        print(f"[toys] rid={rid}  sum(lam+)={lam_pos.sum():.6g}  sum(lam-)={lam_neg.sum():.6g}")

        for itoy in range(args.n_toys):
            idx, w = sample_toy_indices_from_lambda_signed(lam,rng=rng)
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w

    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    param_tag = "_".join(f"{k}_{v}" for k, v in sorted(dynamic_kwargs.items())) if dynamic_kwargs else "nominal"
    filename = f"toys_{param_tag}_N{args.n_toys}.npz"
    toy_out = os.path.join(out_dir, filename)

    np.savez(toy_out, **store)
    print(f"[toys] Saved {args.n_toys} toys to {toy_out}")


if __name__ == "__main__":
    main()
