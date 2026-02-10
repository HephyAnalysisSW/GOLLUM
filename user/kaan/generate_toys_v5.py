#!/usr/bin/env python
"""
Generate unbinned toy datasets from N2LL Asimov cache.

Supports signed-weight toy sampling (for NLO negative weights) via a "scale" (weight quantum) parameter:
- scale = 1.0 (default): weights are ±1 and Poisson means are Poisson(sum λ±)
- scale = 0.5: weights are ±0.5 and Poisson means are Poisson(sum λ± / 0.5)  -> ~2x more draws on average
- in general: each draw has weight ±scale and you draw ~1/scale more events so expectation stays the same

Usage examples:
    python generate_toys_v3.py configs/unbinned_merged.yaml --mode poisson --n-toys 1
    python generate_toys_v3.py configs/unbinned_merged.yaml --mode poisson --n-toys 10 --scale 0.5 --c1 0.5 --c2 1.0 --nu_pu 0.2
"""

import argparse
import os
import numpy as np

import common.yaml_loader as yaml_loader
import common.user as user

import fit.Likelihood as Likelihood


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

    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis)

    # nuisance values
    nu_vals = {}
    for p in getattr(hypothesis, "parameters", []):
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


def sample_toy_indices_from_lambda_signed(
    lam: np.ndarray,
    rng: np.random.Generator,
    mode: str = "poisson",
    fixed_N: int | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Signed bootstrap sampling for possibly negative lam (NLO weights).

    Decompose λ into λ+ and λ-:
        λ+ = max(λ, 0),  λ- = max(-λ, 0)

    Then sample two independent Poisson processes and assign weights ±scale.

    scale parameter:
      - Each drawn event carries weight +scale (from λ+) or -scale (from λ-).
      - To keep expectations invariant when changing scale, the Poisson means are scaled as:
            N+ ~ Poisson(sum(λ+) / scale)
            N- ~ Poisson(sum(λ-) / scale)
        (so smaller scale => more draws, smaller per-draw weight)

    If mode == "fixed", the total number of draws is fixed_N (split between + and - in proportion
    to sums of λ+ and λ-), and the per-draw weights are still ±scale.

    Returns
    -------
    indices : (Ndraw,) int
    weights : (Ndraw,) float in {+scale, -scale}
    """
    lam = np.asarray(lam, dtype=np.float64)

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"[toys] scale must be positive, got {scale}")

    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())
    tot_abs = tot_pos + tot_neg
    if tot_abs <= 0.0:
        raise RuntimeError("[toys] Sum of |λ_i| is non-positive.")

    if mode == "poisson":
        # IMPORTANT: divide by scale so expected signed sum remains invariant under scale changes
        mean_pos = tot_pos / scale
        mean_neg = tot_neg / scale
        N_pos = int(rng.poisson(mean_pos)) if mean_pos > 0 else 0
        N_neg = int(rng.poisson(mean_neg)) if mean_neg > 0 else 0

    elif mode == "fixed":
        if fixed_N is None:
            raise ValueError("mode='fixed' requires fixed_N.")
        fixed_N = int(fixed_N)
        if fixed_N < 0:
            raise ValueError("fixed_N must be >= 0.")
        p_pos = tot_pos / tot_abs
        N_pos = int(rng.binomial(fixed_N, p_pos))
        N_neg = fixed_N - N_pos

    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    if (N_pos + N_neg) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    out_idx = []
    out_w = []

    if N_pos > 0 and tot_pos > 0:
        p = lam_pos / tot_pos
        idx = rng.choice(len(lam_pos), size=N_pos, replace=True, p=p)
        out_idx.append(idx)
        out_w.append(np.full(N_pos, +scale, dtype=np.float64))

    if N_neg > 0 and tot_neg > 0:
        p = lam_neg / tot_neg
        idx = rng.choice(len(lam_neg), size=N_neg, replace=True, p=p)
        out_idx.append(idx)
        out_w.append(np.full(N_neg, -scale, dtype=np.float64))

    if not out_idx:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    indices = np.concatenate(out_idx).astype(np.int64, copy=False)
    weights = np.concatenate(out_w).astype(np.float64, copy=False)

    # mix signs
    perm = rng.permutation(indices.size)
    return indices[perm], weights[perm]


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate unbinned toys from N2LL Asimov cache.")
    ap.add_argument("config", help="Path to global YAML config.")
    ap.add_argument("--version", default=None, help="Analysis version (used in cache dir name).")
    ap.add_argument("--overwrite-cache", action="store_true", help="Force rebuild of the N2LL cache.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for toy generation.")
    ap.add_argument("--n-toys", type=int, default=1, help="Number of toy datasets to generate.")
    ap.add_argument("--mode", choices=["poisson", "fixed"], default="poisson",
                    help="Toy size: Poisson(sum λ± / scale) or fixed N.")
    ap.add_argument("--fixed-N", type=int, default=None,
                    help="If mode='fixed', use this N_toy per region.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Per-draw weight quantum for signed toys. "
                         "Default 1.0. Use 0.5 to get ~2x draws with ±0.5 weights.")

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

    # mode sanity
    if args.mode == "fixed" and args.fixed_N is None:
        ap.error("mode='fixed' requires --fixed-N to be set.")
    if args.mode == "poisson" and args.fixed_N is not None:
        ap.error("--fixed-N is only valid if mode='fixed'.")

    if not np.isfinite(args.scale) or args.scale <= 0.0:
        ap.error("--scale must be positive.")

    print(f"[toys] mode='{args.mode}', n_toys={args.n_toys}, scale={args.scale}")

    rng = np.random.default_rng(args.seed)

    # Load config + surrogates
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    Likelihood.cfg = cfg
    like_info = Likelihood.load_likelihood(cfg)
    hyp = Likelihood.build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters:")
    hyp.print()

    # Apply dynamic parameter modifications
    valid_param_names = [p.name for p in hyp.parameters]
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
        module_samples="data.samples",
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
            idx, w = sample_toy_indices_from_lambda_signed(
                lam,
                rng=rng,
                mode=args.mode,
                fixed_N=args.fixed_N,
                scale=args.scale,
            )
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w

    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    param_tag = "|".join(f"{k}_{v}" for k, v in sorted(dynamic_kwargs.items())) if dynamic_kwargs else "nominal"
    filename = f"toys_{param_tag}_mode-{args.mode}_N{args.n_toys}_scale{args.scale:g}.npz"
    toy_out = os.path.join(out_dir, filename)

    np.savez(toy_out, **store)
    print(f"[toys] Saved {args.n_toys} toys to {toy_out}")


if __name__ == "__main__":
    main()
