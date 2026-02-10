#!/usr/bin/env python

"""
Generate unbinned toy datasets from N2LL Asimov cache.
One can pass POIs and nuisance parameters via command line arguments to sample according to alternative hypotheses.
If no parameters are passed, toys are generated according to the nominal (null) hypothesis.


Usage example:
    python generate_toys_v3.py configs/unbinned_merged.yaml --mode poisson --n-toys 1
    python generate_toys_v3.py configs/unbinned_merged.yaml --mode poisson --n-toys 1 --c1 0.5 --c2 1.0 --nu_pu 0.2
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
    Compute event-wise rates λ_i(θ) for a single unbinned region:
        dσ(x; c,ν)/dx = dσ(x; 0,0)/dx * (1 + T(x; c,ν))
        w0_i = dσ(x_i; 0,0)/dx * (integrated luminosity)
        λ_i(θ) = w0_i * (1 + T_i(θ))
        

    using the cached Asimov events.

    Returns
    -------
    lam : np.ndarray of shape (N_region,)
    """
    class_ids = n2ll._class_ids_by_region.get(rid, [])
    if not class_ids:
        raise RuntimeError(f"[toys] Region '{rid}' has no classes.")

    # N: total number of Asimov events
    N = n2ll._N_region.get(rid, 0)
    if N == 0:
        raise RuntimeError(f"[toys] Region '{rid}' has N = 0 events.")

    # Current (c, ν) → A-basis and lnN-bias
    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis)

    nu_vals = {}
    for p in getattr(hypothesis, "parameters", []):
        if not p.isPOI:
            nu_vals[p.name] = float(p.val)

    ln_bias = {}
    for cid in class_ids:
        ln_bias[cid] = 0.0
        for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), []):
            nu_val = nu_vals.get(pname, 0.0)
            ln_bias[cid] += log1p_alpha * nu_val

    lam = np.empty(N, dtype=np.float64)
    chunk = n2ll.eval_chunk_size
    first_cid = class_ids[0] # just pick one class to read nominal weights w0 from

    # Loop over cached events in chunks, like Asimov mode
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        T_chunk = n2ll._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
        w0_chunk = n2ll._h5[(rid, first_cid)]["w0"][start:stop]
        lam_chunk = w0_chunk * (1.0 + T_chunk)   # our convention
        lam[start:stop] = lam_chunk

    return lam


def sample_toy_indices_from_lambda_signed(lam: np.ndarray,
                                          rng: np.random.Generator,
                                          mode: str = "poisson",
                                          fixed_N: int | None = None,
                                          scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Signed bootstrap sampling for possibly negative lam.

    Returns
    -------
    indices : (N_toy,) int
    weights : (N_toy,) float   in {+scale, -scale}
    """
    lam = np.asarray(lam, dtype=np.float64)

    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())
    tot_abs = tot_pos + tot_neg
    if tot_abs <= 0:
        raise RuntimeError("[toys] Sum of |λ_i| is non-positive.")

    # choose toy sizes
    if mode == "poisson":
        N_pos = int(rng.poisson(tot_pos)) if tot_pos > 0 else 0
        N_neg = int(rng.poisson(tot_neg)) if tot_neg > 0 else 0
    elif mode == "fixed":
        if fixed_N is None:
            raise ValueError("mode='fixed' requires fixed_N.")
        fixed_N = int(fixed_N)
        if fixed_N < 0:
            raise ValueError("fixed_N must be >= 0.")
        # split fixed_N into + and - parts proportional to tot_pos/tot_abs
        p_pos = tot_pos / tot_abs
        N_pos = int(rng.binomial(fixed_N, p_pos))
        N_neg = fixed_N - N_pos
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    if (N_pos + N_neg) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    out_idx = []
    out_w   = []

    if N_pos > 0:
        p = lam_pos / tot_pos
        idx = rng.choice(len(lam_pos), size=N_pos, replace=True, p=p)
        out_idx.append(idx)
        out_w.append(np.full(N_pos, +scale, dtype=np.float64))

    if N_neg > 0:
        p = lam_neg / tot_neg
        idx = rng.choice(len(lam_neg), size=N_neg, replace=True, p=p)
        out_idx.append(idx)
        out_w.append(np.full(N_neg, -scale, dtype=np.float64))

    indices = np.concatenate(out_idx).astype(np.int64, copy=False)
    weights = np.concatenate(out_w).astype(np.float64, copy=False)

    # shuffle so + and - draws are mixed
    perm = rng.permutation(indices.size)
    return indices[perm], weights[perm]


# ----------------------------------------------------------------------


def main():
    # -------------------------------
    # Parse known + unknown arguments
    # -------------------------------
    ap = argparse.ArgumentParser(description="Generate unbinned toys from N2LL Asimov cache.")
    ap.add_argument("config", help="Path to global YAML config.")
    ap.add_argument("--version", help="Analysis version (used in cache dir name).",
                    default=None)
    ap.add_argument("--overwrite-cache", action="store_true",
                    help="Force rebuild of the N2LL cache.")
    ap.add_argument("--seed", type=int, default=123,
                    help="Random seed for toy generation.")
    ap.add_argument("--n-toys", type=int, default=1,
                    help="Number of toy datasets to generate.")
    ap.add_argument("--mode", choices=["poisson", "fixed"], default="poisson",
                    help="Toy size: Poisson(sum λ_i) or fixed N.")
    ap.add_argument("--fixed-N", type=int, default=None,
                    help="If mode='fixed', use this N_toy per region.")

    # parse known + unknown args
    args, unknown = ap.parse_known_args()

    # -----------------------------------------
    # Parse dynamic parameter modifications
    # -----------------------------------------
    #
    # Accepts --c1 1.0 --c4 0.5 --nu_pu 0.2 ...
    #
    dynamic_kwargs = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith("--"):
            name = key[2:]        # drop leading --
            if (i + 1) < len(unknown):
                dynamic_kwargs[name] = unknown[i + 1]
                i += 2
            else:
                raise RuntimeError(f"Missing value for parameter {key}")
        else:
            raise RuntimeError(f"We expect you to pass POIs, and nuisance parameters with -- prefix.")

    print("\n[dynamic params] Parsed:", dynamic_kwargs)


    # Ensure correct usage of 'fixed' and 'poisson' modes.
    # ---------------------------------------------------
    if args.mode == "fixed" and args.fixed_N is None:
        ap.error("mode='fixed' requires --fixed-N to be set.")
    elif args.mode == "poisson" and args.fixed_N is not None:
        ap.error("--fixed-N is only valid if mode='fixed'.")
    else:
        print(f"[toys] Generating toys in mode '{args.mode}'.")
    # ---------------------------------------------------


    # Set seed for random number generator
    rng = np.random.default_rng(args.seed)


    # Load config and surrogates 
    # ---------------------------------------------------
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config,
                                overwrite=False,
                                prefer_numba=False)

    Likelihood.cfg = cfg

    like_info = Likelihood.load_likelihood(cfg)
    hyp = Likelihood.build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters:")
    hyp.print()



    # Build hypothesis
    # -----------------------------------------

    valid_param_names = [p.name for p in hyp.parameters]
    kwargs = {}
    for k, v in dynamic_kwargs.items():
        if k in valid_param_names:
            kwargs[k] = float(v)
        else:
            raise ValueError(f"Unknown model parameter: {k}\n" +
                             "Check the arguments passed to argparser. Here are the available parameters to the model:\n" +
                             f"{valid_param_names}")


    hyp_test = hyp.cloneModify(**kwargs)
    print("\n[Test hypothesis] Modified parameters:")
    hyp_test.print()




    # Set up N2LL
    # ---------------------------------------------------
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

    # Build cache if needed, then open for runtime
    n2ll.build_cache()
    n2ll.prepare_runtime()

    # For toy generation we don't need to call setAsimov / setObservation.
    # We just use the cached Asimov event set as the support.

    region_ids = [R["id"] for R in n2ll.regions]
    print("\n[toys] Unbinned regions:", region_ids)
    print('Number of toys: ', args.n_toys)

    # Store all toys for this hypothesis
    store = {}


    for rid in region_ids:
        lam = compute_lambda_unbinned_for_region(n2ll, hyp_test, rid)
        for itoy in range(args.n_toys):
            idx, w = sample_toy_indices_from_lambda_signed(
                lam,
                rng=rng,
                mode=args.mode,
                fixed_N=args.fixed_N,
            )
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w
    
    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    param_tag = '|'.join(f"{k}_{v}" for k, v in sorted(dynamic_kwargs.items())) if dynamic_kwargs else "nominal"
    filename = f"toys_{param_tag}_mode-{args.mode}_N{args.n_toys}.npz"
    toy_out = os.path.join(out_dir, filename)

    np.savez(toy_out, **store)
    print(f"[toys] Saved {args.n_toys} toys to {toy_out}")





if __name__ == "__main__":
    main()