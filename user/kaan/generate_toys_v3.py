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
    nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, "parameters", []) if not p.isPOI}

    ln_bias = {
        cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                 for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), []))
        for cid in class_ids
    }

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


def sample_toy_indices_from_lambda(lam: np.ndarray,
                                   rng: np.random.Generator,
                                   mode: str = "poisson",
                                   fixed_N: int | None = None) -> np.ndarray:
    """
    Sample toy indices from a discrete pmf derived from λ_i.

    Parameters
    ----------
    lam : array of λ_i (non-negative)
    rng : np.random.Generator
    mode : "poisson" or "fixed"
        - "poisson": N_toy ~ Poisson(sum(lam))
        - "fixed":   N_toy = fixed_N (required)
    fixed_N : int or None
        If mode == "fixed", use this N_toy.

    Returns
    -------
    indices : np.ndarray of shape (N_toy,)
        Event indices in [0, N_events).
    """
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.clip(lam, 0.0, None)
    total = lam.sum()
    if total <= 0:
        raise RuntimeError("[toys] Sum of λ_i is non-positive.")

    if mode == "poisson":
        N_toy = rng.poisson(total)
        print('total: ', total, '\t', 'N_toy: ', N_toy)
    elif mode == "fixed":
        if fixed_N is None:
            raise ValueError("mode='fixed' requires fixed_N.")
        N_toy = int(fixed_N)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    if N_toy == 0:
        return np.empty(0, dtype=np.int64)

    p = lam / total
    indices = rng.choice(len(lam), size=N_toy, replace=True, p=p)
    return indices

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
            idx = sample_toy_indices_from_lambda(
                lam,
                rng=rng,
                mode=args.mode,
                fixed_N=args.fixed_N,
            )
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
    
    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    param_tag = '|'.join(f"{k}_{v}" for k, v in sorted(dynamic_kwargs.items())) if dynamic_kwargs else "nominal"
    filename = f"toys_{param_tag}_mode-{args.mode}_N{args.n_toys}.npz"
    toy_out = os.path.join(out_dir, filename)

    np.savez(toy_out, **store)
    print(f"[toys] Saved {args.n_toys} toys to {toy_out}")





if __name__ == "__main__":
    main()