#!/usr/bin/env python

"""
Example toy generator using Likelihood.N2LL

Usage:
    python generate_toys.py path/to/config.yaml \
        --version v1 \
        --n-toys 3 \
        --seed 123

This:
  - loads your config & surrogates
  - builds/opens the N2LL cache
  - for each toy:
      * computes per-event rates λ_i(θ) = w0_i * (1 + T_i(θ))
      * samples toy events per region from this discrete pmf
  - prints a small summary

Next step (separate): feed these toys back into N2LL as observations.
"""
import time
t0 = time.time()

import argparse
import os
import numpy as np


import common.yaml_loader as yaml_loader
import common.user as user

import fit.Likelihood as Likelihood
from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL


# ----------------------------------------------------------------------
# Helpers to compute λ_i(θ) and sample toys
# ----------------------------------------------------------------------
def compute_lambda_unbinned_for_region(n2ll: N2LL, hypothesis, rid: str) -> np.ndarray:
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
        print('total:', total)
        print('N_toy:', N_toy)
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


def build_by_class_slices_for_indices(n2ll: N2LL, rid: str, indices: np.ndarray) -> dict:
    """
    For a given region and a set of event indices, collect the per-class
    arrays (g, R, Δ...) from the HDF5 caches.

    Returns
    -------
    by_class : dict
        {
          cid: {
            "g":  (N_toy,),
            "R":  (N_toy, nA),
            "Delta::<sysid>": (N_toy, nB),
            ...
          },
          ...
        }
    """
    by_class = {}
    indices = np.asarray(indices, dtype=np.int64)

    # No events in this toy for this region
    if indices.size == 0:
        for cid in n2ll._class_ids_by_region[rid]:
            by_class[cid] = {
                "g": np.empty(0, dtype=np.float64),
                "R": np.empty((0, 0), dtype=np.float64),
            }
        return by_class

    # h5py wants strictly increasing indices, so we:
    #  - take unique sorted indices for reading
    #  - then expand back to full toy via "inverse" mapping
    unique_idx, inverse = np.unique(indices, return_inverse=True)

    for cid in n2ll._class_ids_by_region[rid]:
        f = n2ll._h5[(rid, cid)]
        meta = n2ll._meta[(rid, cid)]

        comp = {}

        # Read unique rows once
        g_unique = f["g"][unique_idx]
        R_unique = f["R"][unique_idx, :]

        # Expand back to N_toy using inverse
        comp["g"] = g_unique[inverse]
        comp["R"] = R_unique[inverse, :]

        # Same trick for all Delta groups
        for gm in meta.get("delta_groups", []):
            dset_name = gm.get("dset", f"Delta::{gm['id']}")
            D_unique = f[dset_name][unique_idx, :]
            comp[dset_name] = D_unique[inverse, :]

        by_class[cid] = comp

    return by_class



# Likelihood evaluation with toy data
# ----------------------------------------------------------------------
def set_observation_from_toy(n2ll: N2LL, toy_data: dict) -> None:
    """
    Put a toy dataset into N2LL in observation mode so that n2ll(hypothesis)
    evaluates -2 log L for that toy.

    toy_data: dict[rid] -> {
        "by_class": {...},   # output of build_by_class_slices_for_indices
        "w": np.ndarray,     # (N_toy,)
    }
    """
    import numpy as np

    # Disable any Asimov state
    n2ll._asimov_hypothesis_set = False
    n2ll._asimov_active = False
    n2ll._asimov_hyp = None
    n2ll._asimov_T.clear()
    n2ll._binned_asimov_lambda.clear()

    # Reset observation containers
    n2ll._obs_unbinned = {}
    n2ll._obs_binned = {}

    # Flag that we’re now in observed-data mode
    n2ll._observation_set = True

    # Fill unbinned observation blocks
    for rid, block in toy_data.items():
        n2ll._obs_unbinned[rid] = {
            "by_class": block["by_class"],
            "w": np.asarray(block["w"], dtype=np.float64),
        }





# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
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
    args = ap.parse_args()

    if args.mode == "fixed" and args.fixed_N is None:
        ap.error("mode='fixed' requires --fixed-N to be set.")
    elif args.mode == "poisson" and args.fixed_N is not None:
        ap.error("--fixed-N is only valid if mode='fixed'.")
    else:
        print(f"[toys] Generating toys in mode '{args.mode}'.")

    rng = np.random.default_rng(args.seed)

    # --- load config + surrogates (like Likelihood.__main__) ---
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config,
                                overwrite=False,
                                prefer_numba=False)

    # Make cfg visible to Likelihood.N2LL.build_cache(), which expects a
    # module-global variable named `cfg` (hack but effective).
    Likelihood.cfg = cfg

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters:")
    hyp.print()

    # --- define null and test hypotheses ---
    # Null: all params = 0 (already the case)
    hyp_null = hyp.clone()

    # Test: c1 = 1.000
    # (assumes 'c1' is indeed the POI name in your model)
    hyp_test = hyp.cloneModify(c1=1.000)
    print("\n[Test hypothesis] c1 = 1.000:")
    hyp_test.print()




    # --- set up N2LL ---
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

    # Build cache if needed, then open for runtime
    n2ll.build_cache()
    n2ll.prepare_runtime()

    # For toy generation we don't need to call setAsimov / setObservation.
    # We just use the cached Asimov event set as the support.

    region_ids = [R["id"] for R in n2ll.regions]
    print("\n[toys] Unbinned regions:", region_ids)

    # Certain checks...
    print("\n[Debug] POI A-basis per region/class:")
    for rid in region_ids:
        for cid in n2ll._class_ids_by_region[rid]:
            poi_names = n2ll._poi_order[(rid, cid)]
            print(f"  ({rid}, {cid}): poi_names = {poi_names}")

    rid0 = region_ids[0]
    cid0 = n2ll._class_ids_by_region[rid0][0]
    R = np.array(n2ll._h5[(rid0, cid0)]['R'][0:10, :])
    print("[Debug] R head for", rid0, cid0, ":\n", R)

    n2ll.setAsimov()  # A-simov, null (c=0) by default
    val0 = n2ll(hyp_null)
    val1 = n2ll(hyp_test)
    print("Asimov N2LL at c1=0:", val0)
    print("Asimov N2LL at c1=1.000:", val1)


    print('args.n_toys: ', args.n_toys)

    T_toys = []  # store test statistics per toy

    for itoy in range(args.n_toys):
        print(f"\n================= TOY {itoy} =================")

        toy_data = {}  # rid -> {"indices", "by_class", "w"}
        print('region_ids: ', region_ids)
        for rid in region_ids:
            # generate toys under the null (c1 = 0)
            lam = compute_lambda_unbinned_for_region(n2ll, hyp_null, rid)
            
            print('alternative hypothesis weights:')
            print('dσ(x; c,ν)/dx = dσ(x; 0,0)/dx * (1 + T(x; c,ν))')
            print('w0_i = dσ(x_i; 0,0)/dx * (integrated luminosity)')
            print('λ_i(θ) = w0_i * (1 + T_i(θ))')
            print('lam[:5]:', lam[:5])

            idx = sample_toy_indices_from_lambda(
                lam,
                rng=rng,
                mode=args.mode,
                fixed_N=args.fixed_N,
            )

            


            by_class = build_by_class_slices_for_indices(n2ll, rid, idx)
            # For now, unit weights for toy events
            w_toy = np.ones(len(idx), dtype=np.float64)

            toy_data[rid] = {
                "indices": idx,
                "by_class": by_class,
                "w": w_toy,
            }

            print(f"[toys] Region '{rid}': "
                  f"N_events_support={len(lam)}, "
                  f"N_toy={len(idx)}, "
                  f"λ_sum={lam.sum():.3g}")

        # At this point `toy_data` holds one full toy dataset.
        # You can:
        #   - save it to disk (np.savez, pickle, HDF5, ...)
        #   - or in the next step, plug it back into n2ll._obs_unbinned
        #     and call n2ll(hypothesis) to evaluate the likelihood.
        #
        # Example: save a very small summary to npz (optional)
        out_dir = os.path.join(user.output_directory, "toys")
        os.makedirs(out_dir, exist_ok=True)
        toy_out = os.path.join(out_dir, f"toy_{itoy:04d}.npz")

        # Store only indices + weights as a demo; by_class is nested and
        # would need pickling or h5 if you want everything.
        np.savez(
            toy_out,
            **{
                f"{rid}_indices": toy_data[rid]["indices"]
                for rid in toy_data.keys()
            },
        )
        print(f"[toys] Saved index summary to {toy_out}")

        # Example: set toy as observation and evaluate likelihood (optional)
        set_observation_from_toy(n2ll, toy_data)

        # Optional sanity check: N2LL at null should be ~0
        n2ll_null = n2ll(hyp_null)
        print(f"[toys] Toy {itoy}: -2 log L(null, c1=0) = {n2ll_null:.6f}")

        # Test statistic: evaluate at c1 = 1.000
        T_toy = n2ll(hyp_test)
        T_toys.append(T_toy)
        print(f"[toys] Toy {itoy}: T(c1=1.000) = {T_toy:.6f}")


        
        # Example: append to a list or save per toy
        # (here just simple npy store)
        out_dir = os.path.join(user.output_directory, "toys")
        os.makedirs(out_dir, exist_ok=True)
        ll_out = os.path.join(out_dir, f"toy_{itoy:04d}_T_c1_1.000.npy")
        np.save(ll_out, np.array([T_toy], dtype=np.float64))
        print(f"[toys] Saved T(c1=1.000) for toy {itoy} to {ll_out}")




if __name__ == "__main__":
    main()
