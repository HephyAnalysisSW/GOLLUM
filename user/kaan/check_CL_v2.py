#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

import common.yaml_loader as yaml_loader
import common.user as user

import fit.Likelihood as Likelihood
from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)


# --- copy of your helper to get λ_i(θ) per region ---
def compute_lambda_unbinned_for_region(n2ll: N2LL, hypothesis, rid: str) -> np.ndarray:
    """
    Compute event-wise rates λ_i(θ) for a single unbinned region:

        λ_i(θ) = w0_i * (1 + T_i(θ))

    using the cached Asimov events.

    Returns
    -------
    lam : np.ndarray of shape (N_region,)
    """
    class_ids = n2ll._class_ids_by_region.get(rid, [])
    if not class_ids:
        raise RuntimeError(f"[toys] Region '{rid}' has no classes.")

    N = n2ll._N_region.get(rid, 0)
    if N == 0:
        raise RuntimeError(f"[toys] Region '{rid}' has N = 0 events.")

    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis)
    nu_vals = {
        p.name: float(p.val)
        for p in getattr(hypothesis, "parameters", [])
        if not p.isPOI
    }

    ln_bias = {
        cid: sum(
            log1p_alpha * nu_vals.get(pname, 0.0)
            for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), [])
        )
        for cid in class_ids
    }

    lam = np.empty(N, dtype=np.float64)
    chunk = n2ll.eval_chunk_size
    first_cid = class_ids[0]

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        T_chunk = n2ll._compute_T_chunk(
            rid, cA_per_class, nuA_per_group, ln_bias, start, stop
        )
        w0_chunk = n2ll._h5[(rid, first_cid)]["w0"][start:stop]
        lam_chunk = w0_chunk * (1.0 + T_chunk)
        lam[start:stop] = lam_chunk

    return lam


def load_toys(toy_dir: str, pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(toy_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No toy files matching '{pattern}' found in {toy_dir}")
    vals = [np.load(p).ravel() for p in paths]
    return np.concatenate(vals)


def infer_toy_nevents_per_region(toy_dir: str, region_ids) -> dict[str, int]:
    """
    Look at toy_XXXX.npz (with '{rid}_indices') and infer N_toy per region.
    Assumes all toys use the same N_toy per region.
    """
    npz_paths = sorted(glob.glob(os.path.join(toy_dir, "toy_*.npz")))
    if not npz_paths:
        raise RuntimeError(f"No toy_*.npz files found in {toy_dir}")
    sample = np.load(npz_paths[0])
    nevents = {}
    for rid in region_ids:
        key = f"{rid}_indices"
        if key not in sample:
            raise RuntimeError(f"Key '{key}' not found in {npz_paths[0]}")
        nevents[rid] = int(len(sample[key]))
    return nevents


def main():
    ap = argparse.ArgumentParser(
        description="Compare Asimov T(c1) vs toys scaled up to full luminosity."
    )
    ap.add_argument("config", help="Path to global YAML config.")
    ap.add_argument(
        "--version",
        help="Analysis version (used in cache dir name).",
        default=None,
    )
    ap.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Force rebuild of the N2LL cache.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05 = 95% CL).",
    )
    ap.add_argument(
        "--toy-dir",
        default=os.path.join(user.output_directory, "toys"),
        help="Directory containing toy_*_T_c1_X.npy and toy_*.npz.",
    )
    ap.add_argument(
        "--c1",
        type=float,
        default=1e-3,
        help="Test value of c1 (must match what you used in toy generation).",
    )
    ap.add_argument(
        "--toy-pattern",
        default="toy_*_T_c1_1e-3.npy",
        help="Glob pattern for saved toy test statistics.",
    )
    args = ap.parse_args()

    # --- load cfg + surrogates ---
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(
        cfg, args.config, overwrite=False, prefer_numba=False
    )

    # Make cfg visible to Likelihood.N2LL.build_cache
    Likelihood.cfg = cfg

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters (all 0):")
    hyp.print()

    hyp_null = hyp.clone()
    hyp_test = hyp.cloneModify(c1=args.c1)

    print("\n[Null hypothesis] (c1 = 0):")
    hyp_null.print()
    print(f"\n[Test hypothesis] (c1 = {args.c1}):")
    hyp_test.print()

    # --- N2LL setup ---
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
    print("\n[rejection] Unbinned regions:", region_ids)

    # =========================================================
    # 1) Asimov dataset under null, evaluate T_Asimov at c1
    # =========================================================
    print("\n[Asimov] Building Asimov dataset under NULL (c1 = 0)...")
    n2ll.setAsimov(hyp_null)
    T_asimov = n2ll(hyp_test)
    print(f"[Asimov] T_Asimov (c1 = {args.c1} | Asimov(c1=0)) = {T_asimov:.6f}")

    # =========================================================
    # 2) Compute L_full (Asimov) and L_toy (per toy)
    # =========================================================
    # L_full: sum_i λ_i under null, across all unbinned regions
    L_full = 0.0
    for rid in region_ids:
        lam = compute_lambda_unbinned_for_region(n2ll, hyp_null, rid)
        L_full += float(lam.sum())
        print(f"[Lumi] Region '{rid}': lam.sum() = {lam.sum():.6f}")
    print(f"[Lumi] Total L_full (Asimov) = {L_full:.6f}")

    # L_toy: since each toy event has w=1, total weight per toy is just
    # the total number of toy events across regions
    nevents_per_region = infer_toy_nevents_per_region(args.toy_dir, region_ids)
    L_toy = float(sum(nevents_per_region.values()))
    print(f"[Lumi] nevents_per_region (from toy_*.npz): {nevents_per_region}")
    print(f"[Lumi] Total L_toy (per toy) = {L_toy:.6f}")

    if L_toy <= 0 or L_full <= 0:
        raise RuntimeError(f"Non-positive luminosity: L_toy={L_toy}, L_full={L_full}")

    # Scale factor to boost toys up to full luminosity:
    #   T_toy_scaled = T_toy * (L_full / L_toy)
    scale_toy_to_full = L_full / L_toy
    print(f"[Lumi] scale_toy_to_full = {scale_toy_to_full:.6g}")

    # =========================================================
    # 3) Load toys and scale them up
    # =========================================================
    print(f"\n[rejection] Loading toys from: {args.toy_dir}")
    T_toys = load_toys(args.toy_dir, pattern=args.toy_pattern)
    print(f"[rejection] Loaded {len(T_toys)} toy values.")
    print(
        f"[rejection] Toy stats (unscaled): "
        f"mean={T_toys.mean():.6e}, median={np.median(T_toys):.6e}, "
        f"min={T_toys.min():.3e}, max={T_toys.max():.3e}"
    )

    T_toys_scaled = T_toys * scale_toy_to_full
    print(
        f"[rejection] Toy stats (scaled to full lumi): "
        f"mean={T_toys_scaled.mean():.6f}, "
        f"median={np.median(T_toys_scaled):.6f}"
    )

    # =========================================================
    # 4) p-value: p = P(T_toy_scaled >= T_Asimov | H0)
    # =========================================================
    alpha = args.alpha
    p = float(np.mean(T_toys_scaled >= T_asimov))

    print(f"[rejection] T_Asimov (full lumi) = {T_asimov:.6f}")
    print(f"[rejection] p-value (using scaled toys) = {p:.6g}")

    if p < alpha:
        print(
            f"[rejection] p < {alpha:.3g} → "
            f"REJECT c1 = {args.c1:g} at ≈ {(1 - alpha) * 100:.1f}% CL "
            f"(full-luminosity Asimov, toys scaled up)."
        )
    else:
        print(
            f"[rejection] p ≥ {alpha:.3g} → "
            f"do NOT reject c1 = {args.c1:g} at {(1 - alpha) * 100:.1f}% CL "
            f"(full-luminosity Asimov, toys scaled up)."
        )


if __name__ == "__main__":
    main()
