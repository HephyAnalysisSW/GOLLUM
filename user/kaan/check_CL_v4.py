#!/usr/bin/env python

import sys
import os
import glob
import argparse
import numpy as np

from user.kaan.generate_toys_v3 import compute_lambda_unbinned_for_region

import common.yaml_loader as yaml_loader
import common.user as common_user

import fit.Likelihood as Likelihood
from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)



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


def main(config,
         version=None,
         overwrite_cache=False,
         alpha=0.05,
         toy_dir=os.path.join(common_user.output_directory, "toys"),
         c1=1e-3,
         toy_pattern="toy_*_T_c1_1e-3.npy"):
    

    # --- load cfg + surrogates ---
    cfg = yaml_loader.load_yaml(config)
    yaml_loader.print_summary(cfg, config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(
        cfg, config, overwrite=False, prefer_numba=False
    )

    # Make cfg visible to Likelihood.N2LL.build_cache
    Likelihood.cfg = cfg

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters (all 0):")
    hyp.print()

    hyp_null = hyp.clone()
    hyp_test = hyp.cloneModify(c1=c1)

    print("\n[Null hypothesis] (c1 = 0):")
    hyp_null.print()
    print(f"\n[Test hypothesis] (c1 = {c1}):")
    hyp_test.print()

    # --- N2LL setup ---
    base = os.path.splitext(os.path.basename(config))[0]
    version = version or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)

    n2ll = N2LL(
        likelihood=like_info,
        module_samples="data.samples",
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=overwrite_cache,
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
    print(f"[Asimov] T_Asimov (c1 = {c1} | Asimov(c1=0)) = {T_asimov:.6f}")

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
    nevents_per_region = infer_toy_nevents_per_region(toy_dir, region_ids)
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
    print(f"\n[rejection] Loading toys from: {toy_dir}")
    T_toys = load_toys(toy_dir, pattern=toy_pattern)
    print(T_toys)
    exit()
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
    p = float(np.mean(T_toys_scaled >= T_asimov))

    print(f"[rejection] T_Asimov (full lumi) = {T_asimov:.6f}")
    print(f"[rejection] p-value (using scaled toys) = {p:.6g}")

    if p < alpha:
        print(
            f"[rejection] p < {alpha:.3g} → "
            f"REJECT c1 = {c1:g} at ≈ {(1 - alpha) * 100:.1f}% CL "
            f"(full-luminosity Asimov, toys scaled up)."
        )
    else:
        print(
            f"[rejection] p ≥ {alpha:.3g} → "
            f"do NOT reject c1 = {c1:g} at {(1 - alpha) * 100:.1f}% CL "
            f"(full-luminosity Asimov, toys scaled up)."
        )


if __name__ == "__main__":
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
        default=os.path.join(common_user.output_directory, "toys"),
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


    main(args.config,
         args.version,
         args.overwrite_cache,
         args.alpha,
         args.toy_dir,
         args.c1,
         args.toy_pattern)
