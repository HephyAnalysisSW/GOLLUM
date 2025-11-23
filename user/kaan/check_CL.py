#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

import common.yaml_loader as yaml_loader
import common.user as user

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)


def load_toys(toy_dir: str, pattern: str = "toy_*_T_c1_1e-4.npy") -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(toy_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No toy files matching '{pattern}' found in {toy_dir}")
    vals = [np.load(p).ravel() for p in paths]
    return np.concatenate(vals)


def main():
    ap = argparse.ArgumentParser(
        description="Compute T_obs from Asimov and compare to toys for c1=1e-4."
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
        help="Directory containing toy_*_T_c1_1e-4.npy.",
    )
    args = ap.parse_args()

    # --- load config + surrogates ---
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(
        cfg,
        args.config,
        overwrite=False,
        prefer_numba=False,
    )

    # Make cfg visible to Likelihood.N2LL.build_cache (same hack as before)
    import fit.Likelihood as Likelihood
    Likelihood.cfg = cfg

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    print("\n[Hypothesis] Initial parameters (all 0):")
    hyp.print()

    # --- define null and test hypotheses ---
    hyp_null = hyp.clone()             # all params = 0
    hyp_test = hyp.cloneModify(c1=1e-4) # test point c1 = 1

    print("\n[Null hypothesis] (c1 = 0):")
    hyp_null.print()
    print("\n[Test hypothesis] (c1 = 1e-4):")
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

    n2ll.build_cache()
    n2ll.prepare_runtime()

    # =========================================================
    # 1) Build Asimov dataset under the null: hyp_null (c1 = 0)
    # =========================================================
    print("\n[Asimov] Building Asimov dataset under the NULL (c1 = 0)...")
    n2ll.setAsimov(hyp_null)

    # Now n2ll(hypothesis) is -2 log L(hypothesis | Asimov(hyp_null))
    T_obs = n2ll(hyp_test)
    print(f"[Asimov] T_obs (c1 = 1e-4 | Asimov(c1 = 0)) = {T_obs:.6f}")

    # (optional) save it so other scripts can reuse it
    toy_dir = args.toy_dir
    os.makedirs(toy_dir, exist_ok=True)
    T_obs_path = os.path.join(toy_dir, "T_obs_c1_1e-4_Asimov_c1_0.npy")
    np.save(T_obs_path, np.array([T_obs], dtype=np.float64))
    print(f"[Asimov] Saved T_obs to {T_obs_path}")

    # =========================================================
    # 2) Load the toys generated under the null (c1 = 0)
    #    using your previous toy generator script, which
    #    saved toy_*_T_c1_1e-4.npy for each toy.
    # =========================================================
    print(f"\n[rejection] Loading toys from: {toy_dir}")
    T_toys = load_toys(toy_dir, pattern="toy_*_T_c1_1e-4.npy")
    print(f"[rejection] Loaded {len(T_toys)} toy values.")
    print(
        f"[rejection] Toy stats: mean={T_toys.mean():.6f}, "
        f"median={np.median(T_toys):.6f}"
    )

    # =========================================================
    # 3) Compute p-value: p = P(T >= T_obs | H0: c1 = 0)
    # =========================================================
    alpha = args.alpha
    p = float(np.mean(T_toys >= T_obs))
    print(f"[rejection] T_obs = {T_obs:.6f}")
    print(f"[rejection] p-value = {p:.6g}")

    if p < alpha:
        print(
            f"[rejection] p < {alpha:.3g} → "
            f"REJECT c1 = 1e-4 at ≈ { (1 - alpha) * 100:.1f}% CL (Asimov-expected)."
        )
    else:
        print(
            f"[rejection] p ≥ {alpha:.3g} → "
            f"do NOT reject c1 = 1e-4 at { (1 - alpha) * 100:.1f}% CL (Asimov-expected)."
        )


if __name__ == "__main__":
    main()
