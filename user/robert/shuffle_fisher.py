#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np

# Import your likelihood machinery
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)
import common.yaml_loader as yaml_loader


def set_free_values(hyp, names, values):
    """Assign values (array-like) to the corresponding parameter names in hyp."""
    for nm, v in zip(names, values):
        getattr(hyp, nm).val = float(v)


def clone_with_delta(hyp, names, delta):
    """Clone hypothesis and shift free params by delta vector."""
    h = hyp.clone()
    for nm, dv in zip(names, delta):
        getattr(h, nm).val = float(getattr(h, nm).val) + float(dv)
    return h


p = argparse.ArgumentParser(
    description="Compare true -2logL to Fisher quadratic approximation near (0,0) Asimov."
)
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--cache-root", default=None, help="Override cache root (optional)")
p.add_argument("--overwrite", action="store_true", help="Overwrite caches")
p.add_argument("--shuffle", nargs="*", default=[], help="Shuffle these features")
p.add_argument("--module-samples", default="data.samples",
               help="Python module with sample loaders")
p.add_argument("--step", type=float, default=1e-2,
               help="Base step size for parameter probes")
p.add_argument("--n-probes", type=int, default=20,
               help="Number of probe displacements")
p.add_argument("--seed", type=int, default=12345, help="Random seed for probe generation")
p.add_argument("--axis-only", action="store_true",
               help="Use only single-parameter axis probes")
p.add_argument("--fi-step-scale", type=float, default=1e-4,
               help="Step scale for Fisher binned/penalty finite diffs")
args = p.parse_args()

# ---- Load YAML + surrogates ----
cfg = yaml_loader.load_yaml(args.config)
yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

# ---- Likelihood info & hypothesis scaffold ----
like_info = load_likelihood(cfg)
hyp = build_hypothesis_from_likelihood(like_info, name="SR")

# Keep track of the free parameters (not frozen and not ignored)
free_params = [p for p in hyp.parameters
               if not p.isFrozen and not getattr(p, "isIgnored", False)]
free_names  = [p.name for p in free_params]
if not free_names:
    raise RuntimeError("No free parameters to probe.")
print("\n[free] Parameters to probe:")
for nm in free_names:
    print("   -", nm)

# Everyday I'm shuffeling https://youtu.be/KQ6zr6kCPj8?t=73
shuffle_suffix = "_".join( ["shuffle"]+sorted(args.shuffle) )

# ---- Build caches, prepare runtime ----
n2ll = N2LL(
    like_info,
    args.module_samples,
    os.path.join(
        "NN2LCache",
        os.path.splitext(os.path.basename(args.config))[0],
        str(cfg.get("version", "v0"))+"_"+ shuffle_suffix,
    ),
    cache_root=args.cache_root,
    overwrite=args.overwrite,
)
n2ll.shuffle_features = args.shuffle
n2ll.build_cache()
n2ll.prepare_runtime()

# ---- Set A-simov at (0,0) ----
# By convention setAsimov(None) activates A-simov with zero bias term.
n2ll.setAsimov()  # A-simov (c' = 0, ν' = 0)

# ---- Baseline -2logL at (0,0) ----
# All params were initialized to 0 in build_hypothesis_from_likelihood.
f0 = float(n2ll(hyp))
print(f"\n[baseline] -2 log L (Asimov, at 0) = {f0:.6e}")

# ---- Fisher information at (0,0) ----
# N2LL.fisher_information() returns the regular Fisher matrix I_ab
# (i.e. E[∂_a log L ∂_b log L]). For small displacements δ, the
# quadratic approximation of Δ(-2logL) is δ^T I δ.
print("\n[FI] Computing Fisher information at (0,0) in Asimov A-mode…")
I = n2ll.fisher_information(hyp, step_scale=args.fi_step_scale, verbose=True)

# Numerical symmetrization for safety
I = 0.5 * (I + I.T)

# Some quick diagnostics on I
evals, _ = np.linalg.eigh(I)
print("\n[FI] Eigenvalues of Fisher matrix:")
for i, ev in enumerate(evals):
    print(f"  λ[{i}] = {ev:.6e}")

# ---- Build probe directions ----
rng = np.random.default_rng(args.seed)
npar = len(free_names)
probes = []

if args.axis_only:
    # a handful of ±axis directions
    for i in range(npar):
        d = np.zeros(npar); d[i] = args.step
        probes.append(d.copy())
        d[i] = -args.step
        probes.append(d.copy())
else:
    # Mix of axis and random combos
    # 40% axis probes, 60% random small Gaussian steps scaled to have norm ~ step
    n_axis = max(2 * npar, int(0.4 * args.n_probes))
    n_rand = max(0, args.n_probes - n_axis)

    # axis
    for i in range(npar):
        if len(probes) >= n_axis:
            break
        d = np.zeros(npar); d[i] = args.step
        probes.append(d.copy())
        if len(probes) >= n_axis:
            break
        d[i] = -args.step
        probes.append(d.copy())

    # random combos
    for _ in range(n_rand):
        d = rng.normal(size=npar)
        # scale to have L2 norm ≈ step
        norm = np.linalg.norm(d) + 1e-15
        d = d / norm * args.step
        probes.append(d)

# ---- Evaluate true Δ(-2logL) and quadratic approx δ^T I δ ----
rows = []
max_abs_err = 0.0
max_rel_err = 0.0

print("\n[compare] true Δ(-2logL) vs quadratic δ^T I δ  (Asimov @ 0)")
print(" index | true_d2logL         approx_quad         rel_err       | δ (first 6 comps)")
print("-" * 96)

for idx, d in enumerate(probes):
    h_shift = clone_with_delta(hyp, free_names, d)
    f = float(n2ll(h_shift))
    true_delta = f - f0
    approx = float(d @ (I @ d))  # For -2logL, quadratic coefficient is I
    rel = abs(true_delta - approx) / (abs(true_delta) + 1e-12)

    max_abs_err = max(max_abs_err, abs(true_delta - approx))
    max_rel_err = max(max_rel_err, rel)

    d_preview = " ".join(f"{x:+.2e}" for x in d[:6])
    print(f"{idx:5d} | {true_delta:+.6e}   {approx:+.6e}   {rel:8.3e} | {d_preview}")

    rows.append({
        "index": idx,
        "true_delta": float(true_delta),
        "approx_delta": float(approx),
        "rel_err": float(rel),
        "delta": d.tolist(),
    })

print("\n[max errors]")
print(f"  max |true - approx| = {max_abs_err:.6e}")
print(f"  max relative error = {max_rel_err:.6e}")

