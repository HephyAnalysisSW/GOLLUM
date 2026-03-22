#!/usr/bin/env python3
"""
Generate toys from a target PDF set by first projecting it onto the current POD basis.

Workflow:
1. Read the PDF basis definition from the YAML config.
2. Project the target PDF set onto that basis using gluon-only minimization.
3. Optionally rotate the projected vector with a Fisher rotation JSON.
4. Inject that point into the likelihood and generate signed-weight toys.

This keeps the code simple:
- the target PDF set name is the only required truth argument
- the projection logic is imported from project_pdf_to_pod_basis_v2.py
- the toy generation logic follows the current N2LL runtime
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import lhapdf
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import common.user as user
import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list
import fit.Likelihood as Likelihood
from fit.Modeling import Rotated

from project_pdf_to_pod_basis_v2 import (
    fit_gluon_coefficients,
    rotate_coefficients,
    select_x_grid,
)


def compute_lambda_unbinned_for_region(n2ll: Likelihood.N2LL, hypothesis, rid: str) -> np.ndarray:
    """
    Compute event-wise signed rates lambda_i(theta) for one unbinned region.

    The cached nominal event weights are already stored in the N2LL cache.
    We only need to evaluate the PDF/syst response T(x; theta) and combine it
    with those cached nominal weights.
    """
    class_ids = n2ll._class_ids_by_region.get(rid, [])
    if not class_ids:
        raise RuntimeError(f"[toys] Region '{rid}' has no classes.")

    n_events = n2ll._N_region.get(rid, 0)
    if n_events == 0:
        raise RuntimeError(f"[toys] Region '{rid}' has N = 0 events.")

    cA_per_class = n2ll._assemble_cA_per_class(rid, hypothesis._base)
    nuA_per_group = n2ll._assemble_nuA_groups(rid, hypothesis._base)

    # Collect nuisance values once.
    nu_vals = {}
    for p in getattr(hypothesis._base, "parameters", []):
        if not p.isPOI:
            nu_vals[p.name] = float(p.val)

    # Build the per-class lnN bias once.
    ln_bias = {}
    for cid in class_ids:
        total = 0.0
        for pname, log1p_alpha in n2ll._lnN_by_class.get((rid, cid), []):
            total += log1p_alpha * nu_vals.get(pname, 0.0)
        ln_bias[cid] = total

    lam = np.empty(n_events, dtype=np.float64)
    chunk_size = n2ll.eval_chunk_size
    first_cid = class_ids[0]

    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        rate_shift = n2ll._assemble_rate_shift_per_class(rid, hypothesis._base)
        t_chunk = n2ll._compute_T_chunk(
            rid,
            cA_per_class,
            nuA_per_group,
            ln_bias,
            rate_shift,
            start,
            stop,
        )
        w0_chunk = n2ll._h5[(rid, first_cid)]["w0"][start:stop]
        lam[start:stop] = w0_chunk * (1.0 + t_chunk)

    return lam


def sample_toy_indices_from_lambda_signed(lam: np.ndarray, rng: np.random.Generator):
    """
    Sample signed toys from event-wise signed rates.

    Positive and negative contributions are Poisson-sampled separately and then merged.
    """
    lam = np.asarray(lam, np.float64)
    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())
    if (tot_pos + tot_neg) <= 0.0:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    n_pos = int(rng.poisson(tot_pos)) if tot_pos > 0.0 else 0
    n_neg = int(rng.poisson(tot_neg)) if tot_neg > 0.0 else 0
    if (n_pos + n_neg) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    idxs, ws = [], []
    if n_pos:
        idxs.append(rng.choice(lam.size, size=n_pos, replace=True, p=lam_pos / tot_pos))
        ws.append(np.ones(n_pos, dtype=np.float64))
    if n_neg:
        idxs.append(rng.choice(lam.size, size=n_neg, replace=True, p=lam_neg / tot_neg))
        ws.append(-np.ones(n_neg, dtype=np.float64))

    idx = np.concatenate(idxs).astype(np.int64, copy=False)
    w = np.concatenate(ws).astype(np.float64, copy=False)
    perm = rng.permutation(idx.size)
    return idx[perm], w[perm]


def find_poi_pdf_job(cfg: dict) -> tuple[dict, dict]:
    """
    Find the first POI-dependent BIT job in the YAML and return:
    - the likelihood POI block
    - the matching job block from cfg['jobs']
    """
    poi_block = None
    for region in cfg.get("likelihood", {}).get("regions", []) or []:
        for cls in region.get("classes", []) or []:
            poi = cls.get("POI", None)
            if poi and poi.get("type") == "bit":
                poi_block = poi
                break
        if poi_block is not None:
            break

    if poi_block is None:
        raise RuntimeError("Could not find a POI-dependent BIT definition in the config.")

    job_id = poi_block.get("job")
    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == job_id), None)
    if job is None:
        raise RuntimeError(f"Could not find the BIT job '{job_id}' in cfg['jobs'].")

    return poi_block, job


def build_factory(cfg: dict):
    """Build the sample factory in the same way as fit/Likelihood.py."""
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    return samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate toys from a target PDF set by projecting it onto the config POD basis."
    )
    p.add_argument("config", help="Path to the YAML config.")
    p.add_argument("target_pdf_set", help="Name of the target LHAPDF set, for example PDF4LHC21_mc.")
    p.add_argument("--target-member", type=int, default=0, help="Target LHAPDF member. Default: 0.")
    p.add_argument("--Q", type=float, default=70.0, help="Scale Q used in the PDF projection.")
    p.add_argument("--x-min", type=float, default=3e-3, help="Minimum x used in the PDF projection.")
    p.add_argument("--x-max", type=float, default=0.6, help="Maximum x used in the PDF projection.")
    p.add_argument("--version", default=None, help="Analysis version used in the cache directory.")
    p.add_argument("--overwrite-cache", action="store_true", help="Rebuild the N2LL cache.")
    p.add_argument("--seed", type=int, default=123, help="Random seed for toy generation.")
    p.add_argument("--n-toys", type=int, default=1, help="Number of toy datasets to generate.")
    p.add_argument("--rotate", default=None, help="Optional Fisher rotation JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load config and surrogates
    # ------------------------------------------------------------------
    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)

    missing_cmds = yaml_loader.load_surrogates(cfg, args.config, overwrite=False)
    if missing_cmds:
        print(f"[error] Found {len(missing_cmds)} missing surrogate(s).")
        print("Train or copy the missing artifacts first:")
        for cmd in missing_cmds:
            print(cmd)
        raise RuntimeError("Missing surrogate artifacts. Aborting toy generation.")

    # ------------------------------------------------------------------
    # Read the basis definition directly from the config
    # ------------------------------------------------------------------
    poi_block, poi_job = find_poi_pdf_job(cfg)
    pdf_cfg = poi_job.get("pdf", {}) or {}
    basis_members = list(pdf_cfg.get("pdf_n", []) or [])
    basis_type = pdf_cfg.get("pdf_type", None)
    basis_set = pdf_cfg.get("pdf_basis", None)

    if basis_type != "PODBasis":
        raise RuntimeError(f"This toy generator expects PODBasis, got '{basis_type}'.")
    if not basis_members or basis_set is None:
        raise RuntimeError("The config is missing pdf_n and/or pdf_basis in the BIT job.")

    # ------------------------------------------------------------------
    # Project the target PDF onto the basis
    # ------------------------------------------------------------------
    print("\n[projection] Building the truth vector from the target PDF set...")
    from pdf.PDFParametrization import PDFParametrization

    pdf = PDFParametrization(n=basis_members, typ="PODBasis", basis=basis_set, active_pids="all")
    target_pdf = lhapdf.mkPDF(args.target_pdf_set, args.target_member)
    x_grid = select_x_grid(args.x_min, args.x_max)
    fit_result = fit_gluon_coefficients(pdf, target_pdf, x_grid, args.Q)
    base_coeffs = np.asarray(fit_result["coeffs"], dtype=float)
    base_coeff_map = {f"c{i}": float(val) for i, val in enumerate(base_coeffs)}

    rotated_info = rotate_coefficients(base_coeffs, args.rotate) if args.rotate else None
    injected_map = rotated_info["rotated_map"] if rotated_info is not None else base_coeff_map

    print("[projection] Base coefficients:")
    for name, val in base_coeff_map.items():
        print(f"  {name:>3s} = {val: .8e}")
    if rotated_info is not None:
        print("[projection] Rotated coefficients:")
        for name, val in rotated_info["rotated_map"].items():
            print(f"  {name:>3s} = {val: .8e}")

    # ------------------------------------------------------------------
    # Build the hypothesis and inject the projected point
    # ------------------------------------------------------------------
    Likelihood.cfg = cfg
    like_info = Likelihood.load_likelihood(cfg)
    hyp = Likelihood.build_hypothesis_from_likelihood(like_info, name="SR")

    if args.rotate:
        print("[rotation] Rotation JSON provided. Using rotated POIs in toy generation.")
        hyp = Rotated(hyp, args.rotate, name="Fisher-basis")

    hyp_test = hyp.cloneModify(**injected_map)
    print("\n[truth hypothesis] Injected parameters:")
    hyp_test.print()

    # ------------------------------------------------------------------
    # Build N2LL runtime and generate toys
    # ------------------------------------------------------------------
    base = os.path.splitext(os.path.basename(args.config))[0]
    version = args.version or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)
    factory = build_factory(cfg)

    n2ll = Likelihood.N2LL(
        likelihood=like_info,
        factory=factory,
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=args.overwrite_cache,
    )
    n2ll.build_cache()
    n2ll.prepare_runtime()

    region_ids = [region["id"] for region in n2ll.regions]
    print("\n[toys] Unbinned regions:", region_ids)

    store = {}
    for rid in region_ids:
        lam = compute_lambda_unbinned_for_region(n2ll, hyp_test, rid)
        lam_pos = np.clip(lam, 0.0, None)
        lam_neg = np.clip(-lam, 0.0, None)
        print(f"[toys] rid={rid}  sum(lam+)={lam_pos.sum():.6g}  sum(lam-)={lam_neg.sum():.6g}")

        for itoy in range(args.n_toys):
            if itoy % 100 == 0: print('itoy: ', itoy)
            idx, w = sample_toy_indices_from_lambda_signed(lam, rng=rng)
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w

    # ------------------------------------------------------------------
    # Save toys and the projection metadata used to generate them
    # ------------------------------------------------------------------
    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    rot_tag = "_rotate" if args.rotate else ""
    toy_stem = f"toys_{args.target_pdf_set}_m{args.target_member}{rot_tag}_N{args.n_toys}"
    toy_path = os.path.join(out_dir, f"{toy_stem}.npz")
    meta_path = os.path.join(out_dir, f"{toy_stem}.json")

    np.savez(toy_path, **store)

    meta = {
        "config": os.path.abspath(args.config),
        "target_pdf_set": args.target_pdf_set,
        "target_member": int(args.target_member),
        "Q": float(args.Q),
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "basis_set": basis_set,
        "basis_members": basis_members,
        "base_coefficients": base_coeff_map,
        "rotated_coefficients": None if rotated_info is None else rotated_info["rotated_map"],
        "rotation_file": None if rotated_info is None else args.rotate,
        "gluon_abs_rms": float(fit_result["abs_rms"]),
        "gluon_rel_rms": float(fit_result["rel_rms"]),
        "n_toys": int(args.n_toys),
        "seed": int(args.seed),
        "toy_file": toy_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)

    print(f"\n[toys] Saved {args.n_toys} toys to {toy_path}")
    print(f"[toys] Saved projection metadata to {meta_path}")


if __name__ == "__main__":
    main()
