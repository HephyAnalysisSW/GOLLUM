#!/usr/bin/env python3
"""
Generate nominal toys directly from the nominal MC event weights.

No PDF reweighting is applied.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import common.user as user
import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list


def parse_args():
    parser = argparse.ArgumentParser(description="Generate nominal toys from nominal MC weights.")
    parser.add_argument("config", help="Path to the YAML config.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed. Default: 123.")
    parser.add_argument("--n-toys", type=int, default=1, help="Number of toys to generate. Default: 1.")
    return parser.parse_args()


def build_factory(cfg):
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    return samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )


def iter_region_weight_batches(region, factory):
    for sample_name in (region.get("classifier", {}) or {}).get("asimov", []):
        loader = factory.get(sample_name)

        old_split = None
        if hasattr(loader, "n_split"):
            old_split = loader.n_split
            loader.n_split = 100

        n_shards = len(getattr(loader, "base", loader))
        for shard in range(n_shards):
            _, weights = loader.materialize(shard=shard, what="ow", n=None)
            yield np.asarray(weights, dtype=np.float64)

        if old_split is not None:
            loader.n_split = old_split


def compute_lambda_for_region(region, factory):
    return np.concatenate(list(iter_region_weight_batches(region, factory)), axis=0)


def sample_toy_indices_from_lambda_signed(lam, rng):
    lam = np.asarray(lam, np.float64)
    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())

    n_pos = int(rng.poisson(tot_pos)) if tot_pos > 0.0 else 0
    n_neg = int(rng.poisson(tot_neg)) if tot_neg > 0.0 else 0

    idxs = []
    ws = []

    if n_pos:
        idxs.append(rng.choice(lam.size, size=n_pos, replace=True, p=lam_pos / tot_pos))
        ws.append(np.ones(n_pos, dtype=np.float64))
    if n_neg:
        idxs.append(rng.choice(lam.size, size=n_neg, replace=True, p=lam_neg / tot_neg))
        ws.append(-np.ones(n_neg, dtype=np.float64))
    if not idxs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    idx = np.concatenate(idxs).astype(np.int64, copy=False)
    w = np.concatenate(ws).astype(np.float64, copy=False)
    perm = rng.permutation(idx.size)
    return idx[perm], w[perm]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)

    factory = build_factory(cfg)
    regions = list((cfg.get("likelihood", {}) or {}).get("regions", []) or [])

    store = {}

    for region in regions:
        rid = region["id"]
        lam = compute_lambda_for_region(region, factory)
        lam_pos = np.clip(lam, 0.0, None)
        lam_neg = np.clip(-lam, 0.0, None)

        print(f"[toys] rid={rid}  N={lam.size}  sum(lam+)={lam_pos.sum():.6g}  sum(lam-)={lam_neg.sum():.6g}")

        for itoy in range(args.n_toys):
            idx, w = sample_toy_indices_from_lambda_signed(lam, rng=rng)
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w

    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    toy_path = os.path.join(out_dir, f"toys_nominal_N{args.n_toys}.npz")
    np.savez(toy_path, **store)

    print(f"\n[toys] Saved {args.n_toys} toys to {toy_path}")


if __name__ == "__main__":
    main()
