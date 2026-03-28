#!/usr/bin/env python3

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import common.user as user
import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot x1, x2, and Generator_scalePDF distributions for the events used in toy generation."
    )
    parser.add_argument("config", help="Path to the YAML config.")
    parser.add_argument("--x-bins", type=int, default=60, help="Number of bins for x1/x2. Default: 60.")
    parser.add_argument("--scale-bins", type=int, default=60, help="Number of bins for Generator_scalePDF. Default: 60.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to common.user.output_directory/generator_distributions.",
    )
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


def iter_region_observer_weight_batches(region, factory):
    for sample_name in (region.get("classifier", {}) or {}).get("asimov", []):
        loader = factory.get(sample_name)

        old_split = None
        if hasattr(loader, "n_split"):
            old_split = loader.n_split
            loader.n_split = 100

        n_shards = len(getattr(loader, "base", loader))
        for shard in range(n_shards):
            observers, weights = loader.materialize(shard=shard, what="ow", n=None)
            yield loader.observer_names, np.asarray(observers, dtype=np.float64), np.asarray(weights, dtype=np.float64)

        if old_split is not None:
            loader.n_split = old_split


def collect_region_arrays(region, factory):
    x1_all = []
    x2_all = []
    scale_all = []
    weight_all = []

    for observer_names, observers, weights in iter_region_observer_weight_batches(region, factory):
        idx = {name: i for i, name in enumerate(observer_names)}
        x1_all.append(observers[:, idx["Generator_x1"]])
        x2_all.append(observers[:, idx["Generator_x2"]])
        scale_all.append(observers[:, idx["Generator_scalePDF"]])
        weight_all.append(weights)

    return (
        np.concatenate(x1_all, axis=0),
        np.concatenate(x2_all, axis=0),
        np.concatenate(scale_all, axis=0),
        np.concatenate(weight_all, axis=0),
    )


def positive_log_bins(values, n_bins):
    positive = values[values > 0.0]
    vmin = positive.min()
    vmax = positive.max()
    return np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)


def make_x1_x2_plot(rid, x1, x2, weights, outdir, n_bins):
    bins = positive_log_bins(np.concatenate([x1, x2]), n_bins)

    plt.figure(figsize=(8, 5))
    plt.hist(
        x1,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=2.0,
        color="red",
        label="x1",
    )
    plt.hist(
        x2,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=2.0,
        color="blue",
        label="x2",
    )
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("Weighted events")
    plt.title(f"{rid}: x1 and x2 distributions")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = os.path.join(outdir, f"{rid}_x1_x2_distribution.png")
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[plot] Saved {out_path}")


def make_scale_plot(rid, scale, weights, outdir, n_bins):
    bins = positive_log_bins(scale, n_bins)

    plt.figure(figsize=(8, 5))
    plt.hist(
        scale,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=2.0,
        color="gold",
        label="Generator_scalePDF",
    )
    plt.xscale("log")
    plt.xlabel("Generator_scalePDF")
    plt.ylabel("Weighted events")
    plt.title(f"{rid}: Generator_scalePDF distribution")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = os.path.join(outdir, f"{rid}_scale_distribution.png")
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[plot] Saved {out_path}")


def main():
    args = parse_args()

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)

    factory = build_factory(cfg)
    regions = list((cfg.get("likelihood", {}) or {}).get("regions", []) or [])

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(user.output_directory, "generator_distributions")
    os.makedirs(outdir, exist_ok=True)

    for region in regions:
        rid = region["id"]
        x1, x2, scale, weights = collect_region_arrays(region, factory)

        print(f"[region] {rid}")
        print(f"  n_events   = {len(weights)}")
        print(f"  sum_weight = {weights.sum():.6g}")

        make_x1_x2_plot(rid, x1, x2, weights, outdir, args.x_bins)
        make_scale_plot(rid, scale, weights, outdir, args.scale_bins)


if __name__ == "__main__":
    main()
