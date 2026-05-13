#!/usr/bin/env python
"""Plot TFMC true class probabilities and DCRs across phase-space features."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.syncer as syncer
import common.user as user
import common.yaml_loader as yaml_loader
from data.plot_options import plot_options as PLOT_OPTS

from common.helpers import copyIndexPHP

import mplhep as hep

LOGGER = logging.getLogger(__name__)

def list_jobs_and_exit(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """List TFMC jobs and exit."""
    jobs = [
        j
        for j in (cfg.get("jobs") or [])
        if j.get("type") == "classifier" and j.get("framework") == "tfmc"
    ]
    if not jobs:
        LOGGER.info("No TFMC classifier jobs found in YAML.")
        raise SystemExit(0)

    # script = os.path.basename(__file__)
    for job in jobs:
        LOGGER.info("python %s %s --job %s", __file__, args.config, job["id"])
    raise SystemExit(0)


def resolve_job(cfg: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Resolve one TFMC classifier job from YAML."""
    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == job_id), None)
    if job is None:
        raise RuntimeError(f"Job id '{job_id}' not found.")
    if job.get("type") != "classifier" or job.get("framework") != "tfmc":
        raise RuntimeError(f"Job '{job_id}' is not a TFMC classifier job.")
    return job


def resolve_loaders(
    cfg: dict[str, Any],
    job: dict[str, Any],
) -> tuple[list[str], list[Any], list[str], int]:
    """Create class loaders configured from YAML and return basic feature layout."""
    defaults = cfg.get("defaults", {}) or {}
    module_samples = defaults.get("module_samples", "data.samples")
    samples_mod = importlib.import_module(module_samples)

    class_names = list(job["data"]["classes"])
    loaders: list[Any] = []

    for class_name in class_names:
        if not hasattr(samples_mod, class_name):
            raise RuntimeError(f"Class loader '{class_name}' not found in {module_samples}.")
        loader = getattr(samples_mod, class_name)
        loader.setFeatures(job["features"])
        loaders.append(loader)

    selection = job.get("selection", None)
    selection_features = job.get("selection_features", [])
    print(f"{selection=}, {selection_features=}")
    if selection:
        for loader in loaders:
            loader.addSelection(selection, selection_features)

    feature_names = list(getattr(loaders[0], "feature_names", []))
    if not feature_names:
        raise RuntimeError("First loader has no feature_names set.")
    for loader in loaders[1:]:
        if list(getattr(loader, "feature_names", [])) != feature_names:
            raise RuntimeError("Feature mismatch across class loaders.")
    
    n_split = int(job.get("runtime", {}).get("n_split", 1))
    if n_split:
        for loader in loaders:
            loader.set_n_split(n_split)

    input_dim = len(feature_names)
    return class_names, loaders, feature_names, input_dim


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Compute numerator / denominator with zeros where denominator is zero."""
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def compute_class_weights(weight_sums: np.ndarray) -> np.ndarray:
    """Compute TFMC class weights from per-class inclusive weight sums."""
    mean_weight_sum = float(np.mean(weight_sums))
    return np.where(weight_sums > 0.0, mean_weight_sum / weight_sums, 1.0)


def compute_weighted_dcr_and_error(
    counts: np.ndarray,
    counts_sumw2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin weighted differential cross-section ratios (weighted fractions).
    
    This is the cross-section composition in each bin: the ratio of weighted
    class counts to total weighted counts. Uncertainty comes from sum of squared weights.
    This is what the model actually sees during training.
    """
    totals = counts.sum(axis=1, keepdims=True)
    weighted_fracs = safe_divide(counts, totals)
    
    n_bins, n_classes = counts.shape
    frac_var = np.zeros_like(weighted_fracs, dtype=np.float64)
    
    for i_class in range(n_classes):
        deriv = np.zeros_like(counts, dtype=np.float64)
        deriv[:, i_class] = safe_divide(1.0 - weighted_fracs[:, i_class], totals[:, 0])
        for j_class in range(n_classes):
            if j_class == i_class:
                continue
            deriv[:, j_class] = safe_divide(-weighted_fracs[:, i_class], totals[:, 0])
        
        frac_var[:, i_class] = np.sum((deriv * deriv) * counts_sumw2, axis=1)
    
    frac_err = np.sqrt(np.clip(frac_var, 0.0, None))
    return weighted_fracs, frac_err


def make_plot(
    metric_name: str,
    values: np.ndarray,
    errors: np.ndarray,
    bins: np.ndarray,
    class_names: list[str],
    output_path: str,
    feature_label: str,
) -> None:

    """Plot all feature panels for one metric as a single grid figure."""

    fig, ax = plt.subplots(figsize=(8.0, 8.0))

    plt.style.use("petroff10")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    hep.style.use("CMS")

    centers = 0.5 * (bins[:-1] + bins[1:])

    for class_idx, class_name in enumerate(class_names):
        color = color_cycle[class_idx % len(color_cycle)]
        ax.errorbar(
            centers,
            values[:, class_idx],
            yerr=errors[:, class_idx],
            fmt="o-",
            markersize=3,
            linewidth=1.2,
            capsize=1.8,
            color=color,
            label=class_name,
            alpha=0.9,
        )


    ax.set_xlabel(feature_label)
    ax.set_ylabel(metric_name)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    hep.cms.label("Internal", ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="center", fontsize="small", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)

if __name__ == "__main__":
    """Run TFMC truth plotting workflow."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(
        description="Plot TFMC true class probabilities and DCRs from class loaders."
    )
    parser.add_argument("config", help="Path to global YAML config")
    parser.add_argument("--job", default=None, help="TFMC classifier job id to run")
    parser.add_argument("--small", action="store_true", help="Only first shard for debugging")
    parser.add_argument("--rebin", type=int, default=1, help="Coarsen histograms by this factor")
    parser.add_argument(
        "--out-tag",
        default="truth_only",
        help="Subfolder tag for plots",
    )
    args = parser.parse_args()

    cfg_path = os.path.expanduser(os.path.expandvars(args.config))
    cfg = yaml_loader.load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        raise RuntimeError("Top-level YAML config must be a dictionary.")

    if args.job is None:
        list_jobs_and_exit(cfg, args)

    job = resolve_job(cfg, args.job)

    class_names, loaders, feature_names, _ = resolve_loaders(cfg, job)
    plot_features = [feature for feature in feature_names if feature in PLOT_OPTS]
    if not plot_features:
        raise RuntimeError("No features are present in plot_options for plotting.")

    feature_to_column = {feature: idx for idx, feature in enumerate(feature_names)}

    bins: dict[str, np.ndarray] = {}
    counts_weighted: dict[str, np.ndarray] = {}
    counts_sumw2: dict[str, np.ndarray] = {}

    rebin_factor = max(1, int(args.rebin))
    n_classes = len(class_names)
    for feature in plot_features:
        n_bins, x_min, x_max = PLOT_OPTS[feature]["binning"]
        n_bins = max(1, int(n_bins) // rebin_factor)
        bins[feature] = np.linspace(float(x_min), float(x_max), n_bins + 1)
        counts_weighted[feature] = np.zeros((n_bins, n_classes), dtype=np.float64)
        counts_sumw2[feature] = np.zeros((n_bins, n_classes), dtype=np.float64)

    shard_counts = [len(getattr(loader, "base", loader)) for loader in loaders]
    n_shards = min(shard_counts)
    if args.small:
        n_shards = min(n_shards, 1)

    LOGGER.info("Iterating over %d shard(s) across %d classes.", n_shards, n_classes)

    weight_sums = np.zeros(n_classes, dtype=np.float64)
    event_sums = np.zeros(n_classes, dtype=np.int32)

    for shard in range(n_shards):
        for class_idx, loader in enumerate(loaders):
            X, w = loader.materialize(shard=shard, what="fw")
            weights = w.astype(np.float64, copy=False)
            weights_sq = weights * weights
            
            weight_sums[class_idx] += float(np.sum(weights))
            event_sums[class_idx] += len(weights)

            for feature in plot_features:
                col_idx = feature_to_column[feature]
                edges = bins[feature]
                hist_weighted, _ = np.histogram(X[:, col_idx], bins=edges, weights=weights)
                hist_sumw2, _ = np.histogram(X[:, col_idx], bins=edges, weights=weights_sq)
                counts_weighted[feature][:, class_idx] += hist_weighted
                counts_sumw2[feature][:, class_idx] += hist_sumw2

    LOGGER.info("Class names: %s", class_names)
    LOGGER.info("Computed weighted events: %s", weight_sums.tolist())
    LOGGER.info("Computed unweighted events: %s", event_sums.tolist())

    cfg_base = os.path.splitext(os.path.basename(cfg_path))[0]
    output_dir = os.path.join(user.plot_directory, "TFMC_inputs", cfg_base, f"{job['id']}_{args.out_tag}")
    os.makedirs(output_dir, exist_ok=True)

    for feature in plot_features:
        weighted_dcr, weighted_dcr_err = compute_weighted_dcr_and_error(
            counts_weighted[feature],
            counts_sumw2[feature],
        )
        
        LOGGER.info(
            "%s: weighted_dcr=%s",
            feature,
            np.mean(weighted_dcr, axis=0),
        )

        label_tex_mpl = "$" + PLOT_OPTS[feature]["tex"].replace("#", "\\") + "$"
        make_plot(
            metric_name="weighted_dcr",
            values=weighted_dcr,
            errors=weighted_dcr_err,
            bins=bins[feature],
            class_names=class_names,
            output_path=os.path.join(output_dir, f"truth_weighted_dcr_{feature}.png"),
            feature_label=label_tex_mpl,
        )
    
    copyIndexPHP(output_dir)

    syncer.sync()
    LOGGER.info("Done.")