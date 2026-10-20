#!/usr/bin/env python
"""Plot TFMC training-closure curves for a trained multi-classifier."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.syncer as syncer
import common.user as user
import common.yaml_loader as yaml_loader
from common.helpers import copyIndexPHP
from data.colors import cmap_petroff10_mpl
from data.plot_options import plot_options as PLOT_OPTS
from ML.TFMC.TFMC import TFMC
from data.UIDSplitter import UIDSplitter

LOGGER = logging.getLogger(__name__)
MAKE_PUBLIC_PLOTS = False

def list_jobs_and_exit(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """List TFMC jobs and exit."""
    jobs = [
        job
        for job in (cfg.get("jobs") or [])
        if job.get("type") == "classifier" and job.get("framework") == "tfmc"
    ]
    if not jobs:
        LOGGER.info("No TFMC classifier jobs found in YAML.")
        raise SystemExit(0)

    script = os.path.basename(__file__)
    for job in jobs:
        print(f"python {__file__} {args.config} --job {job['id']}")
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
    """Create class loaders configured from YAML and return feature layout."""
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
    if selection:
        LOGGER.info("Applying selection '%s' with features %s", selection, selection_features)
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

    return class_names, loaders, feature_names, len(feature_names)


def load_model(model_dir: str) -> TFMC:
    """Load the trained TFMC checkpoint, preferring the best checkpoint."""
    try:
        return TFMC.load(model_dir)
    except Exception:
        LOGGER.info("Best checkpoint not available, falling back to the latest checkpoint.")
        return TFMC.load(model_dir, latest_filename="last_checkpoint")


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Compute numerator / denominator with zeros where the denominator is zero."""
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def normalize_histograms(
    truth_counts: np.ndarray,
    truth_sumw2: np.ndarray,
    pred_counts: np.ndarray,
    normalize_classes: bool,
    normalize_bins: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare truth and prediction histograms for plotting."""
    truth = truth_counts.astype(np.float64, copy=True)
    truth_var = truth_sumw2.astype(np.float64, copy=True)
    pred = pred_counts.astype(np.float64, copy=True)

    if normalize_classes:
        truth_norm = truth.sum(axis=0, keepdims=True)
        pred_norm = pred.sum(axis=0, keepdims=True)
        truth = safe_divide(truth, truth_norm)
        truth_var = safe_divide(truth_var, truth_norm * truth_norm)
        pred = safe_divide(pred, pred_norm)

    if normalize_bins:
        truth_total = truth.sum(axis=1, keepdims=True)
        pred_total = pred.sum(axis=1, keepdims=True)
        truth = safe_divide(truth, truth_total)
        truth_var = safe_divide(truth_var, truth_total * truth_total)
        pred = safe_divide(pred, pred_total)

    return truth, np.sqrt(np.clip(truth_var, 0.0, None)), pred


def make_feature_plot(
    feature: str,
    metric_name: str,
    bins: np.ndarray,
    truth_counts: np.ndarray,
    truth_sumw2: np.ndarray,
    pred_counts: np.ndarray,
    class_names: list[str],
    output_path: str,
    normalize_classes: bool,
    normalize_bins: bool,
) -> None:
    """Draw one per-feature closure plot."""
    truth, truth_err, pred = normalize_histograms(
        truth_counts,
        truth_sumw2,
        pred_counts,
        normalize_classes=normalize_classes,
        normalize_bins=normalize_bins,
    )

    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = [cmap_petroff10_mpl[i % len(cmap_petroff10_mpl)] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    hep.style.use("CMS")

    for class_idx, class_name in enumerate(class_names):
        color = colors[class_idx]
        ax.errorbar(
            centers,
            truth[:, class_idx],
            yerr=truth_err[:, class_idx],
            fmt="o",
            markersize=3.8,
            linewidth=1.0,
            capsize=1.6,
            color=color,
            alpha=0.9,
            label="_nolegend_",
        )
        ax.step(
            bins,
            np.r_[pred[:, class_idx], pred[-1, class_idx]],
            where="post",
            color=color,
            linewidth=2.0,
            label=class_name,
        )

    x_title = PLOT_OPTS[feature]["tex"].replace("#", "\\")
    ax.set_xlabel(f"${x_title}$")
    ax.set_ylabel(metric_name if not normalize_bins else f"{metric_name} fraction")

    truth_min = float(np.min(truth)) if truth.size else 0.0
    pred_min = float(np.min(pred)) if pred.size else 0.0
    min_y = max(truth_min, pred_min)
    truth_max = float(np.max(truth)) if truth.size else 0.0
    pred_max = float(np.max(pred)) if pred.size else 0.0
    max_y = max(truth_max, pred_max)
    # if normalize_bins:
    # 	ax.set_ylim(0.0, 1.05)
        
    # else:
    if PLOT_OPTS.get(feature, {}).get("logY", False) or min_y/max_y < 0.01:
        positive_parts = [arr[arr > 0.0] for arr in (truth, pred) if np.any(arr > 0.0)]
        if positive_parts:
            positive_values = np.concatenate(positive_parts)
            ax.set_yscale("log")
            ax.set_ylim(max(float(np.min(positive_values)) / 3.0, 1e-4), max(1.0, 1.2 * max_y))
        else:
            ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.0, 1.2 * max_y if max_y > 0.0 else 1.0)

    ax.grid(True, alpha=0.25)

    hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", ax=ax)

    class_handles = [
        Line2D([], [], color=color, linewidth=2.0, label=class_name)
        for color, class_name in zip(colors, class_names)
    ]

    class_labels = []

    style_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="truth"),
        Line2D([], [], color="black", linewidth=2.0, label="prediction"),
    ]

    from data.plot_options import get_sample_legend
    fig.legend(
        class_handles,
        ["$"+get_sample_legend(class_name).replace('#','\\')+"$" for class_name in class_names],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, len(class_names)),
        frameon=False,
        fontsize=11,
        columnspacing=1.4,
        handlelength=2.0,
    )
    fig.legend(
        style_handles,
        ["truth", "prediction"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        columnspacing=1.8,
    )

    fig.subplots_adjust(top=0.83, bottom=0.15)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the TFMC closure plotting workflow."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Plot TFMC training-closure curves from a trained classifier on a held-out test dataset."
    )
    parser.add_argument("config", help="Path to global YAML config")
    parser.add_argument("--job", default=None, help="TFMC classifier job id to run")
    parser.add_argument("--small", action="store_true", help="Only first shard for debugging")
    parser.add_argument("--rebin", type=int, default=1, help="Coarsen histograms by this factor")
    parser.add_argument("--norm_plot", action="store_true", help="Normalize each class to unit area.")
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
    n_classes = len(class_names)

    cfg_base = os.path.join(cfg.get("version", "default"), job["region"])
    model_dir = os.path.join(user.model_directory, cfg_base, "TFMC", job["id"])
    plot_dir = os.path.join(
        user.plot_directory,
        "TFMC_training_closure",
        cfg_base,
        f"{job['id']}",
    )
    if args.norm_plot:
        plot_dir+="_norm"
        
    os.makedirs(plot_dir, exist_ok=True)
    copyIndexPHP(plot_dir)

    LOGGER.info("Loading TFMC model from %s", model_dir)
    model = load_model(model_dir)
    LOGGER.info("Loaded TFMC model with %d classes and %d inputs.", model.num_classes, model.input_dim)

    rebin_factor = max(1, int(args.rebin))
    bins: dict[str, np.ndarray] = {}
    truth_counts: dict[str, np.ndarray] = {}
    truth_sumw2: dict[str, np.ndarray] = {}
    pred_counts: dict[str, np.ndarray] = {}

    for feature in plot_features:
        n_bins, x_min, x_max = PLOT_OPTS[feature]["binning"]
        n_bins = max(1, int(n_bins) // rebin_factor)
        bins[feature] = np.linspace(float(x_min), float(x_max), n_bins + 1)
        truth_counts[feature] = np.zeros((n_bins, n_classes), dtype=np.float64)
        truth_sumw2[feature] = np.zeros((n_bins, n_classes), dtype=np.float64)
        pred_counts[feature] = np.zeros((n_bins, n_classes), dtype=np.float64)

    shard_counts = [len(getattr(loader, "base", loader)) for loader in loaders]
    n_shards = min(shard_counts)
    if args.small:
        n_shards = min(n_shards, 1)

    # ---------------- UID splitting (YAML-driven, implemented in data/UIDSplitter.py) ----------------
    UID_CFG = (job.get("splitting") or {})
    uid_enabled   = bool(UID_CFG.get("enabled", False))
    uid_fields    = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
    uid_seed      = int(UID_CFG.get("seed", 0))
    uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
    uid_scheme    = (UID_CFG.get("scheme") or {})

    uid_intervals = None
    uid_splitter = None
    if uid_enabled:
        uid_splitter = UIDSplitter(
            uid_fields=tuple(uid_fields),
            seed=uid_seed,
            n_buckets=uid_n_buckets,
        )

        # build bucket intervals (inline; no extra helper, no extra checks)
        keys  = list(uid_scheme.keys())
        fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

        sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
        sizes[-1] += uid_n_buckets - sum(sizes)

        uid_intervals = {}
        lo = 0
        for k, sz in zip(keys, sizes):
            uid_intervals[k] = (lo, lo + int(sz))
            lo += int(sz)

        # evaluating on the whole c2st partition
        # assumes that c2st_train comes before c2st_val
        eval_interval = (uid_intervals["c2st_train"][0], uid_intervals["c2st_val"][1])

        print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
        print(f"[UID] scheme intervals: {uid_intervals}")
        print(f"[UID] PNN eval split 'c2st' -> {eval_interval}")

    LOGGER.info("Iterating over %d shard(s) across %d classes.", n_shards, n_classes)

    for shard in range(n_shards):
        Xs: list[np.ndarray] = []
        Ys: list[np.ndarray] = []
        Ws: list[np.ndarray] = []
        for class_idx, loader in enumerate(loaders):
            X, o, w = loader.materialize(shard=shard, what="fow")
            y = np.zeros((len(X), n_classes), dtype=np.float32)
            y[:, class_idx] = 1.0

            if not uid_enabled:
                Xs.append(X)
                Ys.append(y)
                Ws.append(w)
            else:
                obs_names = loader.observer_names
                uid_idx = [obs_names.index(f) for f in uid_fields]
                O_uid = o[:, uid_idx]

                lo, hi = eval_interval
                m_eval = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)

                Xs.append(X[m_eval]); Ys.append(y[m_eval]); Ws.append(w[m_eval])

        X_all = np.concatenate(Xs, axis=0) if Xs else np.empty((0, len(feature_names)))
        y_all = np.concatenate(Ys, axis=0) if Ys else np.empty((0, n_classes))
        w_all = np.concatenate(Ws, axis=0) if Ws else np.empty((0,))
        if len(X_all) == 0:
            continue

        weights = w_all.astype(np.float64, copy=False)
        weights_sq = weights * weights
        pred = model.predict(X_all).astype(np.float64, copy=False)

        for feature in plot_features:
            col_idx = feature_to_column[feature]
            edges = bins[feature]
            values = X_all[:, col_idx]

            for class_idx in range(n_classes):
                hist_truth, _ = np.histogram(
                    values,
                    bins=edges,
                    weights=weights * y_all[:, class_idx],
                )
                hist_truth_sumw2, _ = np.histogram(
                    values,
                    bins=edges,
                    weights=weights_sq * y_all[:, class_idx],
                )
                hist_pred, _ = np.histogram(
                    values,
                    bins=edges,
                    weights=weights * pred[:, class_idx],
                )

                truth_counts[feature][:, class_idx] += hist_truth
                truth_sumw2[feature][:, class_idx] += hist_truth_sumw2
                pred_counts[feature][:, class_idx] += hist_pred

    metric_name = "DCR"
    metric_tag = "dcr"

    for feature in plot_features:
        raw_path = os.path.join(plot_dir, f"truth_{metric_tag}_{feature}.png")
        make_feature_plot(
            feature=feature,
            metric_name=metric_name,
            bins=bins[feature],
            truth_counts=truth_counts[feature],
            truth_sumw2=truth_sumw2[feature],
            pred_counts=pred_counts[feature],
            class_names=class_names,
            output_path=raw_path,
            normalize_classes=args.norm_plot,
            normalize_bins=False,
        )

        norm_path = os.path.join(plot_dir, f"norm_truth_{metric_tag}_{feature}.png")
        make_feature_plot(
            feature=feature,
            metric_name=metric_name,
            bins=bins[feature],
            truth_counts=truth_counts[feature],
            truth_sumw2=truth_sumw2[feature],
            pred_counts=pred_counts[feature],
            class_names=class_names,
            output_path=norm_path,
            normalize_classes=args.norm_plot,
            normalize_bins=True,
        )

        LOGGER.info("Wrote closure plots for %s", feature)

    copyIndexPHP(plot_dir)
    syncer.sync()
    LOGGER.info("Done. Plots stored in %s", plot_dir)


if __name__ == "__main__":
    main()
