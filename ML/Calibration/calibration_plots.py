#!/usr/bin/env python
"""Plot BIT calibration residuals (truth - prediction) vs predicted coefficient value.

Reads the ``calib_prediction_<split>.csv`` file written by
``calibration_runner.run_calibration`` (via ``pdf_calibration.py`` or
``eft_calibration.py``) -- one ``<label>_truth`` / ``<label>_pred`` column pair per
derivative plus a ``weight`` column -- and draws, per derivative, a two-panel figure:
the weighted mean +/- 1 sigma residual vs the BIT-predicted coefficient value (top
panel), and the weighted event yield per bin (bottom panel).

Works identically for PDF and EFT jobs since it only reads the CSV already saved by
the calibration scripts; the job's ``pdf``/``eft`` block is irrelevant here.

Example
-------
    python ML/Calibration/calibration_plots.py configs/unbinned_v6/unbinned_2018.yaml \
        --job bit_NG_PDF4LHC21_6_...
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import common.user as user
import common.syncer as syncer
import common.helpers as helpers
import common.yaml_loader as yaml_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

hep.style.use("CMS")


# --------------------------------------------------------------------------------
# CLI + config/job loading
# --------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--job", default=None, help="BIT job id (omit to list bit jobs)")
    p.add_argument("--num-bins", type=int, default=25, help="Number of bins in the predicted coefficient")
    p.add_argument("--labels", nargs="+", default=None, help="Restrict to these derivative labels (default: all)")
    return p


def load_cfg_and_job(args):
    """Load the YAML config and select the requested bit job. Lists jobs and exits if ``--job`` is omitted."""
    cfg_path = os.path.expanduser(os.path.expandvars(args.config))
    cfg = yaml_loader.load_yaml(cfg_path)

    if args.job is None:
        jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "bit"]
        if not jobs:
            print("No BIT jobs found.")
            sys.exit(0)
        for j in jobs:
            print(f"python {__file__} {args.config} --job {j['id']}")
        sys.exit(0)

    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
    if job is None or job.get("type") != "bit":
        raise RuntimeError(f"Job '{args.job}' not found or not type 'bit'.")
    return cfg, job


# --------------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------------

def weighted_mean(array, weight):
    return (array * weight).sum() / weight.sum()


def weighted_std(array, weight):
    mean = weighted_mean(array, weight)
    var = weighted_mean((array - mean) ** 2, weight)
    return np.sqrt(np.abs(var))


def sanitize_label(label: str) -> str:
    return label.replace(" * ", "_x_").replace("^", "pow").replace(" ", "")


def bin_calibration(pred, residual, weight, num_bins):
    """Bin ``residual`` by ``pred`` value.

    Returns paired (left-edge, right-edge) point arrays for each populated bin, ready
    for a stair-style ``plot``/``fill_between``/``step`` call, matching the standalone
    notebook this script replaces.
    """
    bins = np.linspace(pred.min(), pred.max(), num_bins + 1)
    which_bin = (pred > bins[:-1].reshape(-1, 1)) & (pred <= bins[1:].reshape(-1, 1))

    paired_pred, paired_mean, paired_std, paired_count, paired_count_err = [], [], [], [], []
    for i in range(num_bins):
        if not np.any(which_bin[i]):
            continue
        res_bin = residual[which_bin[i]]
        w_bin = weight[which_bin[i]]

        mean_w = weighted_mean(res_bin, w_bin)
        std_w = weighted_std(res_bin, w_bin)
        count = w_bin.sum()
        count_err = np.sqrt(np.sum(w_bin ** 2))

        paired_pred += [bins[i], bins[i + 1]]
        paired_mean += [mean_w, mean_w]
        paired_std += [std_w, std_w]
        paired_count += [count, count]
        paired_count_err += [count_err, count_err]

    return tuple(np.array(a) for a in (paired_pred, paired_mean, paired_std, paired_count, paired_count_err))


# --------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------

def plot_calibration(label, pred, truth, weight, out_path, num_bins):
    """Draw the two-panel calibration figure (residual band + weighted yield) for one derivative.

    Returns whether any bin is miscalibrated by more than 1 sigma, i.e. the drawn
    +/- std residual band does not cover zero (``None`` if the derivative was
    skipped and never plotted).
    """
    if np.allclose(pred.min(), pred.max()):
        logger.warning("Derivative %s: predicted values are degenerate, skipping.", label)
        return None

    residual = truth - pred
    paired_pred, paired_mean, paired_std, paired_count, paired_count_err = bin_calibration(
        pred, residual, weight, num_bins
    )
    if len(paired_pred) == 0:
        logger.warning("Derivative %s: no populated bins, skipping.", label)
        return None

    out_of_calibration = bool(np.any(np.abs(paired_mean) > paired_std))

    fig, (ax_top_panel, ax_bottom_panel) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8),
        gridspec_kw={"hspace": 0.0, "height_ratios": [1, 1]},
    )

    # ---- top panel: weighted event yield (log scale) ----
    ax_top_panel.set_yscale("log")
    ax_top_panel.set_ylabel("Weighted events")
    positive_counts = paired_count[paired_count > 0]
    if positive_counts.size:
        ax_top_panel.set_ylim(positive_counts.min() * 0.8, positive_counts.max() * 1.6)
    ax_top_panel.fill_between(
        paired_pred,
        np.clip(paired_count - paired_count_err, 1e-12, None),
        paired_count + paired_count_err,
        step="post", color="#a6c8a6", alpha=0.3, linewidth=0,
    )
    ax_top_panel.step(paired_pred, paired_count, where="post", color="k", label="Calibration dataset")
    ax_top_panel.legend(frameon=False, fontsize=12, loc="lower right")
    ax_top_panel.tick_params(axis="x", which="both", labelbottom=False)

    # ---- bottom panel: residual mean +/- std vs predicted coefficient ----
    ax_bottom_panel.plot(paired_pred, paired_mean, color="k", label=r"$\langle R - \hat{R} \rangle$")
    ax_bottom_panel.fill_between(
        paired_pred, paired_mean + paired_std, paired_mean - paired_std,
        color="#a6c8a6", alpha=0.3, label=r"$\pm 1\sigma$",
    )
    ax_bottom_panel.axhline(0.0, linestyle="--", color="k")
    ax_bottom_panel.set_ylabel(r"$\langle R - \hat{R} \rangle$")
    ax_bottom_panel.set_xlabel(rf"$\hat{{R}}$ ({label})")
    ax_bottom_panel.legend(frameon=False, fontsize=12, loc="upper right")

    y_max = float(np.max(np.abs(paired_mean) + paired_std))
    y_pad = 1.2 * max(y_max, 1e-3)
    ax_bottom_panel.set_ylim(-y_pad, y_pad)

    for ax in (ax_top_panel, ax_bottom_panel):
        ax.set_xlim(paired_pred.min(), paired_pred.max())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(axis="x", which="both", direction="in", top=True)
        ax.tick_params(axis="both", labelsize=12)

    hep.cms.label("Internal", data=False, ax=ax_top_panel, loc=0)

    plt.subplots_adjust(hspace=0.0)
    plt.savefig(out_path + ".png", bbox_inches="tight", dpi=200)
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)

    return out_of_calibration


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    cfg, job = load_cfg_and_job(args)

    cfg_base = os.path.join(cfg.get("version", "default"), job["region"])
    model_dir = os.path.join(user.model_directory, cfg_base, "BIT", job["id"])

    csv_path = os.path.join(model_dir, f"calib_prediction.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Run pdf_calibration.py / eft_calibration.py for job '{job['id']}' first."
        )

    df = pd.read_csv(csv_path, index_col=0)
    if "weight" not in df.columns:
        raise RuntimeError(f"{csv_path} has no 'weight' column.")
    weight = df["weight"].to_numpy()

    der_labels_list = []
    for col in df.columns:
        if col.endswith("_truth"):
            label = col[: -len("_truth")]
            if f"{label}_pred" not in df.columns:
                raise RuntimeError(f"{csv_path}: column '{col}' has no matching '{label}_pred' column.")
            der_labels_list.append(label)
    if not der_labels_list:
        raise RuntimeError(f"{csv_path} has no '<label>_truth' columns.")

    selected = list(args.labels) if args.labels else list(der_labels_list)
    unknown = set(selected) - set(der_labels_list)
    if unknown:
        raise RuntimeError(f"Requested labels not found: {sorted(unknown)}. Available: {der_labels_list}")

    out_dir = os.path.join(
        user.plot_directory, "BIT-calibration",
        cfg.get("version", "default"), job["region"], job["id"],
    )
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    out_of_calibration = {}
    for label in selected:
        truth = df[f"{label}_truth"].to_numpy()
        pred = df[f"{label}_pred"].to_numpy()
        out_path = os.path.join(out_dir, sanitize_label(label))
        flagged = plot_calibration(label, pred, truth, weight, out_path, args.num_bins)
        logger.info("Wrote %s.png / .pdf", out_path)
        if flagged is not None:
            out_of_calibration[label] = flagged

    flagged_labels = [label for label, flagged in out_of_calibration.items() if flagged]
    if flagged_labels:
        logger.warning("Miscalibrated by >1 sigma in at least one bin: %s", flagged_labels)
    flags_path = os.path.join(out_dir, "out_of_calibration.json")
    with open(flags_path, "w") as f:
        json.dump(out_of_calibration, f, indent=2)
    logger.info("Wrote calibration flags -> %s", flags_path)

    helpers.copyIndexPHP(out_dir)


if __name__ == "__main__":
    main()
    syncer.sync()
