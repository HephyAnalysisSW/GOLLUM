#!/usr/bin/env python
"""Combine, analyze and plot the fit results of toy pseudo-experiments.

Reads the per-toy fit JSONs written by fit/Likelihood.py, groups them by the
injected point and the toy-generation source (cache/truth), and produces, for
each such configuration:
  - the distribution of the best-fit value of every free parameter,
  - the distribution of fval, the optimization target at the minimum,
  - pull distributions (value - truth)/error, testing bias and error calibration,
  - the distribution of the fitted uncertainty, compared to the toy-to-toy RMS,
  - a 2D scatter of the first two free parameters with the mean fitted 1-sigma
    covariance ellipse overlaid on the empirical one.

A summary table (n_toys, truth, mean, RMS, mean error, pull mean/width,
coverage, fval mean/RMS) is printed and written as JSON and CSV.

Run from the repo root:
    python fit/analyze_toy_fits.py --inputDir output_SBIEFT/
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, '..')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mplhep as hep

import common.user as user
import common.syncer as syncer # noqa: F401  -- import hooks plt.savefig and syncs www/ to EOS on exit
from common.helpers import copyIndexPHP
from common.logger import get_logger

plt.style.use("petroff10")
hep.style.use("CMS")

# number of decimal places in print scripts
PRECISION = 4

def load_groups(input_dir: str) -> dict:
    """Load every toy fit JSON under input_dir, grouped by (point, source).

    Returns {(point, source): list of fit payloads}. Fits without a 'toy' block
    are a hard error: they carry no injected truth, so pulls cannot be formed.
    Re-run fit/Likelihood.py --toyFile, which writes the 'toy' block directly.
    """
    fit_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*_toy*_fit.json"), recursive=True))
    groups = {}
    for fit_path in fit_paths:
        with open(fit_path) as fit_file:
            payload = json.load(fit_file)
        if "toy" not in payload:
            raise RuntimeError(f"{fit_path} has no 'toy' block; regenerate this fit with fit/Likelihood.py.")
        groups.setdefault((payload["toy"]["point"], payload["toy"]["source"]), []).append(payload)
    return groups


def stack_group(payloads: list) -> dict:
    """Turn a list of fit payloads into arrays, checking they are commensurate."""
    parameters = payloads[0]["free_parameter_order"]
    truth = payloads[0]["toy"]["hypothesis"]
    for i, payload in enumerate(payloads):
        if payload["free_parameter_order"] != parameters:
            raise RuntimeError("Parameter order differs between toys of the same configuration.")
        if payload["toy"]["hypothesis"] != truth:
            raise RuntimeError(f"Injected hypothesis {payload['toy']['hypothesis']} differs between toys of the same configuration {truth} for fit number {payload['toy']['seed']}.")

    values = np.array([[p["parameters"][i]["value"] for i in range(len(parameters))] for p in payloads])
    errors = np.array([[p["parameters"][i]["error"] for i in range(len(parameters))] for p in payloads])
    return {
        "parameters": parameters,
        "truth": np.array([truth[name] for name in parameters]),
        "values": values,
        "errors": errors,
        "fval": np.array([p["fval"] for p in payloads]),
        "covariance": np.array([p["covariance"]["matrix"] for p in payloads]),
    }


def plot_histogram(array, xlabel, legend_lines, out_path, vlines=()):
    """Step histogram with a text legend and optional labelled vertical lines."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(array, bins=25, histtype="step", color="black", linewidth=1.5)
    for position, label, color in vlines:
        ax.axvline(position, color=color, linestyle="--", linewidth=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Toys")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [plt.Line2D([], [], linestyle="none")] * len(legend_lines),
              labels + legend_lines, fontsize=14, loc="best")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_pull(pulls, parameter, out_path):
    """Pull distribution with a unit Gaussian overlaid, normalised to the entries."""
    fig, ax = plt.subplots(figsize=(8, 8))
    counts, edges, _ = ax.hist(pulls, bins=25, range=(-5, 5), histtype="step",
                               color="black", linewidth=1.5)
    grid = np.linspace(-5, 5, 200)
    bin_width = edges[1] - edges[0]
    ax.plot(grid, len(pulls) * bin_width * np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi),
            color="#ffa90e", linewidth=1.5, label="unit Gaussian")
    ax.set_xlabel(f"({parameter} - truth) / error")
    ax.set_ylabel("Toys")
    ax.legend([plt.Line2D([], [], linestyle="none")] * 2 + ax.get_legend_handles_labels()[0],
              [f"mean = {pulls.mean():.3f} +/- {pulls.std(ddof=1) / np.sqrt(len(pulls)):.3f}",
               f"width = {pulls.std(ddof=1):.3f}", "unit Gaussian"],
              fontsize=14, loc="best")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def covariance_ellipse(center, covariance, color, label):
    """1-sigma ellipse of a 2x2 covariance matrix, as a matplotlib patch."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    return Ellipse(center, 2 * np.sqrt(eigenvalues[-1]), 2 * np.sqrt(eigenvalues[0]),
                   angle=angle, facecolor="none", edgecolor=color, linewidth=2, label=label)


def plot_scatter(stacked, out_path):
    """Best-fit points of the first two parameters, with fitted and empirical ellipses."""
    first, second = 0, 1
    x_values, y_values = stacked["values"][:, first], stacked["values"][:, second]
    mean_covariance = stacked["covariance"].mean(axis=0)[np.ix_([first, second], [first, second])]
    empirical_covariance = np.cov(x_values, y_values)
    center = (x_values.mean(), y_values.mean())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_values, y_values, s=12, color="black", alpha=0.6)
    ax.add_patch(covariance_ellipse(center, mean_covariance, "#ffa90e", "mean fitted covariance"))
    ax.add_patch(covariance_ellipse(center, empirical_covariance, "#3f90da", "empirical scatter"))
    ax.plot(stacked["truth"][first], stacked["truth"][second], marker="*", markersize=18,
            color="red", linestyle="none", label="injected truth")
    ax.set_xlabel(stacked["parameters"][first])
    ax.set_ylabel(stacked["parameters"][second])
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def summarize(point, source, stacked) -> list:
    """One summary row per parameter of a configuration."""
    rows = []
    n_toys = len(stacked["fval"])
    for index, parameter in enumerate(stacked["parameters"]):
        values, errors = stacked["values"][:, index], stacked["errors"][:, index]
        truth = stacked["truth"][index]
        pulls = (values - truth) / errors
        rows.append({
            "point": (point),
            "source": (source),
            "parameter": (parameter),
            "n_toys": (n_toys),
            "truth": round(float(truth),PRECISION),
            "mean": round(float(values.mean()),PRECISION),
            "RMS": round(float(values.std(ddof=1)),PRECISION),
            "mean_error": round(float(errors.mean()),PRECISION),
            "pull_mean": round(float(pulls.mean()),PRECISION),
            "pull_mean_uncertainty": round(float(pulls.std(ddof=1) / np.sqrt(n_toys)),PRECISION),
            "pull_width": round(float(pulls.std(ddof=1)),PRECISION),
            "coverage": round(float(np.mean(np.abs(values - truth) < errors)),PRECISION),
            "fval_mean": round(float(stacked["fval"].mean()),PRECISION),
            "fval_RMS": round(float(stacked["fval"].std(ddof=1)),PRECISION),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputDir", required=True,
                        help="Directory holding the per-toy fit JSONs (searched recursively).")
    parser.add_argument("--outputName", default="",
                        help="Subfolder of user.plot_directory/toy_fits_summary to write plots and summary to.")
    args = parser.parse_args()

    logger = get_logger("INFO")

    groups = load_groups(args.inputDir)
    logger.info("Loaded %d toy fits in %d configurations", sum(map(len, groups.values())), len(groups))

    base_dir = os.path.join(user.plot_directory, "toy_fits_summary", args.outputName)
    os.makedirs(base_dir, exist_ok=True)
    copyIndexPHP(base_dir)

    summary_rows = []
    for (point, source), payloads in sorted(groups.items()):
        stacked = stack_group(payloads)
        out_dir = os.path.join(base_dir, f"{point}_{source}")
        os.makedirs(out_dir, exist_ok=True)
        copyIndexPHP(out_dir)
        logger.info("%s / %s: %d toys, parameters %s",
                    point, source, len(payloads), ", ".join(stacked["parameters"]))

        for index, parameter in enumerate(stacked["parameters"]):
            values, errors = stacked["values"][:, index], stacked["errors"][:, index]
            truth = stacked["truth"][index]

            plot_histogram(values, f"best-fit {parameter}",
                           [f"mean = {values.mean():.4f}", f"RMS = {values.std(ddof=1):.4f}"],
                           os.path.join(out_dir, f"bestfit_{parameter}.png"),
                           vlines=[(truth, f"injected {truth:g}", "red")])

            plot_pull((values - truth) / errors, parameter,
                      os.path.join(out_dir, f"pull_{parameter}.png"))

            plot_histogram(errors, f"fitted uncertainty on {parameter}",
                           [f"mean = {errors.mean():.4f}", f"median = {np.median(errors):.4f}"],
                           os.path.join(out_dir, f"error_{parameter}.png"),
                           vlines=[(values.std(ddof=1), "RMS of best-fit values", "#ffa90e")])

        plot_histogram(stacked["fval"], "fval at the minimum",
                       [f"mean = {stacked['fval'].mean():.4f}",
                        f"RMS = {stacked['fval'].std(ddof=1):.4f}"],
                       os.path.join(out_dir, "fval.png"))

        if len(stacked["parameters"]) >= 2:
            plot_scatter(stacked, os.path.join(
                out_dir, f"scatter_{stacked['parameters'][0]}_{stacked['parameters'][1]}.png"))

        summary_rows += summarize(point, source, stacked)

    with open(os.path.join(base_dir, "summary.json"), "w") as summary_file:
        json.dump(summary_rows, summary_file, indent=2)
    with open(os.path.join(base_dir, "summary.csv"), "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    syncer.file_sync_storage.append(os.path.join(base_dir, "summary.json"))
    syncer.file_sync_storage.append(os.path.join(base_dir, "summary.csv"))

    header = f"{'point':>12} {'source':>7} {'par':>7} {'N':>4} {'truth':>7} {'mean':>9} " \
             f"{'RMS':>9} {'<err>':>9} {'pull mean':>11} {'pull width':>10} {'cover':>6}"
    logger.info(header)
    for row in summary_rows:
        logger.info("%12s %7s %7s %4d %7.3f %9.4f %9.4f %9.4f %6.3f+/-%.3f %10.3f %6.3f",
                    row["point"], row["source"], row["parameter"], row["n_toys"], row["truth"],
                    row["mean"], row["RMS"], row["mean_error"], row["pull_mean"],
                    row["pull_mean_uncertainty"], row["pull_width"], row["coverage"])
    logger.info("Plots and summary written to %s", base_dir)

    syncer.sync()

if __name__ == "__main__":
    main()
