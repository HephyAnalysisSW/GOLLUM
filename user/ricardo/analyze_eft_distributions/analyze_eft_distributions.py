#!/usr/bin/env python
"""Summarize EFT modification `bin_contents.json` outputs into flat CSV reports.

Consumes `bin_contents.json` files produced by `plot/bit/eft_modification_plot.py`
(written by `plot/bit/modification_plotter.py::_write_analysis_json`), one file
per (job, term order). For every (feature, derivative) pair it computes summary
statistics of the per-bin EFT coefficient (ratio of modified to SM weight) and
flags variables with large or wildly fluctuating coefficients as candidates for
an upper cut, so BIT training is not destabilized by rare large-weight events.

Each bin also carries a per-bin coefficient uncertainty (MC-statistical, from
the derivative weight's own sumw2 -- see `modification_plotter.py`), used here
to compute a pull (`|coeff| / unc`) per bin. This separates "large because of
real physical structure" (large coeff, large pull) from "large because the
per-event derivative weight is just noisy" (large coeff, small pull) -- the
latter flagged as `noisy_bin_frac` per (feature, derivative).

Also computes a neighbor pull between geometrically adjacent used bins
(`|coeff[i+1] - coeff[i]| / sqrt(unc[i]^2 + unc[i+1]^2)`, skipping pairs
separated by a bin dropped for low statistics), flagging roughness/instability
that a single-bin pull can miss -- two consecutive bins can each look fine on
their own while jumping between each other by far more than their combined
uncertainty allows.

Takes plain file/directory paths -- no dependency on the YAML config, so it
works on any job you add later without touching this script.

Usage
-----
    # single file
    python user/ricardo/analyze_eft_distributions.py \
        www/BIT-modification/unbinned_v3_eft_ND/SR_2016/bit_TT01j2l_EFT_2016_ctG_only/lin/bin_contents.json

    # a directory: recursively finds every bin_contents.json under it
    python user/ricardo/analyze_eft_distributions.py \
        www/BIT-modification/unbinned_v3_eft_ND/SR_2016

    # several paths (files and/or directories) at once, custom thresholds
    python user/ricardo/analyze_eft_distributions.py \
        www/BIT-modification/unbinned_v3_eft_ND/SR_2016/bit_TT01j2l_EFT_2016_ctG_only \
        www/BIT-modification/unbinned_v3_eft_ND/SR_2016/bit_TT01j2l_EFT_2016_ct_ML4EFT \
        --sensitivity-threshold 0.1 --ratio-threshold 5.0 --min-stat 20 --pull-threshold 3.0

Job id and term order (lin/quad) are inferred from the path
(`.../<job_id>/<lin|quad>/bin_contents.json`), matching the layout written by
`modification_plotter.py`.

Outputs three CSVs in --out-dir (default: this script's directory):
    - eft_sensitivity_detail.csv      : one row per (job, terms, feature, derivative)
    - eft_sensitivity_summary.csv     : one row per feature, worst case across all
                                         files scanned
    - eft_sensitivity_by_derivative.csv : one row per derivative, worst feature plus
                                         the noisy_bin_frac/pull averaged over all
                                         features it appears in
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import common.syncer as syncer
import common.user as user


# number of decimal places in output table
PRECISION: int = 3

def _find_bin_contents(paths: list[str]) -> list[str]:
    """Expand a list of file/directory paths into a flat list of bin_contents.json files."""
    found = []
    for path in paths:
        if os.path.isfile(path):
            found.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "bin_contents.json" in files:
                    found.append(os.path.join(root, "bin_contents.json"))
        else:
            raise RuntimeError(f"Path not found: {path}")
    return sorted(set(found))

def _analyze_file(path: str, min_stat: float, sensitivity_threshold: float, pull_threshold: float) -> list[dict]:
    """Return per-(feature, derivative) statistics rows for one bin_contents.json."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for feature, feat_data in data.items():
        sm = feat_data["nominal_histogram"]
        bin_centers = feat_data["bin_centers"]
        for key, values in feat_data.items():
            if key in ("bin_centers", "nominal_histogram"):
                continue
            derivative = key
            coeff_vals = values["coeff"]
            unc_vals = values["unc"]

            used = [
                (bc, c, u, s) for bc, c, u, s in zip(bin_centers, coeff_vals, unc_vals, sm) if s >= min_stat
            ]
            if not used:
                rows.append({
                    "feature": feature, "derivative": derivative,
                    "n_bins_used": 0, "max_abs_coeff": 0.0, "flagged_bin_center": None,
                    "mean_abs_coeff": 0.0, "std_abs_coeff": 0.0, "max_to_mean_ratio": 0.0,
                    "mean_abs_pull": 0.0, "max_abs_pull": 0.0, "noisy_bin_frac": 0.0,
                    "max_neighbor_pull": 0.0, "neighbor_jump_frac": 0.0,
                })
                continue

            abs_vals = [abs(c) for _, c, _, _ in used]
            max_abs = max(abs_vals)
            max_idx = abs_vals.index(max_abs)
            mean_abs = sum(abs_vals) / len(abs_vals)
            var = sum((a - mean_abs) ** 2 for a in abs_vals) / len(abs_vals)
            std_abs = var ** 0.5
            ratio = (max_abs / mean_abs) if mean_abs > 1e-10 else 0.0

            # pull: how many "sigma" a bin's coefficient sits from zero, given its own
            # per-bin MC-statistical uncertainty. A large coefficient with a small pull
            # is consistent with statistical noise in the derivative weight, not real
            # structure.
            pulls = [(abs(c) / u if u > 1e-12 else 0.0) for _, c, u, _ in used]
            mean_abs_pull = sum(pulls) / len(pulls)
            max_abs_pull = max(pulls)
            noisy_bins = sum(
                1 for (_, c, _, _), pull in zip(used, pulls)
                if abs(c) >= sensitivity_threshold and pull < pull_threshold
            )
            noisy_bin_frac = noisy_bins / len(used)

            # neighbor pull: is the jump between two *geometrically adjacent* bins
            # (skipping pairs separated by a bin dropped for low SM statistics) larger
            # than their combined per-bin uncertainty can explain -- catches a rough,
            # unstable coefficient even where each bin looks fine on its own.
            bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.0
            neighbor_pulls = []
            for (bc0, c0, u0, _), (bc1, c1, u1, _) in zip(used, used[1:]):
                if bin_width <= 0.0 or abs((bc1 - bc0) - bin_width) > 1e-6 * bin_width:
                    continue
                combined_unc = (u0 ** 2 + u1 ** 2) ** 0.5
                neighbor_pulls.append(abs(c1 - c0) / combined_unc if combined_unc > 1e-12 else 0.0)
            max_neighbor_pull = max(neighbor_pulls) if neighbor_pulls else 0.0
            neighbor_jump_frac = (
                sum(1 for p in neighbor_pulls if p >= pull_threshold) / len(neighbor_pulls)
                if neighbor_pulls else 0.0
            )

            rows.append({
                "feature": feature,
                "derivative": derivative,
                "n_bins_used": len(used),
                "max_abs_coeff": round(max_abs,PRECISION),
                "flagged_bin_center": round(used[max_idx][0], PRECISION),
                "mean_abs_coeff": round(mean_abs, PRECISION),
                "std_abs_coeff": round(std_abs, PRECISION),
                "max_to_mean_ratio": round(ratio,PRECISION),
                "mean_abs_pull": round(mean_abs_pull, PRECISION),
                "max_abs_pull": round(max_abs_pull, PRECISION),
                "noisy_bin_frac": round(noisy_bin_frac, PRECISION),
                "max_neighbor_pull": round(max_neighbor_pull, PRECISION),
                "neighbor_jump_frac": round(neighbor_jump_frac, PRECISION),
            })
    return rows


def _recommend(max_abs_coeff: float, max_to_mean_ratio: float,
                sensitivity_threshold: float, ratio_threshold: float) -> str:
    if max_abs_coeff < sensitivity_threshold:
        return "keep"
    if max_to_mean_ratio >= ratio_threshold:
        return "cut"
    return "monitor"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                         help="One or more bin_contents.json files, or directories to search recursively")
    parser.add_argument("--sensitivity-threshold", type=float, default=0.1,
                         help="Min |coefficient| for a variable to be considered sensitive to an operator")
    parser.add_argument("--ratio-threshold", type=float, default=5.0,
                         help="Min max/mean(|coeff|) ratio to flag a variable for an upper cut")
    parser.add_argument("--min-stat", type=float, default=20.0,
                         help="Minimum SM event count in a bin for it to be used in statistics (avoids low-stat artifacts)")
    parser.add_argument("--pull-threshold", type=float, default=3.0,
                         help="Min |coeff|/unc for a bin to count as statistically significant, not just noisy")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                         help="Directory to write the output CSVs")
    args = parser.parse_args()

    json_files = _find_bin_contents(args.paths)
    if not json_files:
        raise RuntimeError("No bin_contents.json files found under the given paths.")
    logger.info("Found %d bin_contents.json files.", len(json_files))

    detail_rows = []
    for path in json_files:
        job_id = path.split("/")[-5]
        for row in _analyze_file(path, args.min_stat, args.sensitivity_threshold, args.pull_threshold):
            row["job_id"] = job_id
            row["recommendation"] = _recommend(
                row["max_abs_coeff"], row["max_to_mean_ratio"],
                args.sensitivity_threshold, args.ratio_threshold,
            )
            detail_rows.append(row)

    detail_cols = [
        "job_id", "feature", "derivative", "n_bins_used",
        "max_abs_coeff", "flagged_bin_center", "mean_abs_coeff", "std_abs_coeff",
        "max_to_mean_ratio", "mean_abs_pull", "max_abs_pull", "noisy_bin_frac",
        "max_neighbor_pull", "neighbor_jump_frac", "recommendation",
    ]
    detail_path = os.path.join(args.out_dir, f"eft_sensitivity_detail_minstat{str(args.min_stat).replace('.','p')}.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_cols)
        writer.writeheader()
        # storing in descending order of max absolute coefficient
        for row in sorted(detail_rows, key=lambda r: -r["max_abs_coeff"]):
            writer.writerow(row)
    logger.info("Wrote %d detail rows to %s", len(detail_rows), detail_path)

    # ---- per-feature worst-case summary across all files scanned ----
    by_feature: dict[str, dict] = {}
    for row in detail_rows:
        feat = row["feature"]
        best = by_feature.get(feat)
        if best is None or row["max_abs_coeff"] > best["max_abs_coeff"]:
            by_feature[feat] = row

    summary_cols = [
        "feature", "worst_job_id", "worst_derivative",
        "max_abs_coeff", "flagged_bin_center", "max_to_mean_ratio", "recommendation",
    ]
    summary_path = os.path.join(args.out_dir, f"eft_sensitivity_summary_minstat{str(args.min_stat).replace('.','p')}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for feat, row in sorted(by_feature.items(), key=lambda kv: -kv[1]["max_abs_coeff"]):
            writer.writerow({
                "feature": feat,
                "worst_job_id": row["job_id"],
                "worst_derivative": row["derivative"],
                "max_abs_coeff": row["max_abs_coeff"],
                "flagged_bin_center": row["flagged_bin_center"],
                "max_to_mean_ratio": row["max_to_mean_ratio"],
                "recommendation": row["recommendation"],
            })
    logger.info("Wrote %d summary rows to %s", len(by_feature), summary_path)

    n_cut = sum(1 for r in by_feature.values() if r["recommendation"] == "cut")
    n_monitor = sum(1 for r in by_feature.values() if r["recommendation"] == "monitor")
    n_keep = sum(1 for r in by_feature.values() if r["recommendation"] == "keep")
    logger.info("Recommendation breakdown (worst case per feature): cut=%d monitor=%d keep=%d",
                n_cut, n_monitor, n_keep)

    # ---- per-derivative summary across all features it appears in ----
    # Transpose of the per-feature summary above: is this *operator* noisy across
    # (most of) the features it's projected onto, not just in one worst-case bin.
    by_derivative: dict[str, list[dict]] = {}
    for row in detail_rows:
        by_derivative.setdefault(row["derivative"], []).append(row)

    by_derivative_path = os.path.join(args.out_dir, f"eft_sensitivity_by_derivative_minstat{str(args.min_stat).replace('.','p')}.csv")
    by_derivative_cols = [
        "derivative", "n_features", "worst_feature", "worst_job_id",
        "max_abs_coeff", "mean_noisy_bin_frac", "mean_abs_pull",
        "mean_neighbor_jump_frac", "max_neighbor_pull",
    ]
    with open(by_derivative_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=by_derivative_cols)
        writer.writeheader()
        by_derivative_summary = {}
        for derivative, rows in by_derivative.items():
            worst = max(rows, key=lambda r: r["max_abs_coeff"])
            by_derivative_summary[derivative] = {
                "derivative": derivative,
                "n_features": len(rows),
                "worst_feature": worst["feature"],
                "worst_job_id": worst["job_id"],
                "max_abs_coeff": worst["max_abs_coeff"],
                "mean_noisy_bin_frac": round(sum(r["noisy_bin_frac"] for r in rows) / len(rows), PRECISION),
                "mean_abs_pull": round(sum(r["mean_abs_pull"] for r in rows) / len(rows), PRECISION),
                "mean_neighbor_jump_frac": round(sum(r["neighbor_jump_frac"] for r in rows) / len(rows), PRECISION),
                "max_neighbor_pull": max(r["max_neighbor_pull"] for r in rows),
            }
        for derivative, row in sorted(by_derivative_summary.items(), key=lambda kv: -kv[1]["mean_noisy_bin_frac"]):
            writer.writerow(row)
    logger.info("Wrote %d per-derivative rows to %s", len(by_derivative_summary), by_derivative_path)

    # adding files by hand to single object to send them to LXPLUS/EOS
    syncer.file_sync_storage.extend([summary_path, detail_path, by_derivative_path])
    syncer.sync()


if __name__ == "__main__":
    main()
