#!/usr/bin/env python
"""Summarize EFT modification `bin_contents.json` outputs into flat CSV reports.

Consumes `bin_contents.json` files produced by `plot/bit/eft_modification_plot.py`
(written by `plot/bit/modification_plotter.py::_write_analysis_json`), one file
per (job, term order). For every (feature, derivative) pair it computes summary
statistics of the per-bin EFT coefficient (ratio of modified to SM weight) and
flags variables with large or wildly fluctuating coefficients as candidates for
an upper cut, so BIT training is not destabilized by rare large-weight events.

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
        --sensitivity-threshold 0.1 --ratio-threshold 5.0 --min-stat 20

Job id and term order (lin/quad) are inferred from the path
(`.../<job_id>/<lin|quad>/bin_contents.json`), matching the layout written by
`modification_plotter.py`.

Outputs two CSVs in --out-dir (default: this script's directory):
    - eft_sensitivity_detail.csv  : one row per (job, terms, feature, derivative)
    - eft_sensitivity_summary.csv : one row per feature, worst case across all
                                     files scanned
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

TERM_DIRS = ("lin", "quad")

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


def _job_and_term_from_path(path: str) -> tuple[str, str]:
    """Infer (job_id, term_order) from a '.../<job_id>/<lin|quad>/bin_contents.json' path."""
    term_dir = os.path.basename(os.path.dirname(path))
    job_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    term_order = term_dir if term_dir in TERM_DIRS else "unknown"
    job_id = job_dir if term_dir in TERM_DIRS else term_dir
    return job_id, term_order


def _analyze_file(path: str, min_stat: float) -> list[dict]:
    """Return per-(feature, derivative) statistics rows for one bin_contents.json."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for feature, feat_data in data.items():
        sm = feat_data["sm_histogram"]
        bin_centers = feat_data["bin_centers"]
        for key, values in feat_data.items():
            if key in ("bin_centers", "sm_histogram"):
                continue
            derivative = key

            used = [(bc, v, s) for bc, v, s in zip(bin_centers, values, sm) if s >= min_stat]
            if not used:
                rows.append({
                    "feature": feature, "derivative": derivative,
                    "n_bins_used": 0, "max_abs_coeff": 0.0, "flagged_bin_center": None,
                    "mean_abs_coeff": 0.0, "std_abs_coeff": 0.0, "max_to_mean_ratio": 0.0,
                })
                continue

            abs_vals = [abs(v) for _, v, _ in used]
            max_abs = max(abs_vals)
            max_idx = abs_vals.index(max_abs)
            mean_abs = sum(abs_vals) / len(abs_vals)
            var = sum((a - mean_abs) ** 2 for a in abs_vals) / len(abs_vals)
            std_abs = var ** 0.5
            ratio = (max_abs / mean_abs) if mean_abs > 1e-10 else 0.0

            rows.append({
                "feature": feature,
                "derivative": derivative,
                "n_bins_used": len(used),
                "max_abs_coeff": round(max_abs,PRECISION),
                "flagged_bin_center": round(used[max_idx][0], PRECISION),
                "mean_abs_coeff": round(mean_abs, PRECISION),
                "std_abs_coeff": round(std_abs, PRECISION),
                "max_to_mean_ratio": round(ratio,PRECISION),
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
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                         help="Directory to write the two output CSVs")
    args = parser.parse_args()

    json_files = _find_bin_contents(args.paths)
    if not json_files:
        raise RuntimeError("No bin_contents.json files found under the given paths.")
    logger.info("Found %d bin_contents.json files.", len(json_files))

    detail_rows = []
    for path in json_files:
        job_id, term_order = _job_and_term_from_path(path)
        for row in _analyze_file(path, args.min_stat):
            row["job_id"] = job_id
            row["term_order"] = term_order
            row["recommendation"] = _recommend(
                row["max_abs_coeff"], row["max_to_mean_ratio"],
                args.sensitivity_threshold, args.ratio_threshold,
            )
            detail_rows.append(row)

    detail_cols = [
        "job_id", "term_order", "feature", "derivative", "n_bins_used",
        "max_abs_coeff", "flagged_bin_center", "mean_abs_coeff", "std_abs_coeff",
        "max_to_mean_ratio", "recommendation",
    ]
    detail_path = os.path.join(args.out_dir, "eft_sensitivity_detail.txt")
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
        "feature", "worst_job_id", "worst_term_order", "worst_derivative",
        "max_abs_coeff", "flagged_bin_center", "max_to_mean_ratio", "recommendation",
    ]
    summary_path = os.path.join(args.out_dir, "eft_sensitivity_summary.txt")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for feat, row in sorted(by_feature.items(), key=lambda kv: -kv[1]["max_abs_coeff"]):
            writer.writerow({
                "feature": feat,
                "worst_job_id": row["job_id"],
                "worst_term_order": row["term_order"],
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
    
    # adding files by hand to single object to send them to LXPLUS/EOS
    syncer.file_sync_storage.extend([summary_path, detail_path])
    syncer.sync()


if __name__ == "__main__":
    main()
