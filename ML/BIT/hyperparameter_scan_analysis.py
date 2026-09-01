#!/usr/bin/env python
"""Scan the calibration bias of BIT hyperparameter variants.

For every non-standard BIT job in a config (every 'bit' job besides the one
given with --standard-job), this script:
  1. builds a single-job variant config that points the region's POI at that job,
  2. generates a no-Poisson truth toy at --toy-point (retrying once, since the
     first run against a fresh surrogate cache can crash from a double
     materialization: build_cache() streams the sample once, then truth-mode
     toy generation streams it again independently; the retry finds the cache
     already built and only pays the second pass),
  3. fits that toy with the variant config and reads back --poi,
and prints/saves a summary of the fitted values across all variants.

Example:
    python user/ricardo/bit_hyperparameter_scan.py \\
        configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_cQj18.yaml \\
        --standard-job bit_TT01j2l_EFT_2016_cQj18 \\
        --toy-spec configs/unbinned_v7_eft_genpoint/toys_BIT_closure_test_2016.yaml \\
        --toy-point cQj18_sm \\
        --poi cQj18
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import h5py
import yaml

import common.user as user
import common.yaml_loader as yaml_loader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_poi_class(cfg: dict, standard_job_id: str) -> dict:
    """Find the class dict in cfg['likelihood']['regions'] whose POI.job is standard_job_id."""
    for region in cfg["likelihood"]["regions"]:
        for cls in region["classes"]:
            if (cls.get("POI") or {}).get("job") == standard_job_id:
                return cls
    raise RuntimeError(f"No class references POI.job == {standard_job_id!r} in config")


def run_with_retry(cmd: list[str], retries: int = 1) -> None:
    """Run a subprocess command, retrying on failure.

    Needed because the first run against a not-yet-built surrogate cache directory
    can crash (see module docstring); a second identical run finds the cache
    already built and succeeds.
    """
    attempt = 0
    while True:
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode == 0:
            return
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"Command failed after {attempt} attempt(s): {' '.join(cmd)}")
        logger.warning("Command failed (attempt %d), retrying: %s", attempt, " ".join(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the base YAML config holding the standard job and its variants")
    parser.add_argument("--standard-job", required=True, help="id of the standard/baseline BIT job")
    parser.add_argument("--toy-spec", required=True, help="Path to the toy spec YAML")
    parser.add_argument("--toy-point", required=True, help="Name of the point in the toy spec to generate")
    parser.add_argument("--scan-dir", default=None,
                         help="Where to write per-job configs/toys (default: <user.output_directory>/bit_scan/<config_basename>)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate toys/fits even if they already exist")
    args = parser.parse_args()

    cfg = yaml_loader.load_yaml(args.config)
    all_bit_jobs = [job for job in cfg["jobs"] if job.get("type") == "bit"]
    variant_jobs = [job for job in all_bit_jobs if job["id"] != args.standard_job]
    if not variant_jobs:
        raise RuntimeError(f"No non-standard 'bit' jobs found besides {args.standard_job!r}")

    config_base = os.path.splitext(os.path.basename(args.config))[0]
    scan_dir = args.scan_dir or os.path.join(REPO_ROOT, "user", "ricardo", "bit_scan_configs", config_base)
    os.makedirs(scan_dir, exist_ok=True)

    results = []
    for job in variant_jobs:
        job_id = job["id"]
        logger.info("=== %s ===", job_id)

        # single-job variant config: same likelihood, POI redirected to this job,
        # and only this job's artifact needs to exist on disk
        variant_cfg = copy.deepcopy(cfg)
        find_poi_class(variant_cfg, args.standard_job)["POI"]["job"] = job_id
        variant_cfg["jobs"] = [job]

        # named after job_id so ToyGenerator's and Likelihood's surrogate cache
        # (keyed off the config basename / --base) is the same directory for both
        variant_config_path = os.path.join(scan_dir, f"{job_id}.yaml")
        with open(variant_config_path, "w") as f:
            yaml.safe_dump(variant_cfg, f, sort_keys=False)

        # ToyGenerator.py always appends "_no_poisson" to --outputDir under --no_poisson
        toy_dir = os.path.join(scan_dir, job_id, "toys") + "_no_poisson"
        toy_path = os.path.join(toy_dir, "toy0.h5")
        if args.overwrite or not os.path.exists(toy_path):
            toy_cmd = [
                "python", "fit/ToyGenerator.py", variant_config_path,
                "--toySpec", args.toy_spec,
                "--toyPoint", args.toy_point,
                "--seeds", "0",
                "--outputDir", os.path.join(scan_dir, job_id, "toys"),
                "--no_poisson",
            ]
            if args.overwrite:
                toy_cmd += ["--overwrite", "toy"]
            run_with_retry(toy_cmd)

        with h5py.File(toy_path, "r") as toy_f:
            toy_point = str(toy_f["meta"].attrs.get("point", "")) or "toy"
            toy_seed = int(toy_f["meta"].attrs["seed"])
            toy_source = str(toy_f["meta"].attrs.get("source", ""))
        if "no_poisson" in toy_path:
            toy_source += "_no_poisson"

        fit_cmd = [
            "python", "fit/Likelihood.py", variant_config_path,
            "--toyFile", toy_path,
            "--base", job_id,
        ]
        if args.overwrite:
            fit_cmd += ["--overwrite", "fit"]
        run_with_retry(fit_cmd)

        # replicate fit/Likelihood.py's out_path construction (base=job_id via --base)
        version = str(variant_cfg.get("version", "v0"))
        suffix = f"_{toy_point}_{toy_source}_toy{toy_seed}"
        fit_out_path = os.path.join(
            user.output_directory, f"{job_id}_{toy_point}_{toy_source}_toy_fits",
            f"{job_id}_{version}{suffix}_fit.json",
        )
        fit_result = json.load(open(fit_out_path))
        for p in fit_result["parameters"]:
            results.append({"job": job_id,"name": p["name"],"value": p["value"], "error": p["error"]})
            logger.info("%s: %s = %+.4f +/- %.4f", job_id, p["name"], p["value"], p["error"])

    logger.info("\n=== Summary ===")
    for r in results:
        logger.info("%-45s: %-45s = %+.4f +/- %.4f", r["job"], r["name"], r["value"], r["error"])

    summary_path = os.path.join(scan_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
