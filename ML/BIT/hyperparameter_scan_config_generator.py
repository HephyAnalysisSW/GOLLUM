#!/usr/bin/env python
"""Generate BIT hyperparameter variant jobs from a standard job, one at a time.

Reads a config's standard BIT job, and for each hyperparameter in
HYPERPARAMETER_POOLS samples --n-samples candidate values (each variant job
changes exactly one hyperparameter away from the standard job). Writes:
  - a config with the new job blocks appended to the standard job's config,
  - a shell script with one `ML/BIT/eft_bit_training.py --job <id> --debug`
    line per new job, ready to be piped to the `submit` command.

Feed the output config + job ids into bit_hyperparameter_scan.py afterward to
check calibration.

Example:
    python user/ricardo/generate_bit_hyperparameter_variants.py \\
        configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_cQj18.yaml \\
        --standard-job bit_TT01j2l_EFT_2016_cQj18 \\
        --n-samples 3 --seed 0 \\
        --output-config configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_cQj18_random_scan.yaml \\
        --output-script user/ricardo/launch_cQj18_random_scan.sh
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys

sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import yaml

import common.yaml_loader as yaml_loader

# candidate values per hyperparameter, one changed at a time relative to the standard job
HYPERPARAMETER_POOLS = {
    "n_bins": [32, 64, 128, 256, 512, 1024],
    "min_size": [1, 5, 10, 25, 50, 100, 200],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    "n_trees": [100, 200, 400, 800],
    "quantile_bins": [True, False],
    "split_mode": ["binned", "exact"],
}

# framework defaults for hyperparameters the standard job may leave unset (see the
# handles listed as comments in configs/unbinned_v7_eft_genpoint/*.yaml)
FRAMEWORK_DEFAULTS = {"max_depth": 4, "min_size": 50}

def sanitize(value) -> str:
    """Turn a hyperparameter value into a job-id-safe token, e.g. 0.3 -> '0p3'."""
    return str(value).replace(".", "p").replace("-", "m")


def find_job(cfg: dict, job_id: str) -> dict:
    """Return the job dict in cfg['jobs'] with id == job_id."""
    for job in cfg["jobs"]:
        if job.get("id") == job_id:
            return job
    raise RuntimeError(f"No job with id {job_id!r} in config")


def build_variant_jobs(standard_job: dict, n_samples: int, rng: random.Random, existing_ids: set[str]) -> list[dict]:
    """One variant job per sampled value per hyperparameter, changing only that hyperparameter."""
    variants = []
    for param, pool in HYPERPARAMETER_POOLS.items():
        current_value = standard_job["model"].get(param, FRAMEWORK_DEFAULTS.get(param))
        candidates = [v for v in pool if v != current_value]
        sampled = rng.sample(candidates, k=min(n_samples, len(candidates)))
        for value in sampled:
            job_id = f"{standard_job['id']}_{param}_{sanitize(value)}"
            if job_id in existing_ids:
                continue
            variant = copy.deepcopy(standard_job)
            variant["id"] = job_id
            variant["model"][param] = value
            variant["output"]["filename"] = f"{job_id}.pkl"
            variants.append(variant)
            existing_ids.add(job_id)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the base YAML config holding the standard BIT job")
    parser.add_argument("--standard-job", required=True, help="id of the standard/baseline BIT job")
    parser.add_argument("--n-samples", type=int, default=3, help="Candidate values to sample per hyperparameter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--output-config", required=True, help="Path to write the config with the new job blocks appended")
    parser.add_argument("--output-script", required=True, help="Path to write the shell script of training commands")
    args = parser.parse_args()

    cfg = yaml_loader.load_yaml(args.config)
    standard_job = find_job(cfg, args.standard_job)
    existing_ids = {job["id"] for job in cfg["jobs"]}

    rng = random.Random(args.seed)
    variant_jobs = build_variant_jobs(standard_job, args.n_samples, rng, existing_ids)

    cfg["jobs"] = cfg["jobs"] + variant_jobs
    os.makedirs(os.path.dirname(os.path.abspath(args.output_config)), exist_ok=True)
    with open(args.output_config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {len(variant_jobs)} new job blocks to {args.output_config}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_script)), exist_ok=True)
    with open(args.output_script, "w") as f:
        for job in variant_jobs:
            if "n_ensemble" in job:
                n_ensemble = job.get("n_ensemble")
                for i_ensemble in range(n_ensemble):
                    f.write(f"python ML/BIT/eft_bit_training.py {args.output_config} --job {job['id']} --i_ensemble {i_ensemble}  --debug\n")
            else:
                f.write(f"python ML/BIT/eft_bit_training.py {args.output_config} --job {job['id']} --debug\n")
    
    print(f"Wrote training commands to {args.output_script}")


if __name__ == "__main__":
    main()
