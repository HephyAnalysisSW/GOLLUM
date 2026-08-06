"""
check_weights_toys.py

Utility script used during toy-generation to compare weight arrays produced by
two different reweighting mechanisms (cache-based exact reweighting vs.
truth/BIT-based reweighting). Intended to be run from the command line.

What it does
- Loads one or more global YAML configs (project and sample configuration).
- Builds and prepares an N2LL likelihood object and related factory for samples.
- Loads a toy specification YAML describing points and splitting.
- Either loads existing saved numpy weight arrays or generates a toy and
    extracts the relevant weight array and saves it.
- Prints a short summary comparing sums of weights when both cache and truth
    arrays exist.

Usage (example):
    python check_weights_toys.py config.yaml --toySpec toy_spec.yaml --toyPoint POINT_NAME --outputDir out_dir
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import importlib
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import h5py

# project root (this file lives in fit/) + ML/Calibration (for the shared UID-split helper)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "ML", "Calibration"))

from fit.Likelihood import N2LL, expand_pois_linear_quadratic, build_hypothesis_from_likelihood
from fit.Modeling import Hypothesis
from ML.Calibration import calibration_runner as cr  # _uid_split_interval

# common.derivative_providers is imported lazily inside _materialize_truth_weights
# (see below): it pulls in data/samples_eft.py, which eagerly constructs RDataLoaders
# for every declared EFT sample at import time, so importing it at module level would
# make cache-mode generation depend on the EFT sample data being reachable even though
# cache mode never touches it.

logger = logging.getLogger(__name__)

from fit.ToyGenerator import _parse_injection, _hypothesis_from_point, generate_toy, _parse_seeds

if __name__ == "__main__":
    import argparse
    import common.yaml_loader as yaml_loader
    from fit.Likelihood import load_likelihood
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Compare weights from exact and BIT-based reweighting. Uses code developed for toy generation for convenience.")
    p.add_argument("configs", nargs="+", help="Path to one or more global YAML configs")
    p.add_argument("--toySpec", required=True, help="Path to the toy spec YAML")
    p.add_argument("--toyPoint", required=True, help="Name of the point in the spec to generate")
    p.add_argument("--outputDir", required=True, help="Directory to write the numpy arrays.")
    p.add_argument("--overwrite", action="store_true", help="Remake the weight arrays")

    args = p.parse_args()
        
    list_configs = []
    for config_path in args.configs:
        aux_cfg = yaml_loader.load_yaml(config_path)
        yaml_loader.print_summary(aux_cfg, config_path, yaml_loader._INCLUDE_TRACE)
        yaml_loader.load_surrogates(aux_cfg, config_path, overwrite=False)
        list_configs.append(aux_cfg)
    cfg = yaml_loader.combine_configs(list_configs)

    like_info = load_likelihood(cfg)

    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    from common.yaml_loader import _resolve_features_list
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    factory = samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )

    base = "_".join(os.path.splitext(os.path.basename(c))[0] for c in args.configs)
    n2ll = N2LL(
        like_info, factory=factory,
        cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
        cache_root=None,
    )

    spec = yaml_loader.load_yaml(args.toySpec)
    spec_source = spec.get("source", "cache")
    spec_split = spec.get("split", ("c2st_train","c2st_val"))
    if isinstance(spec_split, str):
        logger.info("spec_split is a string")
        spec_split = [part.strip() for part in spec_split.split(",") if part.strip()]
    else:
        logger.info("spec_split is a list")
    spec_throw_nuisances = bool(spec.get("throw_nuisances", False))
    spec_allow_negative = bool(spec.get("allow_negative_weights", False))

    n2ll.shuffle_features = None
    n2ll.build_cache()
    n2ll.prepare_runtime()

    n2ll.version = cfg.get("version")
    n2ll._toy_splitting_defaults = (cfg.get("defaults") or {}).get("splitting")
    n2ll._toy_jobs_by_id = {j["id"]: j for j in (cfg.get("jobs") or []) if j.get("id")}

    point = next((pt for pt in (spec.get("points") or []) if pt.get("name") == args.toyPoint), None)
    if point is None:
        available = [pt.get("name") for pt in (spec.get("points") or [])]
        raise RuntimeError(f"Point '{args.toyPoint}' not found in {args.toySpec}. Available: {available}")

    if spec_split is None:
        spec_split_str = "all"
    else:
        spec_split_str = "_".join(spec_split)

    weights_cache_path = os.path.join(args.outputDir, f"{args.toyPoint}_cache_weights_noPoisson.npy")
    weights_truth_path = os.path.join(args.outputDir, f"{args.toyPoint}_truth_{spec_split_str}_weights_noPoisson.npy")

    if os.path.exists(weights_cache_path) and os.path.exists(weights_truth_path) and not args.overwrite:

        weights_cache = np.load(weights_cache_path)
        weights_truth = np.load(weights_truth_path)

        delta_sumw = np.sum(weights_cache) - np.sum(weights_truth)

        logger.info(f"{args.toyPoint}: sum of weights cache= {np.sum(weights_cache)}; sum of weights truth= {np.sum(weights_truth)} (partition {spec_split_str}); delta= {delta_sumw}")
    else:
        truth_sources = _parse_injection(point.get("injection")) if spec_source == "truth" else None
        hypothesis = _hypothesis_from_point(n2ll, point.get("hypothesis"))

        toy = generate_toy(
            n2ll, seed=0, source=spec_source, hypothesis=hypothesis, truth_sources=truth_sources,
            split=spec_split, throw_nuisances=spec_throw_nuisances, allow_negative_weights=spec_allow_negative, debug=True
        )
        toy["point"] = args.toyPoint

        os.makedirs(args.outputDir, exist_ok=True)

        # works when there's only the one process
        output_path = None
        if spec_source == "cache":
            weight_array = toy["diagnostics"]['unbinned']['SR_2016']['w_cache_all']
            output_path = weights_cache_path
        elif spec_source == "truth":
            weight_array = toy["diagnostics"]['unbinned']['SR_2016']['TT01j2l_EFT_2016']['w_truth']
            output_path = weights_truth_path

        np.save(output_path, weight_array)

        logger.info(f"Weights stored in {output_path}")



