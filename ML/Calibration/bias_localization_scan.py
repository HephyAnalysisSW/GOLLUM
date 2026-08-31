#!/usr/bin/env python
"""Localize the BIT closure bias in a kinematic variable via a truth/BIT splice scan.

See user/ricardo/claude/bit_cQj18_calibration_bias_investigation.md ("Concrete next
test"). Generates one debug truth-mode Asimov toy in-process (pseudo-data = exact truth
weights at the injected point), then sweeps a threshold in a chosen kinematic variable:
for events above the threshold the per-event model ratio is replaced by the exact truth
derivative ratio; below it the BIT prediction is kept. For each threshold the single POI
is refit with the existing N2LL, and c_hat(threshold) is plotted.

The two limits are built-in validation anchors:
  - threshold -> +inf : no event is truth-corrected  -> all-BIT -> reproduces the closure
                        bias (c_hat ~ 0.2-0.3 for cQj18).
  - threshold -> -inf : every event is truth-corrected -> should collapse toward the
                        injected value (c_hat ~ 0), up to the all-BIT Asimov yield term.

Where c_hat collapses as the threshold is lowered localizes the region of phase space
whose BIT shape error sources the bias.

The splice is a top-level np.where on the toy's by_class['R'] performed *before* setToy,
so the fit machinery (rate expansion, NLL) is reused unchanged and the scan is a loop.
Run from the repo root:

    python ML/BIT/bias_localization_scan.py configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_cQj18.yaml \
        --toySpec configs/unbinned_v7_eft_genpoint/toys_BIT_closure_test_2016APV.yaml \
        --toyPoint cQj18_sm --var tr_ttbar_mass

NB: this only makes sense for 1D scans.
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import importlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# repo root (this file lives in fit/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

import common.user as user
import common.syncer as syncer
import common.helpers as helpers
import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list
from fit.Likelihood import load_likelihood, N2LL, build_hypothesis_from_likelihood
from fit.ToyGenerator import (
    generate_toy, nominal_hypothesis, _parse_injection, _hypothesis_from_point,
    _region_feature_union, _find_region,
)
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
hep.style.use("CMS")


def build_n2ll(config_paths: list[str]):
    """Combine configs, load surrogates, and return (cfg, runtime-ready N2LL).

    Mirrors the setup in ToyGenerator.__main__ so the toy this script generates and the
    fit it runs share exactly one likelihood object.
    """
    cfgs = []
    for path in config_paths:
        cfg_single = yaml_loader.load_yaml(path)
        yaml_loader.load_surrogates(cfg_single, path, overwrite=False)
        cfgs.append(cfg_single)
    cfg = yaml_loader.combine_configs(cfgs)

    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    factory = samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )
    like_info = load_likelihood(cfg)
    base = "_".join(os.path.splitext(os.path.basename(p))[0] for p in config_paths)
    n2ll = N2LL(
        like_info, factory=factory,
        cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
        cache_root=None, overwrite=False,
    )
    n2ll.shuffle_features = None
    n2ll.build_cache()
    n2ll.prepare_runtime()
    n2ll.version = cfg.get("version")
    n2ll._toy_splitting_defaults = (cfg.get("defaults") or {}).get("splitting")
    n2ll._toy_jobs_by_id = {j["id"]: j for j in (cfg.get("jobs") or []) if j.get("id")}
    return cfg, n2ll, base


def splice_ratio(bit_R: np.ndarray, truth_R: np.ndarray, var: np.ndarray, threshold: float) -> np.ndarray:
    """Return the per-event model ratio matrix to hand the fit for one threshold.

    bit_R, truth_R : (N, nA) BIT-predicted and exact-truth derivative ratios (same basis).
    var            : (N,) the kinematic variable each event is thresholded on.
    threshold      : scalar; np.inf -> all BIT, -np.inf -> all truth.
    Returns        : (N, nA) spliced ratio matrix.
    """
    use_truth = var > threshold  # (N,); strict > gives all-BIT at +inf, all-truth at -inf
    return np.where(use_truth[:, None], truth_R, bit_R)


def scan_one_threshold(n2ll, toy, rid, cid, hyp, poi_name, bit_R, truth_R, var, threshold, bounds):
    """Splice by_class['R'] at one threshold, refit the POI, return c_hat."""
    toy["unbinned_blocks"][rid]["by_class"][cid]["R"] = splice_ratio(bit_R, truth_R, var, threshold)
    n2ll.setToy(toy, hyp)

    from fit.Likelihood import run_autograd_fit, run_iminuit_fit

    # freezing POIs when given
    if poi_name:
        for poi in hyp.POIs:
            if poi.name != poi_name:
                poi.isFrozen = True
        bounds = bounds[0:]

    try:
        result = run_autograd_fit(
            n2ll,
            hyp,
            print_every=1,
            do_migrad=True,
            bounds=bounds
        )

        return result.values

    except RuntimeError:
        logger.warning("threshold=%g: no finite NLL minimum inside bounds; c_hat unreliable.",
                       threshold)
        return [float("nan") for poi in hyp.POIs]        


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="+", help="Path to one or more global YAML configs")
    p.add_argument("--toySpec", required=True, help="Path to the toy spec YAML (source: truth)")
    p.add_argument("--toyPoint", required=True, help="Name of the injection point in the spec")
    p.add_argument("--var", default="tr_ttbar_mass", help="Kinematic feature to threshold on")
    p.add_argument("--poi", default=None, help="POI to fit (default: the single POI of the likelihood)")
    p.add_argument("--n_thresholds", type=int, default=25, help="Number of finite thresholds (quantiles of var)")
    p.add_argument("--poi_bounds", type=float, nargs=2, default=[-1.0, 1.0], help="Bounds for the 1D POI fit")
    p.add_argument("--n_split", type=int, default=None, help="Loader file-shard split for materialization")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg, n2ll, base = build_n2ll(args.configs)

    # --- generate one debug truth-mode Asimov toy in-process ---
    spec = yaml_loader.load_yaml(args.toySpec)
    if spec.get("source", "cache") != "truth":
        raise RuntimeError(f"{args.toySpec}: localization scan requires a truth-mode spec.")
    spec_split = spec.get("split", ("c2st_train", "c2st_val"))
    if isinstance(spec_split, str):
        spec_split = [part.strip() for part in spec_split.split(",") if part.strip()]

    point = next((pt for pt in (spec.get("points") or []) if pt.get("name") == args.toyPoint), None)
    if point is None:
        available = [pt.get("name") for pt in (spec.get("points") or [])]
        raise RuntimeError(f"Point '{args.toyPoint}' not found in {args.toySpec}. Available: {available}")

    # generate_toy consults args.n_split inside ToyGenerator via its module-level `args`
    import fit.ToyGenerator as toygen
    toygen.args = args

    truth_sources = _parse_injection(point.get("injection"))
    hypothesis = _hypothesis_from_point(n2ll, point.get("hypothesis"))
    toy = generate_toy(
        n2ll, seed=0, source="truth", hypothesis=hypothesis, truth_sources=truth_sources,
        split=spec_split, allow_negative_weights=bool(spec.get("allow_negative_weights", False)),
        no_poisson=True, debug=True,
    )

    # --- extract the single region/class arrays ---
    unbinned = toy["unbinned_blocks"]
    if len(unbinned) != 1:
        raise RuntimeError(f"Scan supports a single unbinned region; got {list(unbinned)}.")
    rid = next(iter(unbinned))
    block = unbinned[rid]
    by_class = block["by_class"]
    if len(by_class) != 1:
        raise RuntimeError(f"Scan supports a single class; region {rid} has classes {list(by_class)}.")
    cid = next(iter(by_class))

    diag_cid = toy["diagnostics"]["unbinned"][rid][cid]
    if "truth_R" not in diag_cid:
        raise RuntimeError("Toy carries no diagnostics truth_R; was it generated with debug=True on the "
                           "derivative (coefficients) route?")
    bit_R = np.asarray(by_class[cid]["R"], dtype=np.float64)
    truth_R = np.asarray(diag_cid["truth_R"], dtype=np.float64)
    if bit_R.shape != truth_R.shape:
        raise RuntimeError(f"bit_R {bit_R.shape} vs truth_R {truth_R.shape} shape mismatch.")

    feature_names = _region_feature_union(_find_region(n2ll, rid))
    if args.var not in feature_names:
        raise RuntimeError(f"--var '{args.var}' not in region features {feature_names}.")
    var = np.asarray(block["X"], dtype=np.float64)[:, feature_names.index(args.var)]

    # --- POI and fit hypothesis ---
    #hyp = build_hypothesis_from_likelihood(n2ll.lk, name="scan")
    hyp = nominal_hypothesis(n2ll, name="scan")

    # --- threshold grid: quantiles of var, bracketed by the all-BIT / all-truth anchors ---
    finite_thresholds = np.quantile(var, np.linspace(0.0, 1.0, args.n_thresholds))
    thresholds = np.concatenate(([-np.inf], finite_thresholds, [np.inf]))

    bounds = [tuple(args.poi_bounds)]
    c_hat = np.array([
        scan_one_threshold(n2ll, toy, rid, cid, nominal_hypothesis(n2ll, name="scan"),  args.poi, bit_R, truth_R, var, float(thr), bounds)
        for thr in thresholds
    ])
    for thr, chat in zip(thresholds, c_hat):
        logger.info(f"threshold({args.var}) = {thr:.6g} -> c_hat = {chat}")

    # --- plot ---
    out_dir = os.path.join(user.plot_directory, "BIT-localization", base, cfg.get("version", "default"), rid, args.toyPoint)
    os.makedirs(out_dir, exist_ok=True)

    for i_poi, poi in enumerate([poi for poi in hyp.POIs if not poi.isFrozen]):
        
        finite = np.isfinite(thresholds)
        c_hat_poi = c_hat[:,i_poi]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(thresholds[finite], c_hat_poi[finite], marker="o", color="k", label=r"$\hat{c}(\mathrm{threshold})$")
        ax.axhline(c_hat_poi[thresholds == np.inf][0], ls="--", color="tab:red", label="all-BIT (thr $\\to +\\infty$)")
        ax.axhline(c_hat_poi[thresholds == -np.inf][0], ls="--", color="tab:blue", label="all-truth (thr $\\to -\\infty$)")
        ax.axhline(0.0, ls=":", color="0.5")
        ax.set_xlabel(f"{args.var} threshold (truth used above, BIT below)")
        ax.set_ylabel(rf"$\hat{{{poi.name}}}$")
        ax.legend(frameon=False, fontsize=12)
        hep.cms.label("Internal", data=False, ax=ax, loc=0)
        out_path = os.path.join(out_dir, f"localization_{poi.name}_{args.var}")
        plt.savefig(out_path + ".png", bbox_inches="tight", dpi=200)
        plt.savefig(out_path + ".pdf", bbox_inches="tight")
        logger.info("Wrote %s.png / .pdf", out_path)

        np.savez(out_path + ".npz", thresholds=thresholds, c_hat=c_hat_poi, var=args.var, poi=poi.name)
    helpers.copyIndexPHP(out_dir)


if __name__ == "__main__":
    main()
    syncer.sync()
