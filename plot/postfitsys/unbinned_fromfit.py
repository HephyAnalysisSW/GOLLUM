#!/usr/bin/env python3
"""Postfit plots for unbinned fits using default feature binnings."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from array import array
from typing import Any

import numpy as np
import ROOT
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
import common.yaml_loader as yaml_loader

from common.yaml_loader import _resolve_features_list
from data.colors import get_color
from data.plot_options import plot_options, get_sample_legend
from fit.Likelihood import build_hypothesis_from_likelihood, expand_pois_linear_quadratic, load_likelihood, nuis_to_A_vector
from fit.Modeling import Rotated

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_KEEPALIVE = []


def _safe_name(text: object) -> str:
    """Return a ROOT-safe object name."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(text))


def _hist_with_flow(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram values with under/overflow merged into first/last bins."""
    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, values, side="right") - 1
    out = np.bincount(idx[(idx >= 0) & (idx < n_bins)], weights=weights[(idx >= 0) & (idx < n_bins)], minlength=n_bins).astype(np.float64)

    under = idx < 0
    if np.any(under):
        out[0] += np.sum(weights[under])
    over = idx >= n_bins
    if np.any(over):
        out[-1] += np.sum(weights[over])
    return out


def build_class_shard_cache(
    cls: dict[str, Any],
    event_features: np.ndarray,
    nominal_weights: np.ndarray,
    feature_to_index: dict[str, int],
) -> dict[str, Any]:
    """Build per-shard surrogate cache so later evaluations are fast dot products."""
    poi = cls.get("POI", {}) or {}
    poi_params = list(poi.get("parameters", []) or [])
    poi_predictor = poi.get("predictor", None)

    n_events = event_features.shape[0]
    if poi_predictor is None:
        r_a = np.empty((n_events, 0), dtype=np.float64)
    else:
        poi_feature_names = list(getattr(poi_predictor, "feature_names", []) or [])
        if poi_feature_names:
            missing = [name for name in poi_feature_names if name not in feature_to_index]
            if missing:
                raise RuntimeError(
                    f"Missing POI predictor features for class {cls.get('id', '?')}: {missing}"
                )
            poi_cols = [feature_to_index[name] for name in poi_feature_names]
            events_for_poi = event_features[:, poi_cols]
        else:
            events_for_poi = event_features

        r_a = np.asarray(poi_predictor.predict(events_for_poi), dtype=np.float64)
        if r_a.ndim == 1:
            r_a = r_a[:, None]

    syst_caches = []
    lnN_terms = []
    for syst in cls.get("systematics", []) or []:
        syst_type = syst.get("type")

        if syst_type in ("pnn", "icp", "icph"):
            syst_predictor = syst.get("predictor", None)
            if syst_predictor is None:
                raise RuntimeError(
                    f"Missing {syst_type} predictor for {cls.get('id', '?')}/{syst.get('id', '?')}"
                )

            syst_feature_names = list(getattr(syst_predictor, "feature_names", []) or [])
            if syst_feature_names:
                missing = [name for name in syst_feature_names if name not in feature_to_index]
                if missing:
                    raise RuntimeError(
                        f"Missing {syst_type} predictor features in {cls.get('id', '?')}/{syst.get('id', '?')}: {missing}"
                    )
                syst_cols = [feature_to_index[name] for name in syst_feature_names]
                events_for_syst = event_features[:, syst_cols]
            else:
                events_for_syst = event_features

            if hasattr(syst_predictor, "deltaA"):
                delta_a = np.asarray(syst_predictor.deltaA(events_for_syst), dtype=np.float64)
            else:
                delta_a = np.asarray(syst_predictor.predict(events_for_syst), dtype=np.float64)
            if delta_a.ndim == 1:
                delta_a = delta_a[:, None]

            syst_caches.append(
                {
                    "id": syst.get("id", "?"),
                    "type": syst_type,
                    "delta_a": delta_a,
                    "params": list(syst.get("parameters", []) or []),
                    "combinations": [tuple(item) for item in (syst.get("combinations", []) or [])],
                }
            )

        elif syst_type == "lnN":
            syst_params = list(syst.get("parameters", []) or [])
            if len(syst_params) != 1:
                raise RuntimeError(f"Bad lnN systematic definition: {syst}")
            alpha = float(syst.get("value", 0.0))
            lnN_terms.append((syst_params[0], float(np.log1p(alpha))))

    return {
        "features": np.asarray(event_features, dtype=np.float64),
        "weights": np.asarray(nominal_weights, dtype=np.float64),
        "poi_params": poi_params,
        "reference_point": dict(getattr(poi_predictor, "expansion_point", {}) or {}),
        "r_a": r_a,
        "syst_caches": syst_caches,
        "lnN_terms": lnN_terms,
    }


def evaluate_class_cached(shard_cache: dict[str, Any], h_base: Any) -> np.ndarray:
    """Evaluate event weights from cached surrogate tensors for a parameter point."""
    event_weights = shard_cache["weights"]
    n_events = len(event_weights)

    poi_params = shard_cache["poi_params"]
    reference_point = shard_cache["reference_point"]
    r_a = shard_cache["r_a"]
    if r_a.shape[1] > 0:
        poi_values = {name: float(h_base[name].val) for name in poi_params}
        c_a = expand_pois_linear_quadratic(poi_params, poi_values, reference_point)
        if r_a.shape[1] != len(c_a):
            raise RuntimeError(
                f"POI basis mismatch: predictor columns {r_a.shape[1]} != c_A length {len(c_a)}"
            )
        c_dot_r = r_a @ c_a
    else:
        c_dot_r = np.zeros(n_events, dtype=np.float64)

    expo = np.zeros(n_events, dtype=np.float64)
    for syst_cache in shard_cache["syst_caches"]:
        params = syst_cache["params"]
        combinations = syst_cache["combinations"]
        nu_values = {name: float(h_base[name].val) for name in params}
        nu_a = nuis_to_A_vector(params, combinations, nu_values)
        delta_a = syst_cache["delta_a"]
        if delta_a.shape[1] != len(nu_a):
            raise RuntimeError(
                f"{syst_cache['type']} basis mismatch in {syst_cache['id']}: "
                f"predictor columns {delta_a.shape[1]} != nu_A length {len(nu_a)}"
            )
        expo += delta_a @ nu_a

    for param_name, log1p_alpha in shard_cache["lnN_terms"]:
        expo += float(log1p_alpha * float(h_base[param_name].val))

    response = (1.0 + c_dot_r) * np.exp(expo)
    return event_weights * response

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple postfit unbinned template plots from an external fit result")
    p.add_argument("fit_result", help="Fit JSON file")
    p.add_argument("--configs", nargs="+",default=None, help="Path to one or more configs.", required=True)
    p.add_argument("--base", help="Base name for output directories")
    p.add_argument("--rotate", default=None, help="Rotation JSON, same logic as Likelihood.py")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--n-toys", default=1000, type=int, help="Number of covariance toys")
    p.add_argument("--seed", default=42, type=int, help="Random seed")
    p.add_argument("--min_ratio", type=float, help="Minimum of ratio pad.")
    p.add_argument("--max_ratio", type=float, help="Maximum of ratio pad.")
    p.add_argument("--prefit", action="store_true", help="Creates prefit plots.")
    args = p.parse_args()

    fit: Any = json.load(open(args.fit_result))


    logger.info("fit_result : %s", args.fit_result)
    logger.info("config     : %s", args.configs)
    logger.info("rotate     : %s", args.rotate)

    # doing it this way, since print_summary and load_surrogates
    # use the path of the configs to give info to the user
    list_configs = []
    for config_path in args.configs:
        aux_cfg = yaml_loader.load_yaml(config_path)
        yaml_loader.print_summary(aux_cfg, config_path, yaml_loader._INCLUDE_TRACE)
        yaml_loader.load_surrogates(aux_cfg, config_path, overwrite=False)

        list_configs.append(aux_cfg)

    cfg = yaml_loader.combine_configs(list_configs)

    like_info: Any = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    if not like_info.get("regions", []):
        raise RuntimeError("Config does not define unbinned regions in likelihood.regions")

    default_feature_tokens = cfg.get("defaults", {}).get("default_features", []) or []
    default_features = _resolve_features_list(default_feature_tokens)
    plot_features = [feature for feature in default_features if feature in plot_options and "binning" in plot_options[feature]]
    if not plot_features:
        raise RuntimeError("No default feature has plot_options binning. Nothing to plot.")
    feature_to_index = {name: i for i, name in enumerate(default_features)}

    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    factory = samples_mod.Factory(
        features=default_features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )

    is_data_fit = "data" in args.fit_result
    rotated = bool(args.rotate)
    hyp_for_fit = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

    fit_names = list(fit["free_parameter_order"])
    fit_param_map = {p_item["name"]: float(p_item["value"]) for p_item in fit["parameters"]}
    fit_cov = np.asarray(fit["covariance"]["matrix"], dtype=np.float64)

    all_unfrozen = [par for par in hyp_for_fit.parameters if not par.isFrozen]
    all_unfrozen_names = [par.name for par in all_unfrozen]
    poi_name_set = {par.name for par in hyp_for_fit.POIs}

    missing_in_fit = [n for n in all_unfrozen_names if n not in fit_param_map or n not in fit_names]
    missing_pois_in_fit = [n for n in missing_in_fit if n in poi_name_set]
    missing_other_in_fit = [n for n in missing_in_fit if n not in poi_name_set]

    if missing_pois_in_fit:
        logger.warning("These POIs are missing in fit JSON; setting to zero and not sampling: %s", missing_pois_in_fit)
        for n in missing_pois_in_fit:
            if hasattr(hyp_for_fit, n):
                getattr(hyp_for_fit, n).val = 0.0
            if hasattr(hyp, n):
                getattr(hyp, n).val = 0.0

    if missing_other_in_fit:
        raise RuntimeError(f"These active non-POI parameters are missing in fit JSON: {missing_other_in_fit}")

    extra_in_fit = [n for n in fit_names if n not in all_unfrozen_names]
    if extra_in_fit:
        logger.warning("Ignoring parameters present in fit JSON but not active here: %s", extra_in_fit)

    active_params = [par for par in all_unfrozen if par.name in fit_param_map and par.name in fit_names]
    active_names = [par.name for par in active_params]

    fit_index = {name: idx for idx, name in enumerate(fit_names)}
    cov_index = [fit_index[name] for name in active_names]
    cov_active = fit_cov[np.ix_(cov_index, cov_index)]
    mean_active = np.asarray([fit_param_map[name] for name in active_names], dtype=np.float64)

    best_fit_hyp = hyp_for_fit.cloneModify(**{name: fit_param_map[name] for name in active_names})
    best_base_hyp = best_fit_hyp._base

    np.random.seed(args.seed)
    if len(active_names) > 0 and args.n_toys > 0:
        if args.prefit:
            theta_samples = np.random.normal(loc=0.0, scale=1.0, size=(args.n_toys, len(active_params)))
        else:
            theta_samples = np.random.multivariate_normal(mean_active, cov_active, size=args.n_toys)
    else:
        theta_samples = np.zeros((0, len(active_names)), dtype=np.float64)

    # base from mangling together configs or given by user
    base_list = []
    for config_path in args.configs:
        base_list.append(os.path.splitext(os.path.basename(config_path))[0])

    base = "_".join(base_list) 

    if args.base:
        base = args.base

    base_fit_result = os.path.splitext(os.path.basename(args.fit_result))[0]
    version = str(cfg.get("version", "v0"))
    suffix = "_rotate" if rotated else ""
    if args.outdir is None:
        outdir = os.path.join(user.plot_directory, "unbinned_postfit_fromfit", base, f"{version}{suffix}", f"from_{base_fit_result}")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    logger.info("output dir : %s", outdir)
    logger.info("n_toys     : %d", args.n_toys)

    region_canvases: dict[str, ROOT.TCanvas] = {}

    for region in like_info.get("regions", []) or []:
        region_id = region["id"]
        logger.info("plotting region %s", region_id)

        class_infos = []
        for cls in region.get("classes", []) or []:
            class_id = cls["id"]
            sample_name = cls.get("sample", class_id)
            loader = factory.get(sample_name)
            color = get_color(sample_name) if callable(get_color) else ROOT.kGray + 1

            class_infos.append(
                {
                    "id": class_id,
                    "sample": sample_name,
                    "cls": cls,
                    "loader": loader,
                    "color": color,
                }
            )

        if not class_infos:
            logger.warning("Region %s has no classes, skipping.", region_id)
            continue

        if (not is_data_fit) and ("data" in region):
            logger.warning(
                "data sample in region %s, but fit result from MC-only fit. Plotting data as total MC.",
                region_id,
            )

        if is_data_fit and "data" in region:
            data_loader = factory.get(region["data"]["sample"])
        else:
            data_loader = None
            if is_data_fit:
                raise RuntimeError(f"Fit output looks like data fit, but region '{region_id}' has no data sample")

        if is_data_fit:
            assert data_loader is not None
            data_shards_f = []
            for shard_index in range(len(data_loader)):
                (shard_features,) = data_loader.materialize(shard=shard_index, what="f", feature_names=default_features)
                shard_features = np.asarray(shard_features, dtype=np.float64)
                if shard_features.size == 0:
                    continue
                data_shards_f.append(shard_features)
        else:
            data_shards_f = []

        for class_info in tqdm(class_infos, desc=f"[{region_id}] materialize classes", leave=False):
            loader = class_info["loader"]
            shards_fw = []
            shard_caches = []
            for shard_index in range(len(loader)):
                shard_features, shard_weights = loader.materialize(shard=shard_index, what="fw", feature_names=default_features)
                shard_features = np.asarray(shard_features, dtype=np.float64)
                shard_weights = np.asarray(shard_weights, dtype=np.float64)
                if shard_features.size == 0:
                    continue
                shards_fw.append((shard_features, shard_weights))
                shard_caches.append(
                    build_class_shard_cache(
                        class_info["cls"],
                        shard_features,
                        shard_weights,
                        feature_to_index,
                    )
                )
            class_info["shards_fw"] = shards_fw
            class_info["shard_caches"] = shard_caches

        for feature_name in plot_features:
            feature_index = feature_to_index[feature_name]
            if feature_name not in plot_options:
                raise RuntimeError(f"Feature '{feature_name}' is missing in data.plot_options")
            if "binning" not in plot_options[feature_name]:
                raise RuntimeError(f"Feature '{feature_name}' has no 'binning' entry in data.plot_options")
            binning = plot_options[feature_name]["binning"]
            if isinstance(binning, (list, tuple)) and len(binning) == 3 and isinstance(binning[0], int):
                n_bins_cfg, x_low, x_high = binning
                edges = np.linspace(float(x_low), float(x_high), int(n_bins_cfg) + 1, dtype=np.float64)
            else:
                edges = np.asarray(list(binning), dtype=np.float64)
            x_title = plot_options.get(feature_name, {}).get("tex", feature_name)
            log_y = bool(plot_options.get(feature_name, {}).get("logY", False))
            n_bins = len(edges) - 1

            logger.info(f"{feature_name=}")
            total_central = np.zeros(n_bins, dtype=np.float64)
            class_central_hists = []
            class_labels = []

            logging.info("evaluating nominal prediction")
            for class_info in class_infos:
                class_hist = np.zeros(n_bins, dtype=np.float64)

                for shard_cache in class_info["shard_caches"]:
                    if args.prefit:
                        event_weights = evaluate_class_cached(shard_cache, hyp_for_fit)
                    else:
                        event_weights = evaluate_class_cached(shard_cache, best_base_hyp)
                    shard_features = shard_cache["features"]
                    class_hist += _hist_with_flow(shard_features[:, feature_index], event_weights, edges)

                total_central += class_hist
                class_central_hists.append(class_hist)
                class_labels.append(get_sample_legend(class_info["sample"]))

            logging.info(f"building histograms for sampled toy points, {len(theta_samples)=}")
            if len(theta_samples) > 0:
                total_samples = np.zeros((len(theta_samples), n_bins), dtype=np.float64)

                for itoy, theta in enumerate(
                    tqdm(theta_samples, desc=f"[{region_id}/{feature_name}] toys", leave=False)
                ):
                    toy_pars = {name: float(theta[i]) for i, name in enumerate(active_names)}
                    toy_hyp = hyp_for_fit.cloneModify(**toy_pars)
                    toy_base = toy_hyp._base

                    toy_total = np.zeros(n_bins, dtype=np.float64)
                    for class_info in class_infos:
                        class_hist_toy = np.zeros(n_bins, dtype=np.float64)

                        for shard_cache in class_info["shard_caches"]:
                            event_weights_toy = evaluate_class_cached(shard_cache, toy_base)
                            shard_features = shard_cache["features"]
                            class_hist_toy += _hist_with_flow(shard_features[:, feature_index], event_weights_toy, edges)

                        toy_total += class_hist_toy

                    total_samples[itoy, :] = toy_total

                q_low = np.quantile(total_samples, 0.16, axis=0)
                q_high = np.quantile(total_samples, 0.84, axis=0)
            else:
                q_low = total_central.copy()
                q_high = total_central.copy()

            class_hists = []
            for class_info, values in zip(class_infos, class_central_hists):
                h_cls = ROOT.TH1F(
                    f"h_postfit_{_safe_name(region_id)}_{_safe_name(feature_name)}_{_safe_name(class_info['id'])}",
                    "",
                    len(edges) - 1,
                    array("d", edges),
                )
                h_cls.SetDirectory(0)
                for ibin, value in enumerate(values, start=1):
                    h_cls.SetBinContent(ibin, float(value))
                h_cls.SetLineColor(ROOT.kBlack)
                h_cls.SetFillColor(class_info["color"])
                h_cls.SetLineWidth(1)
                class_hists.append(h_cls)

            h_total = ROOT.TH1F(
                f"h_postfit_total_{_safe_name(region_id)}_{_safe_name(feature_name)}",
                "",
                len(edges) - 1,
                array("d", edges),
            )
            h_total.SetDirectory(0)
            for ibin, value in enumerate(total_central, start=1):
                h_total.SetBinContent(ibin, float(value))

            h_unc = h_total.Clone(f"h_postfit_unc_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_unc.SetDirectory(0)
            for ibin, (nominal, low, high) in enumerate(zip(total_central, q_low, q_high), start=1):
                err = max(abs(nominal - low), abs(high - nominal))
                h_unc.SetBinError(ibin, float(err))
            h_unc.SetFillColor(ROOT.kGray + 1)
            h_unc.SetFillStyle(3345)
            h_unc.SetLineWidth(0)
            h_unc.SetMarkerSize(0)

            h_unc_up = h_total.Clone(f"h_postfit_unc_up_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_unc_down = h_total.Clone(f"h_postfit_unc_down_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_unc_up.SetDirectory(0)
            h_unc_down.SetDirectory(0)
            for ibin in range(1, n_bins + 1):
                nominal = h_total.GetBinContent(ibin)
                err = h_unc.GetBinError(ibin)
                h_unc_up.SetBinContent(ibin, nominal + err)
                h_unc_down.SetBinContent(ibin, max(0.0, nominal - err))
                h_unc_up.SetBinError(ibin, 0.0)
                h_unc_down.SetBinError(ibin, 0.0)
            h_unc_up.SetLineColor(ROOT.kGray + 2)
            h_unc_down.SetLineColor(ROOT.kGray + 2)
            h_unc_up.SetLineWidth(1)
            h_unc_down.SetLineWidth(1)
            h_unc_up.SetFillStyle(0)
            h_unc_down.SetFillStyle(0)

            h_data = h_total.Clone(f"h_postfit_data_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_data.SetDirectory(0)
            if is_data_fit:
                for ibin in range(1, n_bins + 1):
                    h_data.SetBinContent(ibin, 0.0)
                for shard_features in data_shards_f:
                    unit_weights = np.ones(shard_features.shape[0], dtype=np.float64)
                    data_counts = _hist_with_flow(shard_features[:, feature_index], unit_weights, edges)
                    for ibin, value in enumerate(data_counts, start=1):
                        h_data.SetBinContent(ibin, h_data.GetBinContent(ibin) + float(value))
            for ibin in range(1, n_bins + 1):
                y_val = h_data.GetBinContent(ibin)
                h_data.SetBinError(ibin, np.sqrt(max(0.0, y_val)))
            h_data.SetMarkerStyle(ROOT.kFullCircle)
            h_data.SetMarkerSize(1.0)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetFillStyle(0)

            h_ratio_data = h_data.Clone(f"h_postfit_ratio_data_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_ratio_data.SetDirectory(0)

            max_dev_data = 0.0
            for ibin in range(1, n_bins + 1):
                nominal = h_total.GetBinContent(ibin)
                y_val = h_data.GetBinContent(ibin)
                y_err = h_data.GetBinError(ibin)
                if nominal > 0.0:
                    rel = max(abs(y_val + y_err - nominal), abs(y_val - y_err - nominal)) / nominal
                    h_ratio_data.SetBinContent(ibin, y_val / nominal)
                    h_ratio_data.SetBinError(ibin, y_err / nominal)
                    max_dev_data = max(max_dev_data, rel)
                else:
                    h_ratio_data.SetBinContent(ibin, 0.0)
                    h_ratio_data.SetBinError(ibin, 0.0)

            h_ratio_data.SetMarkerStyle(ROOT.kFullCircle)
            h_ratio_data.SetMarkerSize(1.0)
            h_ratio_data.SetLineColor(ROOT.kBlack)
            h_ratio_data.SetMarkerColor(ROOT.kBlack)
            h_ratio_data.SetLineWidth(1)
            h_ratio_data.SetFillStyle(0)

            integrals = [h.Integral(1, n_bins) for h in class_hists]
            order = sorted(range(len(class_hists)), key=lambda idx: integrals[idx])
            class_hists_sorted = [class_hists[idx] for idx in order]
            class_labels_sorted = [class_labels[idx] for idx in order]

            hs = ROOT.THStack(f"stack_postfit_{_safe_name(region_id)}_{_safe_name(feature_name)}", "")
            for h_cls in class_hists_sorted:
                hs.Add(h_cls, "hist")

            canvas_name = f"c_postfit_{_safe_name(region_id)}_{_safe_name(feature_name)}"
            c = ROOT.TCanvas(canvas_name, canvas_name, 900, 900)
            pad_top = ROOT.TPad(c.GetName() + "_top", c.GetName() + "_top", 0.0, 0.30, 1.0, 1.0)
            pad_bottom = ROOT.TPad(c.GetName() + "_bottom", c.GetName() + "_bottom", 0.0, 0.00, 1.0, 0.30)

            pad_top.SetBottomMargin(0.0)
            pad_top.SetTopMargin(0.08)
            pad_top.SetLeftMargin(0.10)
            pad_top.SetRightMargin(0.05)
            pad_top.SetTicks(1, 1)

            pad_bottom.SetTopMargin(0.0)
            pad_bottom.SetBottomMargin(0.30)
            pad_bottom.SetLeftMargin(0.10)
            pad_bottom.SetRightMargin(0.05)
            pad_bottom.SetTicks(1, 1)

            pad_top.Draw()
            pad_bottom.Draw()

            pad_top.cd()
            if log_y:
                pad_top.SetLogy(True)

            hs.Draw("HIST")
            hs.GetYaxis().SetTitle("Events")
            hs.GetYaxis().SetTitleSize(0.05)
            hs.GetYaxis().SetTitleOffset(1.1)
            hs.GetYaxis().SetLabelSize(0.045)
            hs.GetXaxis().SetLabelSize(0)
            hs.GetXaxis().SetTitleSize(0)

            max_y = max(hs.GetMaximum(), h_data.GetMaximum())
            if log_y:
                hs.SetMinimum(0.5)
                hs.SetMaximum(10.0 * max_y if max_y > 0 else 1.0)
            else:
                hs.SetMinimum(0.0)
                hs.SetMaximum(1.5 * max_y if max_y > 0 else 1.0)

            h_unc.Draw("E2 SAME")
            h_unc_up.Draw("HIST SAME")
            h_unc_down.Draw("HIST SAME")
            h_data.Draw("E SAME")

            leg = ROOT.TLegend(0.50, 0.60, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetNColumns(2)
            if is_data_fit:
                leg.AddEntry(h_data, "Data", "lep")
            else:
                leg.AddEntry(h_data, "Data (Asimov)", "lep")
            for h_cls, label_text in zip(class_hists_sorted, class_labels_sorted):
                leg.AddEntry(h_cls, label_text, "f")
            leg.AddEntry(h_unc, "Uncertainty", "f")
            leg.Draw()

            title = ROOT.TLatex()
            title.SetNDC(True)
            title.SetTextSize(0.035)
            title.DrawLatex(0.12, 0.93, f"{region_id} - {feature_name}")

            pad_bottom.cd()

            h_ratio = h_total.Clone(f"h_postfit_ratio_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_ratio.SetDirectory(0)
            h_ratio.Divide(h_total)
            h_ratio.SetLineColor(ROOT.kBlack)
            h_ratio.SetLineWidth(2)
            h_ratio.SetTitle("")
            h_ratio.GetYaxis().SetTitle("data / MC" if is_data_fit else "var / nominal")
            h_ratio.GetYaxis().SetNdivisions(505)
            h_ratio.GetYaxis().SetTitleSize(0.09)
            h_ratio.GetYaxis().SetTitleOffset(0.5)
            h_ratio.GetYaxis().SetLabelSize(0.08)
            h_ratio.GetXaxis().SetTitle(x_title)
            h_ratio.GetXaxis().SetTitleSize(0.10)
            h_ratio.GetXaxis().SetLabelSize(0.08)

            ratio_boxes = []
            h_ratio_up = h_ratio.Clone(f"h_postfit_ratio_up_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_ratio_down = h_ratio.Clone(f"h_postfit_ratio_down_{_safe_name(region_id)}_{_safe_name(feature_name)}")
            h_ratio_up.SetDirectory(0)
            h_ratio_down.SetDirectory(0)
            h_ratio_up.SetFillStyle(0)
            h_ratio_down.SetFillStyle(0)
            h_ratio_up.SetLineColor(ROOT.kGray + 2)
            h_ratio_down.SetLineColor(ROOT.kGray + 2)
            h_ratio_up.SetLineWidth(1)
            h_ratio_down.SetLineWidth(1)

            max_dev = 0.0
            for ibin in range(1, n_bins + 1):
                x1 = float(edges[ibin - 1])
                x2 = float(edges[ibin])
                nominal = h_total.GetBinContent(ibin)
                err = h_unc.GetBinError(ibin)
                if nominal > 0.0:
                    rel = err / nominal
                    y_low = 1.0 - rel
                    y_high = 1.0 + rel

                    box = ROOT.TBox(x1, y_low, x2, y_high)
                    box.SetFillColor(ROOT.kGray + 1)
                    box.SetFillStyle(3345)
                    box.SetLineWidth(0)
                    ratio_boxes.append(box)

                    h_ratio_up.SetBinContent(ibin, y_high)
                    h_ratio_down.SetBinContent(ibin, y_low)
                    max_dev = max(max_dev, rel)
                else:
                    h_ratio_up.SetBinContent(ibin, 1.0)
                    h_ratio_down.SetBinContent(ibin, 1.0)

            if is_data_fit:
                max_dev = max(max_dev, max_dev_data)

            if max_dev <= 0.0:
                r_min, r_max = 0.9, 1.1
            else:
                half_range = 1.3 * max_dev
                r_min = 1.0 - half_range
                r_max = 1.0 + half_range

            if args.min_ratio:
                h_ratio.SetMinimum(args.min_ratio)
            else:
                h_ratio.SetMinimum(r_min)

            if args.max_ratio:
                h_ratio.SetMaximum(args.max_ratio)
            else:
                h_ratio.SetMaximum(r_max)
                
            h_ratio.Draw("HIST")
            for box in ratio_boxes:
                box.Draw("SAME")
            h_ratio_up.Draw("HIST SAME")
            h_ratio_down.Draw("HIST SAME")
            h_ratio_data.Draw("E1 X0 SAME")

            unity = ROOT.TLine(float(edges[0]), 1.0, float(edges[-1]), 1.0)
            unity.SetLineStyle(ROOT.kDashed)
            unity.SetLineColor(ROOT.kBlack)
            unity.Draw("SAME")

            c.cd()
            c.Update()

            plot_dir = os.path.join(outdir, f"{_safe_name(region_id)}/")
            helpers.copyIndexPHP(plot_dir) 
            out_name = f"{plot_dir}/{_safe_name(feature_name)}"

            if args.prefit:
                out_name += f"__prefit"
            else:
                out_name += f"__postfit"

            c.SaveAs(f"{out_name}.png")
            c.SaveAs(f"{out_name}.pdf")

            region_canvases[f"{region_id}::{feature_name}"] = c
            _KEEPALIVE.extend(
                [
                    c,
                    pad_top,
                    pad_bottom,
                    hs,
                    h_total,
                    h_unc,
                    h_unc_up,
                    h_unc_down,
                    h_data,
                    h_ratio,
                    h_ratio_data,
                    h_ratio_up,
                    h_ratio_down,
                    leg,
                    title,
                    unity,
                ]
                + class_hists_sorted
                + ratio_boxes
            )

        syncer.sync()
    logger.info("done, produced %d plots", len(region_canvases))
