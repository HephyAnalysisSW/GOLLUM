"""Compare data and MC and derive a reweighting for Drell-Yan."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import common.user as user
from common.helpers import copyIndexPHP
from common.plot_helpers import edges_from_binning
from data.plot_options import plot_options
from data.samples_RunII import BASE_DIRECTORY, Factory, process_labels
import common.syncer as syncer

logger = logging.getLogger(__name__)

hep.style.use("CMS")


DEFAULT_FEATURES = ["ht", "dilep_mass", "dilep_pt", "lep0_pt", "lep1_pt", "nSelJet", "nBJet"]
DEFAULT_PROCESSES = ["TTLep_pow", "SingleTop", "TTSemi_pow", "DrellYan_LO_HTbinned"]
DEFAULT_ERAS = ["2016", "2016APV", "2017", "2018"]
DEFAULT_REWEIGHT_FEATURE = "ht"

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the HT reweighting workflow."""
    parser = argparse.ArgumentParser(description="Compare data and MC and derive a reweighting of the Drell-Yan sample.")
    parser.add_argument("--eras", nargs="+", default=["RunII"], choices=DEFAULT_ERAS)
    parser.add_argument("--processes", nargs="+", default=DEFAULT_PROCESSES)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--selection")
    parser.add_argument(
        "--selection_branches",
        nargs="+",
        default=None,
        help="Branches needed by --selection. Required when --selection is used.",
    )
    parser.add_argument(
        "--derive_reweighting",
        action="store_true",
        help="Derive a normalized reweighting curve for the Drell-Yan sample.",
    )
    parser.add_argument(
        "--reweight_feature",
        default=DEFAULT_REWEIGHT_FEATURE,
        help="Feature used to derive the reweighting curve.",
    )
    parser.add_argument(
        "--output_tag",
        default=None,
        help="Optional tag appended to the output directory name.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """Create a filesystem-friendly tag from free-form text."""
    text = text.strip()
    text = text.replace(">=","GEQ")
    text = text.replace("<=","LEQ")
    text = text.replace("==","")
    text = text.replace("<","LT")
    text = text.replace(">","GT")
    if not text:
        return "default"
    
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "default"


def hist_with_flow(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill a 1D histogram and fold under/overflow into the edge bins."""
    nbins = len(edges) - 1
    indices = np.searchsorted(edges, values, side="right") - 1

    in_range = (indices >= 0) & (indices < nbins)
    sumw = np.bincount(indices[in_range], weights=weights[in_range], minlength=nbins).astype(np.float64)
    sumw2 = np.bincount(indices[in_range], weights=weights[in_range] ** 2, minlength=nbins).astype(np.float64)

    under = indices < 0
    if np.any(under):
        under_weights = weights[under]
        sumw[0] += np.sum(under_weights)
        sumw2[0] += np.sum(under_weights * under_weights)

    over = indices >= nbins
    if np.any(over):
        over_weights = weights[over]
        sumw[-1] += np.sum(over_weights)
        sumw2[-1] += np.sum(over_weights * over_weights)

    return sumw, sumw2


def normalize_histogram(sumw: np.ndarray) -> np.ndarray:
    """Return a unit-area histogram, guarding against empty input."""
    total = float(np.sum(sumw))
    if total <= 0.0:
        return np.zeros_like(sumw, dtype=np.float64)
    return np.asarray(sumw, dtype=np.float64) / total


def sanitize_selection_branches(selection: str | None, selection_branches: list[str] | None) -> list[str] | None:
    """Require explicit selection branches when a custom selection is provided."""
    if selection and not selection_branches:
        raise RuntimeError("Provide --selection-branches when using --selection so the loader can request the right branches.")
    return selection_branches


def configure_loader(loader, features: list[str], selection: str | None, selection_branches: list[str] | None):
    """Clear the built-in selection and configure the loader for this script."""
    # Clear any selections that the loader came with (must be done before
    # materialization). We do this to ensure the CLI-provided `--selection`
    # fully replaces defaults.

    branches = list(features)
    if selection and selection_branches:
        branches += list(selection_branches)

    ordered_branches = list(dict.fromkeys(branches))
    loader.setFeatures(feature_names=ordered_branches)

    # if selections not given, uses the default selection
    if selection:
        loader.clearSelections()
        loader.addSelection(selection, required_branches=selection_branches)

    return loader


def load_histogram(loader, feature: str, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate a weighted histogram over all shards of a loader."""
    sumw = np.zeros(len(edges) - 1, dtype=np.float64)
    sumw2 = np.zeros(len(edges) - 1, dtype=np.float64)

    for shard in range(len(loader)):
        values = loader.features(shard=shard, feature_names=[feature])[:, 0].astype(np.float64, copy=False)
        weights = loader.weight_vector(shard=shard).astype(np.float64, copy=False)
        mask = np.isfinite(values) & np.isfinite(weights)
        shard_sumw, shard_sumw2 = hist_with_flow(values[mask], weights[mask], edges)
        sumw += shard_sumw
        sumw2 += shard_sumw2

    return sumw, sumw2


def plot_feature(output_path: str, feature: str, edges: np.ndarray, data_sumw: np.ndarray, data_sumw2: np.ndarray,
                 mc_sumw_by_process: dict[str, np.ndarray], mc_sumw2_by_process: dict[str, np.ndarray],
                 logy: bool) -> None:
    """Draw a data/MC comparison for one feature."""
    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    yerr_data = np.sqrt(np.maximum(data_sumw2, 0.0))

    stack_order = list(mc_sumw_by_process.keys())
    stack_order.sort(key=lambda proc: float(np.sum(mc_sumw_by_process[proc])))

    stack_components = []
    stack_labels = []
    stack_colours = []
    for process in stack_order:
        stack_components.append(mc_sumw_by_process[process])
        process_label = process_labels.get(process, process)
        if "#" in process_label:
            process_label = "$" + process_label.replace("#","\\") + "$"
        stack_labels.append(process_label)
        from data.colors import get_color_mpl
        stack_colours.append(get_color_mpl(process))

    ax_top.hist(
        [centers] * len(stack_components),
        bins=edges,
        weights=stack_components,
        histtype="barstacked",
        label=stack_labels,
        color=stack_colours,
    )

    total_mc = np.sum(np.vstack(list(mc_sumw_by_process.values())), axis=0) if mc_sumw_by_process else np.zeros_like(data_sumw)
    total_mc_err = np.sqrt(np.sum(np.vstack(list(mc_sumw2_by_process.values())), axis=0)) if mc_sumw2_by_process else np.zeros_like(data_sumw)

    ax_top.stairs(total_mc, edges, color="black", linewidth=0.8)

    ax_top.fill_between(
        edges[:-1],
        total_mc - total_mc_err,
        total_mc + total_mc_err,
        step="post",
        color="0.5",
        alpha=0.25,
        label="MC stat. unc.",
    )

    ax_top.errorbar(
        centers,
        data_sumw,
        yerr=yerr_data,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
        markersize=4.5,
        linewidth=1.0,
        label="Data",
    )

    if logy:
        ax_top.set_yscale("log")

    ax_top.set_ylabel("Events")
    ax_top.legend(frameon=False, ncol=2, fontsize=10)
    ax_top.grid(False)

    ratio = np.divide(data_sumw, total_mc, out=np.zeros_like(data_sumw), where=total_mc > 0)
    ratio_err = np.divide(yerr_data, total_mc, out=np.zeros_like(yerr_data), where=total_mc > 0)

    ax_ratio.axhline(1.0, color="0.2", linewidth=1.0)
    ax_ratio.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        xerr=0.5 * widths,
        fmt="o",
        color="black",
        markersize=4.0,
        linewidth=1.0,
    )

    max_dev = np.max(np.abs(ratio - 1.0))
    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    ax_ratio.set_ylim(r_min, r_max)
    ax_ratio.set_ylabel("Data/MC")
    xlabel = "$" + plot_options[feature]["tex"].replace("#","\\") + "$"
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.grid(False)

    hep.cms.label("Internal", data=True, ax=ax_top, loc=0)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def plot_reweighting(output_path: str, feature: str, edges: np.ndarray, data_norm: np.ndarray, mc_norm: np.ndarray,
                     dy_norm: np.ndarray, weights: np.ndarray) -> None:
    """Draw the derived reweighting curve and the weighted comparison."""
    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio = np.divide(data_norm, mc_norm, out=np.zeros_like(data_norm), where=mc_norm > 0)

    ax_top.step(centers, ratio, where="mid", color="black", linewidth=2.0, label="Raw data/MC shape ratio")
    ax_top.step(centers, weights, where="mid", color="#d73027", linewidth=2.0, label="Normalized DY weight")
    ax_top.set_ylabel("Weight")
    ax_top.legend(frameon=False)
    ax_top.grid(False)

    weighted_dy = dy_norm * weights
    weighted_mc = (mc_norm - dy_norm) + weighted_dy
    weighted_ratio = np.divide(data_norm, weighted_mc, out=np.zeros_like(data_norm), where=weighted_mc > 0)

    ax_ratio.axhline(1.0, color="0.2", linewidth=1.0)
    ax_ratio.step(centers, weighted_ratio, where="mid", color="black", linewidth=2.0)
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.set_ylabel("Data/MC")
    xlabel = "$" + plot_options[feature]["tex"].replace("#","\\") + "$"
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.grid(False)

    hep.cms.label("Internal", data=True, ax=ax_top, loc=0)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the data/MC comparison and optional reweighting derivation."""
    args = parse_args()
    selection_branches = sanitize_selection_branches(args.selection, args.selection_branches)

    features = list(dict.fromkeys(list(args.features) + [args.reweight_feature]))
    for feature in features:
        if feature not in plot_options:
            raise RuntimeError(f"Feature {feature!r} is not defined in data.plot_options.")

    plot_root = Path(user.plot_directory) / "DY_reweighting" 
    output_root = Path(user.output_directory) / "DY_reweighting" 
    output_tag = (args.output_tag or slugify(args.selection or "default_selection"))+"_"+"_".join(args.eras)
    plot_dir = plot_root / output_tag
    output_dir = output_root / output_tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    copyIndexPHP(str(plot_dir))

    dy_samples = [process for process in args.processes if (("DrellYan" in process) or ("DY" in process))]
    if len(dy_samples) == 0:
        raise ValueError("You don't have a Drell Yan sample in your process list")
    
    if len(dy_samples) > 1:
        raise ValueError("You have more than one Drell Yan sample in your process list. If using samples with multiple slices, define first a group in data.samples_RunII-")
    
    dy_process = dy_samples[0]

    factory = Factory(BASE_DIRECTORY=str(BASE_DIRECTORY))

    data_loaders = {}
    mc_loaders = {}

    logger.info("Configuring loaders.")
    for era in args.eras:
        data_loader = configure_loader(factory.get(f"Data_{era}"), features, args.selection, selection_branches)
        data_loaders[era] = data_loader

        for process in args.processes:
            loader = configure_loader(factory.get(f"{process}_{era}"), features, args.selection, selection_branches)
            mc_loaders[(era, process)] = loader

    feature_edges = {feature: edges_from_binning(plot_options[feature]["binning"]) for feature in features}
    feature_logy = {feature: bool(plot_options[feature].get("logY", False)) for feature in features}

    logger.info("Looping over features for plotting")
    for feature in features:
        edges = feature_edges[feature]

        data_sumw = np.zeros(len(edges) - 1, dtype=np.float64)
        data_sumw2 = np.zeros(len(edges) - 1, dtype=np.float64)
        mc_sumw_by_process = {process: np.zeros(len(edges) - 1, dtype=np.float64) for process in args.processes}
        mc_sumw2_by_process = {process: np.zeros(len(edges) - 1, dtype=np.float64) for process in args.processes}

        for era in args.eras:
            era_data_sumw, era_data_sumw2 = load_histogram(data_loaders[era], feature, edges)
            data_sumw += era_data_sumw
            data_sumw2 += era_data_sumw2

            for process in args.processes:
                era_mc_sumw, era_mc_sumw2 = load_histogram(mc_loaders[(era, process)], feature, edges)
                mc_sumw_by_process[process] += era_mc_sumw
                mc_sumw2_by_process[process] += era_mc_sumw2

        plot_path = plot_dir / f"{feature}.pdf"
        plot_feature(
            str(plot_path),
            feature,
            edges,
            data_sumw,
            data_sumw2,
            mc_sumw_by_process,
            mc_sumw2_by_process,
            feature_logy[feature],
        )

        if feature == args.reweight_feature and args.derive_reweighting:
            data_norm = normalize_histogram(data_sumw)
            mc_total_sumw = np.sum(np.vstack(list(mc_sumw_by_process.values())), axis=0)
            mc_norm = normalize_histogram(mc_total_sumw)

            dy_norm = normalize_histogram(mc_sumw_by_process[dy_process])
            ratio = np.divide(data_norm, mc_norm, out=np.zeros_like(data_norm), where=mc_norm > 0)

            if np.any((mc_norm > 0) & ~np.isfinite(ratio)):
                raise RuntimeError("Derived ratio contains non-finite values.")

            raw_yield = float(np.sum(dy_norm * ratio))
            if raw_yield <= 0.0:
                raise RuntimeError("Could not normalize the derived DY weights because the raw yield is non-positive.")
            weights = ratio / raw_yield

            weighted_path = plot_dir / f"{feature}_reweighted.pdf"
            plot_reweighting(str(weighted_path), feature, edges, data_norm, mc_norm, dy_norm, weights)

            np.savez(
                output_dir / f"{feature}_reweighting.npz",
                feature=feature,
                selection=args.selection,
                selection_branches=np.array(selection_branches or [], dtype=object),
                edges=edges,
                data_normalized=data_norm,
                mc_normalized=mc_norm,
                dy_normalized=dy_norm,
                raw_ratio=ratio,
                weights=weights,
            )

            with open(output_dir / f"{feature}_reweighting.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "feature": feature,
                        "selection": args.selection,
                        "selection_branches": selection_branches or [],
                        "edges": edges.tolist(),
                        "weights": weights.tolist(),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )

    logger.info(f"Wrote plots and derived weights to: {plot_dir}")


if __name__ == "__main__":
    main()
    syncer.sync()
