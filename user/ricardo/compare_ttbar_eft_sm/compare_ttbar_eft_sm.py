"""Compare kinematic distributions between the nominal ttbar sample and the EFT sample at the SM point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import common.user as user
from common.helpers import copyIndexPHP
import common.syncer as syncer
from data.plot_options import plot_options
from data.samples_RunII import Factory 
import data.samples_eft as samples_eft

logger = logging.getLogger(__name__)

hep.style.use("CMS")


from data.observables import TOP_KINEMATICS, LEPTON_KINEMATICS, ASYMMETRY, BASIC_EVENT, SPIN_CORRELATION
DEFAULT_FEATURES = TOP_KINEMATICS + LEPTON_KINEMATICS + ASYMMETRY + BASIC_EVENT + SPIN_CORRELATION
DEFAULT_ERAS = ["2016", "2016APV", "2018", "RunII"] # 2017 not included as we don't yet have the ntuples for that


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the nominal-ttbar-vs-EFT-SM comparison."""
    parser = argparse.ArgumentParser(
        description="Compare kinematic distributions between the nominal ttbar sample and the EFT sample at the SM point."
    )
    parser.add_argument("--era", default="2018", choices=DEFAULT_ERAS)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, choices=DEFAULT_FEATURES)
    parser.add_argument("--output_tag", default=None, help="Optional tag appended to the output directory name.")
    parser.add_argument("--norm", action="store_true", help="Each plot is normalized to unit area.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: single feature, extra logging")
    return parser.parse_args()


def edges_from_binning(binning: list) -> np.ndarray:
    """Convert a plot_options binning specification into bin edges."""
    nbins, x_min, x_max = binning
    return np.linspace(float(x_min), float(x_max), int(nbins) + 1, dtype=np.float64)


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


def normalize_histogram(sumw: np.ndarray, sumw2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit-area histogram and correspondingly rescaled variance, guarding against empty input."""
    total = float(np.sum(sumw))
    if total <= 0.0:
        return np.zeros_like(sumw, dtype=np.float64), np.zeros_like(sumw2, dtype=np.float64)
    return sumw / total, sumw2 / (total * total)


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


def plot_comparison(
    plot_dir: str,
    feature: str,
    edges: np.ndarray,
    nominal_sumw: np.ndarray,
    nominal_sumw2: np.ndarray,
    eft_sumw: np.ndarray,
    eft_sumw2: np.ndarray,
    logy: bool,
    normalize: bool
) -> None:
    """Draw a shape (unit-area) comparison between the nominal ttbar sample and the EFT sample at the SM point."""
    fig, (ax_top, ax_ratio, ax_mc_stat) = plt.subplots(
        3,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0, 1.0]},
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    
    if normalize:
        nominal, nominal_var = normalize_histogram(nominal_sumw, nominal_sumw2)
        eft, eft_var = normalize_histogram(eft_sumw, eft_sumw2)
    else:
        nominal, nominal_var = (nominal_sumw, nominal_sumw2)
        eft, eft_var = (eft_sumw, eft_sumw2)
        
    nominal_err = np.sqrt(np.maximum(nominal_var, 0.0))
    eft_err = np.sqrt(np.maximum(eft_var, 0.0))

    ax_top.stairs(nominal, edges, color="black", linewidth=1.4, label=f"Nominal $t\\bar{{t}}, yield: {np.sum(nominal):.0f}$")
    ax_top.fill_between(
        edges[:-1],
        nominal - nominal_err,
        nominal + nominal_err,
        step="post",
        color="0.5",
        alpha=0.25,
    )

    ax_top.stairs(eft, edges, color="#d73027", linewidth=1.4, label=f"EFT (SM point), yield: {np.sum(eft):.0f}")
    ax_top.fill_between(
        edges[:-1],
        eft - eft_err,
        eft + eft_err,
        step="post",
        color="#d73027",
        alpha=0.15,
    )

    if logy:
        ax_top.set_yscale("log")

    if normalize:
        ax_top.set_ylabel("Normalized to unit area")
    else:
        ax_top.set_ylabel("Events")

    ax_top.legend(frameon=False, fontsize=10)
    ax_top.grid(False)

    ratio = np.divide(nominal, eft, out=np.zeros_like(nominal), where=eft > 0)
    ratio_err = np.divide(nominal_err, eft, out=np.zeros_like(nominal_err), where=eft > 0)

    ax_ratio.axhline(1.0, color="0.2", linewidth=1.0)
    ax_ratio.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        fmt="o",
        color="black",
        markersize=4.0,
        linewidth=1.0,
    )

    finite_ratio = ratio[np.isfinite(ratio) & (eft > 0)]
    max_dev = np.max(np.abs(finite_ratio - 1.0)) if finite_ratio.size else 0.0
    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    ax_ratio.set_ylim(r_min, r_max)
    ax_ratio.set_ylabel("Nominal/EFT", fontsize=8)
    ax_ratio.grid(False)

    # MC stat uncertainty ratio subplot
    rel_mc_stat_nominal = np.divide(nominal_err,nominal, out=np.zeros_like(nominal_err), where=(nominal > 0))
    rel_mc_stat_eft = np.divide(eft_err,eft, out=np.zeros_like(eft_err), where=(eft > 0))
    ax_mc_stat.axhline(1.0, color="0.2", linewidth=1.0)
    ax_mc_stat.stairs(rel_mc_stat_nominal, edges, color="black", linewidth=1.4, label="Nominal")
    ax_mc_stat.stairs(rel_mc_stat_eft, edges, color="#d73027", linewidth=1.4, label="EFT (SM point)")

    finite_rel_mc_stat = np.concatenate([rel_mc_stat_nominal[np.isfinite(rel_mc_stat_nominal) & (nominal_err > 0)],
                    rel_mc_stat_eft[np.isfinite(rel_mc_stat_eft) & (eft_err > 0)]]) 
    max_dev = np.max(np.abs(finite_rel_mc_stat)) if finite_rel_mc_stat.size else 0.0

    if max_dev <= 0.0:
        r_max = 1.1
    else:
        r_max = 1.3 * max_dev

    ax_mc_stat.set_ylim(-0.01, r_max)
    ax_mc_stat.set_ylabel("Relative MCStat unc.", fontsize=8)
    ax_mc_stat.grid(False)
    xlabel = "$" + plot_options[feature]["tex"].replace("#", "\\") + "$"
    ax_mc_stat.set_xlabel(xlabel)

    # final settings

    hep.cms.label("Internal", data=False, ax=ax_top, loc=0)
    fig.tight_layout()
    if normalize:
        output_path = plot_dir + f"/norm_{feature}"
    else:
        output_path = plot_dir + f"/{feature}"
    plt.savefig(output_path+".pdf")
    plt.savefig(output_path+".png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Compare kinematic distributions between the nominal ttbar sample and the EFT sample at the SM point."""
    args = parse_args()

    factory = Factory()
    nominal_loader = factory.get(f"TTLep_pow_{args.era}")
    eft_loader = getattr(samples_eft, f"TT01j2l_EFT_{args.era}")

    nominal_loader.setFeatures(DEFAULT_FEATURES)
    eft_loader.setFeatures(DEFAULT_FEATURES)

    if args.debug:
        logger.info("Printing additional debug info.")
        logger.setLevel(logging.DEBUG)

    output_tag = args.output_tag or args.era
    plot_dir = Path(user.plot_directory) / "ttbar_vs_eft_sm" / output_tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    copyIndexPHP(str(plot_dir))

    logger.info("Looping over features for plotting.")
    for ifeature, feature in enumerate(DEFAULT_FEATURES):
        if args.debug and ifeature > 0:
            break
        edges = edges_from_binning(plot_options[feature]["binning"])
        logy = bool(plot_options[feature].get("logY", False))

        try:
            nominal_sumw, nominal_sumw2 = load_histogram(nominal_loader, feature, edges)
            eft_sumw, eft_sumw2 = load_histogram(eft_loader, feature, edges)
        except KeyError:
            continue

        plot_comparison(
            str(plot_dir),
            feature,
            edges,
            nominal_sumw,
            nominal_sumw2,
            eft_sumw,
            eft_sumw2,
            logy,
            normalize=args.norm
        )

    logger.info(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()
    syncer.sync()
