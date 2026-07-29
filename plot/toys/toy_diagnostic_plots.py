"""Diagnostic plots for generated toys: per-feature distributions overlaid across
toy seeds, for each unbinned region, using the config's default_features and
data/plot_options binning.

For cache-mode toys (model is truth), raw kinematics are not part of the toy file
-- only cache row indices are, to avoid duplicating the surrogate cache -- so
features are recovered by re-streaming the region's samples once, at the union of
all requested toys' kept indices (see
fit.ToyGenerator.materialize_cache_region_features). This is costly by design and
meant for occasional diagnostic use, not routine toy generation.

For truth-mode toys, raw kinematics (X) are already part of the toy file, but only
for the region's own surrogate feature union (whatever the classifier/BIT/PNNs
consume) -- default_features not in that union are skipped with a warning.

Entry point: fit/ToyGenerator.py --plot (see its __main__).
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import common.user as user
from data.plot_options import plot_options as DEFAULT_PLOT_OPTS
from fit.ToyGenerator import _find_region, _region_feature_union, materialize_cache_region_features

# Same as data.colors.cmap_petroff10_mpl, inlined to avoid importing that module's
# ROOT dependency here -- ROOT's cling JIT has been observed to segfault when
# initialized a second time in the same process after fit/ToyGenerator.py has
# already streamed samples (RDataLoader/PyROOT), so this purely-matplotlib
# diagnostic script stays ROOT-free.
COLOR_HEX = ['#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
             '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd']

logger = logging.getLogger(__name__)

hep.style.use("CMS")


def _available_features_for_region(n2ll, region_id: str, toys: list, feature_names: list) -> list:
    """Cache-mode toys can recover any raw feature by name; truth-mode toys are
    limited to whatever the region's own surrogates consume."""
    blocks = [t["unbinned_blocks"][region_id] for t in toys]
    if any(b.get("indices") is not None for b in blocks):
        return list(feature_names)
    region = _find_region(n2ll, region_id)
    union = set(_region_feature_union(region))
    available = [f for f in feature_names if f in union]
    skipped = [f for f in feature_names if f not in union]
    if skipped:
        logger.warning(
            "[toy:plot:%s] Skipping feature(s) %s -- not part of this truth-mode toy's "
            "stored kinematics (region surrogates don't consume them).", region_id, skipped,
        )
    return available


def _region_toy_features(n2ll, region_id: str, toys: list, feature_names: list) -> dict:
    """{seed: (X (n_kept, n_features), w (n_kept,))} for one region, X columns
    ordered as `feature_names`."""
    blocks = [(t["seed"], t["unbinned_blocks"][region_id]) for t in toys]
    cache_blocks = [(seed, b) for seed, b in blocks if b.get("indices") is not None]

    cache_X_by_seed = {}
    if cache_blocks:
        union_indices = np.unique(np.concatenate([np.asarray(b["indices"]) for _, b in cache_blocks]))
        union_X = materialize_cache_region_features(n2ll, region_id, union_indices, feature_names)
        pos_of = {int(idx): i for i, idx in enumerate(union_indices)}
        for seed, b in cache_blocks:
            rows = [pos_of[int(i)] for i in b["indices"]]
            cache_X_by_seed[seed] = union_X[rows]

    region = _find_region(n2ll, region_id)
    truth_feature_union = _region_feature_union(region)

    out = {}
    for seed, b in blocks:
        w = np.asarray(b["w"], dtype=np.float64)
        if seed in cache_X_by_seed:
            X = cache_X_by_seed[seed]
        else:
            full_X = np.asarray(b["X"], dtype=np.float64)
            col_positions = [truth_feature_union.index(f) for f in feature_names]
            X = full_X[:, col_positions]
        out[seed] = (X, w)
    return out


def plot_toy_feature_distributions(n2ll, toys: list, feature_names: list, out_dir: str,
                                    plot_opts: dict = DEFAULT_PLOT_OPTS) -> None:
    """One figure per (unbinned region, feature): each requested toy seed overlaid
    as a weighted step histogram, binned per data/plot_options. Binned regions have
    no per-event kinematics and are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    seeds = sorted(t["seed"] for t in toys)
    colors = {seed: COLOR_HEX[i % len(COLOR_HEX)] for i, seed in enumerate(seeds)}

    for region in n2ll.regions:
        rid = region["id"]
        available = _available_features_for_region(n2ll, rid, toys, feature_names)
        if not available:
            continue
        by_seed = _region_toy_features(n2ll, rid, toys, available)

        region_dir = os.path.join(out_dir, rid)
        os.makedirs(region_dir, exist_ok=True)
        for i_feat, feat in enumerate(available):
            if feat not in plot_opts:
                logger.warning("[toy:plot:%s] No plot_options entry for '%s', skipping.", rid, feat)
                continue
            n_bins, x_lo, x_hi = plot_opts[feat]["binning"]
            edges = np.linspace(x_lo, x_hi, int(n_bins) + 1, dtype=np.float64)
            tex = "$" + plot_opts[feat]["tex"].replace("#", "\\") + "$"

            fig, ax = plt.subplots(figsize=(8.0, 6.5))
            for seed, (X, w) in by_seed.items():
                hist, _ = np.histogram(X[:, i_feat], bins=edges, weights=w)
                ax.stairs(hist, edges, color=colors[seed], linewidth=1.6, label=f"toy {seed}", zorder=2)

            if plot_opts[feat].get("logY", False):
                ax.set_yscale("log")
            ax.set_xlabel(tex)
            ax.set_ylabel("Toy events")
            n_col = 1 if len(seeds) <= 8 else 2
            ax.legend(frameon=False, fontsize=8, ncol=n_col, loc="upper right")

            hep.cms.label("Internal", data=False, ax=ax, loc=0)
            fig.tight_layout()
            stub = os.path.join(region_dir, feat)
            plt.savefig(stub + ".png", dpi=200)
            plt.savefig(stub + ".pdf")
            plt.close(fig)

    logger.info("[toy:plot] Wrote diagnostic plots to %s", out_dir)
