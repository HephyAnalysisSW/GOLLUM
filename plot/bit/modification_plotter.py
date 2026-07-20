#!/usr/bin/env python
"""Encapsulated plotting for BIT coefficient / modification plots.

This module holds the (long, backend-agnostic) boilerplate shared by the EFT
and PDF entry scripts: CLI definition, config/job loading, the loader shard
loop, histogram accumulation, optional BIT overlay, and figure drawing.

The derivative-specific bits live in the entry scripts
(``eft_modification_plot.py`` / ``pdf_modification_plot.py``). Each passes in a
lightweight *provider* object exposing:

  - ``combinations``       : list of canonical derivative tuples, native column
                             order, including the nominal ``()``.
  - ``parameters``         : list of coefficient / operator names.
  - ``required_observers`` : observer branch names the provider needs.
  - ``truth_weight_matrix(G, w, observer_names)`` : returns an ``(N, M)`` matrix
                             of truth weights aligned to ``combinations``
                             (column 0 == nominal weight).

For a combination ``der`` at coefficient value ``c`` the bottom panel draws
    contribution(x) = c**len(der) * (sum_bin w_der) / (sum_bin w_SM),
i.e. the ratio-to-SM of that term. At ``c = 1`` this equals the raw coefficient
a BIT learns, so the plot doubles as a BIT closure plot.
"""

from __future__ import annotations

import os
import sys
import argparse
import importlib
import logging
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mplhep as hep
import numpy as np
from tqdm import tqdm

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common.user as user
import common.yaml_loader as yaml_loader
import common.helpers as helpers
from data.RandomSplitter import RandomSplitter
from data.UIDSplitter import UIDSplitter

from data.plot_options import plot_options as DEFAULT_PLOT_OPTS
from common.derivative_providers import canonical_combination

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

hep.style.use("CMS")

# Petroff 6 palette (color-blind friendly), extended.
from data.colors import cmap_petroff10_mpl
COLOR_HEX = cmap_petroff10_mpl


# --------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------

def derivative_label(der) -> str:
    """Human-readable label for a derivative combination (matplotlib mathtext)."""
    if len(der) == 1:
        return f"{der[0]} (lin)"
    if len(der) == 2 and der[0] == der[1]:
        return f"{der[0]}$^2$ (quad)"
    if len(der) == 2:
        return f"{der[0]}$\\times${der[1]} (mixed)"
    return str(der)


# --------------------------------------------------------------------------------
# CLI + config/job loading
# --------------------------------------------------------------------------------

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Shared argument parser for the modification-plot entry scripts."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--job", default=None, help="BIT job id (omit to list bit jobs)")
    p.add_argument("--with-bit", action="store_true", help="Overlay predictions of the trained BIT model")
    p.add_argument("--max-n-tree", type=int, default=None, help="Use up to this many trees for the BIT prediction")
    p.add_argument("--value", type=float, default=1.0, help="Coefficient value c used to scale the modification (default 1.0)")
    p.add_argument("--terms", choices=["linear", "quadratic", "both"], default="both", help="Which term orders to draw")
    p.add_argument("--mixed", action="store_true", help="Also draw mixed cross terms (op0, op1)")
    p.add_argument("--operators", nargs="+", default=None, help="Restrict to these operators (default: all in the provider)")
    p.add_argument("--split", choices=["all", "train", "valid"], default="all", help="Event split to plot (uses job splitting if enabled). Train: events used in training; valid: events not used in training.")
    p.add_argument("--small", type=int, default=None, help="Stop after roughly this many selected events")
    return p


def load_cfg_and_job(args):
    """Load the YAML config and select the requested bit job.

    Returns ``(cfg, job, samples_mod, module_samples)``. Lists jobs and exits if
    ``--job`` is omitted.
    """
    config_path = os.path.expanduser(os.path.expandvars(args.config))
    cfg = yaml_loader.load_yaml(config_path)
    defaults = cfg.get("defaults", {}) or {}
    module_samples = defaults.get("module_samples", "data.samples")

    if args.job is None:
        _list_jobs_and_exit(cfg, args.config)

    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
    if job is None or job.get("type") != "bit":
        raise RuntimeError(f"Job '{args.job}' not found or not of type 'bit'.")

    samples_mod = importlib.import_module(module_samples)
    return cfg, job, samples_mod, module_samples


def _list_jobs_and_exit(cfg, config_path):
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "bit"]
    if not jobs:
        print("No BIT jobs found.")
        sys.exit(0)
    script = os.path.relpath(sys.argv[0])
    for j in jobs:
        print(f"python {script} {config_path} --job {j['id']}")
    sys.exit(0)


# --------------------------------------------------------------------------------
# BIT overlay
# --------------------------------------------------------------------------------

class _BITPrediction:
    """Wrapper around a trained BIT: predict(X) -> (N, K) aligned to self.derivatives."""

    def __init__(self, model_path, max_n_tree=None):
        from ML.BIT.NumbaBIT import MultiBoostedInformationTree

        self._bit = MultiBoostedInformationTree.load(model_path)
        self.n_trees = len(getattr(self._bit, "trees", []) or [])
        self.max_n_tree = self.n_trees if max_n_tree is None else min(int(max_n_tree), self.n_trees)
        raw = list(getattr(self._bit, "derivatives", []) or [])
        if not raw:
            raise RuntimeError(f"BIT model '{model_path}' has no derivatives; nothing to predict.")
        self.derivatives = [canonical_combination(d) for d in raw]

    def predict(self, X):
        pred = np.asarray(self._bit.predict(X, max_n_tree=self.max_n_tree))
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


# --------------------------------------------------------------------------------
# combination selection
# --------------------------------------------------------------------------------

def _select_combinations(provider, args):
    """Pick which derivative combinations to draw, honoring --terms/--mixed/--operators."""
    param_set = set(args.operators) if args.operators else set(provider.parameters)
    if args.operators:
        unknown = param_set - set(provider.parameters)
        if unknown:
            raise RuntimeError(f"Requested operators not in provider: {sorted(unknown)}")

    selected = []
    for der in provider.combinations:
        if len(der) == 0:
            continue
        if len(der) == 1:
            if args.terms in ("linear", "both") and der[0] in param_set:
                selected.append(der)
        elif len(der) == 2:
            if args.terms not in ("quadratic", "both"):
                continue
            if der[0] == der[1]:
                if der[0] in param_set:
                    selected.append(der)
            elif args.mixed and der[0] in param_set and der[1] in param_set:
                selected.append(der)
    return selected


# --------------------------------------------------------------------------------
# textual analysis output
# --------------------------------------------------------------------------------

def _write_analysis_json(out_dir, plot_feats, selected, truth_num, truth_num_sumw2, sm_hist, feature_edges, value):
    """Write JSON file with bin contents for each feature and derivative combination.

    Each derivative entry is ``{"coeff": [...], "unc": [...]}``: ``coeff`` is the
    usual ratio-to-SM coefficient, ``unc`` is its per-bin MC-statistical
    uncertainty from the numerator alone (``sqrt(sum w_der^2)) / sum(w_SM)``,
    treating the SM denominator as fixed. This is not a full error propagation
    (numerator and denominator are correlated, sharing the same events) but is
    enough to separate "large because of real structure" from "large because
    the per-event derivative weight is noisy" -- see ``analyze_eft_distributions.py``.
    """
    analysis = {}
    for feat in plot_feats:
        analysis[feat] = {}
        sm = sm_hist[feat]
        edges = feature_edges[feat]
        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        analysis[feat]["bin_centers"] = bin_centers.tolist()
        analysis[feat]["sm_histogram"] = sm.tolist()

        for der in selected:
            scale = value ** len(der)
            coeff = np.zeros_like(sm)
            unc = np.zeros_like(sm)
            nz = sm != 0.0
            coeff[nz] = scale * truth_num[feat][der][nz] / sm[nz]
            unc[nz] = scale * np.sqrt(truth_num_sumw2[feat][der][nz]) / sm[nz]

            der_label = derivative_label(der)
            analysis[feat][der_label] = {"coeff": coeff.tolist(), "unc": unc.tolist()}

    # Write JSON
    json_path = os.path.join(out_dir, "bin_contents.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Wrote bin contents to %s", json_path)


# --------------------------------------------------------------------------------
# main entry: accumulate + draw
# --------------------------------------------------------------------------------

def make_modification_plots(cfg, job, samples_mod, args, provider):
    """Draw per-feature SM distribution + per-term modification, using ``provider``.

    Always draws truth (from the provider); overlays BIT predictions if
    ``args.with_bit`` is set. This is the whole shared pipeline; the entry
    scripts only build ``provider``.
    """
    combo_to_col = {der: i for i, der in enumerate(provider.combinations)}
    selected = _select_combinations(provider, args)
    if not selected:
        raise RuntimeError("No derivative combinations selected (check --terms / --operators).")
    logger.info("Selected derivatives: %s", [derivative_label(d) for d in selected])

    # ---- loader ----
    loader_name = job.get("process")
    if not hasattr(samples_mod, loader_name):
        raise RuntimeError(f"Loader/view '{loader_name}' not found in module {samples_mod.__name__}.")
    loader = getattr(samples_mod, loader_name)
    # Shard for bounded memory: one shard per file for the file-splitting loaders
    # (EFT samples), event-based shards otherwise.
    if getattr(loader, "splitting_strategy", None) == "files":
        loader.set_n_split(max(1, len(loader._all_files)))
    else:
        loader.set_n_split(100)

    selection = job.get("selection", None)
    if selection:
        loader.addSelection(selection, job.get("selection_features", []))
        logger.info("Added selection: %s", selection)

    # ---- event split config (checked before setFeatures: UID splitting needs extra observers) ----
    split_cfg = job.get("splitting") or {}
    # plot true deviations on entire dataset 
    split_enabled = bool(split_cfg.get("enabled", False)) and (args.split != "all")
    split_type = split_cfg.get("type", "random")
    uid_fields = tuple(split_cfg.get("uid_fields", ["run", "luminosityBlock", "event"]))

    required_observers = list(provider.required_observers)
    if split_enabled and split_type == "uid":
        required_observers = required_observers + [f for f in uid_fields if f not in required_observers]

    loader.setFeatures(job["features"], observer_names=required_observers)
    feat_names = list(getattr(loader, "feature_names", []) or [])
    obs_names = list(getattr(loader, "observer_names", []) or [])
    if not feat_names:
        raise RuntimeError("Loader has no feature_names.")

    plot_opts = getattr(samples_mod, "plot_options", DEFAULT_PLOT_OPTS)
    plot_feats = [f for f in feat_names if f in plot_opts]
    if not plot_feats:
        raise RuntimeError("No plottable features (none of the job features are in plot_options).")
    logger.info("Plottable features: %s", plot_feats)

    # ---- optional BIT overlay ----
    bit = None
    if args.with_bit:
        cfg_base = os.path.join(cfg.get("version", "default"), job["region"])
        model_dir = os.path.join(user.model_directory, cfg_base, "BIT", job["id"])
        model_path = os.path.join(model_dir, "BIT_best.pkl")
        logger.info("Loading BIT model from %s", model_path)
        bit = _BITPrediction(model_path, max_n_tree=args.max_n_tree)
        logger.info("Loaded BIT: %d trees (using %d)", bit.n_trees, bit.max_n_tree)
        drop = [d for d in selected if d not in set(bit.derivatives)]
        if drop:
            logger.warning("BIT model does not predict %s; drawing truth only for those.",
                           [derivative_label(d) for d in drop])
        
        if not split_enabled or args.split=="all":
            raise RuntimeError("Splitting disabled or checking BIT closure on all the events." \
                                "Enable splitting or run with args.split != 'all' to avoid data leakage.")

        if args.split == "train":
            logger.warning("Checking closure on training dataset, beware not to generalize conclusions." \
                            "To check closure on independent dataset, run with args.split == 'valid'.")

    # ---- event split ----
    splitter = None
    uid_idx = None
    train_interval = val_interval = None
    if split_enabled:
        if split_type == "random":
            splitter = RandomSplitter(
                fraction=float(split_cfg.get("fraction", 0.5)),
                seed=int(split_cfg.get("seed", 0)),
            )
            logger.info("Plotting the '%s' split (random, fraction=%.3f, seed=%d).",
                        args.split, splitter.fraction, splitter.seed)

        elif split_type == "uid":
            uid_seed = int(split_cfg.get("seed", 0))
            uid_n_buckets = int(split_cfg.get("n_buckets", 10000))
            uid_scheme = split_cfg.get("scheme") or {}

            splitter = UIDSplitter(uid_fields=uid_fields, seed=uid_seed, n_buckets=uid_n_buckets)

            # build bucket intervals EXACTLY like PNN / eft_bit_training.py
            keys = list(uid_scheme.keys())
            fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]
            sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
            sizes[-1] += uid_n_buckets - sum(sizes)

            uid_intervals = {}
            lo = 0
            for k, sz in zip(keys, sizes):
                uid_intervals[k] = (lo, lo + int(sz))
                lo += int(sz)
            
            # either plotting data used in training (pnn...) or data not used in training (c2st...)
            # keeping the final_eval partition for downstream testing
            train_interval = (uid_intervals["pnn_train"][0], uid_intervals["pnn_val"][1])
            val_interval = (uid_intervals["c2st_train"][0], uid_intervals["c2st_val"][1])
            uid_idx = [obs_names.index(f) for f in uid_fields]

            logger.info("Plotting the '%s' split (uid, fields=%s, seed=%d, n_buckets=%d).",
                        "pnn (train+val)" if args.split == 'train' else "c2st (train+val)", uid_fields, uid_seed, uid_n_buckets)
            logger.info("UID scheme intervals: %s", uid_intervals)
        else:
            raise RuntimeError(f"Unsupported splitting.type='{split_type}'. Only 'random' and 'uid' are implemented.")
    elif args.split != "all":
        logger.warning("--split %s requested but job has no enabled splitting; plotting all events.", args.split)

    # ---- histogram accumulators ----
    feature_columns = {f: feat_names.index(f) for f in plot_feats}
    feature_edges = {}
    sm_hist = {}
    sm_sumw2 = {}
    truth_num = {}
    truth_num_sumw2 = {}
    pred_num = {}
    for feat in plot_feats:
        n_bins, x_lo, x_hi = plot_opts[feat]["binning"]
        feature_edges[feat] = np.linspace(x_lo, x_hi, int(n_bins) + 1, dtype=np.float64)
        sm_hist[feat] = np.zeros(int(n_bins), dtype=np.float64)
        sm_sumw2[feat] = np.zeros(int(n_bins), dtype=np.float64)
        truth_num[feat] = {der: np.zeros(int(n_bins), dtype=np.float64) for der in selected}
        truth_num_sumw2[feat] = {der: np.zeros(int(n_bins), dtype=np.float64) for der in selected}
        pred_num[feat] = {der: np.zeros(int(n_bins), dtype=np.float64) for der in selected}

    # ---- event loop ----
    selected_events = 0
    for shard in tqdm(range(len(loader)), desc="Shards", unit="shard"):
        X, G, w = loader.materialize(shard=shard, what="fow")
        if len(X) == 0:
            continue

        if splitter is not None:
            if split_type == "random":
                keep = splitter.mask(len(X), shard=shard)
                if args.split == "valid":
                    keep = ~keep
            else:  # uid
                O_uid = G[:, uid_idx]
                lo, hi = train_interval if args.split == "train" else val_interval
                keep = splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)
            X, G, w = X[keep], G[keep], w[keep]
            if len(X) == 0:
                continue

        if args.small is not None:
            remaining = args.small - selected_events
            if remaining <= 0:
                break
            if len(X) > remaining:
                X, G, w = X[:remaining], G[:remaining], w[:remaining]

        X = X.astype(np.float32, copy=False)
        w = w.astype(np.float32, copy=False)

        deriv_w = provider.truth_weight_matrix(G, w, obs_names)
        nominal_w = deriv_w[:, combo_to_col[()]]

        pred = bit.predict(X) if bit is not None else None
        pred_col = (
            {der: bit.derivatives.index(der) for der in selected if der in bit.derivatives}
            if bit is not None else {}
        )

        for feat in plot_feats:
            xvals = X[:, feature_columns[feat]].astype(np.float64, copy=False)
            edges = feature_edges[feat]
            sm_hist[feat] += np.histogram(xvals, bins=edges, weights=nominal_w)[0]
            sm_sumw2[feat] += np.histogram(xvals, bins=edges, weights=nominal_w ** 2)[0]
            for der in selected:
                der_w = deriv_w[:, combo_to_col[der]]
                truth_num[feat][der] += np.histogram(xvals, bins=edges, weights=der_w)[0]
                truth_num_sumw2[feat][der] += np.histogram(xvals, bins=edges, weights=der_w ** 2)[0]
                if der in pred_col:
                    pred_num[feat][der] += np.histogram(
                        xvals, bins=edges, weights=nominal_w * pred[:, pred_col[der]]
                    )[0]

        selected_events += len(X)

    if selected_events == 0:
        raise RuntimeError("No events passed selection / split; nothing to plot.")
    logger.info("Processed %d selected events.", selected_events)

    # ---- output directory ----
    if args.with_bit:
        out_dir = os.path.join(
            user.plot_directory, "BIT-closure",
            cfg.get("version", "default"), job["region"], job["id"],
        )
    else:
        out_dir = os.path.join(
            user.plot_directory, "BIT-modification",
            cfg.get("version", "default"), job["region"], job["id"],
        )

    out_dir = os.path.join(out_dir, f"{args.split}_events")

    if args.operators is None:
        out_dir = os.path.join(out_dir, "all_ops")
    else:
        out_dir = os.path.join(out_dir, "_".join(args.operators))

    if args.terms == "linear":
        out_dir = os.path.join(out_dir, "lin_terms")

    if args.terms == "quadratic":
        out_dir = os.path.join(out_dir, "quad_terms")
        if not args.mixed:
            out_dir += "_no_mix"

    if args.terms == "both":
        if args.mixed:
            out_dir = os.path.join(out_dir, "all_terms")
        else:
            out_dir = os.path.join(out_dir, "linquad_terms_no_mix")

    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    colors = {der: COLOR_HEX[i % len(COLOR_HEX)] for i, der in enumerate(selected)}

    # ---- draw ----
    # Single canvas with a twin y-axis (a la plot/bit/bit_plot.py): the SM
    # distribution is read on the left axis, the derivative coefficients on the
    # right axis. At c = 1 the right-axis curves are the raw derivatives.
    for feat in plot_feats:
        edges = feature_edges[feat]
        sm = sm_hist[feat]
        nz = sm != 0.0
        tex = "$" + plot_opts[feat]["tex"].replace("#", "\\") + "$"

        sm_err = np.sqrt(np.maximum(sm_sumw2[feat], 0.0))

        fig, ax_left = plt.subplots(figsize=(9.0, 7.0))
        ax_right = ax_left.twinx()

        # left axis: SM distribution (filled grey), sitting in the lower portion
        ax_left.stairs(sm, edges, color="0.35", linewidth=1.2, fill=False, zorder=1)
        ax_left.fill_between(edges[:-1], 0.0, sm, step="post", color="0.88", zorder=0)
        # MC statistical uncertainty band on the SM distribution
        ax_left.fill_between(
            edges[:-1],
            np.maximum(sm - sm_err, 0.0),
            sm + sm_err,
            step="post",
            color="0.55",
            alpha=0.45,
            linewidth=0.0,
            zorder=2,
        )
        if plot_opts[feat].get("logY", False):
            ax_left.set_yscale("log")
        else:
            top = float(np.max(sm)) if sm.size else 1.0
            ax_left.set_ylim(0.0, (top if top > 0 else 1.0) * 1.5)
        ax_left.set_ylabel("Number of events (SM)")
        ax_left.set_xlabel(tex)
        ax_left.grid(False)

        # right axis: derivative coefficients (truth solid, BIT dashed)
        ax_right.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
        coeff_values = []
        for der in selected:
            scale = args.value ** len(der)
            coeff_truth = np.zeros_like(sm)
            coeff_truth[nz] = scale * truth_num[feat][der][nz] / sm[nz]
            # per-bin MC statistical uncertainty on the truth coefficient, from the
            # derivative weight's own sumw2 (numerator only, SM denominator treated as
            # fixed -- see _write_analysis_json). Shaded so wildly-fluctuating, poorly
            # constrained phase-space regions are visible at a glance.
            coeff_truth_unc = np.zeros_like(sm)
            coeff_truth_unc[nz] = scale * np.sqrt(truth_num_sumw2[feat][der][nz]) / sm[nz]
            ax_right.fill_between(
                edges[:-1], coeff_truth - coeff_truth_unc, coeff_truth + coeff_truth_unc,
                step="post", color=colors[der], alpha=0.18, linewidth=0.0, zorder=2,
            )
            ax_right.stairs(coeff_truth, edges, color=colors[der], linewidth=2.0, linestyle="--", baseline=None, zorder=3)
            coeff_values.append(coeff_truth[nz] + coeff_truth_unc[nz])
            coeff_values.append(coeff_truth[nz] - coeff_truth_unc[nz])

            if bit is not None and der in bit.derivatives:
                coeff_pred = np.zeros_like(sm)
                coeff_pred[nz] = scale * pred_num[feat][der][nz] / sm[nz]
                ax_right.stairs(coeff_pred, edges, color=colors[der], linewidth=2.0, linestyle="-", baseline=None, zorder=3)
                coeff_values.append(coeff_pred[nz])

        # symmetric-ish padding on the right axis, always including 0
        finite = np.concatenate([v[np.isfinite(v)] for v in coeff_values]) if coeff_values else np.array([0.0, 1.0])
        if finite.size:
            r_min, r_max = min(0.0, float(finite.min())), max(0.0, float(finite.max()))
        else:
            r_min, r_max = 0.0, 1.0
        if r_max <= r_min:
            r_max = r_min + 1.0
        pad = 0.20 * (r_max - r_min)
        ax_right.set_ylim(r_min - pad, r_max + pad)
        coeff_label = "Coefficient" if args.value == 1.0 else r"$c^{k}\times$ coefficient"
        ax_right.set_ylabel(coeff_label)

        # legend: SM stat band + operator colors + truth/BIT linestyles
        handles = [
            Patch(facecolor="0.55", alpha=0.45, label="SM stat. unc."),
            Patch(facecolor="0.3", alpha=0.18, label="truth stat. unc. (per-operator color)"),
        ]
        handles += [Line2D([0], [0], color=colors[der], linewidth=2.0, label=derivative_label(der)) for der in selected]
        handles.append(Line2D([0], [0], color="0.2", linewidth=2.0, linestyle="--", label="truth"))
        if bit is not None:
            handles.append(Line2D([0], [0], color="0.2", linewidth=2.0, linestyle="-", label="BIT"))
        n_col = 1 if len(handles) <= 6 else (2 if len(handles) <= 14 else 3)
        ax_right.legend(handles=handles, frameon=False, fontsize=9, ncol=n_col, loc="upper right")

        ax_left.text(
            0.02, 0.95, f"c = {args.value:g}",
            transform=ax_left.transAxes, va="top", ha="left", fontsize=11,
        )

        hep.cms.label("Internal", data=False, ax=ax_left, loc=0)
        fig.tight_layout()
        stub = os.path.join(out_dir, feat)
        plt.savefig(stub + ".png", dpi=200)
        plt.savefig(stub + ".pdf")
        plt.close(fig)

    helpers.copyIndexPHP(out_dir)
    logger.info("Wrote %d feature plots.", len(plot_feats))

    # ---- write textual analysis files ----
    _write_analysis_json(out_dir, plot_feats, selected, truth_num, truth_num_sumw2, sm_hist, feature_edges, args.value)
