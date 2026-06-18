#!/usr/bin/env python
from __future__ import annotations

import os, sys, argparse, importlib, warnings, math
from array import array

import numpy as np
import tensorflow as tf

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.user as user
import common.syncer as syncer
from common.helpers import copyIndexPHP

from ML.PNN.PNN import PNN
from data.UIDSplitter import UIDSplitter
from tqdm import tqdm

# Plot options (binning, labels, optional y_ratio_range)

from data.plot_options import plot_options as PLOT_OPTS
# from plot.bit.propaganda_plot_options import plot_options as PLOT_OPTS #temporary change

MAKE_PUBLIC_PLOTS = False

# ---------------- args ----------------
p = argparse.ArgumentParser(description="PNN training-closure per-feature plots (YAML-driven) on held-out test dataset")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="PNN job id to run (omit to list)")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--lumi_scale", type=float, default=None, help="Scale lumi?")
p.add_argument("--for_debug", action="store_true", help="Use _for_debug directories")
p.add_argument("--n_split", default=None, help="Set sample split")
p.add_argument("--shape_only", action="store_true", help="Removing impact of total rate variations from ICP to plot shape-only variations.")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
import common.yaml_loader as yaml_loader
CFG = yaml_loader.load_yaml(cfg_path)

D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "pnn"]
    if not jobs:
        print("No PNN jobs found.")
        sys.exit(0)
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {__file__} {args.config} --job {j['id']}")
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "pnn":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'pnn'.")

param_names = list(J.get("parameters", []))
param_map_tex = " ".join([f"#nu_{{{i+1}}}={p}" for i, p in enumerate(param_names)]) if param_names else ""

# ---------------- UID splitting (YAML-driven, implemented in data/UIDSplitter.py) ----------------
UID_CFG = (J.get("splitting") or {})
uid_enabled   = bool(UID_CFG.get("enabled", False))
uid_fields    = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed      = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme    = (UID_CFG.get("scheme") or {})

uid_intervals = None
uid_splitter = None
if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    # build bucket intervals (inline; no extra helper, no extra checks)
    keys  = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)
    eval_key = "final_eval"
    eval_interval = uid_intervals[eval_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] PNN eval split '{eval_key}' -> {eval_interval}")

# ---------------- resolve loaders ----------------
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

samples_mod = importlib.import_module(module_samples)

bp_specs    = J["base_points"]  # list of {coords: [...], loader: "name", optional removeweights/addweights}
base_points = [spec["coords"] for spec in bp_specs]

loaders = []

for i, spec in enumerate(bp_specs):
    nm = spec["loader"]
    if not hasattr(samples_mod, nm):
        raise RuntimeError(f"Loader/view '{nm}' not found in module {module_samples}.")
    base = getattr(samples_mod, nm)

    base.setFeatures(J["features"])

    remove = list(spec.get("removeweights", []) or [])
    add    = list(spec.get("addweights", []) or [])

    if not remove and not add:
        loaders.append(base)
        continue

    if isinstance(base, RDataLoader):
        base_weights = list(base.weight_branches or [])
        root_loader = base

    elif isinstance(base, SelectionView):
        if base._w_override is not None:
            base_weights = list(base._w_override)
        else:
            if not isinstance(base.base, RDataLoader):
                raise RuntimeError(
                    f"SelectionView '{base.name}' has a non-RDataLoader base. "
                    "Layered views are not supported in this job logic."
                )
            base_weights = list(base.base.weight_branches or [])
        root_loader = base.base if isinstance(base.base, RDataLoader) else None
        if root_loader is None:
            raise RuntimeError(
                f"Could not find underlying RDataLoader for SelectionView '{base.name}'."
            )
    else:
        raise RuntimeError(
            f"Loader/view '{nm}' has unsupported type {type(base)} for automatic weight variations."
        )

    new_weights = list(base_weights)

    for w in remove:
        if w in new_weights:
            new_weights.remove(w)
        else:
            warnings.warn(
                f"[job {J.get('id', '<unknown>')}] weight '{w}' requested for removal "
                f"but not found in loader '{nm}' (current weights: {base_weights})."
            )

    for w in add:
        if w not in new_weights:
            new_weights.append(w)

    if hasattr(root_loader, "_requested_branches"):
        for b in add:
            if b not in root_loader._requested_branches:
                root_loader._requested_branches.append(b)

    if root_loader.observer_names is None:
        root_loader.observer_names = list(add)
    else:
        for b in add:
            if b not in root_loader.observer_names:
                root_loader.observer_names.append(b)

    if isinstance(base, RDataLoader):
        vname = f"{nm}_wvar{i}"
        eff_loader = SelectionView(
            base=base,
            name=vname,
            selection_fn=None,
            feature_names=base.feature_names,
            observer_names=base.observer_names,
            selection_feature_names=None,
            weight=new_weights,
        )
    else:
        vname = f"{base.name}_wvar{i}"
        eff_loader = SelectionView(
            base=base.base,
            name=vname,
            selection_fn=base._selection_fns,
            feature_names=base._feature_names,
            observer_names=base._observer_names,
            selection_feature_names=base._sel_feats,
            weight=new_weights,
        )

    loaders.append(eff_loader)

for l in loaders:
    # Reset n_split
    if isinstance(l, SelectionView):
        if args.lumi_scale is not None:
            l.base.weight_rescale = args.lumi_scale 
        if args.n_split:
            l.base.split = args.n_split
    else:
        if args.lumi_scale is not None:
            l.weight_rescale = args.lumi_scale 
        if args.n_split:
            l.split = args.n_split

# ---------------- sanity: same features across loaders ----------------
feat_names = list(getattr(loaders[0], "feature_names", []))
if not feat_names:
    raise RuntimeError("First loader has no feature_names.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != feat_names:
        raise RuntimeError("Feature mismatch across base-point loaders.")

print(f"\nResolved loaders for job '{J.get('id', '<unknown>')}':")
for idx, (spec, L) in enumerate(zip(bp_specs, loaders)):
    print(f"  base point {idx}, coords={spec['coords']}, loader spec='{spec['loader']}':")
    print(L)
    print("-" * 60)

input_dim = len(feat_names)
feat2col  = {f: i for i, f in enumerate(feat_names)}

# ---------------- artifacts: scaler & ICP ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J["region"])

from ML.Scaler.Scaler import Scaler
scaler_id = J["extras"].get("use_scaler", None)
if scaler_id:
    sj     = next(jj for jj in (CFG.get("jobs") or []) if jj.get("id") == scaler_id)
    sname  = sj["output"]["filename"]
    spath  = os.path.join(user.model_directory, cfg_base, "Scaler", sname)
    sc     = Scaler.load(spath)
    scaler_means, scaler_vars = sc.feature_means, sc.feature_variances
    print(f"Loaded Scaler: {spath}")
else:
    scaler_means = np.zeros(input_dim, dtype=np.float64)
    scaler_vars  = np.ones(input_dim,  dtype=np.float64)
    print("No Scaler configured; using identity.")

# ---------------- dirs ----------------
model_dir = os.path.join(
    user.model_directory,
    cfg_base + ("_for_debug" if args.for_debug else ""),
    "PNN",
    J["id"],
)

# ensure there is a checkpoint (we do not display epoch)
latest = tf.train.latest_checkpoint(model_dir)
if not latest:
    raise RuntimeError(f"No checkpoint found in model_dir: {model_dir}")
print(f"Latest checkpoint: {latest}")

# ---------------- load model (same as training script) ----------------
print(f"Trying to load PNN from {model_dir}")
try:
    pnn = PNN.load(model_dir)
    # RB: if model is trained with ICP bias,  
    # it will be loaded with the ICP bias from the saved payload
    # even when deleting the use_icp field from the job config
except Exception as e:
    raise RuntimeError(f"Failed to load PNN from {model_dir}") from e
print("Success!")

pnn.set_scaler(scaler_means, scaler_vars)

icp_id = J["extras"].get("use_icp", None)
if icp_id:
    from ML.ICP.ICP import InclusiveCrosssectionParametrization
    ij       = next(jj for jj in (CFG.get("jobs") or []) if jj.get("id") == icp_id)
    icp_fn   = ij["output"]["filename"]
    icp_path = os.path.join(user.model_directory, cfg_base, "ICP", icp_fn)
    icp      = InclusiveCrosssectionParametrization.load(icp_path)
    print(f"Loaded ICP: {icp_path}")

    _params = list(icp.parameters)
    _combs  = [tuple(c) for c in icp.combinations]
    _DeltaA = np.asarray(icp.DeltaA, dtype=np.float64)
    pnn.set_icp(parameters=_params, combinations=_combs, DeltaA=_DeltaA)

if args.shape_only:

    if not pnn.has_icp():
        raise NotImplementedError("Currently, only allowing shape-only systematics for PNNs trained with ICP bias.")

    print("Removing impact of ICP (shape-only variations).")
    pnn.remove_icp_bias()

# ---------------- helpers ----------------
def iterate_epoch(shard_limit=None):
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        Xs, Ws, Os = [], [], []
        for L in loaders:
            X, O, w = L.materialize(shard=shard, what="fow")
            Xs.append(X)
            Os.append(O)
            Ws.append(w.astype(np.float32, copy=False))
        
        if not uid_enabled:
            yield Xs, Ws
            continue

        # evaluating on 'final_eval' partition only
        # follow structure used in training code
        Xs_eval, Ws_eval = [], []
        for L, X, w, O in zip(loaders, Xs, Ws, Os):
            obs_names = L.observer_names
            uid_idx = [obs_names.index(f) for f in uid_fields]
            O_uid = O[:, uid_idx]

            lo, hi = eval_interval
            m_eval = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)

            Xs_eval.append(X[m_eval]); Ws_eval.append(w[m_eval])
        
        yield Xs_eval, Ws_eval


def nu_tex_from_coords(coords):
    values = [str(int(np.rint(v))) for v in coords]
    return rf"({', '.join(values)})"

def strip_wvar_suffix(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    if "_wvar" in name:
        name = name.split("_wvar", 1)[0]
    return name

def init_histograms(plot_features, n_bp, rebin=1):
    h_true, h_true2, h_pred, bins = {}, {}, {}, {}
    for feat in plot_features:
        n, lo, hi = PLOT_OPTS[feat]["binning"]
        n = max(1, n // max(1, int(rebin)))
        h_true[feat]  = np.zeros((n, n_bp), dtype=np.float64)
        h_true2[feat] = np.zeros((n, n_bp), dtype=np.float64)
        h_pred[feat]  = np.zeros((n, n_bp), dtype=np.float64)
        bins[feat]    = np.linspace(lo, hi, n + 1)
    return h_true, h_true2, h_pred, bins

# ---------------- feature list ----------------
plot_feats = [f for f in feat_names if f in PLOT_OPTS]
if not plot_feats:
    raise RuntimeError("No features found that are present in PLOT_OPTS.")

# ---------------- closure histogram accumulation (MODULE SCOPE) ----------------
VkA     = pnn.VkA
nom_idx = pnn.nominal_base_point_index

rebin       = int(J.get("runtime", {}).get("rebin", 1))
shard_limit = 1 if args.small else None

true_h, true_h2, pred_h, bins = init_histograms(plot_feats, n_bp=len(base_points), rebin=rebin)

print("Accumulating histograms ...")
for Xs, Ws in tqdm(iterate_epoch(shard_limit=shard_limit), desc="Closure", unit="batch"):
    X0, w0 = Xs[nom_idx], Ws[nom_idx]
    if len(X0) == 0:
        continue
    if not all(len(Xi) for Xi in Xs):
        continue

    dA0 = pnn.deltaA(X0)  # (N0, C)

    # nominal column
    for feat in plot_feats:
        col   = feat2col[feat]
        edges = bins[feat]

        ht0, _   = np.histogram(X0[:, col], bins=edges, weights=w0)
        ht0_2, _ = np.histogram(X0[:, col], bins=edges, weights=(w0 * w0))

        true_h[feat][:, nom_idx]  += ht0
        true_h2[feat][:, nom_idx] += ht0_2
        pred_h[feat][:, nom_idx]  += ht0

    # other base points
    for i_bp, (Xi, wi) in enumerate(zip(Xs, Ws)):
        if i_bp == nom_idx:
            continue
        vk = VkA[i_bp]
        pred_w = w0 * np.exp(dA0 @ vk)

        for feat in plot_feats:
            col   = feat2col[feat]
            edges = bins[feat]

            ht, _  = np.histogram(Xi[:, col], bins=edges, weights=wi)
            ht2, _ = np.histogram(Xi[:, col], bins=edges, weights=(wi * wi))
            hp, _  = np.histogram(X0[:, col], bins=edges, weights=pred_w)

            true_h[feat][:, i_bp]  += ht
            true_h2[feat][:, i_bp] += ht2
            pred_h[feat][:, i_bp]  += hp

# ---------------- plotting (matplotlib + mplhep, per-feature) ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from data.plot_options import get_sample_legend, get_short_parameter_name
if param_names:
    label_param_names_latex = [r"\nu_{"+get_short_parameter_name(param_name)+"}" for param_name in param_names]
    param_info_text = r"$\nu = (" + ",".join(label_param_names_latex) + ")$"
else:
    raise ValueError("No parameters defined for this job.")

sample_legend = "$"+get_sample_legend(strip_wvar_suffix(str(getattr(loaders[0], 'name', bp_specs[0].get('loader', 'bp0')))))+"$"

legend_mapping_line = f"{sample_legend}, {param_info_text}"
legend_mapping_line = legend_mapping_line.replace("#","\\")

# legend layout
legend_columns = 3
n_entries = len(base_points)
n_cols = 3
n_rows = int(math.ceil(n_entries / float(n_cols)))

if n_rows <= 1:
    axes_top = 0.85
    main_legend_y = 0.93
    truth_pred_y = 0.905
elif n_rows == 2:
    axes_top = 0.80
    main_legend_y = 0.925
    truth_pred_y = 0.895
else:
    axes_top = 0.75
    main_legend_y = 0.915
    truth_pred_y = 0.895

# CMS style only has the 6-color Petroff scheme
# stored colors from 10 color scheme in cmap_petroff10_mpl
from data.colors import cmap_petroff10_mpl
hep.style.use("CMS")

colors = [cmap_petroff10_mpl[i] for i in range(n_entries)]
colors[nom_idx] = "black"

feature_keep = {}

version_name = os.path.split(cfg_base)[0]
lumi_by_era = {
    "2016APV": 19.50,
    "2016": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "Run 2": 137.62
}

lumi_era = "Run 2"
for era in lumi_by_era:
    if era in version_name:
        lumi_era = era
        break

def make_pnn_closure_plots():

    plot_dir = os.path.join(
        user.plot_directory,
        "PNN_training_closure",
        cfg_base + ("_for_debug" if args.for_debug else ""),
        J["id"],
    )

    if args.shape_only:
        plot_dir += "_shape"

    print(f"Writing per-feature closure plots to: {plot_dir}")

    os.makedirs(plot_dir, exist_ok=True)
    copyIndexPHP(plot_dir)

    for feat in plot_feats:
        # changing ROOT latex format (in PLOT_OPTS) to mpl latex format
        x_title = PLOT_OPTS.get(feat, {}).get("tex", feat).replace("#","\\")
        x_title = fr"${{{x_title}}}$"
        logY = PLOT_OPTS.get(feat, {}).get("logY", False)

        edges = np.asarray(bins[feat], dtype=np.float64)
        n_bins = len(edges) - 1
        x_min, x_max = float(edges[0]), float(edges[-1])
        centers = 0.5 * (edges[1:] + edges[:-1])
        widths = edges[1:] - edges[:-1]

        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.03)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
        fig.subplots_adjust(top=axes_top)

        # Figure-level header above the plot: sample + nuisance parameters.
        fig.text(
            0.5,
            0.965,
            legend_mapping_line,
            ha="center",
            va="top",
            fontsize=18,
            weight="bold",
        )

        # Top: histograms
        max_y = 0.0
        handles = []
        labels = []

        for k, nu in enumerate(base_points):
            y = true_h[feat][:, k].astype(np.float64)
            y2 = true_h2[feat][:, k].astype(np.float64)
            y_pred = pred_h[feat][:, k].astype(np.float64)

            err = np.sqrt(y2)

            # plot predicted as stepped line
            # use edges for step plotting
            step_x = np.concatenate([edges[:-1], edges[-1:]])
            step_y = np.concatenate([y_pred, y_pred[-1:]])
            h_line, = ax_top.step(step_x, step_y, where="post", color=colors[k], linewidth=2)

            # plot truth as markers with errorbars at bin centers
            h_err = ax_top.errorbar(centers, y, yerr=err, fmt="o", color=colors[k], markersize=4, label=nu_tex_from_coords(nu))

            handles.append(h_line)
            labels.append(nu_tex_from_coords(nu))

            max_y = max(max_y, np.nanmax(y))

        if logY:
            ax_top.set_yscale("log")
            y_min = max(0.1, 0.3)
            y_max = max(1.0, 1.2 * max_y) if max_y > 0 else 1.0
            ax_top.set_ylim(y_min, y_max)
        else:
            ax_top.set_ylim(0.0, 1.2 * max_y if max_y > 0 else 1.0)

        ax_top.set_ylabel("Events")
        ax_top.tick_params(labelbottom=False)

        # CMS label area
        hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, year = lumi_era, ax=ax_top, loc=0, fontsize=14)
        # hep.mpl_magic()

        # Bottom: ratios to nominal
        h_nom = true_h[feat][:, nom_idx].astype(np.float64)
        denom = h_nom.copy()
        denom[denom == 0] = np.nan

        max_dev = 0.0
        for k in range(n_entries):
            y = true_h[feat][:, k].astype(np.float64)
            y_pred = pred_h[feat][:, k].astype(np.float64)

            r_true = y / denom
            r_pred = y_pred / denom

            err = np.sqrt(true_h2[feat][:, k].astype(np.float64))
            r_err = err / denom

            ax_bot.errorbar(centers, r_true, yerr=r_err, fmt="o", color=colors[k], markersize=4)
            step_x = np.concatenate([edges[:-1], edges[-1:]])
            step_y = np.concatenate([r_pred, r_pred[-1:]])
            ax_bot.step(step_x, step_y, where="post", color=colors[k], linewidth=2)

            # compute max deviation
            valid = np.isfinite(r_true)
            if np.any(valid):
                max_dev = max(max_dev, np.nanmax(np.abs(r_true[valid] - 1.0)))
            validp = np.isfinite(r_pred)
            if np.any(validp):
                max_dev = max(max_dev, np.nanmax(np.abs(r_pred[validp] - 1.0)))

        if max_dev <= 0.0:
            r_min, r_max = 0.9, 1.1
        else:
            half_range = 1.3 * max_dev
            r_min = 1.0 - half_range
            r_max = 1.0 + half_range

        ax_bot.set_ylim(r_min, r_max)
        ax_bot.set_ylabel("var / nominal")
        ax_bot.set_xlabel(x_title)
        # ax_bot.axhline(1.0, color="k", linestyle="--")

        # Figure-level legend in the same top area as the mapping line.
        fig.legend(
            handles,
            labels,
            ncol=n_cols,
            loc="upper center",
            bbox_to_anchor=(0.5, main_legend_y),
            frameon=False,
            fontsize=18,
            handlelength=2.0,
            columnspacing=1.4,
        )

        # Marker/line meaning centered below the basis-point legend.
        truth_pred_handles = [
            Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="truth"),
            Line2D([], [], color="black", linestyle="-", linewidth=2, label="prediction"),
        ]
        
        fig.legend(
            truth_pred_handles,
            ["truth", "prediction"],
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, truth_pred_y),
            frameon=False,
            fontsize=14,
            handlelength=2.0,
            columnspacing=1.8,
        )

        out_png = os.path.join(plot_dir, f"{feat}.png")
        out_pdf = os.path.join(plot_dir, f"{feat}.pdf")
        plt.savefig(out_png, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        feature_keep[feat] = True


def make_pnn_chi2_plots():

    plot_dir = os.path.join(
        user.plot_directory,
        "PNN_prediction_diagnostics",
        cfg_base + ("_for_debug" if args.for_debug else ""),
        J["id"],
    )

    if args.shape_only:
        plot_dir += "_shape"
        
    copyIndexPHP(plot_dir)

    for feat in plot_feats:
        # changing ROOT latex format (in PLOT_OPTS) to mpl latex format
        x_title = PLOT_OPTS.get(feat, {}).get("tex", feat).replace("#","\\")
        x_title = fr"${{{x_title}}}$"
        logY = PLOT_OPTS.get(feat, {}).get("logY", False)

        edges = np.asarray(bins[feat], dtype=np.float64)
        n_bins = len(edges) - 1
        x_min, x_max = float(edges[0]), float(edges[-1])
        centers = 0.5 * (edges[1:] + edges[:-1])
        widths = edges[1:] - edges[:-1]

        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.03)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
        fig.subplots_adjust(top=axes_top)

        # Figure-level header above the plot: sample + nuisance parameters.
        fig.text(
            0.5,
            0.965,
            legend_mapping_line,
            ha="center",
            va="top",
            fontsize=18,
            weight="bold",
        )

        # Top: histograms
        max_y = 0.0
        handles = []
        labels = []

        for k, nu in enumerate(base_points):

            y = true_h[feat][:, k].astype(np.float64)
            y2 = true_h2[feat][:, k].astype(np.float64)
            y_pred = pred_h[feat][:, k].astype(np.float64)

            err = np.sqrt(y2)

            # plot predicted as stepped line
            # use edges for step plotting
            step_x = np.concatenate([edges[:-1], edges[-1:]])
            step_y = np.concatenate([y_pred, y_pred[-1:]])
            h_line, = ax_top.step(step_x, step_y, where="post", color=colors[k], linewidth=2)

            # plot truth as markers with errorbars at bin centers
            # h_err = ax_top.errorbar(centers, y, yerr=err, fmt="o", color=colors[k], markersize=4, label=nu_tex_from_coords(nu))

            handles.append(h_line)
            labels.append(nu_tex_from_coords(nu))

            max_y = max(max_y, np.nanmax(y))

        if logY:
            ax_top.set_yscale("log")
            y_min = max(0.1, 0.3)
            y_max = max(1.0, 1.2 * max_y) if max_y > 0 else 1.0
            ax_top.set_ylim(y_min, y_max)
        else:
            ax_top.set_ylim(0.0, 1.2 * max_y if max_y > 0 else 1.0)

        ax_top.set_ylabel("Events")
        ax_top.tick_params(labelbottom=False)

        # CMS label area
        hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, year = lumi_era, ax=ax_top, loc=0, fontsize=14)
        # hep.mpl_magic()

        # Bottom: ratios to nominal
        h_nom = true_h[feat][:, nom_idx].astype(np.float64)
        denom = h_nom.copy()
        denom[denom == 0] = np.nan

        max_dev = 0.0
        for k in range(n_entries):

            y = true_h[feat][:, k].astype(np.float64)
            y_pred = pred_h[feat][:, k].astype(np.float64)
            # err = np.sqrt(true_h2[feat][:, k].astype(np.float64))

            from ML.TFMC.tfmc_plot_true_model import safe_divide
            chi_2 = safe_divide(np.square((y_pred - h_nom)),h_nom)

            step_x = np.concatenate([edges[:-1], edges[-1:]])
            step_y = np.concatenate([chi_2, chi_2[-1:]])
            ax_bot.step(step_x, step_y, where="post", color=colors[k], linewidth=2)

            validp = np.isfinite(chi_2)
            if np.any(validp):
                max_dev = max(max_dev, np.nanmax(np.abs(chi_2[validp])))

        if max_dev <= 0.0:
            r_max = 0.1
        else:
            half_range = 1.3 * max_dev
            r_max = half_range

        ax_bot.set_ylim(0.0, r_max)
        ax_bot.set_ylabel(r"$\Delta^2(var,nom)/\sigma_{nom}^2$")
        ax_bot.set_xlabel(x_title)

        # Figure-level legend in the same top area as the mapping line.
        fig.legend(
            handles,
            labels,
            ncol=n_cols,
            loc="upper center",
            bbox_to_anchor=(0.5, main_legend_y),
            frameon=False,
            fontsize=18,
            handlelength=2.0,
            columnspacing=1.4,
        )

        out_png = os.path.join(plot_dir, f"{feat}.png")
        out_pdf = os.path.join(plot_dir, f"{feat}.pdf")
        plt.savefig(out_png, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        feature_keep[feat] = True

make_pnn_closure_plots()

make_pnn_chi2_plots()

syncer.sync()
print("Done.")

