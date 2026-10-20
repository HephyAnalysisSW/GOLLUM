#!/usr/bin/env python
"""ICPH training-closure plot (YAML-driven): compares truth per-bin yields,
accumulated directly from the base-point loaders, against the yields
predicted by the trained ICPH object, for every base point of a job."""
from __future__ import annotations

import os, sys, math, argparse, importlib, warnings

import numpy as np

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.user as user
import common.syncer as syncer
from common.helpers import copyIndexPHP

from ML.ICPH.ICPH import InclusiveCrosssectionParametrizationHistogram
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView
from tqdm import tqdm

MAKE_PUBLIC_PLOTS = False

# ---------------- args ----------------
p = argparse.ArgumentParser(description="ICPH training-closure plot (YAML-driven): truth vs prediction per bin")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="ICPH job id to run (omit to list)")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--for_debug", action="store_true", help="Use _for_debug directories")
p.add_argument("--n_split", default=None, help="Set sample split")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
import common.yaml_loader as yaml_loader
CFG = yaml_loader.load_yaml(cfg_path)

D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")


def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "icph"]
    if not jobs:
        print("No ICPH jobs found.")
        sys.exit(0)
    for j in jobs:
        print(f"python {__file__} {args.config} --job {j['id']}")
    sys.exit(0)


if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "icph":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'icph'.")

params       = list(J["parameters"])
combinations = [tuple(c) for c in J["combinations"]]
bp_specs     = list(J["base_points"] or [])
if not bp_specs:
    raise RuntimeError(f"ICPH job '{J['id']}' has no base_points defined.")

binning_spec = J.get("binning", [])
if len(binning_spec) not in (1, 2):
    raise RuntimeError("ICPH: only 1D or 2D binning supported")
axis_names = [b[0] for b in binning_spec]
bin_edges  = [np.asarray(b[1], dtype=float) for b in binning_spec]

base_points = []
nominal_base_point = None
nominal_index = None
for i, spec in enumerate(bp_specs):
    coords = tuple(spec["coords"])
    base_points.append(coords)
    if spec.get("nominal", False):
        if nominal_base_point is not None:
            raise RuntimeError(f"ICPH job '{J['id']}' has multiple nominal base points.")
        nominal_base_point = coords
        nominal_index = i
if nominal_base_point is None:
    raise RuntimeError(f"ICPH job '{J['id']}' has no nominal base point.")

# ---------------- resolve loaders (weight variations per base point) ----------------
samples_mod = importlib.import_module(module_samples)
loaders: list[object] = []

for i, spec in enumerate(bp_specs):
    loader_name = spec.get("loader", None)
    if loader_name is None:
        raise RuntimeError(f"ICPH job '{J['id']}', base_points[{i}] has no 'loader'.")
    if not hasattr(samples_mod, loader_name):
        raise RuntimeError(f"Loader/view '{loader_name}' not found in module '{module_samples}'.")

    base = getattr(samples_mod, loader_name)
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
        root_loader = base.base
    else:
        raise RuntimeError(
            f"Loader/view '{loader_name}' has unsupported type {type(base)} for automatic weight variations."
        )

    new_weights = list(base_weights)

    for w in remove:
        if w in new_weights:
            new_weights.remove(w)
        else:
            warnings.warn(
                f"[job {J.get('id', '<unknown>')}] weight '{w}' requested for removal "
                f"but not found in loader '{loader_name}' (current weights: {base_weights})."
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
        vname = f"{loader_name}_wvar{i}"
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

sel   = J.get("selection", None)
sel_f = J.get("selection_features", [])
if sel:
    for loader in loaders:
        if isinstance(loader, RDataLoader):
            loader.addSelection(sel, sel_f)
        else:
            loader.base.addSelection(sel, sel_f)

if args.n_split:
    for l in loaders:
        if isinstance(l, RDataLoader):
            l.set_n_split(args.n_split)
        else:
            l.base.set_n_split(args.n_split)

print(f"\nResolved loaders for ICPH job '{J.get('id', '<unknown>')}':")
for i, (spec, L) in enumerate(zip(bp_specs, loaders)):
    print(f"  base point {i}, coords={spec['coords']}, loader='{spec['loader']}', nominal={bool(spec.get('nominal', False))}:")
    print(L)
    print("-" * 60)

# ---------------- load trained ICPH ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J["region"])
out = J.get("output", {}) or {}
filename = out.get("filename", f"ICPH_{J['id']}.pkl")
icph_path = os.path.join(user.model_directory, cfg_base + ("_for_debug" if args.for_debug else ""), "ICPH", filename)

print(f"Loading ICPH from {icph_path}")
icph = InclusiveCrosssectionParametrizationHistogram.load(icph_path)
print(icph)

# ---------------- accumulate truth histograms per base point ----------------
axis1_name = axis_names[0]
axis2_name = axis_names[1] if len(axis_names) == 2 else None
hist_shape = tuple(len(e) - 1 for e in bin_edges)
n_bins_flat = int(np.prod(hist_shape))

true_h  = np.zeros((n_bins_flat, len(base_points)), dtype=np.float64)
true_h2 = np.zeros((n_bins_flat, len(base_points)), dtype=np.float64)

print("Accumulating histograms ...")
for i_bp, (spec, loader) in enumerate(zip(bp_specs, loaders)):
    n_shards = len(getattr(loader, "base", loader))
    if args.small:
        n_shards = min(n_shards, 1)
    for shard in tqdm(range(n_shards), desc=f"base point {i_bp}", unit="shard"):
        X, G, w = loader.materialize(shard=shard, what="fow")

        vals1 = X[:, loader.feature_names.index(axis1_name)] if axis1_name in loader.feature_names else G[:, loader.observer_names.index(axis1_name)]
        if axis2_name is not None:
            vals2 = X[:, loader.feature_names.index(axis2_name)] if axis2_name in loader.feature_names else G[:, loader.observer_names.index(axis2_name)]
            hist,  _, _ = np.histogram2d(vals1, vals2, bins=(bin_edges[0], bin_edges[1]), weights=w)
            hist2, _, _ = np.histogram2d(vals1, vals2, bins=(bin_edges[0], bin_edges[1]), weights=(w * w))
        else:
            hist,  _ = np.histogram(vals1, bins=bin_edges[0], weights=w)
            hist2, _ = np.histogram(vals1, bins=bin_edges[0], weights=(w * w))

        true_h[:, i_bp]  += hist.reshape(-1)
        true_h2[:, i_bp] += hist2.reshape(-1)

# ---------------- predicted histograms from ICPH ----------------
nom_true = true_h[:, nominal_index].reshape(hist_shape)
nominal_vec = np.array(nominal_base_point, dtype=np.float64)

pred_h = np.zeros((n_bins_flat, len(base_points)), dtype=np.float64)
for i_bp, coords in enumerate(base_points):
    nu_vec = np.array(coords, dtype=np.float64) - nominal_vec
    ratio = icph.predict(nu_vec)
    pred_h[:, i_bp] = (nom_true * ratio).reshape(-1)

# ---------------- plotting (matplotlib + mplhep) ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from data.plot_options import plot_options as PLOT_OPTS
from data.plot_options import get_sample_legend, get_short_parameter_name
from data.colors import cmap_petroff10_mpl

hep.style.use("CMS")


def nu_tex_from_coords(coords):
    values = [str(int(np.rint(v))) for v in coords]
    return rf"({', '.join(values)})"


def strip_wvar_suffix(name: str) -> str:
    if "_wvar" in name:
        name = name.split("_wvar", 1)[0]
    return name


label_param_names_latex = [r"\nu_{" + get_short_parameter_name(param_name) + "}" for param_name in params]
param_info_text = r"$\nu = (" + ",".join(label_param_names_latex) + ")$"

sample_legend = "$" + get_sample_legend(strip_wvar_suffix(bp_specs[nominal_index]["loader"])) + "$"
legend_mapping_line = f"{sample_legend}, {param_info_text}"
legend_mapping_line = legend_mapping_line.replace("#", "\\")

n_entries = len(base_points)
n_cols = 3
n_rows = int(math.ceil(n_entries / float(n_cols)))

has_group_row = len(hist_shape) == 2

if n_rows <= 1:
    axes_top_margin = 0.75 if has_group_row else 0.85
    main_legend_y = 0.90 if has_group_row else 0.93
    truth_pred_y = 0.865 if has_group_row else 0.905
elif n_rows == 2:
    axes_top_margin = 0.70 if has_group_row else 0.80
    main_legend_y = 0.895 if has_group_row else 0.925
    truth_pred_y = 0.86 if has_group_row else 0.895
else:
    axes_top_margin = 0.65 if has_group_row else 0.75
    main_legend_y = 0.885 if has_group_row else 0.915
    truth_pred_y = 0.855 if has_group_row else 0.895

colors = [cmap_petroff10_mpl[i] for i in range(n_entries)]
colors[nominal_index] = "black"

version_name = os.path.split(cfg_base)[0]
lumi_by_era = {
    "2016APV": 19.50,
    "2016": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "Run 2": 137.62,
}
lumi_era = "Run 2"
for era in lumi_by_era:
    if era in version_name:
        lumi_era = era
        break

def mpl_tex(name: str) -> str:
    tex = PLOT_OPTS.get(name, {}).get("tex", name)
    return tex.replace("#", "\\")


def fmt_edge(v: float) -> str:
    return f"{v:g}"


# ---------------- group layout: outer axis (if 2D) split into side-by-side panels ----------------
if len(hist_shape) == 2:
    n_groups, n_inner = hist_shape
    group_edges, inner_edges = bin_edges
    outer_name, inner_name = axis_names
else:
    n_groups, n_inner = 1, hist_shape[0]
    group_edges, inner_edges = None, bin_edges[0]
    outer_name, inner_name = None, axis_names[0]

arr_true  = true_h.T.reshape((n_entries, n_groups, n_inner))
arr_true2 = true_h2.T.reshape((n_entries, n_groups, n_inner))
arr_pred  = pred_h.T.reshape((n_entries, n_groups, n_inner))

inner_centers = 0.5 * (inner_edges[1:] + inner_edges[:-1])
inner_step_x  = np.concatenate([inner_edges[:-1], inner_edges[-1:]])

plot_dir = os.path.join(
    user.plot_directory,
    "ICPH_training_closure",
    cfg_base + ("_for_debug" if args.for_debug else ""),
    # J["id"],
)
print(f"Writing closure plot to: {plot_dir}")
os.makedirs(plot_dir, exist_ok=True)
copyIndexPHP(plot_dir)

fig = plt.figure(figsize=(4 * n_groups + 4, 12))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.03)
gs_top = gs[0].subgridspec(1, n_groups, wspace=0.0)
gs_bot = gs[1].subgridspec(1, n_groups, wspace=0.0)

axes_top = [fig.add_subplot(gs_top[0, 0])]
axes_bot = [fig.add_subplot(gs_bot[0, 0], sharex=axes_top[0])]
for g in range(1, n_groups):
    axes_top.append(fig.add_subplot(gs_top[0, g], sharey=axes_top[0]))
    axes_bot.append(fig.add_subplot(gs_bot[0, g], sharex=axes_top[g], sharey=axes_bot[0]))

fig.subplots_adjust(top=axes_top_margin)
fig.text(0.5, 0.965, legend_mapping_line, ha="center", va="top", fontsize=18, weight="bold")

max_y = 0.0
handles = []
labels = []

for g in range(n_groups):
    ax_top = axes_top[g]
    for k, nu in enumerate(base_points):
        y      = arr_true[k, g]
        y2     = arr_true2[k, g]
        y_pred = arr_pred[k, g]
        err    = np.sqrt(y2)

        step_y = np.concatenate([y_pred, y_pred[-1:]])
        h_line, = ax_top.step(inner_step_x, step_y, where="post", color=colors[k], linewidth=2)
        ax_top.errorbar(inner_centers, y, yerr=err, fmt="o", color=colors[k], markersize=4)

        if g == 0:
            handles.append(h_line)
            labels.append(nu_tex_from_coords(nu))

        max_y = max(max_y, np.nanmax(y), np.nanmax(y_pred))

    ax_top.set_xticks([inner_edges[0], inner_edges[-1]])
    ax_top.set_xticklabels([fmt_edge(inner_edges[0]), fmt_edge(inner_edges[-1])])
    ax_top.tick_params(labelbottom=False)
    if g > 0:
        ax_top.tick_params(labelleft=False)
        ax_top.spines["left"].set_linestyle("--")
    if g < n_groups - 1:
        ax_top.spines["right"].set_linestyle("--")

    if group_edges is not None:
        lo, hi = group_edges[g], group_edges[g + 1]
        ax_top.text(
            0.5, 1.02,
            rf"${fmt_edge(lo)} \leq {mpl_tex(outer_name)} < {fmt_edge(hi)}$",
            transform=ax_top.transAxes, ha="center", va="bottom", fontsize=11,
        )

axes_top[0].set_ylim(0.0, 1.2 * max_y if max_y > 0 else 1.0)
axes_top[0].set_ylabel("Events")

hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, year=lumi_era, ax=axes_top[0], loc=2, fontsize=14)

h_nom_arr = arr_true[nominal_index]
denom_arr = h_nom_arr.copy()
denom_arr[denom_arr == 0] = np.nan

max_dev = 0.0
for g in range(n_groups):
    ax_bot = axes_bot[g]
    denom = denom_arr[g]
    for k in range(n_entries):
        y      = arr_true[k, g]
        y_pred = arr_pred[k, g]
        err    = np.sqrt(arr_true2[k, g])

        r_true = y / denom
        r_pred = y_pred / denom
        r_err  = err / denom

        ax_bot.errorbar(inner_centers, r_true, yerr=r_err, fmt="o", color=colors[k], markersize=4)
        step_y = np.concatenate([r_pred, r_pred[-1:]])
        ax_bot.step(inner_step_x, step_y, where="post", color=colors[k], linewidth=2)

        valid = np.isfinite(r_true)
        if np.any(valid):
            max_dev = max(max_dev, np.nanmax(np.abs(r_true[valid] - 1.0)))
        validp = np.isfinite(r_pred)
        if np.any(validp):
            max_dev = max(max_dev, np.nanmax(np.abs(r_pred[validp] - 1.0)))

    ax_bot.set_xticks([inner_edges[0], inner_edges[-1]])
    ax_bot.set_xticklabels([fmt_edge(inner_edges[0]), fmt_edge(inner_edges[-1])])
    if g > 0:
        ax_bot.tick_params(labelleft=False)
        ax_bot.spines["left"].set_linestyle("--")
    if g < n_groups - 1:
        ax_bot.spines["right"].set_linestyle("--")

if max_dev <= 0.0:
    r_min, r_max = 0.9, 1.1
else:
    half_range = 1.3 * max_dev
    r_min = 1.0 - half_range
    r_max = 1.0 + half_range

axes_bot[0].set_ylim(r_min, r_max)
axes_bot[0].set_ylabel("var / nominal")
fig.text(0.5, 0.04, f"${mpl_tex(inner_name)}$", ha="center", va="top", fontsize=18)

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

out_png = os.path.join(plot_dir, f"{J['id']}.png")
out_pdf = os.path.join(plot_dir, f"{J['id']}.pdf")
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

syncer.sync()
print("Done.")
