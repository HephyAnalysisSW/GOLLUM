#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import math
import argparse
import importlib
import itertools

import numpy as np
import ROOT
from tqdm import tqdm

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.user as user
import common.yaml_loader as yaml_loader

from pdf.PDFParametrization import PDFParametrization
from ML.BIT.NumbaBIT import MultiBoostedInformationTree
from data.UIDSplitter import UIDSplitter
from data.plot_options import plot_options as PLOT_OPTS


# --------------------------------------------------------------------------------
# hard-coded knobs
# --------------------------------------------------------------------------------

# Show all available derivatives by default.
# To restrict the output, uncomment and edit, e.g.
# SHOW_ONLY = [("c1",), ("c2",), ("c1", "c1"), ("c1", "c2")]
SHOW_ONLY = None

# --small will stop after this many selected events
SMALL_MAX_EVENTS = 20000

# More discernible hard-coded colors (Wong/Okabe-Ito style + a few extras)
COLOR_HEX = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#882255",  # wine
    "#44AA99",  # teal
    "#AA4499",  # magenta
    "#117733",  # dark green
    "#332288",  # dark blue
]

# Keep the plot logic in the main scope; only tiny utilities live here.
def canonical_derivative(der):
    der = tuple(der)
    if len(der) <= 1:
        return der
    return tuple(sorted(der))

def derivative_label(der):
    if len(der) == 1:
        return der[0]
    if len(der) == 2 and der[0] == der[1]:
        return f"{der[0]}^{{2}}"
    if len(der) == 2:
        return f"{der[0]}#times{der[1]}"
    return str(der)

def make_root_hist(name, title, edges, values):
    h = ROOT.TH1D(name, title, len(edges) - 1, edges.astype(np.float64))
    for i_bin, val in enumerate(values, start=1):
        h.SetBinContent(i_bin, float(val))
    return h

def map_right_to_left(values, right_min, right_max, left_max):
    out = np.zeros_like(values, dtype=np.float64)
    if right_max <= right_min:
        return out
    out[:] = left_max * (values - right_min) / (right_max - right_min)
    return out


# --------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------

p = argparse.ArgumentParser(description="BIT plotting from a trained model (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--small", action="store_true", help=f"Only use the first {SMALL_MAX_EVENTS} selected events")
args = p.parse_args()


# --------------------------------------------------------------------------------
# ROOT style
# --------------------------------------------------------------------------------

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleBorderSize(0)
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)
ROOT.gStyle.SetLegendBorderSize(0)


# --------------------------------------------------------------------------------
# cfg / job selection
# --------------------------------------------------------------------------------

cfg_path = os.path.expanduser(os.path.expandvars(args.config))
CFG = yaml_loader.load_yaml(cfg_path)
D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "bit"]
    if not jobs:
        print("No BIT jobs found.")
        sys.exit(0)

    script = os.path.basename(__file__)
    flags = []
    if args.small:
        flags.append("--small")

    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}".strip())
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "bit":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'bit'.")


# --------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------

samples_mod = importlib.import_module(module_samples)
loader_name = J.get("process")
if not hasattr(samples_mod, loader_name):
    raise RuntimeError(f"Loader/view '{loader_name}' not found in module {module_samples}.")
L = getattr(samples_mod, loader_name)

sel = J.get("selection", None)
sel_f = J.get("selection_features", [])
if sel:
    L.addSelection(sel, sel_f)
    print(f"Added selection to loader: {sel} with selection_features {sel_f}")

L.setFeatures(J["features"])

feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")

GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

plot_feats = [f for f in feat_names if f in PLOT_OPTS]
if not plot_feats:
    print("No plottable features found in data.plot_options. Nothing to do.")
    sys.exit(0)

print(f"Plottable features: {plot_feats}")


# --------------------------------------------------------------------------------
# UID split config
# --------------------------------------------------------------------------------

UID_CFG = (J.get("splitting") or {})
uid_enabled = bool(UID_CFG.get("enabled", False))
uid_fields = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme = (UID_CFG.get("scheme") or {})

uid_splitter = None
train_interval = None

if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    keys = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)

    bit_train_key = "pnn_train"
    if bit_train_key not in uid_intervals:
        raise RuntimeError(f"UID split enabled, but '{bit_train_key}' was not found in the splitting scheme.")

    train_interval = uid_intervals[bit_train_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] plotting split '{bit_train_key}' -> {train_interval}")
else:
    print("[UID] disabled -> plotting all selected events")


# --------------------------------------------------------------------------------
# PDF parametrization
# --------------------------------------------------------------------------------

pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
pdf_rescale_pod_amplitudes = J.get("pdf", {}).get("rescale_pod_amplitudes", True)

pdf = PDFParametrization(
    n=pdf_n,
    typ=pdf_type,
    basis=pdf_basis,
    rescale_pod_amplitudes=pdf_rescale_pod_amplitudes,
)

combos = [canonical_derivative(c) for c in pdf.combinations]
combo_to_idx = {c: i for i, c in enumerate(combos)}

if () not in combo_to_idx:
    raise RuntimeError("Could not find the nominal coefficient () in PDF combinations.")

nominal_idx = combo_to_idx[()]


# --------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------

cfg_base = os.path.join(CFG.get("version", "default"), J["region"])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))

print(f"Trying to load BIT from {model_path}")
try:
    bit = MultiBoostedInformationTree.load(model_path)
except Exception as e:
    print(f"Could not load BIT model from '{model_path}'.")
    print(f"Reason: {e}")
    sys.exit(1)

n_loaded_trees = len(getattr(bit, "trees", []) or [])
print(f"Loaded BIT model successfully: {n_loaded_trees} trees")

model_derivatives_raw = list(getattr(bit, "derivatives", []) or [])
if not model_derivatives_raw:
    print("Loaded BIT model, but bit.derivatives is empty. Nothing to plot.")
    sys.exit(1)

model_derivatives = [canonical_derivative(d) for d in model_derivatives_raw]
pred_derivative_to_idx = {}
for i_der, der in enumerate(model_derivatives):
    pred_derivative_to_idx[der] = i_der

if SHOW_ONLY is None:
    plot_derivatives = [d for d in model_derivatives if d in combo_to_idx]
else:
    plot_derivatives = []
    for der in SHOW_ONLY:
        cder = canonical_derivative(der)
        if cder not in combo_to_idx:
            print(f"Requested derivative {der} not available in PDF combinations -> skipping")
            continue
        if cder not in pred_derivative_to_idx:
            print(f"Requested derivative {der} not available in loaded model -> skipping")
            continue
        if cder not in plot_derivatives:
            plot_derivatives.append(cder)

if not plot_derivatives:
    print("No derivatives selected for plotting.")
    sys.exit(1)

print("Plotting derivatives:")
for der in plot_derivatives:
    print(f"  {der}")


# --------------------------------------------------------------------------------
# output directory
# --------------------------------------------------------------------------------

out_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"], "coefficients")
if args.small:
    out_dir = os.path.join(out_dir, "small")
os.makedirs(out_dir, exist_ok=True)

print(f"Output directory: {out_dir}")


# --------------------------------------------------------------------------------
# histogram containers
# --------------------------------------------------------------------------------

feature_columns = {f: feat_names.index(f) for f in plot_feats}
feature_edges = {}
nominal_hists = {}
truth_num_hists = {}
pred_num_hists = {}

for feat in plot_feats:
    n_bins, x_lo, x_hi = PLOT_OPTS[feat]["binning"]
    edges = np.linspace(x_lo, x_hi, n_bins + 1, dtype=np.float64)
    feature_edges[feat] = edges
    nominal_hists[feat] = np.zeros(n_bins, dtype=np.float64)
    truth_num_hists[feat] = {der: np.zeros(n_bins, dtype=np.float64) for der in plot_derivatives}
    pred_num_hists[feat] = {der: np.zeros(n_bins, dtype=np.float64) for der in plot_derivatives}

root_colors = {}
for i_der, der in enumerate(plot_derivatives):
    root_colors[der] = ROOT.TColor.GetColor(COLOR_HEX[i_der % len(COLOR_HEX)])


# --------------------------------------------------------------------------------
# event loop
# --------------------------------------------------------------------------------

on2idx = {n: i for i, n in enumerate(obs_names)}

i_Q = on2idx["Generator_scalePDF"]
i_x1 = on2idx["Generator_x1"]
i_x2 = on2idx["Generator_x2"]
i_id1 = on2idx["Generator_id1"]
i_id2 = on2idx["Generator_id2"]

if uid_enabled:
    uid_idx = [on2idx[f] for f in uid_fields]
    lo_tr, hi_tr = train_interval

selected_events = 0
n_shards = len(L)

for shard in tqdm(range(n_shards), desc="Shards", unit="shard"):
    X, G, w = L.materialize(shard=shard, what="fow")

    if len(X) == 0:
        continue

    if uid_enabled:
        O_uid = G[:, uid_idx]
        mask = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_tr, hi_tr)
    else:
        mask = np.ones(len(X), dtype=bool)

    if not np.any(mask):
        continue

    X = X[mask].astype(np.float32, copy=False)
    G = G[mask]
    w = w[mask].astype(np.float32, copy=False)

    if args.small:
        remaining = SMALL_MAX_EVENTS - selected_events
        if remaining <= 0:
            break
        if len(X) > remaining:
            X = X[:remaining]
            G = G[:remaining]
            w = w[:remaining]

    if len(X) == 0:
        continue

    Q = G[:, i_Q].astype(np.float32, copy=False)
    x1 = G[:, i_x1].astype(np.float32, copy=False)
    x2 = G[:, i_x2].astype(np.float32, copy=False)
    id1 = G[:, i_id1].astype(np.int32, copy=False)
    id2 = G[:, i_id2].astype(np.int32, copy=False)

    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q).astype(np.float32, copy=False)
    deriv_w = deriv * w.reshape(-1, 1)

    nominal_w = deriv_w[:, nominal_idx]

    pred = np.asarray(bit.predict(X, max_n_tree=n_loaded_trees))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    for feat in plot_feats:
        xvals = X[:, feature_columns[feat]].astype(np.float64, copy=False)
        edges = feature_edges[feat]

        nominal_hists[feat] += np.histogram(xvals, bins=edges, weights=nominal_w)[0]

        for der in plot_derivatives:
            truth_col = combo_to_idx[der]
            pred_col = pred_derivative_to_idx[der]

            truth_num_hists[feat][der] += np.histogram(
                xvals,
                bins=edges,
                weights=deriv_w[:, truth_col],
            )[0]

            pred_num_hists[feat][der] += np.histogram(
                xvals,
                bins=edges,
                weights=nominal_w * pred[:, pred_col],
            )[0]

    selected_events += len(X)

if selected_events == 0:
    print("No events passed the selection / split. Nothing to plot.")
    sys.exit(1)

print(f"Processed {selected_events} selected events")


# --------------------------------------------------------------------------------
# plots: one feature per canvas
# --------------------------------------------------------------------------------

for feat in plot_feats:
    edges = feature_edges[feat]
    n_bins = len(edges) - 1
    x_lo = edges[0]
    x_hi = edges[-1]
    x_title = PLOT_OPTS[feat]["tex"]

    nominal = nominal_hists[feat]

    coeff_truth = {}
    coeff_pred = {}
    coeff_values = []

    nz = nominal != 0.0

    for der in plot_derivatives:
        ct = np.zeros_like(nominal, dtype=np.float64)
        cp = np.zeros_like(nominal, dtype=np.float64)

        if np.any(nz):
            ct[nz] = truth_num_hists[feat][der][nz] / nominal[nz]
            cp[nz] = pred_num_hists[feat][der][nz] / nominal[nz]

        coeff_truth[der] = ct
        coeff_pred[der] = cp

        coeff_values.append(ct[np.isfinite(ct)])
        coeff_values.append(cp[np.isfinite(cp)])

    if coeff_values:
        coeff_values = np.concatenate([v for v in coeff_values if len(v) > 0], axis=0) if any(len(v) > 0 for v in coeff_values) else np.array([0.0, 1.0])
    else:
        coeff_values = np.array([0.0, 1.0])

    finite = np.isfinite(coeff_values)
    if finite.any():
        right_min = float(np.min(coeff_values[finite]))
        right_max = float(np.max(coeff_values[finite]))
    else:
        right_min, right_max = -1.0, 1.0

    if right_max <= right_min:
        width = 1.0 if right_max == right_min else abs(right_max - right_min)
        right_min -= 0.5 * width
        right_max += 0.5 * width

    right_pad = 0.18 * (right_max - right_min)
    right_min -= right_pad
    right_max += right_pad

    left_max = float(np.max(nominal)) if len(nominal) else 0.0
    if left_max <= 0.0:
        left_max = 1.0
    left_max *= 1.25

    canvas = ROOT.TCanvas(f"c_{feat}", feat, 900, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.14)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.08)

    frame = ROOT.TH1D(f"frame_{feat}", "", n_bins, edges.astype(np.float64))
    frame.SetMinimum(0.0)
    frame.SetMaximum(left_max)
    frame.GetXaxis().SetTitle(x_title)
    frame.GetYaxis().SetTitle("Number of events")
    frame.GetXaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetTitleOffset(1.20)
    frame.Draw("axis")

    h_nominal = make_root_hist(f"h_nominal_{feat}", "", edges, nominal)
    h_nominal.SetLineColor(ROOT.kGray + 2)
    h_nominal.SetLineWidth(2)
    h_nominal.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
    h_nominal.Draw("hist same")

    drawn_objects = [frame, h_nominal]

    for der in plot_derivatives:
        y_truth_left = map_right_to_left(coeff_truth[der], right_min, right_max, left_max)
        y_pred_left = map_right_to_left(coeff_pred[der], right_min, right_max, left_max)

        h_truth = make_root_hist(f"h_truth_{feat}_{str(der)}", "", edges, y_truth_left)
        h_pred = make_root_hist(f"h_pred_{feat}_{str(der)}", "", edges, y_pred_left)

        color = root_colors[der]

        h_truth.SetLineColor(color)
        h_truth.SetLineStyle(2)
        h_truth.SetLineWidth(3)
        h_truth.SetMarkerStyle(0)

        h_pred.SetLineColor(color)
        h_pred.SetLineStyle(1)
        h_pred.SetLineWidth(3)
        h_pred.SetMarkerStyle(0)

        h_truth.Draw("hist same")
        h_pred.Draw("hist same")

        drawn_objects.extend([h_truth, h_pred])

    right_axis = ROOT.TGaxis(x_hi, 0.0, x_hi, left_max, right_min, right_max, 510, "+L")
    right_axis.SetTitle("Polynomial coefficient")
    right_axis.SetLabelSize(0.042)
    right_axis.SetTitleSize(0.050)
    right_axis.SetTitleOffset(1.15)
    right_axis.SetLineWidth(2)
    right_axis.Draw()

    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextSize(0.040)
    title.DrawLatex(0.13, 0.94, f"{J['id']}   ({selected_events} events{' , small' if args.small else ''})")
    drawn_objects.append(title)
    drawn_objects.append(right_axis)

    canvas.RedrawAxis()
    canvas.Modified()
    canvas.Update()

    file_stub = os.path.join(out_dir, feat)
    canvas.SaveAs(file_stub + ".png")
    canvas.SaveAs(file_stub + ".pdf")
    canvas.Close()


# --------------------------------------------------------------------------------
# separate legend plot
# --------------------------------------------------------------------------------

legend_canvas = ROOT.TCanvas("c_legend", "legend", 1400, 900)
legend_canvas.SetLeftMargin(0.03)
legend_canvas.SetRightMargin(0.03)
legend_canvas.SetBottomMargin(0.03)
legend_canvas.SetTopMargin(0.10)

legend_frame = ROOT.TH1D("legend_frame", "", 1, 0.0, 1.0)
legend_frame.SetMinimum(0.0)
legend_frame.SetMaximum(1.0)
legend_frame.GetXaxis().SetLabelSize(0.0)
legend_frame.GetYaxis().SetLabelSize(0.0)
legend_frame.GetXaxis().SetTickLength(0.0)
legend_frame.GetYaxis().SetTickLength(0.0)
legend_frame.Draw("axis")

legend_title = ROOT.TLatex()
legend_title.SetNDC()
legend_title.SetTextSize(0.040)
legend_title.DrawLatex(0.03, 0.95, f"Legend for {J['id']}")

legend_entries = 1 + 2 * len(plot_derivatives)
if legend_entries > 30:
    n_cols = 4
elif legend_entries > 18:
    n_cols = 3
elif legend_entries > 8:
    n_cols = 2
else:
    n_cols = 1

legend = ROOT.TLegend(0.03, 0.08, 0.97, 0.90)
legend.SetNColumns(n_cols)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.030)

legend_objects = [legend_frame, legend_title, legend]

dummy_nominal = ROOT.TH1D("dummy_nominal", "", 1, 0.0, 1.0)
dummy_nominal.SetLineColor(ROOT.kGray + 2)
dummy_nominal.SetLineWidth(2)
dummy_nominal.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
legend.AddEntry(dummy_nominal, "SM distribution", "lf")
legend_objects.append(dummy_nominal)

for der in plot_derivatives:
    label = derivative_label(der)
    color = root_colors[der]

    dummy_truth = ROOT.TH1D(f"dummy_truth_{str(der)}", "", 1, 0.0, 1.0)
    dummy_pred = ROOT.TH1D(f"dummy_pred_{str(der)}", "", 1, 0.0, 1.0)

    dummy_truth.SetLineColor(color)
    dummy_truth.SetLineStyle(2)
    dummy_truth.SetLineWidth(3)

    dummy_pred.SetLineColor(color)
    dummy_pred.SetLineStyle(1)
    dummy_pred.SetLineWidth(3)

    legend.AddEntry(dummy_truth, f"truth  {label}", "l")
    legend.AddEntry(dummy_pred,  f"BIT  {label}",   "l")

    legend_objects.extend([dummy_truth, dummy_pred])

legend.Draw()

legend_canvas.Modified()
legend_canvas.Update()

legend_stub = os.path.join(out_dir, "legend")
legend_canvas.SaveAs(legend_stub + ".png")
legend_canvas.SaveAs(legend_stub + ".pdf")
legend_canvas.Close()

print("Done.")
