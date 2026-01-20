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
from tqdm import tqdm

# Plot options (binning, labels, optional y_ratio_range)
from data.plot_options import plot_options as PLOT_OPTS

# ---------------- args ----------------
p = argparse.ArgumentParser(description="PNN training-closure per-feature plots (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="PNN job id to run (omit to list)")
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
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "pnn"]
    if not jobs:
        print("No PNN jobs found.")
        sys.exit(0)
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} --job {j['id']}")
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "pnn":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'pnn'.")

param_names = list(J.get("parameters", []))
param_map_tex = " ".join([f"#nu_{{{i+1}}}={p}" for i, p in enumerate(param_names)]) if param_names else ""

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

# Reset n_split
if args.n_split:
    for l in loaders:
        if isinstance(l, SelectionView):
            l.base.split = args.n_split
        else:
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

cfg_path = cfg_base
if args.for_debug:
    cfg_base += "_for_debug"

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
    cfg_path,
    "PNN",
    J["id"],
)

plot_dir = os.path.join(
    user.plot_directory,
    "training_closure",
    cfg_path,
    "PNN",
    J["id"],
)
os.makedirs(plot_dir, exist_ok=True)
copyIndexPHP(plot_dir)

# ensure there is a checkpoint (we do not display epoch)
latest = tf.train.latest_checkpoint(model_dir)
if not latest:
    raise RuntimeError(f"No checkpoint found in model_dir: {model_dir}")
print(f"Latest checkpoint: {latest}")

# ---------------- load model (same as training script) ----------------
print(f"Trying to load PNN from {model_dir}")
try:
    pnn = PNN.load(model_dir)
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

# ---------------- helpers ----------------
def iterate_epoch(shard_limit=None):
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        Xs, Ws = [], []
        for L in loaders:
            X, w = L.materialize(shard=shard, what="fw")
            Xs.append(X)
            Ws.append(w.astype(np.float32, copy=False))
        yield Xs, Ws

def nu_tex_from_coords(coords):
    parts = []
    for i, v in enumerate(coords):
        iv = int(np.rint(v))
        parts.append(f"#nu_{{{i+1}}} = {iv:+d}")
    return ", ".join(parts)

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

# ---------------- plotting (INLINE, per-feature) ----------------
import ROOT
ROOT.gStyle.SetOptStat(0)
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
    ROOT.setTDRStyle()
except Exception:
    pass

import cmsstyle
legend_columns = 4
colors = [
#    ROOT.kRed + 1,
#    ROOT.kBlue + 1,
#    ROOT.kGreen + 2,
#    ROOT.kMagenta + 1,
#    ROOT.kOrange + 1,
#    ROOT.kCyan + 1,
    cmsstyle.p10.kBlue,
    cmsstyle.p10.kYellow,
    cmsstyle.p10.kRed,
    cmsstyle.p10.kAsh,
    cmsstyle.p10.kViolet,
    cmsstyle.p10.kBrown,
    cmsstyle.p10.kOrange,
    cmsstyle.p10.kGreen,
    cmsstyle.p10.kGray,
    cmsstyle.p10.kCyan,
]

if len(base_points) > len(colors):
    colors = (colors * (len(base_points) // len(colors) + 1))[:len(base_points)]
colors[nom_idx] = ROOT.kBlack

# one-line mapping: first loader only
lname0 = getattr(loaders[0], "name", None) or bp_specs[0].get("loader", "bp0")
lname0 = strip_wvar_suffix(str(lname0))
legend_mapping_line = f"{lname0} {param_map_tex}".strip()

# legend size tuning based on number of rows
n_entries = 2 * len(base_points)  # truth + pred
n_cols = max(1, int(legend_columns))
n_rows = int(math.ceil(n_entries / float(n_cols)))

# legend-pad height in canvas NDC (clamped)
# (mapping line + legend; scales smoothly with rows)
legend_pad_h = 0.15 + 0.04 * n_rows
legend_pad_h = max(0.16, min(0.34, legend_pad_h))

# legend text size inside pad (clamped)
legend_text_size = 0.22 / max(1.0, n_rows + 1.0)
legend_text_size = max(0.03, min(0.07, legend_text_size))

print(f"Writing per-feature closure plots to: {plot_dir}")

feature_keep = {}

for feat in plot_feats:
    x_title = PLOT_OPTS.get(feat, {}).get("tex", feat)
    logY = PLOT_OPTS.get(feat, {}).get("logY", False)

    edges = np.asarray(bins[feat], dtype=np.float64)
    n_bins = len(edges) - 1
    x_min, x_max = float(edges[0]), float(edges[-1])
    bin_edges_root = array("d", edges.tolist())

    canvas_name = f"closure_{feat}"
    c = ROOT.TCanvas(canvas_name, canvas_name, 800, 1200)

    # pads: legend pad sits directly above the top pad
    y_bottom_top = 0.30
    y_top_top    = 1.0 - legend_pad_h

    padLegend = ROOT.TPad(canvas_name + "_legend", canvas_name + "_legend", 0.0, y_top_top,    1.0, 1.0)
    padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, y_bottom_top, 1.0, y_top_top)
    padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.0,          1.0, y_bottom_top)

    # legend pad: tighter margins
    padLegend.SetBottomMargin(0.00)
    padLegend.SetTopMargin(0.06)
    padLegend.SetLeftMargin(0.10)
    padLegend.SetRightMargin(0.10)
    padLegend.SetFillStyle(0)

    # top pad: maximize plotting area
    padTop.SetBottomMargin(0.0)
    padTop.SetTopMargin(0.00)
    padTop.SetLeftMargin(0.10)
    padTop.SetRightMargin(0.05)

    # bottom pad: unchanged
    padBottom.SetTopMargin(0.0)
    padBottom.SetBottomMargin(0.30)
    padBottom.SetLeftMargin(0.10)
    padBottom.SetRightMargin(0.05)

    padLegend.Draw()
    padTop.Draw()
    padBottom.Draw()

    # legend: place under the mapping line, use most of pad
    # mapping line is at y~0.90, so legend box is below it
    legend = ROOT.TLegend(0.06, 0.05, 0.98, 0.78)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetNColumns(legend_columns)
    legend.SetTextSize(legend_text_size)

    keep = [padLegend, padTop, padBottom, legend]

    # histos
    h_true_abs, h_pred_abs = [], []

    for k, nu in enumerate(base_points):
        ht = ROOT.TH1F(f"h_true_{feat}_{k}", "", n_bins, bin_edges_root)
        hp = ROOT.TH1F(f"h_pred_{feat}_{k}", "", n_bins, bin_edges_root)
        ht.SetDirectory(0)
        hp.SetDirectory(0)

        for ib in range(n_bins):
            y  = float(true_h[feat][ib, k])
            e2 = float(true_h2[feat][ib, k])
            ht.SetBinContent(ib + 1, y)
            ht.SetBinError(ib + 1, math.sqrt(e2) if e2 > 0.0 else 0.0)

            yp = float(pred_h[feat][ib, k])
            hp.SetBinContent(ib + 1, yp)
            hp.SetBinError(ib + 1, 0.0)

        col = colors[k]

        # truth as data with errors
        ht.SetMarkerStyle(ROOT.kFullCircle)
        ht.SetMarkerSize(1.0)
        ht.SetMarkerColor(col)
        ht.SetLineColor(col)
        ht.SetLineWidth(1)
        ht.SetFillStyle(0)

        # pred as solid line, width 2
        hp.SetLineColor(col)
        hp.SetLineStyle(ROOT.kSolid)
        hp.SetLineWidth(2)
        hp.SetMarkerSize(0)

        h_true_abs.append(ht)
        h_pred_abs.append(hp)
        keep.extend([ht, hp])

        nu_tex = nu_tex_from_coords(nu)
        legend.AddEntry(ht, f"{nu_tex} truth", "lep")
        legend.AddEntry(hp, f"{nu_tex} pred",  "l")

    # TOP PAD
    padTop.cd()
    padTop.SetTicks(1, 1)
    if logY:
        padTop.SetLogy(True)

    max_y = 0.0
    for h in h_true_abs + h_pred_abs:
        max_y = max(max_y, float(h.GetMaximum()))

    if logY:
        y_min = 0.3
        y_max = max(1.0, 1.2 * max_y) if max_y > 0 else 1.0
    else:
        y_min = 0.0
        y_max = 1.2 * max_y if max_y > 0 else 1.0

    hframe = ROOT.TH2F(f"hframe_{feat}", "", n_bins, bin_edges_root, 100, y_min, y_max)
    hframe.SetDirectory(0)
    hframe.SetTitle("")
    hframe.GetXaxis().SetTitle(x_title)
    hframe.GetYaxis().SetTitle("Events")
    hframe.GetYaxis().SetTitleSize(0.06)
    hframe.GetYaxis().SetLabelSize(0.045)
    hframe.GetXaxis().SetLabelSize(0.0)
    hframe.GetXaxis().SetTitleSize(0.0)
    hframe.Draw()
    keep.append(hframe)

    for hp in h_pred_abs:
        hp.Draw("HIST SAME")
    for ht in h_true_abs:
        ht.Draw("E1 SAME")

    # BOTTOM PAD (ratios to nominal truth)
    padBottom.cd()
    padBottom.SetTicks(1, 1)

    h_nom_true = h_true_abs[nom_idx]

    h_true_ratio, h_pred_ratio = [], []
    for k, nu in enumerate(base_points):
        rt = h_true_abs[k].Clone(f"h_true_ratio_{feat}_{k}")
        rt.SetDirectory(0)
        rt.Divide(h_nom_true)
        rt.SetMarkerStyle(ROOT.kFullCircle)
        rt.SetMarkerSize(1.0)
        rt.SetMarkerColor(colors[k])
        rt.SetLineColor(colors[k])
        rt.SetLineWidth(1)
        rt.SetFillStyle(0)

        rp = h_pred_abs[k].Clone(f"h_pred_ratio_{feat}_{k}")
        rp.SetDirectory(0)
        rp.Divide(h_nom_true)
        for ib in range(1, n_bins + 1):
            rp.SetBinError(ib, 0.0)
        rp.SetLineColor(colors[k])
        rp.SetLineStyle(ROOT.kSolid)
        rp.SetLineWidth(2)
        rp.SetMarkerSize(0)

        h_true_ratio.append(rt)
        h_pred_ratio.append(rp)
        keep.extend([rt, rp])

    h_ratio_frame = ROOT.TH1F(f"h_ratio_frame_{feat}", "", n_bins, bin_edges_root)
    h_ratio_frame.SetDirectory(0)
    h_ratio_frame.SetTitle("")
    h_ratio_frame.GetYaxis().SetTitle("var / nominal")
    h_ratio_frame.GetYaxis().SetNdivisions(505)
    h_ratio_frame.GetYaxis().SetTitleSize(0.09)
    h_ratio_frame.GetYaxis().SetTitleOffset(0.5)
    h_ratio_frame.GetYaxis().SetLabelSize(0.08)
    h_ratio_frame.GetXaxis().SetTitle(x_title)
    h_ratio_frame.GetXaxis().SetTitleSize(0.10)
    h_ratio_frame.GetXaxis().SetLabelSize(0.08)

    max_dev = 0.0
    for hr in (h_true_ratio + h_pred_ratio):
        for ib in range(1, n_bins + 1):
            val = float(hr.GetBinContent(ib))
            if val != 0.0 and math.isfinite(val):
                max_dev = max(max_dev, abs(val - 1.0))

    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    h_ratio_frame.SetMinimum(r_min)
    h_ratio_frame.SetMaximum(r_max)
    h_ratio_frame.Draw("AXIS")
    keep.append(h_ratio_frame)

    for rp in h_pred_ratio:
        rp.Draw("HIST SAME")
    for rt in h_true_ratio:
        rt.Draw("E1 SAME")

    line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("SAME")
    keep.append(line)

    # LEGEND PAD: mapping line on top, legend below it
    padLegend.cd()

    map_tex = ROOT.TLatex()
    map_tex.SetNDC()
    map_tex.SetTextAlign(13)  # left, top
    map_tex.SetTextSize(0.16)  # mapping line only; pad height varies
    map_tex.DrawLatex(0.04, 0.92, legend_mapping_line)
    keep.append(map_tex)

    legend.Draw()

    c.cd()
    c.Update()

    out_png = os.path.join(plot_dir, f"{feat}.png")
    out_pdf = os.path.join(plot_dir, f"{feat}.pdf")
    c.SaveAs(out_png)
    c.SaveAs(out_pdf)

    feature_keep[feat] = keep

syncer.sync()
print("Done.")

