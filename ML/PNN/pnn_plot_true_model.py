#!/usr/bin/env python
from __future__ import annotations
import os, sys, argparse, importlib, yaml, math
import numpy as np

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer

# Plot options (binning, labels, optional y_ratio_range, logY)
from data.plot_options import plot_options as PLOT_OPTS

# ---------------- args ----------------
p = argparse.ArgumentParser(description="Plot TRUE variations only from a PNN job (no predictions).")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", required=True, help="PNN job id to read base-point loaders from")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--rebin", type=int, default=1, help="Coarsen histograms by this factor")
p.add_argument("--normalized", action="store_true", help="Also produce ratio-to-nominal plots")
p.add_argument("--out-tag", default="truth_only", help="Subfolder tag for plots")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
with open(cfg_path, "r") as f:
    CFG = yaml.safe_load(f) or {}
D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

# ---------------- locate job ----------------
J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "pnn":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'pnn'.")

# ---------------- resolve base-point loaders ----------------
samples_mod = importlib.import_module(module_samples)

bp_specs     = J["base_points"]  # list of {coords: [...], loader: "name"}
base_points  = [list(spec["coords"]) for spec in bp_specs]
loader_names = [spec["loader"] for spec in bp_specs]

loaders = []
for nm in loader_names:
    if not hasattr(samples_mod, nm):
        raise RuntimeError(f"Loader/view '{nm}' not found in module {module_samples}.")
    loaders.append(getattr(samples_mod, nm))

# ---------------- feature layout ----------------
feat_names = list(getattr(loaders[0], "feature_names", []))
if not feat_names:
    raise RuntimeError("First loader has no feature_names.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != feat_names:
        raise RuntimeError("Feature mismatch across base-point loaders.")
feat2col = {f: i for i, f in enumerate(feat_names)}

# Only plot features known to PLOT_OPTS (silently skip others)
plot_feats = [f for f in feat_names if f in PLOT_OPTS]
if not plot_feats:
    print("Warning: no features in plot_options; nothing to plot.")

# ---------------- output dirs ----------------
cfg_base = os.path.splitext(os.path.basename(cfg_path))[0]
plot_dir = os.path.join(user.plot_directory, "PNN", cfg_base, J["id"] + f"_{args.out_tag}")
os.makedirs(plot_dir, exist_ok=True)

# ---------------- set up hist containers (explicit) ----------------
# bins per-feature (after rebin), and histograms: shape (n_bins, n_base_points)
bins   = {}
h_true = {}
for feat in plot_feats:
    n, lo, hi = PLOT_OPTS[feat]['binning']
    n = max(1, n // max(1, int(args.rebin)))
    bins[feat]   = np.linspace(lo, hi, n + 1)
    h_true[feat] = np.zeros((n, len(base_points)), dtype=np.float64)

# ---------------- explicit data loop (no helper wrappers) ----------------
# Choose the number of shards we can iterate jointly
shard_counts = [len(getattr(L, "base", L)) for L in loaders]
n_shards     = min(shard_counts)
if args.small:
    n_shards = min(n_shards, 1)

print(f"[info] Iterating over {n_shards} shard(s) across {len(loaders)} base points.")
for shard in range(n_shards):
    # Materialize (X, w) for each base point explicitly
    Xs, Ws = [], []
    for i_bp, L in enumerate(loaders):
        # SelectionView implements materialize('fw'); base RDataLoader does too
        X, w = L.materialize(shard=shard, what="fw")
        # (X: (N, F), w: (N,))
        Xs.append(X); Ws.append(w.astype(np.float64, copy=False))

        # <-- At this point you can intervene in IPython:
        #     e.g., inspect first few rows/weights:
        # if shard == 0 and i_bp == 0:
        #     import pandas as pd; print(pd.DataFrame(X[:5], columns=feat_names))
        #     print("weights head:", w[:10])

    # Accumulate histograms per-feature, per-basepoint
    for i_bp, (Xi, wi) in enumerate(zip(Xs, Ws)):
        for feat in plot_feats:
            col = feat2col[feat]
            ht, _ = np.histogram(Xi[:, col], bins=bins[feat], weights=wi)
            h_true[feat][:, i_bp] += ht

# ---------------- determine nominal index (0-vector coords) ----------------
nom_idx = None
for i, coords in enumerate(base_points):
    if all(abs(c) < 1e-12 for c in coords):
        nom_idx = i
        break
if nom_idx is None:
    # fall back: try YAML nominal_index if you kept that convention
    if "nominal_index" in J:
        nom_idx = int(J["nominal_index"])
    else:
        # otherwise assume middle
        nom_idx = len(base_points) // 2
print(f"[info] Nominal base-point index = {nom_idx}, coords = {base_points[nom_idx]}")

# ---------------- plotting (ROOT), TRUE ONLY ----------------
import ROOT
try:
    ROOT.gStyle.SetOptStat(0)
    # Try to load TDR (ignore if absent)
    here = os.path.dirname(os.path.realpath(__file__))
    ROOT.gROOT.LoadMacro(os.path.join(here, "../../common/scripts/tdrstyle.C"))
    ROOT.setTDRStyle()
except Exception:
    pass

def draw_panel(normalized: bool):
    n_feat = len(plot_feats)
    n_bp   = len(base_points)

    # Pick colors; set nominal to black
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange+1, ROOT.kMagenta+1, ROOT.kCyan+2,
              ROOT.kViolet+1, ROOT.kAzure+1, ROOT.kPink+7, ROOT.kTeal+3]
    if n_bp > len(colors):
        colors = (colors * (n_bp // len(colors) + 1))[:n_bp]
    colors[nom_idx] = ROOT.kBlack

    # Prepare (optionally normalized) copies
    th = {k: v.copy() for k, v in h_true.items()}
    if normalized:
        for feat in plot_feats:
            ref = th[feat][:, nom_idx].copy()
            ref[ref == 0.0] = 1.0
            th[feat] = th[feat] / ref[:, None]

    total_pads = n_feat + 1
    gx = int(math.ceil(math.sqrt(total_pads)))
    gy = int(math.ceil(total_pads / gx))
    canvas = ROOT.TCanvas("c_truth", "True Variations", 500*gx, 500*gy)
    canvas.Divide(gx, gy)

    keep = []
    for i, feat in enumerate(plot_feats):
        pad = canvas.cd(i + 1)
        pad.SetTicks(1, 1)
        pad.SetBottomMargin(0.15)
        pad.SetLeftMargin(0.15)

        # Log only on raw if requested
        pad.SetLogy((not normalized) and PLOT_OPTS[feat].get("logY", False))

        n_bins, x_min, x_max = PLOT_OPTS[feat]["binning"]
        n_bins = max(1, n_bins // max(1, int(args.rebin)))

        max_y = 0.0
        for k in range(n_bp):
            max_y = max(max_y, th[feat][:, k].max())

        if normalized:
            y_title = "Ratio to nominal"
            # data-driven axis from truth only
            arr = th[feat]
            finite = np.isfinite(arr)
            if not finite.any():
                tmin, tmax = 0.95, 1.05
            else:
                tmin = float(np.min(arr[finite])); tmax = float(np.max(arr[finite]))
                if not np.isfinite(tmin) or not np.isfinite(tmax) or abs(tmax - tmin) < 1e-9:
                    tmin, tmax = 0.95, 1.05
            span = max(1e-9, tmax - tmin)
            pad_frac = 0.10
            y_min = max(0.0, tmin - pad_frac * span)
            y_max = tmax + pad_frac * span
        else:
            y_title = "Weighted counts"
            max_y = 0.0
            for k in range(n_bp):
                max_y = max(max_y, th[feat][:, k].max())
            y_min, y_max = 0.0, (1.2*max_y if max_y > 0 else 1.0)

        hframe = ROOT.TH2F(f"h_{feat}", f";{PLOT_OPTS[feat]['tex']};{y_title}",
                           n_bins, x_min, x_max, 100, y_min, y_max)
        hframe.GetYaxis().SetTitleOffset(1.3)
        hframe.Draw()
        keep.append(hframe)

        # Draw TRUE histograms only (no predictions)
        for k in range(n_bp):
            h = ROOT.TH1F(f"t_{feat}_{k}", "", n_bins, x_min, x_max)
            for b, y in enumerate(th[feat][:, k]): h.SetBinContent(b+1, y)
            h.SetLineColor(colors[k]); h.SetLineStyle(1 if k == nom_idx else 2); h.SetLineWidth(2)
            h.Draw("HIST SAME")
            keep.append(h)

    # Legend
    pad = canvas.cd(n_feat + 1)
    leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9); leg.SetBorderSize(0); leg.SetShadowColor(0)
    leg.SetNColumns(1 + n_bp//20)
    d = []
    for k, nu in enumerate(base_points):
        h = ROOT.TH1F(f"dt_{k}", "", 1, 0, 1)
        h.SetLineColor(colors[k]); h.SetLineStyle(1 if k == nom_idx else 2); h.SetLineWidth(2)
        leg.AddEntry(h, f"{tuple(nu)} (true)", "l")
        d.append(h)
    leg.Draw(); keep.extend(d)

    tag = "norm_" if normalized else ""
    fname = os.path.join(plot_dir, f"{tag}truth_epoch_0000.png")
    for fmt in ["png"]:
        canvas.SaveAs(fname.replace(".png", f".{fmt}"))
    return fname

# Draw raw
raw_png = draw_panel(normalized=False)
print(f"[info] wrote {raw_png}")

# Draw normalized (if requested)
if args.normalized:
    ratio_png = draw_panel(normalized=True)
    print(f"[info] wrote {ratio_png}")

print("[done] Truth-only plotting finished.")

