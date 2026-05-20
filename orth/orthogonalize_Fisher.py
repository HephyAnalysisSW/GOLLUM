#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib

import numpy as np
from tqdm import tqdm

sys.path.insert(0, '..')

from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood
from fit.N2LLExtensions import N2LLExtensions

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

import common.yaml_loader as yaml_loader
import common.syncer as syncer
import common.helpers as helpers
import common.user as user

from data.plot_options import plot_options


p = argparse.ArgumentParser(
    description="Build Fisher eigenbasis in POI space and make diagnostic plots."
)
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--plot_directory", default="FI-orth", help="plot sub-directory")
p.add_argument("--cache-root", default=None, help="Override cache root (optional)")
p.add_argument("--overwrite", action="store_true", help="Overwrite caches")
p.add_argument("--fi-step-scale", type=float, default=1e-4,
               help="Step scale for Fisher binned/penalty finite differences")
args = p.parse_args()

n_sigma_plot = 0.2

# ------------------------------------------------------------------
# Load config and surrogates
# ------------------------------------------------------------------
cfg = yaml_loader.load_yaml(args.config)
yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, args.config, overwrite=False)

like_info = load_likelihood(cfg)
hyp = build_hypothesis_from_likelihood(like_info, name="SR")

# freeze all nuisances
for par in hyp.nuisances:
    par.val = 0.0
    par.freeze()
print("[opts] all nuisances set to 0 and frozen.")

hyp.print()

# free POIs only
poi_free_params = [p for p in hyp.parameters
                   if p.isPOI and not p.isFrozen and not getattr(p, "isIgnored", False)]
poi_names = [p.name for p in poi_free_params]
n_poi = len(poi_names)
if n_poi == 0:
    raise RuntimeError("No free POIs found.")

print("\n[free] POIs to probe:")
for nm in poi_names:
    print("   -", nm)

# ------------------------------------------------------------------
# Build sample factory
# ------------------------------------------------------------------
samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
from common.yaml_loader import _resolve_features_list

default_features = cfg["defaults"].get("default_features", None)
features = _resolve_features_list(default_features) if default_features else None

factory = samples_mod.Factory(
    features=features,
    selection=cfg["defaults"].get("default_selection", None),
    selection_features=cfg["defaults"].get("default_selection_features", None),
)

# ------------------------------------------------------------------
# Build caches and runtime
# ------------------------------------------------------------------
n2ll = N2LLExtensions(
    like_info,
    factory=factory,
    cache_subdir=os.path.join(
        "NN2LCache",
        str(cfg.get("version", "v0")),
        os.path.splitext(os.path.basename(args.config))[0],
    ),
    cache_root=args.cache_root,
    overwrite=args.overwrite,
)
n2ll.build_cache()
n2ll.prepare_runtime()

# ------------------------------------------------------------------
# Null Asimov and baseline
# ------------------------------------------------------------------
n2ll.setAsimov()
f0 = float(n2ll(hyp))
print(f"\n[baseline] -2 log L (Asimov, at 0) = {f0:.6e}")

# ------------------------------------------------------------------
# Fisher information
# ------------------------------------------------------------------
print("\n[FI] Computing Fisher information at (0,0) in Asimov mode...")
M = n2ll.fisher_information(hyp, step_scale=args.fi_step_scale, verbose=True)

# Since nuisances are frozen, M should already be in POI space.
# Still, protect against accidental mismatch.
if M.shape != (n_poi, n_poi):
    raise RuntimeError(
        f"Fisher matrix shape {M.shape} does not match number of free POIs {n_poi}."
    )

M = 0.5 * (M + M.T)

print("\n[rotation] Fisher matrix in POI space:")
print(M)

# ------------------------------------------------------------------
# Stable Fisher eigenbasis
# c = V_new @ d
# d = D @ c
# with D = V_new^T
# ------------------------------------------------------------------
evals, U = np.linalg.eigh(M)          # ascending
order = np.argsort(evals)[::-1]       # descending
evals = evals[order]
U = U[:, order]

# deterministic sign convention
for j in range(U.shape[1]):
    i_max = np.argmax(np.abs(U[:, j]))
    if U[i_max, j] < 0:
        U[:, j] *= -1.0

V_new = U
D_matrix = V_new.T
labels_d = [f"eig_{i}" for i in range(n_poi)]

sigma_d = np.array(
    [(1.0 / np.sqrt(lam)) if lam > 0 else np.inf for lam in evals],
    dtype=np.float64
)
lam_rel = evals / evals[0] if evals[0] > 0 else np.full_like(evals, np.nan)

print("\n=== Stable Fisher eigenbasis summary ===")
print("POI order:", poi_names)
print("\nEigenvalues (descending):")
for j, lam in enumerate(evals):
    print(f"  {labels_d[j]:>8s}: {lam:.6e}")

np.set_printoptions(precision=4, suppress=True)
print("\nEigenvectors V_new (columns are modes in c-space):")
for j, lab in enumerate(labels_d):
    print(f"{lab:>8s}: {V_new[:, j]}")

print("\n[rotation] d = D · c")
print("with c in POI order:", poi_names)
print("D =")
print(D_matrix)

print("\n[rotation] Local 1σ step sizes in d-coordinates:")
for j, sig in enumerate(sigma_d):
    print(f"  {labels_d[j]:>8s}: {sig:.6e}")

print("\n[rotation] Relative eigenvalues λ_j / λ_max:")
for j, r in enumerate(lam_rel):
    print(f"  {labels_d[j]:>8s}: {r:.6e}")

# ------------------------------------------------------------------
# Save basis
# ------------------------------------------------------------------
out_dir = getattr(user, "output_directory", "./outputs")
os.makedirs(out_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.config))[0]
rot_path_json = os.path.join(out_dir, f"eigen_basis_{base_name}_{cfg['version']}.json")
rot_path_npz = os.path.join(out_dir, f"eigen_basis_{base_name}_{cfg['version']}.npz")

payload = {
    "config": os.path.basename(args.config),
    "poi_order": poi_names,
    "basis_labels": labels_d,
    "M": M.tolist(),
    "eigenvalues": evals.tolist(),
    "relative_eigenvalues": lam_rel.tolist(),
    "sigma_d": sigma_d.tolist(),
    "V_new": V_new.tolist(),
    "D": D_matrix.tolist(),
}
with open(rot_path_json, "w") as fjs:
    json.dump(payload, fjs, indent=2)

np.savez(
    rot_path_npz,
    poi_order=np.array(poi_names, dtype=object),
    basis_labels=np.array(labels_d, dtype=object),
    M=M,
    eigenvalues=evals,
    relative_eigenvalues=lam_rel,
    sigma_d=sigma_d,
    V_new=V_new,
    D=D_matrix,
)

# ------------------------------------------------------------------
# Feature plots along eigen-directions
# ------------------------------------------------------------------
plot_directory = os.path.join(
    user.plot_directory,
    args.plot_directory,
    cfg['version'],
    os.path.splitext(os.path.basename(args.config))[0],
)
os.makedirs(plot_directory, exist_ok=True)

print(f"\n[write] Saved Fisher eigenbasis to:\n  {rot_path_json}\n  {rot_path_npz}")

# ------------------------------------------------------------------
# ROOT summary plot of eigensystem
# ------------------------------------------------------------------
c_eig = ROOT.TCanvas("c_fisher_eigensystem", "", 1500, 800)
padL = ROOT.TPad("padL", "padL", 0.00, 0.00, 0.30, 1.00)
padR = ROOT.TPad("padR", "padR", 0.30, 0.00, 1.00, 1.00)

padL.SetLeftMargin(0.08)
padL.SetRightMargin(0.03)
padL.SetTopMargin(0.06)
padL.SetBottomMargin(0.06)

padR.SetLeftMargin(0.14)
padR.SetRightMargin(0.08)
padR.SetTopMargin(0.08)
padR.SetBottomMargin(0.14)
padR.SetLogy(True)

padL.Draw()
padR.Draw()

poi_colors = [
    ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kOrange + 1,
    ROOT.kCyan + 2, ROOT.kViolet + 1, ROOT.kTeal + 1, ROOT.kPink + 1,
    ROOT.kAzure + 2, ROOT.kSpring + 5, ROOT.kYellow + 2, ROOT.kGray + 1,
]

padL.cd()
txt = ROOT.TLatex()
txt.SetNDC(True)
txt.SetTextFont(42)

y = 0.95
txt.SetTextSize(0.040)
txt.DrawLatex(0.04, y, "Fisher eigenbasis summary")
y -= 0.065

txt.SetTextSize(0.032)
txt.DrawLatex(0.04, y, f"N_{{POI}} = {n_poi}")
y -= 0.045
txt.DrawLatex(0.04, y, f"#lambda_{{max}} = {evals[0]:.3e}")
y -= 0.045
txt.DrawLatex(0.04, y, f"#lambda_{{min}} = {evals[-1]:.3e}")
y -= 0.045
txt.DrawLatex(0.04, y, f"cond. #approx {evals[0]/evals[-1]:.3e}" if evals[-1] > 0 else "cond. undefined")
y -= 0.060

txt.SetTextSize(0.030)
txt.DrawLatex(0.04, y, "Eigenvalues:")
y -= 0.045

txt.SetTextSize(0.028)
for k in range(min(n_poi, 8)):
    txt.DrawLatex(0.07, y, f"{labels_d[k]}: {evals[k]:.3e}")
    y -= 0.037

leg = ROOT.TLegend(0.04, 0.04, 0.96, 0.22)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetNColumns(3)
leg_boxes = []
for i, name in enumerate(poi_names):
    b = ROOT.TBox(0, 0, 1, 1)
    b.SetFillColor(poi_colors[i % len(poi_colors)])
    b.SetLineColor(poi_colors[i % len(poi_colors)])
    leg.AddEntry(b, name, "f")
    leg_boxes.append(b)
leg.Draw()

padR.cd()
lam_pos = evals[evals > 0]
if len(lam_pos) == 0:
    raise RuntimeError("No positive eigenvalues to plot.")

ymin = 0.5 * np.min(lam_pos)
ymax = 2.0 * np.max(lam_pos)
log_span = np.log10(ymax / ymin)
half_bar_decades = 0.01 * log_span

def yband(yv):
    return yv / (10.0 ** half_bar_decades), yv * (10.0 ** half_bar_decades)

frame = ROOT.TH2D("frame_eig", "", 100, 0.0, 1.0, 100, ymin, ymax)
frame.SetDirectory(0)
frame.GetXaxis().SetTitle("mode composition fraction  |v_{i}|^{2}")
frame.GetYaxis().SetTitle("Fisher eigenvalue")
frame.GetXaxis().SetTitleSize(0.050)
frame.GetYaxis().SetTitleSize(0.050)
frame.GetXaxis().SetTitleOffset(1.15)
frame.GetYaxis().SetTitleOffset(1.25)
frame.GetXaxis().SetLabelSize(0.042)
frame.GetYaxis().SetLabelSize(0.042)
frame.Draw()

guide_lines = []
bar_boxes = []
mode_labels = []

for k, lam in enumerate(evals):
    y1, y2 = yband(lam)

    gl = ROOT.TLine(0.0, lam, 1.0, lam)
    gl.SetLineColor(ROOT.kGray + 1)
    gl.SetLineStyle(ROOT.kDashed)
    gl.Draw()
    guide_lines.append(gl)

    weights = np.abs(V_new[:, k]) ** 2
    weights /= np.sum(weights)

    x0 = 0.0
    for i, w in enumerate(weights):
        x1 = x0 + float(w)
        box = ROOT.TBox(x0, y1, x1, y2)
        box.SetFillColor(poi_colors[i % len(poi_colors)])
        box.SetLineColor(poi_colors[i % len(poi_colors)])
        box.Draw()
        bar_boxes.append(box)
        x0 = x1

    lab = ROOT.TLatex()
    lab.SetTextFont(42)
    lab.SetTextSize(0.028)
    lab.DrawLatex(1.01, lam, labels_d[k])
    mode_labels.append(lab)

c_eig.Update()

eig_plot_png = os.path.join(plot_directory, f"eigen_basis_{base_name}_{cfg['version']}.png")
eig_plot_pdf = os.path.join(plot_directory, f"eigen_basis_{base_name}_{cfg['version']}.pdf")
c_eig.SaveAs(eig_plot_png)
c_eig.SaveAs(eig_plot_pdf)

print(f"[write] Saved Fisher eigensystem plot to:\n  {eig_plot_png}\n  {eig_plot_pdf}")


basis_colors = [poi_colors[i % len(poi_colors)] for i in range(n_poi)]
stuff = []

print("\n[plots] Making feature distributions with Fisher eigen-directions (per region)…")

def make_hist_feature(weights, feature_idx, feature_name, region_features):
    bins = plot_options[feature_name]['binning']
    arr = region_features[:, feature_idx]
    hist, edges = np.histogram(
        arr,
        np.linspace(bins[1], bins[2], bins[0] + 1),
        weights=weights
    )
    return helpers.make_TH1F((hist, edges))

def clone_and_divide(num, den):
    h = num.Clone(f"{num.GetName()}__ratio")
    h.Divide(den)
    return h

def build_hyp_along_direction(j, alpha_d):
    h_step = hyp.clone()
    c_step = alpha_d * V_new[:, j]
    for a, pname in enumerate(poi_names):
        getattr(h_step, pname).val = float(c_step[a])
    return h_step

for R in n2ll.regions:
    rid = R["id"]
    asimov_list = R.get("_asimov_samples", [])
    if not asimov_list:
        continue

    print(f"\n[plots:{rid}] Collecting A-simov events and nominal weights…")
    region_feats = []
    region_w0s = []
    feature_names_ref = None

    for sname in asimov_list:
        L = n2ll.factory.get(sname)
        feat_names = list(getattr(L, "feature_names", []) or [])
        if feature_names_ref is None:
            feature_names_ref = feat_names
        elif feat_names != feature_names_ref:
            raise RuntimeError(f"[plots:{rid}] Feature mismatch across Asimov samples.")

        nsplits = int(getattr(L, "n_split", 1))
        for shard in range(nsplits):
            X, w0 = L.materialize(shard=shard, what="fw", n=None)
            if X is None or len(X) == 0:
                continue
            region_feats.append(np.asarray(X, dtype=np.float64))
            region_w0s.append(np.asarray(w0, dtype=np.float64))

    if not region_feats:
        print(f"[plots:{rid}] No events found; skipping.")
        continue

    region_features = np.concatenate(region_feats, axis=0)
    w0 = np.concatenate(region_w0s, axis=0)
    print(f"[plots:{rid}] N = {region_features.shape[0]} events; {len(feature_names_ref)} features.")

    _ = n2ll.evaluate_ratio(rid, region_features, feature_names_ref, hyp, cached=True, return_T=False)

    name2idx = {n: i for i, n in enumerate(feature_names_ref)}
    plot_feature_names = [f for f in plot_options.keys() if f in name2idx]
    print(f"[plots:{rid}] #features to plot = {len(plot_feature_names)}")

    for feature_name in tqdm(plot_feature_names, desc=f"[plots:{rid}] features", unit="feat"):
        feature_idx = name2idx[feature_name]

        h_nom = make_hist_feature(w0, feature_idx, feature_name, region_features)
        h_nom.SetLineColor(ROOT.kGray + 2)
        h_nom.SetMarkerStyle(0)
        h_nom.SetLineWidth(2)

        h_dirs = []
        for j in range(n_poi):
            alpha = n_sigma_plot*sigma_d[j] if np.isfinite(sigma_d[j]) else 0.0

            if alpha == 0.0:
                w_step = w0.copy()
            else:
                h_step = build_hyp_along_direction(j, alpha)
                Rj = n2ll.evaluate_ratio(rid, region_features, feature_names_ref, h_step, cached=True, return_T=False)
                w_step = w0 * np.asarray(Rj, dtype=np.float64)

            h_j = make_hist_feature(w_step, feature_idx, feature_name, region_features)
            h_j.SetLineColor(basis_colors[j])
            h_j.SetLineWidth(2)
            h_j.SetMarkerStyle(0)
            h_dirs.append(h_j)

        # yields
        for logY in (False, True):
            c = ROOT.TCanvas(f"c_yield_{rid}_{feature_name}_{int(logY)}", "", 800, 650)
            leg = ROOT.TLegend(0.20, 0.76, 0.90, 0.88)
            leg.SetNColumns(3)
            leg.SetFillStyle(0)
            leg.SetShadowColor(ROOT.kWhite)
            leg.SetBorderSize(0)

            h_nom.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
            h_nom.GetYaxis().SetTitle("events")
            h_nom.Draw("hist")
            leg.AddEntry(h_nom, f"{rid}: nominal", "l")

            for j, hj in enumerate(h_dirs):
                hj.Draw("histsame")
                leg.AddEntry(hj, f"{labels_d[j]} ({n_sigma_plot}#sigma)", "l")

            stuff.append(leg)
            leg.Draw()
            ROOT.gPad.SetLogy(logY)
            c.Update()

            #outdir = os.path.join(plot_directory, rid, "basis_yield", "log" if logY else "lin")
            #os.makedirs(outdir, exist_ok=True)
            helpers.copyIndexPHP(plot_directory)
            c.Print(os.path.join(plot_directory, f"{feature_name}.png"))
            c.Print(os.path.join(plot_directory, f"{feature_name}.pdf"))
            c.Close()

        # ratios
        c = ROOT.TCanvas(f"c_ratio_{rid}_{feature_name}", "", 800, 650)
        leg = ROOT.TLegend(0.20, 0.76, 0.90, 0.88)
        leg.SetNColumns(3)
        leg.SetFillStyle(0)
        leg.SetShadowColor(ROOT.kWhite)
        leg.SetBorderSize(0)

        h_unity = clone_and_divide(h_nom, h_nom)
        h_unity.SetLineColor(ROOT.kGray + 1)
        h_unity.SetLineStyle(2)
        h_unity.SetLineWidth(2)
        h_unity.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
        h_unity.GetYaxis().SetTitle("variation / nominal")
        h_unity.SetMinimum(0.95)
        h_unity.SetMaximum(1.05)
        h_unity.SetTitle("")
        h_unity.Draw("hist")
        leg.AddEntry(h_unity, "nominal", "l")

        for j, hj in enumerate(h_dirs):
            h_ratio = clone_and_divide(hj, h_nom)
            h_ratio.SetLineColor(basis_colors[j])
            h_ratio.SetLineWidth(2)
            h_ratio.SetMarkerStyle(0)
            h_ratio.SetTitle("")
            h_ratio.Draw("histsame")
            stuff.append(h_ratio)
            leg.AddEntry(h_ratio, f"{labels_d[j]} (1#sigma)", "l")

        stuff.append(leg)
        leg.Draw()
        c.Modified()
        c.Update()

        #outdir = os.path.join(plot_directory, rid, "basis_ratio", "lin")
        #os.makedirs(outdir, exist_ok=True)
        #helpers.copyIndexPHP(outdir)
        c.Print(os.path.join(plot_directory, f"{feature_name}.png"))
        c.Print(os.path.join(plot_directory, f"{feature_name}.pdf"))
        c.Close()

print("\n[done] Yield and ratio plots written under:")
print(f"  {os.path.join(plot_directory, '<region>', 'basis_yield')}")
print(f"  {os.path.join(plot_directory, '<region>', 'basis_ratio')}")

# ------------------------------------------------------------------
# Binned template plots along Fisher eigen-directions
# ------------------------------------------------------------------
binned_regions = list(like_info.get("binned", []) or [])
region_binned_canvases = {}

def region_total_binned_template(region, h_eval):
    total = None
    raw_shape = None

    for cls in region.get("classes", []) or []:
        poi = cls.get("POI", {}) or {}
        pred = poi.get("predictor", None)
        if pred is None:
            raise RuntimeError(f"Missing ICH predictor for {region['id']}/{cls.get('id', '?')}")

        par_names = list(poi.get("parameters", []) or [])
        cvec = np.array([float(getattr(h_eval, name).val) for name in par_names], dtype=np.float64)

        vals_raw = np.asarray(pred.predict(cvec), dtype=np.float64)
        if raw_shape is None:
            raw_shape = vals_raw.shape

        if vals_raw.ndim == 1:
            vals = vals_raw.copy()
        elif vals_raw.ndim == 2:
            vals = vals_raw.reshape(-1).copy()
        else:
            raise RuntimeError(
                f"Expected 1D or 2D ICH prediction in region {region['id']}, got shape {vals_raw.shape}"
            )

        if total is None:
            total = np.zeros_like(vals, dtype=np.float64)
        elif vals.shape != total.shape:
            raise RuntimeError(
                f"Inconsistent ICH shape in region {region['id']}: got {vals.shape}, expected {total.shape}"
            )

        total += vals

    if total is None:
        raise RuntimeError(f"Region {region['id']} has no plottable binned classes.")

    return total, raw_shape

if binned_regions:
    print("\n[plots] Making binned rotated-basis template plots...")

for region in binned_regions:
    region_id = region["id"]
    print(f"[plots:binned] Region {region_id}")

    central, raw_shape = region_total_binned_template(region, hyp)
    n_bins = len(central)

    if len(raw_shape) == 2:
        nb1, nb2 = raw_shape
        separators = [i * nb2 + 0.5 for i in range(1, nb1)]
        label_extra = f"unrolled ({nb1} x {nb2})"
    else:
        separators = []
        label_extra = "1D"

    variations = [("nominal", central)]
    for j in range(n_poi):
        alpha = sigma_d[j] if np.isfinite(sigma_d[j]) else 0.0
        if alpha == 0.0:
            vals = central.copy()
        else:
            h_step = build_hyp_along_direction(j, alpha)
            vals, _ = region_total_binned_template(region, h_step)
        variations.append((f"{labels_d[j]} (1#sigma)", vals))

    c = ROOT.TCanvas(f"c_binned_basis_{region_id}", "", 900, 850)
    pad_top = ROOT.TPad(c.GetName() + "_top", c.GetName() + "_top", 0.0, 0.30, 1.0, 1.0)
    pad_bot = ROOT.TPad(c.GetName() + "_bot", c.GetName() + "_bot", 0.0, 0.00, 1.0, 0.30)

    pad_top.SetBottomMargin(0.02)
    pad_top.SetTopMargin(0.08)
    pad_top.SetLeftMargin(0.12)
    pad_top.SetRightMargin(0.04)
    pad_top.SetTicks(1, 1)

    pad_bot.SetTopMargin(0.03)
    pad_bot.SetBottomMargin(0.30)
    pad_bot.SetLeftMargin(0.12)
    pad_bot.SetRightMargin(0.04)
    pad_bot.SetTicks(1, 1)

    pad_top.Draw()
    pad_bot.Draw()

    top_hists = []
    ratio_hists = []
    ymax = 0.0
    positive_min = None
    all_ratio_vals = []

    for idx, (label, vals) in enumerate(variations):
        h = ROOT.TH1D(f"h_binned_top_{region_id}_{idx}", "", n_bins, 0.5, n_bins + 0.5)
        h.SetDirectory(0)
        for ib, val in enumerate(vals, start=1):
            h.SetBinContent(ib, float(val))
            h.SetBinError(ib, 0.0)

        h.SetLineColor(ROOT.kBlack if idx == 0 else basis_colors[idx - 1])
        h.SetLineWidth(3 if idx == 0 else 2)
        h.SetLineStyle(ROOT.kSolid)
        h.SetTitle("")
        top_hists.append(h)

        ymax = max(ymax, float(np.max(vals)) if len(vals) else 0.0)
        pos = vals[vals > 0.0]
        if len(pos):
            cand = float(np.min(pos))
            positive_min = cand if positive_min is None else min(positive_min, cand)

        r = np.ones_like(vals, dtype=np.float64)
        mask = np.abs(central) > 1e-15
        r[mask] = vals[mask] / central[mask]
        all_ratio_vals.append(r)

        hr = ROOT.TH1D(f"h_binned_ratio_{region_id}_{idx}", "", n_bins, 0.5, n_bins + 0.5)
        hr.SetDirectory(0)
        for ib, val in enumerate(r, start=1):
            hr.SetBinContent(ib, float(val))
            hr.SetBinError(ib, 0.0)

        hr.SetLineColor(ROOT.kBlack if idx == 0 else basis_colors[idx - 1])
        hr.SetLineWidth(3 if idx == 0 else 2)
        hr.SetLineStyle(ROOT.kSolid)
        hr.SetTitle("")
        ratio_hists.append(hr)

    pad_top.cd()
    first = top_hists[0]
    ymin = 0.0
    ymax_draw = 1.25 * ymax if ymax > 0.0 else 1.0

    first.SetMinimum(ymin)
    first.SetMaximum(ymax_draw)
    first.GetYaxis().SetTitle("ICH prediction")
    first.GetYaxis().SetTitleSize(0.055)
    first.GetYaxis().SetTitleOffset(1.05)
    first.GetYaxis().SetLabelSize(0.045)
    first.GetXaxis().SetLabelSize(0.0)
    first.GetXaxis().SetTitleSize(0.0)
    first.Draw("HIST")
    for h in top_hists[1:]:
        h.Draw("HIST SAME")

    top_sep_lines = []
    for x in separators:
        line = ROOT.TLine(x, ymin, x, ymax_draw)
        line.SetLineColor(ROOT.kGray + 1)
        line.SetLineStyle(ROOT.kDashed)
        line.SetLineWidth(1)
        line.Draw("SAME")
        top_sep_lines.append(line)

    leg = ROOT.TLegend(0.58, 0.60, 0.94, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(1 if len(variations) <= 7 else 2)
    for h, (label, _) in zip(top_hists, variations):
        leg.AddEntry(h, label, "l")
    leg.Draw()

    label_top = ROOT.TLatex()
    label_top.SetNDC(True)
    label_top.SetTextSize(0.032)
    label_top.DrawLatex(0.11, 0.93, f"{region_id}   |   rotated Fisher basis   |   {label_extra}")

    pad_bot.cd()
    ratio0 = ratio_hists[0]
    ratio_vals = np.concatenate([np.asarray(x, dtype=np.float64) for x in all_ratio_vals])
    finite = ratio_vals[np.isfinite(ratio_vals)]

    if len(finite):
        rmin = float(np.min(finite))
        rmax = float(np.max(finite))
    else:
        rmin, rmax = 0.9, 1.1

    if not np.isfinite(rmin) or not np.isfinite(rmax) or abs(rmax - rmin) < 1e-6:
        rmin, rmax = 0.9, 1.1
    else:
        half = max(abs(rmax - 1.0), abs(1.0 - rmin))
        half *= 1.15
        rmin = 1.0 - half
        rmax = 1.0 + half

    ratio0.SetMinimum(rmin)
    ratio0.SetMaximum(rmax)
    ratio0.GetYaxis().SetTitle("var / nominal")
    ratio0.GetYaxis().SetTitleSize(0.10)
    ratio0.GetYaxis().SetTitleOffset(0.55)
    ratio0.GetYaxis().SetLabelSize(0.085)
    ratio0.GetYaxis().SetNdivisions(505)
    ratio0.GetXaxis().SetTitle("unrolled bin")
    ratio0.GetXaxis().SetTitleSize(0.11)
    ratio0.GetXaxis().SetTitleOffset(1.0)
    ratio0.GetXaxis().SetLabelSize(0.085)
    ratio0.Draw("HIST")
    for h in ratio_hists[1:]:
        h.Draw("HIST SAME")

    bot_sep_lines = []
    for x in separators:
        line = ROOT.TLine(x, rmin, x, rmax)
        line.SetLineColor(ROOT.kGray + 1)
        line.SetLineStyle(ROOT.kDashed)
        line.SetLineWidth(1)
        line.Draw("SAME")
        bot_sep_lines.append(line)

    unit = ROOT.TLine(0.5, 1.0, n_bins + 0.5, 1.0)
    unit.SetLineColor(ROOT.kBlack)
    unit.SetLineStyle(ROOT.kDashed)
    unit.SetLineWidth(1)
    unit.Draw("SAME")

    c.Update()

    out_png = os.path.join(plot_directory, f"{region_id}.png")
    out_pdf = os.path.join(plot_directory, f"{region_id}.pdf")
    c.SaveAs(out_png)
    c.SaveAs(out_pdf)

    print(f"[plots:binned] wrote\n  {out_png}\n  {out_pdf}")

    region_binned_canvases[region_id] = c
    stuff.extend([c, pad_top, pad_bot, leg, label_top, unit] + top_hists + ratio_hists + top_sep_lines + bot_sep_lines)

helpers.copyIndexPHP(plot_directory)
syncer.sync()
