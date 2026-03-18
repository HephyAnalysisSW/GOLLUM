#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np
import importlib

from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your likelihood machinery
sys.path.insert(0, '..')

import common.user as user

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)
from fit.N2LLExtensions import N2LLExtensions 
from common import helpers
import common.yaml_loader as yaml_loader
import common.syncer as syncer
from data.plot_options import plot_options

def set_free_values(hyp, names, values):
    """Assign values (array-like) to the corresponding parameter names in hyp."""
    for nm, v in zip(names, values):
        getattr(hyp, nm).val = float(v)


def clone_with_delta(hyp, names, delta):
    """Clone hypothesis and shift free params by delta vector."""
    h = hyp.clone()
    for nm, dv in zip(names, delta):
        getattr(h, nm).val = float(getattr(h, nm).val) + float(dv)
    return h


p = argparse.ArgumentParser(
    description="Compare true -2logL to Fisher quadratic approximation near (0,0) Asimov."
)
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--plot_directory",     action="store",      default="orthogonalize_diagnosis", help="plot sub-directory")
p.add_argument("--cache-root", default=None, help="Override cache root (optional)")
p.add_argument("--overwrite", action="store_true", help="Overwrite caches")
p.add_argument("--fi-step-scale", type=float, default=1e-4,
               help="Step scale for Fisher binned/penalty finite diffs")
args = p.parse_args()

# ---- Load YAML + surrogates ----
cfg = yaml_loader.load_yaml(args.config)
yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, args.config, overwrite=False)

# ---- Likelihood info & hypothesis scaffold ----
like_info = load_likelihood(cfg)
hyp = build_hypothesis_from_likelihood(like_info, name="SR")

# --- optionally disable all nuisances ---
if True:
    for p in hyp.nuisances:
        p.val = 0.0
        p.freeze()
    print("[opts] --no_syst: all nuisances set to 0 and frozen.")

hyp.print()

# Keep track of the free parameters (not frozen and not ignored)
free_params = [p for p in hyp.parameters
               if not p.isFrozen and not getattr(p, "isIgnored", False)]
free_names  = [p.name for p in free_params]
if not free_names:
    raise RuntimeError("No free parameters to probe.")
print("\n[free] Parameters to probe:")
for nm in free_names:
    print("   -", nm)

# Make sample loader factory from default cfg
samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])

from common.yaml_loader import _resolve_features_list
default_features = cfg["defaults"].get("default_features", None)
features = _resolve_features_list( default_features ) if default_features else None
factory     = samples_mod.Factory( 
    features  = features,
    selection = cfg["defaults"].get("default_selection", None),
    selection_features = cfg["defaults"].get("default_selection_features", None),
    )

# ---- Build caches, prepare runtime ----
n2ll = N2LLExtensions(
    like_info,
    # cfg['defaults']['module_samples'],
    factory = factory,
    cache_subdir = os.path.join(
        "NN2LCache",
        str(cfg.get("version", "v0")),
        os.path.splitext(os.path.basename(args.config))[0],
    ),
    cache_root=args.cache_root,
    overwrite=args.overwrite,
)
n2ll.build_cache()
n2ll.prepare_runtime()

# ---- Set A-simov at (0,0) ----
# By convention setAsimov(None) activates A-simov with zero bias term.
n2ll.setAsimov()  # A-simov (c' = 0, ν' = 0)

# ---- Baseline -2logL at (0,0) ----
# All params were initialized to 0 in build_hypothesis_from_likelihood.
f0 = float(n2ll(hyp))
print(f"\n[baseline] -2 log L (Asimov, at 0) = {f0:.6e}")

# ---- Fisher information at (0,0) ----
# N2LL.fisher_information() returns the regular Fisher matrix I_ab
# (i.e. E[∂_a log L ∂_b log L]). For small displacements δ, the
# quadratic approximation of Δ(-2logL) is δ^T I δ.
print("\n[FI] Computing Fisher information at (0,0) in Asimov A-mode…")
I = n2ll.fisher_information(hyp, step_scale=args.fi_step_scale, verbose=True)

# Numerical symmetrization for safety
I = 0.5 * (I + I.T)

# ================================================================
#   Build rate + shape basis in POI space (Option A coordinates)
#   d are the basis coefficients:  c = V_new @ d  and  d = D @ c
#   => Fisher in d-space is identity.
# ================================================================

free_params = [p for p in hyp.parameters
               if not p.isFrozen and not getattr(p, "isIgnored", False)]
free_names = [p.name for p in free_params]

poi_free_params = [p for p in free_params if p.isPOI]
poi_names = [p.name for p in poi_free_params]

poi_idx = [i for i, p in enumerate(free_params) if p.isPOI]
I_poi = I[np.ix_(poi_idx, poi_idx)]

evals, evecs = np.linalg.eigh(I_poi)

print("Parameter order:")
for i, n in enumerate(poi_names):
    print(f"{i:2d}  {n}")

print("\nEigenvalues of Fisher information (ascending):")
for i, lam in enumerate(evals):
    print(f"{i:2d}  {lam: .6e}")

lam_max = np.max(np.abs(evals))
tol_neg  = 1e-10 * max(1.0, lam_max)
tol_flat = 1e-6  * max(1.0, lam_max)

neg_idx  = [i for i, lam in enumerate(evals) if lam < -tol_neg]
flat_idx = [i for i, lam in enumerate(evals) if abs(lam) <= tol_flat]

print("\nSummary:")
print("  max |lambda|      =", lam_max)
print("  negative modes    =", neg_idx)
print("  near-flat modes   =", flat_idx)

pos = evals[evals > tol_neg]
if len(pos):
    cond = np.max(pos) / np.min(pos)
    print("  condition number  =", cond)
else:
    print("  condition number  = undefined (no positive eigenvalues)")

def print_mode(k, n_show=6):
    lam = evals[k]
    vec = evecs[:, k]
    order = np.argsort(-np.abs(vec))
    print(f"\nMode {k}: lambda = {lam:.6e}")
    print("  dominant components:")
    for j in order[:n_show]:
        print(f"    {poi_names[j]:>10s} : {vec[j]:+.4f}")
    print("  norm check:", np.sum(vec**2))

# inspect the smallest few modes
n_modes_to_show = min(5, len(evals))
for k in range(n_modes_to_show):
    print_mode(k)

sig_modes = 1.0 / np.sqrt(evals)
for k, (lam, sig) in enumerate(zip(evals, sig_modes)):
    print(f"mode {k}: lambda = {lam:.6e},  sigma_mode ≈ {sig:.6e}")


import ROOT
import numpy as np

ROOT.gStyle.SetOptStat(0)

# inputs assumed to exist:
#   evals   : eigenvalues, ascending
#   evecs   : eigenvectors, columns = modes
#   poi_names

# ---------- settings ----------
n_modes = len(evals)
colors = [
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kCyan + 1,
    ROOT.kViolet + 1,
    ROOT.kAzure + 2,
    ROOT.kSpring + 5,
    ROOT.kPink + 7,
]

lam_pos = np.array([x for x in evals if x > 0], dtype=float)
if len(lam_pos) == 0:
    raise RuntimeError("No positive eigenvalues to plot on log scale.")

ymin = 0.5 * np.min(lam_pos)
ymax = 2.0 * np.max(lam_pos)

# thickness of each horizontal bar in log-space
log_span = np.log10(ymax / ymin)
half_bar_decades = 0.025 * log_span

def yband(y):
    return y / (10.0**half_bar_decades), y * (10.0**half_bar_decades)

# ---------- canvas ----------
c_eig = ROOT.TCanvas("c_eigensystem", "c_eigensystem", 1500, 800)

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

# ---------- left pad: text summary ----------
padL.cd()

txt = ROOT.TLatex()
txt.SetNDC(True)
txt.SetTextFont(42)

y = 0.96
txt.SetTextSize(0.040)
txt.DrawLatex(0.04, y, "Fisher eigensystem summary")
y -= 0.06

txt.SetTextSize(0.032)
txt.DrawLatex(0.04, y, f"N_{{POI}} = {len(poi_names)}")
y -= 0.045
txt.DrawLatex(0.04, y, f"#lambda_{{min}} = {evals[0]:.3e}")
y -= 0.045
txt.DrawLatex(0.04, y, f"#lambda_{{max}} = {evals[-1]:.3e}")
y -= 0.045
txt.DrawLatex(0.04, y, f"cond. #approx {evals[-1]/evals[0]:.3e}")
y -= 0.06

tol_flat = 1e-6 * max(1.0, np.max(np.abs(evals)))
flat_idx = [i for i, lam in enumerate(evals) if abs(lam) <= tol_flat]

txt.DrawLatex(0.04, y, f"near-flat modes: {flat_idx}")
y -= 0.06

txt.SetTextSize(0.030)
txt.DrawLatex(0.04, y, "Smallest eigenvalues:")
y -= 0.045

txt.SetTextSize(0.028)
for k in range(min(6, n_modes)):
    txt.DrawLatex(0.07, y, f"m{k}: {evals[k]:.3e}")
    y -= 0.037

y -= 0.03
txt.SetTextSize(0.030)
txt.DrawLatex(0.04, y, "Largest eigenvalues:")
y -= 0.045

txt.SetTextSize(0.028)
for k in range(max(0, n_modes - 3), n_modes):
    txt.DrawLatex(0.07, y, f"m{k}: {evals[k]:.3e}")
    y -= 0.037

# legend for POI colors
leg = ROOT.TLegend(0.04, 0.04, 0.96, 0.22)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetNColumns(3)

leg_boxes = []
for i, name in enumerate(poi_names):
    b = ROOT.TBox(0, 0, 1, 1)
    b.SetFillColor(colors[i % len(colors)])
    b.SetLineColor(colors[i % len(colors)])
    leg.AddEntry(b, name, "f")
    leg_boxes.append(b)

leg.Draw()

# ---------- right pad: eigenvalues + composition ----------
padR.cd()

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

# draw one stacked horizontal bar per eigenmode
mode_labels = []
bar_boxes = []
guide_lines = []

for k, lam in enumerate(evals):
    y1, y2 = yband(lam)

    # thin gray guide line across full width
    gl = ROOT.TLine(0.0, lam, 1.0, lam)
    gl.SetLineColor(ROOT.kGray + 1)
    gl.SetLineStyle(ROOT.kDashed)
    gl.Draw()
    guide_lines.append(gl)

    vec = evecs[:, k]
    weights = np.abs(vec)**2
    weights = weights / np.sum(weights)

    x0 = 0.0
    for i, w in enumerate(weights):
        x1 = x0 + float(w)
        box = ROOT.TBox(x0, y1, x1, y2)
        box.SetFillColor(colors[i % len(colors)])
        box.SetLineColor(colors[i % len(colors)])
        box.Draw()
        bar_boxes.append(box)
        x0 = x1

    # mode label on the right
    lab = ROOT.TLatex()
    lab.SetTextFont(42)
    lab.SetTextSize(0.028)
    lab.DrawLatex(1.01, lam, f"m{k}")
    mode_labels.append(lab)

c_eig.cd()
c_eig.Update()

plot_directory = os.path.join(
    user.plot_directory, args.plot_directory, cfg['version'],
)
helpers.copyIndexPHP(plot_directory)
os.makedirs(plot_directory, exist_ok=True)

c_eig.Print(os.path.join(plot_directory,f"{os.path.splitext(os.path.basename(args.config))[0]}.png")); 
c_eig.Close()
syncer.sync()
