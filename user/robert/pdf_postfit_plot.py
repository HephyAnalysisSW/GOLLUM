#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-file overlays of post-fit Chebyshev PDFs (one figure per JSON).

Hard-coded inputs:
  /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/
    binned_merged_v1_fit.json
    binned_merged_v1_nosyst_fit.json
    unbinned_merged_v1_fit.json
    unbinned_merged_v1_nosyst_fit.json

For each file:
  * auto-detect POIs c0..cN
  * build POI covariance (marginal by default; conditional with --conditional)
  * sample c ~ N(mu, Σ_POI)
  * evaluate f(x) for pid (default: 21) on linear/log grids
  * draw 16% / 50% / 84% quantile *lines* (no bands), y-range fixed to [0.8, 1.2]
  * save to user.plot_directory/<plot_directory>/<basename>/{basename}_{linear,log}.(png|pdf)
"""

import os, sys, argparse, json, re
import numpy as np
import ROOT

# Repo-relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pdf.PDFParametrization import PDFParametrization
from common import helpers
import common.user as user
import common.syncer as syncer

# ----------------------------- CLI -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--plot_directory", default="CHEB_PERFILE", help="subdir under user.plot_directory")
ap.add_argument("--nsamples",       type=int, default=2000, help="# Gaussian posterior samples per fit")
ap.add_argument("--nX",             type=int, default=400,  help="# x points per curve")
#ap.add_argument("--xlog_min",       type=float, default=1e-4, help="min x for log-x grid (>0)")
ap.add_argument("--pid",            type=int, default=21,   help="PDF parton id to evaluate with")
ap.add_argument("--seed",           type=int, default=1337, help="random seed")
ap.add_argument("--conditional", action="store_true",
                help="use conditional POI covariance given nuis fixed at MLE (Schur complement)")
args = ap.parse_args()

np.random.seed(args.seed)
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)

# ----------------------------- Hard-coded input files -----------------------------
INPUT_FILES = [
    "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_merged_v1_fit.json",
    "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/binned_merged_v1_nosyst_fit.json",
    "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_merged_v1_fit.json",
    "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/unbinned_merged_v1_nosyst_fit.json",
]

# Hard-coded colors per base filename (fallback palette used if key missing)
COLOR_BY_BASE = {
    "binned_merged_v1_fit":        ROOT.kAzure + 1,
    "binned_merged_v1_nosyst_fit": ROOT.kRed   + 1,
    "unbinned_merged_v1_fit":      ROOT.kGreen + 2,
    "unbinned_merged_v1_nosyst_fit": ROOT.kMagenta + 2,
}
FALLBACK_COLORS = [ROOT.kOrange+1, ROOT.kCyan+2, ROOT.kViolet+2, ROOT.kGray+2]

# ----------------------------- I/O helpers -----------------------------
def load_fit(path):
    with open(path, "r") as f:
        return json.load(f)

def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]

# ----------------------------- POI helpers -----------------------------
def detect_poi_names(params):
    """
    From a list of {"name":..., "value":...}, extract ['c0',..., 'cN'].
    Accept names matching ^c(\d+)$; returns dense list up to max index.
    """
    idxs = []
    for p in params:
        m = re.fullmatch(r"c(\d+)", p["name"])
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        return []
    n = max(idxs)
    return [f"c{i}" for i in range(n + 1)]

def subvector(names_full, values_full, wanted):
    name2idx = {n: i for i, n in enumerate(names_full)}
    out = np.zeros(len(wanted), dtype=float)
    for j, w in enumerate(wanted):
        out[j] = float(values_full[name2idx[w]]) if w in name2idx else 0.0
    return out

def psd_cholesky(S):
    """Robust factor for (semi-)definite covariance matrices."""
    evals, evecs = np.linalg.eigh(S)
    evals = np.clip(evals, 0.0, None)
    return evecs @ np.diag(np.sqrt(evals))

# ----------------------------- Output root -----------------------------
plot_root = os.path.join(user.plot_directory, args.plot_directory)
os.makedirs(plot_root, exist_ok=True)
helpers.copyIndexPHP(plot_root)

# ----------------------------- X grids -----------------------------
x_lin = np.linspace(0.0, 1.0, args.nX, dtype=float)
x_log = np.geomspace(0.003, 1.0, args.nX, dtype=float)

# ----------------------------- Per-file processing -----------------------------
def process_file(path, fallback_color_idx=0):
    if not os.path.exists(path):
        print(f"[warn] Missing: {path} — skipping.")
        return

    data = load_fit(path)
    base = basename_noext(path)

    # color mapping
    color = COLOR_BY_BASE.get(base, FALLBACK_COLORS[fallback_color_idx % len(FALLBACK_COLORS)])

    params      = data.get("parameters", [])
    order_full  = (data.get("covariance") or {}).get("order", [])
    cov_full    = (data.get("covariance") or {}).get("matrix", [])

    if not params or not order_full or not cov_full:
        print(f"[warn] Incomplete payload in {path}; skipping.")
        return

    # Auto-detect POIs (or use provided, if present)
    poi_names = data.get("poi_names", None)
    if not poi_names:
        poi_names = detect_poi_names(params)
    if not poi_names:
        print(f"[warn] No POIs c_i found in {path}; skipping.")
        return

    names_full  = [p["name"] for p in params]
    values_full = [p["value"] for p in params]
    mu_c = subvector(names_full, values_full, poi_names)

    cov_full_np   = np.asarray(cov_full, dtype=float)
    name2idx_full = {n:i for i,n in enumerate(order_full)}

    # Indices for POIs and nuisances present in the covariance
    idx_c = np.array([name2idx_full[p] for p in poi_names if p in name2idx_full], dtype=int)
    idx_n = np.array([i for i,nm in enumerate(order_full) if nm not in poi_names], dtype=int)

    S_c_marg = cov_full_np[np.ix_(idx_c, idx_c)]
    S_c = S_c_marg
    cov_kind = "marginal"

    if args.conditional and idx_n.size > 0:
        S_cn = cov_full_np[np.ix_(idx_c, idx_n)]
        S_nc = cov_full_np[np.ix_(idx_n, idx_c)]
        S_nn = cov_full_np[np.ix_(idx_n, idx_n)]
        Snn_inv = np.linalg.pinv(S_nn)
        S_c = S_c_marg - S_cn @ Snn_inv @ S_nc
        cov_kind = "conditional (nuisances fixed at MLE)"

    # ---- diagnostics / prints ----
    print(f"\n[fit] {base}")
    print(f"  POIs ({len(poi_names)}): {', '.join(poi_names)}")
    print(f"  Covariance (full) size: {len(order_full)} x {len(order_full)}")
    print(f"  Using {cov_kind} POI covariance: shape {S_c.shape}")
    print(f"  POI indices in full order: {idx_c.tolist()}")
    if args.conditional:
        print(f"  Nuisance indices (conditioned on): {idx_n.tolist()}")
    eig = np.linalg.eigvalsh(S_c)
    print(f"  POI cov eig min/max: {eig.min():.3e} / {eig.max():.3e}")
    print(f"  ||mu_c||_2: {np.linalg.norm(mu_c):.3e}")
    print(f"  Color: {color} (ROOT index)")

    # Sample c ~ N(mu_c, S_c)
    L = psd_cholesky(S_c)
    Z = np.random.normal(0.0, 1.0, size=(args.nsamples, len(poi_names)))
    C_samples = (L @ Z.T).T + mu_c

    # PDF of order len(c)-1
    pdf = PDFParametrization(n=len(poi_names) - 1)

    # Evaluate on grids (use current API: evaluate(x, id, coeffs))
    F_lin = np.array([pdf.evaluate(x_lin, args.pid, c) for c in C_samples], dtype=float)
    F_log = np.array([pdf.evaluate(x_log, args.pid, c) for c in C_samples], dtype=float)

    # 1σ quantiles + median
    q16_lin, q50_lin, q84_lin = np.quantile(F_lin, [0.16, 0.50, 0.84], axis=0)
    q16_log, q50_log, q84_log = np.quantile(F_log, [0.16, 0.50, 0.84], axis=0)

    # Output dir per file
    out_dir = os.path.join(plot_root)
    os.makedirs(out_dir, exist_ok=True)
    helpers.copyIndexPHP(out_dir)

    # Draw helper (single panel)
    def draw_panel(x, q16, q50, q84, x_is_log, suffix):
        y_lo, y_hi = 0.8, 1.2
        c = ROOT.TCanvas(f"c_{base}_{suffix}", "", 900, 700)
        if x_is_log:
            c.SetLogx(True)

        x_lo = float(x[0])
        x_hi = float(x[-1])
        frame = ROOT.TH2F(f"frame_{base}_{suffix}", "", 100, x_lo, x_hi, 100, y_lo, y_hi)
        frame.GetXaxis().SetTitle("x")
        frame.GetYaxis().SetTitle("f(x)")
        frame.Draw()

        # Lines: 1σ lower/upper (solid), median (thicker, solid)
        g_l1  = ROOT.TGraph(len(x), x.astype('f8'), q16.astype('f8'))
        g_u1  = ROOT.TGraph(len(x), x.astype('f8'), q84.astype('f8'))
        g_med = ROOT.TGraph(len(x), x.astype('f8'), q50.astype('f8'))
        for g, lw in [(g_l1, 2), (g_u1, 2), (g_med, 3)]:
            g.SetLineColor(color); g.SetLineStyle(1); g.SetLineWidth(lw)

        g_l1.Draw("L SAME"); g_u1.Draw("L SAME"); g_med.Draw("L SAME")

        # Small legend explaining styles
        leg = ROOT.TLegend(0.12, 0.86, 0.95, 0.98)
        leg.SetNColumns(3); leg.SetFillStyle(0); leg.SetBorderSize(0); leg.SetTextSize(0.030)
        leg.AddEntry(g_med, "median", "l")
        leg.AddEntry(g_l1,  "16%",    "l")
        leg.AddEntry(g_u1,  "84%",    "l")
        leg.Draw()

        out_png = os.path.join(out_dir, f"{base}_{suffix}.png")
        out_pdf = os.path.join(out_dir, f"{base}_{suffix}.pdf")
        c.Print(out_png)
        c.Print(out_pdf)
        c.Close()
        print("Wrote:", out_png)
        print("Wrote:", out_pdf)

    # Render per file
    draw_panel(x_lin, q16_lin, q50_lin, q84_lin, x_is_log=False, suffix="linear")
    draw_panel(x_log, q16_log, q50_log, q84_log, x_is_log=True,  suffix="log")

# ----------------------------- Run over files -----------------------------
for i, path in enumerate(INPUT_FILES):
    process_file(path, fallback_color_idx=i)

# Sync (if your workflow mirrors outputs)
syncer.sync()

