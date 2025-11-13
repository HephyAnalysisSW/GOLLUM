#!/usr/bin/env python

# 2D feature-feature panels with z = weighted median of x (or leg-specific x) per 2D bin

import ROOT
import numpy as np
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# ROOT / style
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

# Helpers & infra
from common import helpers
import common.syncer as syncer
import common.user as user

from data.plot_options import plot_options

# NEW: data interface (RDataLoader + SelectionView definitions live here)
from data import samples as ds

# Parser
import argparse
argParser = argparse.ArgumentParser(description="2D feature-feature panels with z=median x")
argParser.add_argument("--plot_directory",  action="store", default="PDF_xmuF_map", help="plot sub-directory")
argParser.add_argument("--nTraining",       action="store", default=-1, type=int,   help="number of events to read (-1 = all)")
args = argParser.parse_args()

# Plot dir
plot_directory = os.path.join(user.plot_directory, args.plot_directory, "tt2l_nominal")
os.makedirs(plot_directory, exist_ok=True)

# -------------------- data IO --------------------
# Stream all shards from the nominal tt2l loader using 'fwo'
Xs, Ws, Os = [], [], []
tt2l = ds.tt2l
n_shards = len(getattr(tt2l, "base", tt2l))
for shard in range(n_shards):
    X, W, O = tt2l.materialize(shard=shard, what="fwo", n=None)
    if X is None or len(X) == 0:
        continue
    Xs.append(np.asarray(X, dtype=np.float64))
    Ws.append(np.asarray(W, dtype=np.float64))
    Os.append(np.asarray(O, dtype=np.float64))

if not Xs:
    training_features  = np.empty((0,0), dtype=np.float64)
    w0                 = np.empty((0,),   dtype=np.float64)
    training_observers = np.empty((0,0), dtype=np.float64)
else:
    training_features  = np.concatenate(Xs, axis=0)
    w0                 = np.concatenate(Ws, axis=0)
    training_observers = np.concatenate(Os, axis=0)

# Optional cap by --nTraining
if args.nTraining is not None and args.nTraining > 0 and training_features.shape[0] > args.nTraining:
    n = args.nTraining
    training_features  = training_features[:n]
    w0                 = w0[:n]
    training_observers = training_observers[:n]

# Observers & helpers (names as in data/observables.py)
obs_names = tt2l.observer_names
ix_x1  = obs_names.index("Generator_x1")
ix_x2  = obs_names.index("Generator_x2")
ix_id1 = obs_names.index("Generator_id1")
ix_id2 = obs_names.index("Generator_id2")

x1   = training_observers[:, ix_x1]
x2   = training_observers[:, ix_x2]
id1  = training_observers[:, ix_id1].astype(int)
id2  = training_observers[:, ix_id2].astype(int)

# keep-alive scratch
try:
    stuff
except NameError:
    stuff = []

# ---------------- Process-class masks ----------------
abs_qids = np.array([1,2,3,4,5,6], dtype=int)

GG = (id1 == 21) & (id2 == 21)
QG = ((id1 == 21) & np.isin(np.abs(id2), abs_qids)) | ((id2 == 21) & np.isin(np.abs(id1), abs_qids))
QQ = (np.isin(np.abs(id1), abs_qids) & np.isin(np.abs(id2), abs_qids) & (id1 * id2 < 0))  # q qbar

QG_q  = ((id1==21) & np.isin(id2,  abs_qids)) | (np.isin(id1,  abs_qids) & (id2==21))
QG_qb = ((id1==21) & np.isin(id2, -abs_qids)) | (np.isin(id1, -abs_qids) & (id2==21))

# per-leg x for all needed cases
x_gluon_leg_q = np.where((id1==21) & np.isin(id2, abs_qids), x1,
                   np.where((id2==21) & np.isin(id1, abs_qids), x2, np.nan))
x_quark_leg_q = np.where((id1==21) & np.isin(id2, abs_qids), x2,
                   np.where((id2==21) & np.isin(id1, abs_qids), x1, np.nan))

x_gluon_leg_qb  = np.where((id1==21) & np.isin(id2, -abs_qids), x1,
                     np.where((id2==21) & np.isin(id1, -abs_qids), x2, np.nan))
x_antiquark_leg = np.where((id1==21) & np.isin(id2, -abs_qids), x2,
                     np.where((id2==21) & np.isin(id1, -abs_qids), x1, np.nan))

x_quark_qq  = np.where((id1>0) & QQ, x1, np.where((id2>0) & QQ, x2, np.nan))
x_qbar_qq   = np.where((id1<0) & QQ, x1, np.where((id2<0) & QQ, x2, np.nan))

# ---------------- Colors ----------------
col_gg = ROOT.kGreen + 2
col_qg = ROOT.kBlue  + 1
col_qq = ROOT.kRed   + 1
quant_cols = [ROOT.kGray+2, ROOT.kAzure+2, ROOT.kMagenta+1, ROOT.kOrange+1, ROOT.kCyan+2]

# ---------------- Output dir ----------------
outdir = os.path.join(plot_directory, "feature2D_panels_ttbar")
os.makedirs(outdir, exist_ok=True)
helpers.copyIndexPHP(outdir)

# ---------------- Utility: weighted median in 2D bins ----------------
def weighted_quantile(y, w, q):
    """Return weighted quantile of y with positive weights w in [0,1]."""
    if len(y) == 0:
        return float('nan')
    order = np.argsort(y)
    y_sorted = y[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    if cdf[-1] <= 0:
        return float('nan')
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, y_sorted))

def median_map_2d(ix, namex, iy, namey, values_full, weights_full, base_mask=None, q=0.50):
    """Compute TH2F with z = weighted median(values_full) in 2D bins of (feature x, feature y)."""
    binsx = plot_options[namex]['binning']
    binsy = plot_options[namey]['binning']
    edgesx = np.linspace(binsx[1], binsx[2], binsx[0] + 1)
    edgesy = np.linspace(binsy[1], binsy[2], binsy[0] + 1)

    fx = training_features[:, ix]
    fy = training_features[:, iy]

    if base_mask is None:
        base_mask = np.ones_like(weights_full, dtype=bool)
    valid = base_mask & np.isfinite(values_full) & np.isfinite(weights_full) & np.isfinite(fx) & np.isfinite(fy)

    h2 = ROOT.TH2F(f"h2_{namex}_{namey}_{q}", "", binsx[0], binsx[1], binsx[2], binsy[0], binsy[1], binsy[2])
    h2.SetDirectory(0)
    h2.GetXaxis().SetTitle(plot_options[namex]['tex'])
    h2.GetYaxis().SetTitle(plot_options[namey]['tex'])
    h2.GetZaxis().SetTitle("median x")
    h2.SetMinimum(1e-6); h2.SetMaximum(0.4)

    # loop over bins
    for ixbin in range(binsx[0]):
        xlo, xhi = edgesx[ixbin], edgesx[ixbin+1]
        selx = (fx >= xlo) & (fx < xhi)
        for iybin in range(binsy[0]):
            ylo, yhi = edgesy[iybin], edgesy[iybin+1]
            sely = (fy >= ylo) & (fy < yhi)
            sel = valid & selx & sely
            if not np.any(sel):
                continue
            yvals = values_full[sel]
            wvals = weights_full[sel]
            med = weighted_quantile(yvals, wvals, q)
            if not np.isfinite(med):
                continue
            # ROOT bins are 1-indexed
            h2.SetBinContent(ixbin+1, iybin+1, float(med))
    return h2, edgesx, edgesy

# ---------------- Hard-coded feature lists (edit these) ----------------
# These names must exist in tt2l.feature_names and in plot_options
x_features = [
    "tr_ttbar_pt",
    "tr_ttbar_mass",
    "ht",
    "recoLep01_pt",
    "recoLep01_mass",
    "tr_top_pt",
    "tr_topBar_pt",
]
y_features = [
    "tr_ttbar_eta",
    "tr_ttbar_dEta",
    "recoLep_dPhi",
    "recoLep_dEta",
    "tr_top_eta",
    "tr_topBar_eta",
]

# Map feature name -> index once
feat_idx = {name:i for i, name in enumerate(tt2l.feature_names)}

# ================== LOOP OVER FEATURE PAIRS (4 pads) ==================
for fx_name in x_features:
    if fx_name not in feat_idx or fx_name not in plot_options:
        print(f"[warn] x feature '{fx_name}' not found; skipping")
        continue
    for fy_name in y_features:
        if fy_name not in feat_idx or fy_name not in plot_options:
            print(f"[warn] y feature '{fy_name}' not found; skipping")
            continue
        ix = feat_idx[fx_name]
        iy = feat_idx[fy_name]

        c = ROOT.TCanvas(f"c2D_{fx_name}_{fy_name}_panel", "", 2400, 520)
        c.Divide(4, 1)

        # ---------- Pad A (gg): z = median(x1) ----------
        c.cd(1)
        ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetGridx(True); ROOT.gPad.SetGridy(True)
        h2_gg, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x1, w0, base_mask=GG, q=0.50)
        h2_gg.GetZaxis().SetTitle("median x_{1} (gg)")
        h2_gg.Draw("COLZ")
        tex2 = ROOT.TLatex(); tex2.SetNDC(); tex2.SetTextFont(42); tex2.SetTextSize(0.05); tex2.DrawLatex(0.20, 0.88, "gg")
        stuff += [tex2]

        # ---------- Pad B (qg): filled = gluon-leg, contour = quark-leg ----------
        c.cd(2)
        ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetGridx(True); ROOT.gPad.SetGridy(True)
        h2_qg_glu, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_gluon_leg_q, w0, base_mask=QG_q, q=0.50)
        h2_qg_glu.GetZaxis().SetTitle("median x (qg, gluon-leg)")
        h2_qg_glu.Draw("COLZ")
        h2_qg_qua, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_quark_leg_q,  w0, base_mask=QG_q, q=0.50)
        h2_qg_qua.SetContour(10); h2_qg_qua.Draw("CONT3 SAME")

        tex3 = ROOT.TLatex(); tex3.SetNDC(); tex3.SetTextFont(42); tex3.SetTextSize(0.05); tex3.DrawLatex(0.20, 0.88, "qg")
        leg3 = ROOT.TLegend(0.20, 0.80, 0.98, 0.94); leg3.SetNColumns(2); leg3.SetFillStyle(0); leg3.SetBorderSize(0)
        dbox3 = ROOT.TH2F("dbox3","",1,0,1,1,0,1); dbox3.SetFillColor(ROOT.kGray+1)
        dlin3 = ROOT.TH1F("dlin3","",1,0,1); dlin3.SetLineColor(ROOT.kBlack); dlin3.SetLineWidth(2)
        leg3.AddEntry(dbox3, "filled: gluon-leg", "f")
        leg3.AddEntry(dlin3, "contour: quark-leg", "l")
        leg3.Draw()
        stuff += [tex3, dbox3, dlin3]

        # ---------- Pad C (q̄g): filled = gluon-leg, contour = anti-quark-leg ----------
        c.cd(3)
        ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetGridx(True); ROOT.gPad.SetGridy(True)
        h2_qbg_glu, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_gluon_leg_qb,  w0, base_mask=QG_qb, q=0.50)
        h2_qbg_glu.GetZaxis().SetTitle("median x (q#bar{g}, gluon-leg)")
        h2_qbg_glu.Draw("COLZ")
        h2_qbg_ant, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_antiquark_leg, w0, base_mask=QG_qb, q=0.50)
        h2_qbg_ant.SetContour(10); h2_qbg_ant.Draw("CONT3 SAME")

        tex4 = ROOT.TLatex(); tex4.SetNDC(); tex4.SetTextFont(42); tex4.SetTextSize(0.05); tex4.DrawLatex(0.20, 0.88, "#bar{q}g")
        leg4 = ROOT.TLegend(0.20, 0.80, 0.98, 0.94); leg4.SetNColumns(2); leg4.SetFillStyle(0); leg4.SetBorderSize(0)
        dbox4 = ROOT.TH2F("dbox4","",1,0,1,1,0,1); dbox4.SetFillColor(ROOT.kGray+1)
        dlin4 = ROOT.TH1F("dlin4","",1,0,1); dlin4.SetLineColor(ROOT.kBlack); dlin4.SetLineWidth(2)
        leg4.AddEntry(dbox4, "filled: gluon-leg", "f")
        leg4.AddEntry(dlin4, "contour: anti-quark-leg", "l")
        leg4.Draw()
        stuff += [tex4, dbox4, dlin4]

        # ---------- Pad D (q q̄): filled = quark-leg, contour = anti-quark-leg ----------
        c.cd(4)
        ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetGridx(True); ROOT.gPad.SetGridy(True)
        h2_qq_qua, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_quark_qq, w0, base_mask=QQ, q=0.50)
        h2_qq_qua.GetZaxis().SetTitle("median x (q#bar{q}, quark-leg)")
        h2_qq_qua.Draw("COLZ")
        h2_qq_ant, _, _ = median_map_2d(ix, fx_name, iy, fy_name, x_qbar_qq,  w0, base_mask=QQ, q=0.50)
        h2_qq_ant.SetContour(10); h2_qq_ant.Draw("CONT3 SAME")

        tex5 = ROOT.TLatex(); tex5.SetNDC(); tex5.SetTextFont(42); tex5.SetTextSize(0.05); tex5.DrawLatex(0.20, 0.88, "q#bar{q}")
        leg5 = ROOT.TLegend(0.20, 0.80, 0.98, 0.94); leg5.SetNColumns(2); leg5.SetFillStyle(0); leg5.SetBorderSize(0)
        dbox5 = ROOT.TH2F("dbox5","",1,0,1,1,0,1); dbox5.SetFillColor(ROOT.kGray+1)
        dlin5 = ROOT.TH1F("dlin5","",1,0,1); dlin5.SetLineColor(ROOT.kBlack); dlin5.SetLineWidth(2)
        leg5.AddEntry(dbox5, "filled: quark-leg", "f")
        leg5.AddEntry(dlin5, "contour: anti-quark-leg", "l")
        leg5.Draw()
        stuff += [tex5, dbox5, dlin5]

        # Save
        base = f"{fx_name}_vs_{fy_name}_panel2D"
        c.Print(os.path.join(outdir, f"{base}.pdf"))
        c.Print(os.path.join(outdir, f"{base}.png"))
        c.Close()

