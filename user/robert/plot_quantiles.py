#!/usr/bin/env python

# Standalone x vs mu_F coverage plot, updated to use data.samples RDataLoader/SelectionView

import ROOT
import numpy as np
import os, sys
from math import log10
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

from data.samples import tt2l  # base loader (RDataLoader) with observers/features/weights
from data.plot_options import plot_options

# -------------------- args --------------------
import argparse
argParser = argparse.ArgumentParser(description="x–muF coverage plot (new data interface)")
argParser.add_argument("--plot_directory",  action="store", default="PDF_xmuF_map", help="plot sub-directory")
argParser.add_argument("--model",           action="store", default="TT2l_PDF",     help="kept for folder naming")
argParser.add_argument("--nTraining",       action="store", default=-1, type=int,   help="number of events to read (-1 = all)")
args, _ = argParser.parse_known_args(sys.argv[1:])


# -------------------- data IO (new) --------------------
# Load arrays from the nominal tt2l sample via direct streaming
Xs, Ws, Os = [], [], []
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

# names (from loader)
feature_names = list(getattr(tt2l, "feature_names", []) or [])
observer_names = list(getattr(tt2l, "observer_names", []) or [])

# Sanity: required observers must exist
for req in ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"]:
    if req not in observer_names:
        raise RuntimeError(f"Observer '{req}' not available in this sample. Have: {observer_names}")

ix_x1  = observer_names.index("Generator_x1")
ix_x2  = observer_names.index("Generator_x2")
ix_id1 = observer_names.index("Generator_id1")
ix_id2 = observer_names.index("Generator_id2")

x1  = training_observers[:, ix_x1]
x2  = training_observers[:, ix_x2]
id1 = training_observers[:, ix_id1].astype(int, copy=False)
id2 = training_observers[:, ix_id2].astype(int, copy=False)

# -------------------- process masks (as before) --------------------
GG = (id1 == 21) & (id2 == 21)

_abs_flv = np.array([1,2,3,4,5,6], dtype=int)
QG = ((id1 == 21) & np.isin(np.abs(id2), _abs_flv)) | ((id2 == 21) & np.isin(np.abs(id1), _abs_flv))
QQ = (np.isin(np.abs(id1), _abs_flv) & np.isin(np.abs(id2), _abs_flv) & (id1 * id2 < 0))

QG_q  = ((id1==21) & np.isin(id2,  [1,2,3,4,5,6])) | (np.isin(id1,  [1,2,3,4,5,6]) & (id2==21))
QG_qb = ((id1==21) & np.isin(id2, [-1,-2,-3,-4,-5,-6])) | (np.isin(id1, [-1,-2,-3,-4,-5,-6]) & (id2==21))

x_gluon_leg_q = np.where((id1==21) & np.isin(id2,[ 1, 2, 3, 4, 5, 6]), x1,
                   np.where((id2==21) & np.isin(id1,[ 1, 2, 3, 4, 5, 6]), x2, np.nan))
x_quark_leg_q = np.where((id1==21) & np.isin(id2,[ 1, 2, 3, 4, 5, 6]), x2,
                   np.where((id2==21) & np.isin(id1,[ 1, 2, 3, 4, 5, 6]), x1, np.nan))

x_gluon_leg_qb  = np.where((id1==21) & np.isin(id2,[-1,-2,-3,-4,-5,-6]), x1,
                     np.where((id2==21) & np.isin(id1,[-1,-2,-3,-4,-5,-6]), x2, np.nan))
x_antiquark_leg = np.where((id1==21) & np.isin(id2,[-1,-2,-3,-4,-5,-6]), x2,
                     np.where((id2==21) & np.isin(id1,[-1,-2,-3,-4,-5,-6]), x1, np.nan))

x_quark_qq  = np.where((id1>0) & QQ, x1, np.where((id2>0) & QQ, x2, np.nan))
x_qbar_qq   = np.where((id1<0) & QQ, x1, np.where((id2<0) & QQ, x2, np.nan))

# ensure we have a scratch list for ROOT objects we must keep alive
try:
    stuff
except NameError:
    stuff = []

# -------------------- utilities (unchanged) --------------------
def make_th1_feature(mask, feature_idx, feature_name, color, title_suffix=""):
    """Weighted 1D feature histogram for a boolean mask."""
    bins = plot_options[feature_name]['binning']
    arr  = training_features[:, feature_idx][mask]
    ww   = w0[mask]
    h    = helpers.make_TH1F(np.histogram(arr, np.linspace(bins[1], bins[2], bins[0] + 1), weights=ww))
    h.SetDirectory(0)
    h.SetLineColor(color); h.SetLineWidth(2); h.SetMarkerStyle(0)
    if title_suffix:
        h.SetTitle(title_suffix)
    return h

def quantiles_per_bin(feature_idx, feature_name, values_full, weights_full,
                      base_mask=None, quantiles=(0.05, 0.32, 0.50, 0.68, 0.95)):
    """
    Compute weighted quantiles of 'values_full' in bins of the feature.
    All inputs are full-length (one per event). Use 'base_mask' to restrict.
    Returns (centers, edges, [TGraph for each quantile]).
    """
    bins    = plot_options[feature_name]['binning']
    edges   = np.linspace(bins[1], bins[2], bins[0] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fvals = training_features[:, feature_idx]
    if base_mask is None:
        base_mask = np.ones_like(weights_full, dtype=bool)

    valid = base_mask & np.isfinite(values_full) & np.isfinite(weights_full)

    graphs = []
    for q in quantiles:
        xs, ys = [], []
        for i in range(len(edges) - 1):
            sel_bin = valid & (fvals >= edges[i]) & (fvals < edges[i + 1])
            if not np.any(sel_bin):
                continue
            vy = values_full[sel_bin]
            wy = weights_full[sel_bin]
            if vy.size == 0 or wy.sum() <= 0:
                continue

            order     = np.argsort(vy)
            vy_sorted = vy[order]
            wy_sorted = wy[order]
            cdf       = np.cumsum(wy_sorted) / np.sum(wy_sorted)
            yq        = np.interp(q, cdf, vy_sorted)

            xs.append(centers[i]); ys.append(yq)

        graphs.append(ROOT.TGraph(len(xs), np.array(xs, dtype='d'), np.array(ys, dtype='d')))
    return centers, edges, graphs

# -------------------- style --------------------
col_gg = ROOT.kGreen + 2
col_qg = ROOT.kBlue  + 1
col_qq = ROOT.kRed   + 1
quant_cols = [ROOT.kGray+2, ROOT.kAzure+2, ROOT.kMagenta+1, ROOT.kOrange+1, ROOT.kCyan+2]

# -------------------- output dirs --------------------
plot_directory = os.path.join(user.plot_directory, args.plot_directory, args.model)
os.makedirs(plot_directory, exist_ok=True)

outdir = os.path.join(plot_directory, "feature_panels_ttbar_2")
os.makedirs(outdir, exist_ok=True)
helpers.copyIndexPHP(outdir)

# ================== LOOP OVER FEATURES (5 pads) ==================
for feature_idx, feature_name in enumerate(feature_names):
    if feature_name not in plot_options:
        # silently skip unknown features (no binning config)
        continue

    bins   = plot_options[feature_name]['binning']
    edges  = np.linspace(bins[1], bins[2], bins[0] + 1)

    c = ROOT.TCanvas(f"c_{feature_name}_panel", "", 2500, 520)
    c.Divide(5, 1)

    # -------- Pad 1: distribution split into gg, qg(total), qq̄ --------
    c.cd(1)
    ROOT.gPad.SetLogy(True)

    h_gg = make_th1_feature(GG, feature_idx, feature_name, col_gg, "gg")
    h_qg = make_th1_feature(QG, feature_idx, feature_name, col_qg, "qg")
    h_qq = make_th1_feature(QQ, feature_idx, feature_name, col_qq, "q#bar{q}")

    h_gg.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    h_gg.GetYaxis().SetTitle("events (weighted)")

    maxy = max(h.GetMaximum() for h in (h_gg, h_qg, h_qq)) * 5.0
    h_gg.GetYaxis().SetRangeUser(max(1e-9, 0.5), maxy)

    h_gg.Draw("hist")
    h_qg.Draw("hist same")
    h_qq.Draw("hist same")

    leg1 = ROOT.TLegend(0.18, 0.86, 1, 0.96)
    leg1.SetNColumns(3); leg1.SetFillStyle(0); leg1.SetBorderSize(0); leg1.SetTextSize(0.035)
    leg1.AddEntry(h_gg, "gg", "l")
    leg1.AddEntry(h_qg, "qg", "l")
    leg1.AddEntry(h_qq, "q#bar{q}", "l")
    leg1.Draw()

    labels_pct = ["5%", "32%", "50%", "68%", "95%"]

    def make_quantile_legend(x1=0.18, y1=0.77, x2=1.0, y2=0.85, text_size=0.032):
        leg = ROOT.TLegend(x1, y1, x2, y2)
        leg.SetNColumns(5); leg.SetFillStyle(0); leg.SetBorderSize(0); leg.SetTextSize(text_size)
        return leg

    # -------- Pad 2: quantiles of x1 for gg vs feature --------
    c.cd(2); ROOT.gPad.SetGridy(True)
    _, edges2, graphs2 = quantiles_per_bin(feature_idx, feature_name, x1, w0, base_mask=GG)
    frame2 = ROOT.TH1F(f"frame2_{feature_name}", "", len(edges2) - 1, edges2)
    frame2.SetMinimum(1e-6); frame2.SetMaximum(1.0)
    frame2.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame2.GetYaxis().SetTitle("x_{1} quantiles (gg)")
    frame2.Draw()
    for i, gr in enumerate(graphs2):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.Draw("L SAME")
    tex2 = ROOT.TLatex(); tex2.SetNDC(); tex2.SetTextFont(42); tex2.SetTextSize(0.05); tex2.DrawLatex(0.20, 0.88, "gg")
    leg_q2 = make_quantile_legend()
    d_q2=[]
    for i, lab in enumerate(labels_pct):
        d=ROOT.TH1F(f"dummy_q2_{i}_{feature_name}","",1,0,1); d.SetLineColor(quant_cols[i%len(quant_cols)]); d.SetLineStyle(1); d.SetLineWidth(2)
        leg_q2.AddEntry(d, lab, "l"); d_q2.append(d)
    stuff += [tex2] + d_q2
    leg_q2.Draw()

    # -------- Pad 3: qg — gluon-leg (solid) vs quark-leg (dashed) --------
    c.cd(3); ROOT.gPad.SetGridy(True)
    _, edges3, graphs_g_q = quantiles_per_bin(feature_idx, feature_name, x_gluon_leg_q, w0, base_mask=QG_q)
    _, _,      graphs_q_q = quantiles_per_bin(feature_idx, feature_name, x_quark_leg_q, w0, base_mask=QG_q)

    frame3 = ROOT.TH1F(f"frame3_{feature_name}", "", len(edges3) - 1, edges3)
    frame3.SetMinimum(1e-6); frame3.SetMaximum(1.0)
    frame3.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame3.GetYaxis().SetTitle("x quantiles (qg)")
    frame3.Draw()

    for i, gr in enumerate(graphs_g_q):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(1); gr.Draw("L SAME")
    for i, gr in enumerate(graphs_q_q):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(7); gr.Draw("L SAME")

    tex3 = ROOT.TLatex(); tex3.SetNDC(); tex3.SetTextFont(42); tex3.SetTextSize(0.05); tex3.DrawLatex(0.20, 0.88, "qg")
    leg3 = ROOT.TLegend(0.25, 0.82, 1., 0.96); leg3.SetNColumns(2); leg3.SetFillStyle(0); leg3.SetBorderSize(0)
    dsol3 = ROOT.TH1F(f"dummy_sol3_{feature_name}","",1,0,1); dsol3.SetLineColor(ROOT.kBlack); dsol3.SetLineStyle(1); dsol3.SetLineWidth(2)
    ddas3 = ROOT.TH1F(f"dummy_das3_{feature_name}","",1,0,1); ddas3.SetLineColor(ROOT.kBlack); ddas3.SetLineStyle(7); ddas3.SetLineWidth(2)
    leg3.AddEntry(dsol3, "solid: gluon-leg", "l"); leg3.AddEntry(ddas3, "dashed: quark-leg", "l"); leg3.Draw()
    leg_q3 = make_quantile_legend()
    d_q3=[]
    for i, lab in enumerate(labels_pct):
        d=ROOT.TH1F(f"dummy_q3_{i}_{feature_name}","",1,0,1); d.SetLineColor(quant_cols[i%len(quant_cols)]); d.SetLineStyle(1); d.SetLineWidth(2)
        leg_q3.AddEntry(d, lab, "l"); d_q3.append(d)
    stuff += [tex3, dsol3, ddas3] + d_q3
    leg_q3.Draw()

    # -------- Pad 4: q̄g — gluon-leg (solid) vs anti-quark-leg (dashed) --------
    c.cd(4); ROOT.gPad.SetGridy(True)
    _, edges4a, graphs_g_qb = quantiles_per_bin(feature_idx, feature_name, x_gluon_leg_qb,  w0, base_mask=QG_qb)
    _, _,       graphs_a_qb = quantiles_per_bin(feature_idx, feature_name, x_antiquark_leg, w0, base_mask=QG_qb)

    frame4a = ROOT.TH1F(f"frame4a_{feature_name}", "", len(edges4a) - 1, edges4a)
    frame4a.SetMinimum(1e-6); frame4a.SetMaximum(1.0)
    frame4a.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame4a.GetYaxis().SetTitle("x quantiles (#bar{q}g)")
    frame4a.Draw()

    for i, gr in enumerate(graphs_g_qb):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(1); gr.Draw("L SAME")
    for i, gr in enumerate(graphs_a_qb):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(7); gr.Draw("L SAME")

    tex4 = ROOT.TLatex(); tex4.SetNDC(); tex4.SetTextFont(42); tex4.SetTextSize(0.05); tex4.DrawLatex(0.20, 0.88, "#bar{q}g")
    leg4 = ROOT.TLegend(0.25, 0.82, 1., 0.96); leg4.SetNColumns(2); leg4.SetFillStyle(0); leg4.SetBorderSize(0)
    dsol4 = ROOT.TH1F(f"dummy_sol4_{feature_name}","",1,0,1); dsol4.SetLineColor(ROOT.kBlack); dsol4.SetLineStyle(1); dsol4.SetLineWidth(2)
    ddas4 = ROOT.TH1F(f"dummy_das4_{feature_name}","",1,0,1); ddas4.SetLineColor(ROOT.kBlack); ddas4.SetLineStyle(7); ddas4.SetLineWidth(2)
    leg4.AddEntry(dsol4, "solid: gluon-leg", "l"); leg4.AddEntry(ddas4, "dashed: anti-quark-leg", "l"); leg4.Draw()
    leg_q4 = make_quantile_legend()
    d_q4=[]
    for i, lab in enumerate(labels_pct):
        d=ROOT.TH1F(f"dummy_q4_{i}_{feature_name}","",1,0,1); d.SetLineColor(quant_cols[i%len(quant_cols)]); d.SetLineStyle(1); d.SetLineWidth(2)
        leg_q4.AddEntry(d, lab, "l"); d_q4.append(d)
    stuff += [tex4, dsol4, ddas4] + d_q4
    leg_q4.Draw()

    # -------- Pad 5: qq̄ — quark-leg (solid) vs anti-quark-leg (dashed) --------
    c.cd(5); ROOT.gPad.SetGridy(True)
    _, edges5, graphs_q5 = quantiles_per_bin(feature_idx, feature_name, x_quark_qq, w0, base_mask=QQ)
    _, _,     graphs_a5 = quantiles_per_bin(feature_idx, feature_name, x_qbar_qq,  w0, base_mask=QQ)

    frame5 = ROOT.TH1F(f"frame5_{feature_name}", "", len(edges5) - 1, edges5)
    frame5.SetMinimum(1e-6); frame5.SetMaximum(1.0)
    frame5.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame5.GetYaxis().SetTitle("x quantiles (q#bar{q})")
    frame5.Draw()

    for i, gr in enumerate(graphs_q5):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(1); gr.Draw("L SAME")
    for i, gr in enumerate(graphs_a5):
        gr.SetLineColor(quant_cols[i % len(quant_cols)]); gr.SetLineWidth(2); gr.SetLineStyle(7); gr.Draw("L SAME")

    tex5 = ROOT.TLatex(); tex5.SetNDC(); tex5.SetTextFont(42); tex5.SetTextSize(0.05); tex5.DrawLatex(0.20, 0.88, "q#bar{q}")
    leg5 = ROOT.TLegend(0.25, 0.82, 1., 0.96); leg5.SetNColumns(2); leg5.SetFillStyle(0); leg5.SetBorderSize(0)
    dsol5 = ROOT.TH1F(f"dummy_sol5_{feature_name}","",1,0,1); dsol5.SetLineColor(ROOT.kBlack); dsol5.SetLineStyle(1); dsol5.SetLineWidth(2)
    ddas5 = ROOT.TH1F(f"dummy_das5_{feature_name}","",1,0,1); ddas5.SetLineColor(ROOT.kBlack); ddas5.SetLineStyle(7); ddas5.SetLineWidth(2)
    leg5.AddEntry(dsol5, "solid: quark-leg", "l"); leg5.AddEntry(ddas5, "dashed: anti-quark-leg", "l"); leg5.Draw()
    leg_q5 = make_quantile_legend()
    d_q5=[]
    for i, lab in enumerate(labels_pct):
        d=ROOT.TH1F(f"dummy_q5_{i}_{feature_name}","",1,0,1); d.SetLineColor(quant_cols[i%len(quant_cols)]); d.SetLineStyle(1); d.SetLineWidth(2)
        leg_q5.AddEntry(d, lab, "l"); d_q5.append(d)
    stuff += [tex5, dsol5, ddas5] + d_q5
    leg_q5.Draw()

    # -------- Save canvas --------
    c.Print(os.path.join(outdir, f"{feature_name}_panel.pdf"))
    c.Print(os.path.join(outdir, f"{feature_name}_panel.png"))
    c.Close()

