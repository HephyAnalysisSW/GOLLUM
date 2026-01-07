#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from array import array
import os
import argparse

import ROOT
import cmsstyle

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user   as user
import common.syncer as syncer
import common.helpers as helpers

from data.plot_options import plot_options

from PDFParametrization import PDFParametrization
from PDFTemplate import PDFShapeTemplate
from FeatureTemplate import FeatureTemplate

ROOT.gROOT.SetBatch(True)


def pid_to_short(pid: int) -> str:
    m = {
        21: "g",
        1: "d",  2: "u",  3: "s",  4: "c",  5: "b",
        -1: "dbar", -2: "ubar", -3: "sbar", -4: "cbar", -5: "bbar",
    }
    return m.get(pid, f"pid{pid}")


def pid_to_tex(pid: int) -> str:
    m = {
        21: "g",
        1: "d",  2: "u",  3: "s",  4: "c",  5: "b",
        -1: "#bar{d}", -2: "#bar{u}", -3: "#bar{s}", -4: "#bar{c}", -5: "#bar{b}",
    }
    return m.get(pid, f"pid={pid}")

def fmt_edge(x: float) -> str:
    # integers like 3000.0 -> "3000", but 0.5 -> "0.5"
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.1f}"

# =============================================================================
# Argparser
# =============================================================================

parser = argparse.ArgumentParser("Plot 1D correlation: PDF(x) vs feature-bin yield.")
parser.add_argument("--pdf-set", dest="pdf_setname", default="NNPDF31_nnlo_hessian_pdfas", help="LHAPDF set name")
parser.add_argument("--postfix", dest="postfix", default=None, help="Append to filename.")
parser.add_argument("--pid", dest="pid", type=int, default=21, help="LHAPDF PID for PDF (e.g. 21=g)")
parser.add_argument("--Q", dest="Q", type=float, default=100.0, help="Scale Q [GeV] for the PDF(x,Q) template")

parser.add_argument("--logxmin", dest="logxmin", type=float, default=-1.99, help="log10(xmin) for PDF x-axis")
parser.add_argument("--logxmax", dest="logxmax", type=float, default=0.0, help="log10(xmax) for PDF x-axis")
parser.add_argument("--nx", dest="nx", type=int, default=200, help="Number of x bins for PDF template")

parser.add_argument("--sample", dest="sample", default="TTLep_pow_2018", help="Sample loader name from data.samples_RunII")
parser.add_argument("--feature", dest="feature", default="tr_ttbar_mass", help="Feature name passed to loader.setFeatures([...])")
parser.add_argument("--fmin", dest="fmin", type=float, default=345.0, help="Feature histogram min (raise to avoid empty first bin)")
parser.add_argument("--fmax", dest="fmax", type=float, default=1600.0, help="Feature histogram max")
parser.add_argument("--nbins", dest="nbins", type=int, default=10, help="Number of feature bins (coarse bins to overlay)")
parser.add_argument("--max-events", dest="max_events", type=int, default=None, help="Load all shards but keep only first N events (debug)")
parser.add_argument("--bin-edges", dest="bin_edges", default=None,  help='Optional comma-separated feature bin edges')
parser.add_argument("--normalize_feature", action="store_true", help="Use normalized distributions?")
parser.add_argument("--replicas", action="store_true", help="Treat members as MC replicas: subtract replica mean (use mem=1..end as ensemble).")
parser.add_argument("--abs", action="store_true", help="Take the absolute value of the feature")
parser.add_argument("--absEtaMin", type=float, default=None, help="Minimum |tr_ttbar_eta|")
parser.add_argument("--absEtaMax", type=float, default=None,  help="Maximum |tr_ttbar_eta|")

args = parser.parse_args()

pdf_setname = args.pdf_setname
pid = args.pid
Q_pdf = args.Q

x_edges = np.logspace(args.logxmin, args.logxmax, args.nx + 1)
x_centers = np.exp(0.5 * (np.log(x_edges[:-1]) + np.log(x_edges[1:])))

if args.bin_edges is not None:
    feature_edges = np.array([float(x) for x in args.bin_edges.split(",")], dtype=float)
else:
    feature_edges = np.linspace(args.fmin, args.fmax, args.nbins + 1)


include_alphas_members = False  # PDF-only

# =============================================================================
# Instantiate templates
# =============================================================================

pdf = PDFParametrization(pdf_setname, include_alphas_members=include_alphas_members)

A = PDFShapeTemplate(pdf, pid=pid, x_edges=x_edges, Q=Q_pdf, name=f"{pid_to_tex(pid)}(x,Q)")
B = FeatureTemplate(
    pdf=pdf,
    sample=args.sample,
    feature=args.feature,
    bin_edges=feature_edges,
    module_samples="data.samples_RunII",
    name=f"{args.sample}:{args.feature}",
    max_events=args.max_events,
    use_abs=args.abs,
    absEtaMin=args.absEtaMin,
    absEtaMax=args.absEtaMax,
)

# =============================================================================
# Compute templates (A: x-binned PDF; B: feature-binned yields)
# =============================================================================

nmem = min(A.n_members, B.n_members)
A_templates = []
B_templates = []

for m in range(nmem):
    print(f"Constructing template {m}")
    A_templates.append(A.get_template(m))
    B_template = B.get_template(m)
    if args.normalize_feature:
        B_template/=B_template.sum() 
    B_templates.append(B_template)

A_templates = np.stack(A_templates, axis=0)  # (nmem, nX)
B_templates = np.stack(B_templates, axis=0)  # (nmem, nF)

nX = A_templates.shape[1]
nF = B_templates.shape[1]

# =============================================================================
# Correlation matrix rho(x_bin, feature_bin)
# =============================================================================

if args.replicas:
    # Replicas ensemble: use members 1..end, subtract replica mean
    A_ens = A_templates[1:]
    B_ens = B_templates[1:]

    Abar = np.mean(A_ens, axis=0)
    Bbar = np.mean(B_ens, axis=0)

    dA = A_ens - Abar
    dB = B_ens - Bbar
else:
    # Hessian/symmhessian: use members 1..end, subtract central (member 0)
    A0 = A_templates[0]
    B0 = B_templates[0]

    dA = A_templates[1:] - A0
    dB = B_templates[1:] - B0

cov = dA.T @ dB
varA = np.sum(dA * dA, axis=0)
varB = np.sum(dB * dB, axis=0)

den = np.sqrt(varA[:, None] * varB[None, :])
rho = np.where(den > 0.0, cov / den, 0.0)
rho = np.clip(rho, -1.0, 1.0)

# =============================================================================
# Plot: overlay rho(x) curves for each feature bin
# =============================================================================

ROOT.gStyle.SetOptStat(0)
ROOT.TGaxis.SetMaxDigits(3)

colors = [
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

c = ROOT.TCanvas("c", "c", 900, 850)
c.SetRightMargin(0.05)
c.SetLeftMargin(0.12)
c.SetBottomMargin(0.16)
c.SetTopMargin(0.13)
c.SetLogx(True)

frame = ROOT.TH1D("frame", ";x;#rho(PDF(x), bin)", 1, float(x_edges[0]), float(x_edges[-1]))
frame.SetMinimum(-1.0)
frame.SetMaximum(+1.0)
frame.GetXaxis().SetMoreLogLabels(False)
frame.GetXaxis().SetNoExponent(False)
frame.Draw("AXIS")

# --- replace the legend definition block with this ---

leg = ROOT.TLegend(0.16, 0.18, 0.9, 0.35)  # twice as wide, half as tall (keep lower-left anchored)
leg.SetNColumns(2)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.030)

graphs = []
for ib in range(nF):
    y = rho[:, ib]
    gr = ROOT.TGraph(nX, array('d', x_centers.tolist()), array('d', y.astype(float).tolist()))
    gr.SetLineWidth(2)
    gr.SetLineColor(int(colors[ib % len(colors)]))
    gr.SetMarkerColor(int(colors[ib % len(colors)]))
    gr.SetMarkerStyle(0)
    gr.Draw("L SAME")
    graphs.append(gr)

    lo = feature_edges[ib]
    hi = feature_edges[ib + 1]

    f_name = plot_options[args.feature]["tex"]
    if args.abs:
        f_name = f"|{f_name}|"
    leg.AddEntry(gr, f"{fmt_edge(lo)}#leq {f_name} < {fmt_edge(hi)}", "l")

leg.Draw()

lat = ROOT.TLatex()
lat.SetNDC(True)
lat.SetTextFont(42)
lat.SetTextSize(0.035)

lat.DrawLatex(0.12, 0.965, f"#bf{{PDF feature correlation}}  {pdf_setname}")
lat.SetTextSize(0.030)
lat.DrawLatex(0.12, 0.925, f"PDF: {pid_to_tex(pid)}(x, Q={Q_pdf:.0f} GeV)   Sample: {args.sample}")
lat.DrawLatex(0.12, 0.895, f"Feature: {args.feature}   bins={args.nbins}   max_events={args.max_events}")

# =============================================================================
# Output
# =============================================================================

out_dir = "pdf_feature_correlations"
postfix = "_"+args.postfix if args.postfix is not None else ""
out_base = f"corr1d{postfix}_{pdf_setname}_{pid_to_short(pid)}_{args.sample}_{args.feature}"
plot_directory = os.path.join(user.plot_directory, out_dir)
os.makedirs(plot_directory, exist_ok=True)
print(f"[info] plots will be written under: {plot_directory}")
helpers.copyIndexPHP(plot_directory)

c.SaveAs(os.path.join(plot_directory, out_base + ".png"))
c.SaveAs(os.path.join(plot_directory, out_base + ".pdf"))

print("Done")
syncer.sync()

