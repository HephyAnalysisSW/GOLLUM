#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from array import array
import os
import argparse

import ROOT
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user   as user
import common.syncer as syncer
import common.helpers as helpers

from PDFParametrization import PDFParametrization
from PDFTemplate import PDFShapeTemplate

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


# =============================================================================
# Argparser
# =============================================================================

parser = argparse.ArgumentParser("Plot PDF-induced correlation heatmaps (symmhessian).")
parser.add_argument("--pdf-set", dest="pdf_setname", default="NNPDF31_nnlo_hessian_pdfas", help="LHAPDF set name")
parser.add_argument("--pidA", dest="pid_A", type=int, default=21, help="LHAPDF PID for observable A (e.g. 21=g)")
parser.add_argument("--pidB", dest="pid_B", type=int, default=21, help="LHAPDF PID for observable B (e.g. 21=g)")
parser.add_argument("--Q", dest="Q", type=float, default=100.0, help="Scale Q [GeV] at which PDFs are evaluated")

parser.add_argument("--logxminA", dest="logxminA", type=float, default=-3.99, help="log10(xmin) for A axis")
parser.add_argument("--logxmaxA", dest="logxmaxA", type=float, default=0.0, help="log10(xmax) for A axis")
parser.add_argument("--logxminB", dest="logxminB", type=float, default=-3.99, help="log10(xmin) for B axis")
parser.add_argument("--logxmaxB", dest="logxmaxB", type=float, default=0.0, help="log10(xmax) for B axis")

parser.add_argument("--nbins", dest="nbins", type=int, default=200, help="Number of bins (fine) per axis")

parser.add_argument("--replicas", action="store_true",
                    help="Treat members as MC replicas: subtract replica mean (use mem=1..end as ensemble).")

args = parser.parse_args()

pdf_setname = args.pdf_setname
pid_A = args.pid_A
pid_B = args.pid_B
Q = args.Q

x_edges_A = np.logspace(args.logxminA, args.logxmaxA, args.nbins + 1)
x_edges_B = np.logspace(args.logxminB, args.logxmaxB, args.nbins + 1)

include_alphas_members = False  # PDF-only for now: central + symmhessian eigens

# =============================================================================
# Configuration: labels
# =============================================================================

tex_label = "PDF correlation"
plot_lines = [
#    f"{pdf_setname}",
#    f"A: {pid_to_tex(pid_A)}(x, Q={Q:.0f} GeV)   B: {pid_to_tex(pid_B)}(x, Q={Q:.0f} GeV)",
#    f"x(A): [10^{{{args.logxminA:.2f}}}, 10^{{{args.logxmaxA:.2f}}}]   x(B): [10^{{{args.logxminB:.2f}}}, 10^{{{args.logxmaxB:.2f}}}]",
#    "Correlation from symmhessian eigenmembers (PDF-only)",
]

# =============================================================================
# Instantiate providers
# =============================================================================

pdf = PDFParametrization(pdf_setname, include_alphas_members=include_alphas_members)
A_provider = PDFShapeTemplate(pdf, pid=pid_A, x_edges=x_edges_A, Q=Q, name=f"{pid_to_tex(pid_A)}(x,Q)")
B_provider = PDFShapeTemplate(pdf, pid=pid_B, x_edges=x_edges_B, Q=Q, name=f"{pid_to_tex(pid_B)}(x,Q)")

# =============================================================================
# Template computation (later: cache this when templates become expensive)
# =============================================================================

nmem = min(A_provider.n_members, B_provider.n_members)
A_templates = []
B_templates = []

for m in range(nmem):
    A_templates.append(A_provider.get_template(m))
    B_templates.append(B_provider.get_template(m))

A_templates = np.stack(A_templates, axis=0)  # (nmem, nA)
B_templates = np.stack(B_templates, axis=0)  # (nmem, nB)

nA = A_templates.shape[1]
nB = B_templates.shape[1]

# =============================================================================
# Compute correlations and plot (2D heatmap)
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

    dA = A_templates[1:] - A0  # (nmem-1, nA)
    dB = B_templates[1:] - B0  # (nmem-1, nB)

cov = dA.T @ dB  # (nA, nB)
varA = np.sum(dA * dA, axis=0)  # (nA,)
varB = np.sum(dB * dB, axis=0)  # (nB,)

den = np.sqrt(varA[:, None] * varB[None, :])
rho = np.where(den > 0.0, cov / den, 0.0)
rho = np.clip(rho, -1.0, 1.0)

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetNumberContours(100)
ROOT.gStyle.SetPalette(ROOT.kViridis)
ROOT.TGaxis.SetMaxDigits(3)

xA = array('d', A_provider.get_x_edges().tolist())
xB = array('d', B_provider.get_x_edges().tolist())

hname = "hCorr"
htitle = f";{A_provider.name};{B_provider.name}"
h = ROOT.TH2D(hname, htitle, nA, xA, nB, xB)

for ia in range(nA):
    for ib in range(nB):
        h.SetBinContent(ia + 1, ib + 1, float(rho[ia, ib]))

h.SetMinimum(-1.0)
h.SetMaximum(+1.0)

h.GetXaxis().SetMoreLogLabels(False)
h.GetYaxis().SetMoreLogLabels(False)
h.GetXaxis().SetNoExponent(False)
h.GetYaxis().SetNoExponent(False)

c = ROOT.TCanvas("c", "c", 900, 850)
c.SetRightMargin(0.16)
c.SetLeftMargin(0.12)
c.SetBottomMargin(0.16)
c.SetTopMargin(0.08)
c.SetLogx(True)
c.SetLogy(True)

h.Draw("COLZ")

# Diagonal dashed line (lower-left to upper-right in axis coordinates)
xmin = float(xA[0])
xmax = float(xA[-1])
ymin = float(xB[0])
ymax = float(xB[-1])

diag = ROOT.TLine(xmin, ymin, xmax, ymax)
diag.SetLineStyle(2)   # dashed
diag.SetLineWidth(2)
diag.SetLineColor(ROOT.kBlack)
diag.Draw("SAME")

lat = ROOT.TLatex()
lat.SetNDC(True)
lat.SetTextFont(42)
lat.SetTextSize(0.035)

lat.DrawLatex(0.12, 0.965, f"#bf{{{tex_label}}} {pdf_setname}")

y0 = 0.92
dy = 0.04
for i, line in enumerate(plot_lines):
    lat.DrawLatex(0.12, y0 - i * dy, line)

# Output: reflect flavors
out_dir = f"pdf_correlations/{pdf_setname}"

out_base = f"corr2d_{pid_to_short(pid_A)}_{pid_to_short(pid_B)}"
plot_directory = os.path.join(user.plot_directory, out_dir)
os.makedirs(plot_directory, exist_ok=True)
print(f"[info] plots will be written under: {plot_directory}")
helpers.copyIndexPHP(plot_directory)

c.SaveAs(os.path.join(plot_directory, out_base + ".png"))
c.SaveAs(os.path.join(plot_directory, out_base + ".pdf"))

print("Done")
syncer.sync()

