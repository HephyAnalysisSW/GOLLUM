#!/usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import ROOT

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
import samples_postprocessed


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


FEATURES = [
    ("dy_born_mll", "m_{ll}^{Born} [GeV]", 60, 50, 130),
    ("dy_born_yll", "y_{ll}^{Born}", 50, -5, 5),
    ("dy_born_abs_yll", "|y_{ll}^{Born}|", 50, 0, 5),
    ("dy_born_ptll", "p_{T,ll}^{Born} [GeV]", 50, 0, 200),
    ("dy_born_qt_over_m", "q_{T}/m_{ll}^{Born}", 50, 0, 2.0),
    ("dy_ptll", "p_{T,ll}^{dressed} [GeV]", 50, 0, 200),
    ("dy_qt_over_m", "q_{T}/m_{ll}^{dressed}", 50, 0, 2.0),
    ("cs_costheta", "cos#theta_{CS}^{dressed}", 50, -1, 1),
    ("cs_phi", "#phi_{CS}^{dressed}", 50, -math.pi, math.pi),
    ("truth_quark_direction", "true quark direction", 3, -1.5, 1.5),
    ("truth_flavour_label", "truth flavour label", 9, -0.5, 8.5),
    ("gen_x1", "Generator x_{1}", 50, 0, 1),
    ("gen_x2", "Generator x_{2}", 50, 0, 1),
    ("gen_scalePDF", "Generator Q [GeV]", 50, 0, 250),
]

AFB_MASS_BINS = np.array([50, 60, 70, 80, 86, 91, 96, 106, 120, 150, 200], dtype=np.float64)

# CMS-style unrolled A4 binning:
# outer blocks are |Y| bins; inside each block the x-coordinate is M.
A4_MASS_BINS = np.array([60, 70, 80, 86, 91, 96, 106, 120, 133], dtype=np.float64)
A4_ABSY_BINS = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.7, 3.0, 3.4], dtype=np.float64)


def sanitize(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def fill_hist(hist, values, weights):
    finite = np.isfinite(values) & np.isfinite(weights) & (values > -998)
    values = np.ascontiguousarray(values[finite], dtype=np.float64)
    weights = np.ascontiguousarray(weights[finite], dtype=np.float64)
    if len(values):
        hist.FillN(len(values), values, weights)


def materialized(sample):
    if not hasattr(sample, "_dy_materialized"):
        features, weights = sample.materialize(0, "fw")
        sample._dy_materialized = (features, weights)
    return sample._dy_materialized


def require_feature(sample, branch):
    if branch not in sample.feature_names:
        raise RuntimeError(f"Required branch '{branch}' not found in sample {sample.name}")
    return sample.feature_names.index(branch)


def make_stack_plot(plot_dir, selected_samples, feature):
    branch, xtitle, nbins, xmin, xmax = feature
    canvas = ROOT.TCanvas(f"c_{sanitize(branch)}", branch, 800, 700)
    canvas.SetLogy(True)
    stack = ROOT.THStack(f"stack_{sanitize(branch)}", f";{xtitle};Weighted events")
    legend = ROOT.TLegend(0.58, 0.72, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    stuff = [stack, legend]

    for sample in selected_samples:
        features, weights = materialized(sample)
        values = features[:, sample.feature_names.index(branch)]
        hist = ROOT.TH1F(f"h_{sanitize(branch)}_{sample.name}", "", nbins, xmin, xmax)
        hist.SetFillColor(sample.color)
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        fill_hist(hist, values, weights)
        stack.Add(hist)
        legend.AddEntry(hist, sample.tex_name, "f")
        stuff.append(hist)

    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(xtitle)
    stack.GetYaxis().SetTitle("Weighted events")
    legend.Draw()

    base = os.path.join(plot_dir, sanitize(branch))
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    stack.Write("stack")
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


def compute_a4_unrolled(sample):
    """Compute unrolled A4(M, |Y|) using Born-level branches only.

    Definition:
        A4 = 4 * < sign(y_ll^Born) * cos(theta_CS^Born) >

    Error:
        sumw2 weighted-mean error,

        Var(<c>) = sum_i w_i^2 (c_i - <c>)^2 / (sum_i w_i)^2

        sigma(A4) = 4 * sqrt(Var(<c>)).
    """
    features, weights = materialized(sample)

    i_mll = require_feature(sample, "dy_born_mll")
    i_yll = require_feature(sample, "dy_born_yll")
    i_abs_yll = require_feature(sample, "dy_born_abs_yll")
    i_costheta = require_feature(sample, "cs_born_costheta")

    mll = features[:, i_mll]
    yll = features[:, i_yll]
    abs_yll = features[:, i_abs_yll]
    costheta = features[:, i_costheta]

    signed_costheta = np.sign(yll) * costheta

    n_y = len(A4_ABSY_BINS) - 1
    n_m = len(A4_MASS_BINS) - 1

    a4 = np.full((n_y, n_m), np.nan, dtype=np.float64)
    a4_err = np.full((n_y, n_m), np.nan, dtype=np.float64)
    sumw = np.zeros((n_y, n_m), dtype=np.float64)
    sumw2 = np.zeros((n_y, n_m), dtype=np.float64)

    finite = (
        np.isfinite(mll)
        & np.isfinite(yll)
        & np.isfinite(abs_yll)
        & np.isfinite(costheta)
        & np.isfinite(weights)
        & (mll > -998)
        & (abs_yll > -998)
        & (costheta > -998)
        & (np.sign(yll) != 0)
    )

    for iy, (ylo, yhi) in enumerate(zip(A4_ABSY_BINS[:-1], A4_ABSY_BINS[1:])):
        for im, (mlo, mhi) in enumerate(zip(A4_MASS_BINS[:-1], A4_MASS_BINS[1:])):
            in_bin = finite & (abs_yll >= ylo) & (abs_yll < yhi) & (mll >= mlo) & (mll < mhi)
            if not np.any(in_bin):
                continue

            w = weights[in_bin]
            c = signed_costheta[in_bin]

            sw = np.sum(w)
            sw2 = np.sum(w * w)

            if sw == 0:
                continue

            mean_c = np.sum(w * c) / sw
            a4[iy, im] = 4.0 * mean_c
            sumw[iy, im] = sw
            sumw2[iy, im] = sw2

            # Sumw2 error for the weighted mean of c.
            # This includes the covariance between numerator and denominator
            # through the residual form.
            var_mean_c = np.sum((w * w) * (c - mean_c) * (c - mean_c)) / (sw * sw)
            a4_err[iy, im] = 4.0 * math.sqrt(max(0.0, var_mean_c))

    return a4, a4_err, sumw, sumw2


def mass_to_unrolled_x(iy, mass):
    mmin = A4_MASS_BINS[0]
    mmax = A4_MASS_BINS[-1]
    return iy + (mass - mmin) / (mmax - mmin)


def make_graph_for_a4_values(values, errors, sample_name, iy, color, marker_style):
    graph = ROOT.TGraphErrors()
    graph.SetName(f"g_A4_unrolled_{sanitize(sample_name)}_iy{iy}")
    graph.SetLineColor(color)
    graph.SetMarkerColor(color)
    graph.SetMarkerStyle(marker_style)
    graph.SetMarkerSize(0.65)
    graph.SetLineWidth(2)

    ip = 0
    for im, (mlo, mhi) in enumerate(zip(A4_MASS_BINS[:-1], A4_MASS_BINS[1:])):
        value = values[iy, im]
        error = errors[iy, im]
        if not np.isfinite(value):
            continue

        x = mass_to_unrolled_x(iy, 0.5 * (mlo + mhi))
        ex = 0.0
        ey = error if np.isfinite(error) else 0.0

        graph.SetPoint(ip, x, value)
        graph.SetPointError(ip, ex, ey)
        ip += 1

    return graph if ip > 0 else None


def draw_unrolled_guides(ymin, ymax):
    stuff = []
    n_y = len(A4_ABSY_BINS) - 1
    yrange = ymax - ymin

    for iy in range(1, n_y):
        line = ROOT.TLine(iy, ymin, iy, ymax)
        line.SetLineStyle(3)
        line.SetLineWidth(1)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
        stuff.append(line)

    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.026)
    stuff.append(latex)

    y_text = ymax - 0.11 * yrange
    for iy, (ylo, yhi) in enumerate(zip(A4_ABSY_BINS[:-1], A4_ABSY_BINS[1:])):
        latex.DrawLatex(iy + 0.5, y_text, f"#splitline{{|Y|}}{{{ylo:.1f}-{yhi:.1f}}}")

    latex.SetTextAngle(90)
    latex.SetTextSize(0.024)
    y_mass = ymin + 0.09 * yrange
    for iy in range(n_y):
        latex.DrawLatex(iy + 0.10, y_mass, f"{A4_MASS_BINS[0]:.0f}")
        latex.DrawLatex(iy + 0.90, y_mass, f"{A4_MASS_BINS[-1]:.0f}")
    latex.SetTextAngle(0)

    return stuff


def make_unrolled_a4_plot(plot_dir, selected_samples):
    if not selected_samples:
        return

    n_y = len(A4_ABSY_BINS) - 1

    canvas = ROOT.TCanvas("c_A4_unrolled", "A4 unrolled", 1300, 720)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.10)
    canvas.SetBottomMargin(0.18)

    stuff = [canvas]

    ymin, ymax = -1.25, 2.25

    frame = ROOT.TH2F(
        "frame_A4_unrolled",
        "; ;A_{4}",
        n_y,
        0.0,
        float(n_y),
        100,
        ymin,
        ymax,
    )
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTickLength(0.02)
    frame.GetYaxis().SetTitleSize(0.048)
    frame.GetYaxis().SetLabelSize(0.040)
    frame.GetYaxis().SetTitleOffset(0.88)
    frame.Draw()
    stuff.append(frame)

    cms = ROOT.TLatex()
    cms.SetNDC(True)
    cms.SetTextFont(62)
    cms.SetTextSize(0.040)
    cms.DrawLatex(0.11, 0.94, "CMS")
    stuff.append(cms)

    sim = ROOT.TLatex()
    sim.SetNDC(True)
    sim.SetTextFont(52)
    sim.SetTextSize(0.037)
    sim.DrawLatex(0.165, 0.94, "Simulation")
    stuff.append(sim)

    energy = ROOT.TLatex()
    energy.SetNDC(True)
    energy.SetTextFont(42)
    energy.SetTextSize(0.040)
    energy.SetTextAlign(31)
    energy.DrawLatex(0.95, 0.94, "13 TeV")
    stuff.append(energy)

    legend = ROOT.TLegend(0.34, 0.92, 0.62, 0.975)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.026)
    stuff.append(legend)

    stuff += draw_unrolled_guides(ymin, ymax)

    for i_sample, sample in enumerate(selected_samples):
        a4, a4_err, sumw, sumw2 = compute_a4_unrolled(sample)

        for iy in range(n_y):
            graph = make_graph_for_a4_values(
                a4,
                a4_err,
                sample.name,
                iy,
                sample.color,
                20 + i_sample,
            )
            if graph is None:
                continue

            graph.Draw("PZ SAME")
            stuff.append(graph)

            if iy == 0:
                legend.AddEntry(graph, sample.tex_name, "pe")

    legend.Draw()

    mass_label = ROOT.TLatex()
    mass_label.SetNDC(True)
    mass_label.SetTextFont(42)
    mass_label.SetTextSize(0.038)
    mass_label.SetTextAlign(31)
    mass_label.DrawLatex(0.95, 0.060, "M (GeV)")
    stuff.append(mass_label)

    meta_text = (
        "A4_definition=4*weighted_mean(sign(dy_born_yll)*cs_born_costheta); "
        "branches=dy_born_mll,dy_born_yll,dy_born_abs_yll,cs_born_costheta; "
        "error=sumw2_residual_weighted_mean"
    )
    meta_named = ROOT.TNamed("metadata", meta_text)
    stuff.append(meta_named)

    base = os.path.join(plot_dir, "A4_unrolled")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    meta_named.Write()
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()

    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


def make_afb_plot(plot_dir, selected_samples):
    canvas = ROOT.TCanvas("c_afb_born", "AFB Born", 800, 700)
    canvas.SetGrid()
    legend = ROOT.TLegend(0.52, 0.72, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    frame = ROOT.TH2F("frame_afb", ";m_{ll}^{Born} [GeV];A_{FB}^{Born}", 10, AFB_MASS_BINS[0], AFB_MASS_BINS[-1], 100, -1, 1)
    frame.Draw()
    stuff = [frame, legend]

    for i_sample, sample in enumerate(selected_samples):
        features, weights = materialized(sample)
        i_mll = sample.feature_names.index("dy_born_mll")
        i_costheta = sample.feature_names.index("cs_born_costheta")
        i_qdir = sample.feature_names.index("truth_quark_direction")
        mll = features[:, i_mll]
        costheta_trueq = features[:, i_costheta] * features[:, i_qdir]

        xvals, yvals, yerrs = [], [], []
        for lo, hi in zip(AFB_MASS_BINS[:-1], AFB_MASS_BINS[1:]):
            in_bin = (mll >= lo) & (mll < hi) & (costheta_trueq != 0)
            fwd = np.sum(weights[in_bin & (costheta_trueq > 0)])
            bwd = np.sum(weights[in_bin & (costheta_trueq < 0)])
            total = fwd + bwd
            if total == 0:
                continue
            afb = (fwd - bwd) / total
            bin_weights = weights[in_bin]
            neff = (np.sum(bin_weights) ** 2) / np.sum(bin_weights ** 2) if np.sum(bin_weights ** 2) > 0 else 0.0
            err = math.sqrt(max(0.0, (1.0 - afb * afb) / neff)) if neff > 0 else 0.0
            xvals.append(0.5 * (lo + hi))
            yvals.append(afb)
            yerrs.append(err)

        graph = ROOT.TGraphErrors(len(xvals))
        for ip, (x, y, ey) in enumerate(zip(xvals, yvals, yerrs)):
            graph.SetPoint(ip, x, y)
            graph.SetPointError(ip, 0.0, ey)
        graph.SetLineColor(sample.color)
        graph.SetMarkerColor(sample.color)
        graph.SetMarkerStyle(20 + i_sample)
        graph.SetLineWidth(2)
        graph.Draw("P SAME")
        legend.AddEntry(graph, sample.tex_name, "lp")
        stuff.append(graph)

    legend.Draw()
    base = os.path.join(plot_dir, "AFB_born_trueq_vs_mll")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    for obj in stuff:
        obj.Write()
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


parser = argparse.ArgumentParser()
parser.add_argument("--samples", nargs="+", default=["DYJetsToLL_M50_LO_UL17"], help="Samples from samples_postprocessed.py")
parser.add_argument("--max-files", type=int, default=2, help="Files per sample")
parser.add_argument("--level", choices=["both", "fiducial", "parton"], default="both", help="Selection level to plot")
parser.add_argument("--small", action="store_true", help="Use one file per sample and write to a _small directory")
args = parser.parse_args()

if args.small:
    args.max_files = 1

helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features"))

levels = ["fiducial", "parton"] if args.level == "both" else [args.level]

for level in levels:
    selected_samples = []
    for sample_name in args.samples:
        if sample_name not in samples_postprocessed.samples_by_name:
            raise RuntimeError(f"Unknown sample '{sample_name}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")
        sample = samples_postprocessed.samples_by_name[sample_name].get_loader(max_files=args.max_files, selection=level)
        selected_samples.append(sample)

    label = "_".join(sample.name for sample in selected_samples)
    if args.small:
        label += "_small"

    plot_dir = os.path.join(user.plot_directory, "DY", "gen_features", level, label)
    os.makedirs(plot_dir, exist_ok=True)
    helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features", level))
    helpers.copyIndexPHP(plot_dir)

    for feature in FEATURES:
        make_stack_plot(plot_dir, selected_samples, feature)

    make_afb_plot(plot_dir, selected_samples)
    make_unrolled_a4_plot(plot_dir, selected_samples)

syncer.sync()
