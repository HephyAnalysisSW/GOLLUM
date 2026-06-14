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
import eft_reweighting
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
A4_MASS_BINS = np.array([60, 70, 80, 86, 91, 96, 106, 120, 133], dtype=np.float64)
A4_ABSY_BINS = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.7, 3.0, 3.4], dtype=np.float64)
EFT_COLORS = [ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2]
EFT_MARKERS = [24, 25, 26, 32, 27, 28]


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


def target_weights(sample, nominal_weights, eft_point):
    label, values = eft_point
    if not hasattr(sample, "_eft_ratio_cache"):
        sample._eft_ratio_cache = {}
    key = tuple(sorted(values.items()))
    if key not in sample._eft_ratio_cache:
        ar = sample.load_selection_shard(0)
        ratio = eft_reweighting.eft_weight(
            sample.vector_branch(ar, "LHEReweightingWeight"),
            config="auto",
            **values,
        )
        sample._eft_ratio_cache[key] = ratio
        valid = np.isfinite(ratio)
        if np.any(valid):
            print(
                f"[EFT] {sample.name} {label}: valid={np.count_nonzero(valid)}/{len(ratio)} "
                f"target_weight_min={np.min(ratio[valid]):.4g} target_weight_max={np.max(ratio[valid]):.4g}"
            )
        else:
            print(f"[EFT] {sample.name} {label}: valid=0/{len(ratio)}")
    ratio = sample._eft_ratio_cache[key]
    if len(ratio) != len(nominal_weights):
        raise RuntimeError(f"EFT ratio length mismatch for {sample.name}: {len(ratio)} != {len(nominal_weights)}")
    return nominal_weights * ratio


def require_feature(sample, branch):
    if branch not in sample.feature_names:
        raise RuntimeError(f"Required branch '{branch}' not found in sample {sample.name}")
    return sample.feature_names.index(branch)


def make_stack_plot(plot_dir, selected_samples, feature, eft_points):
    branch, xtitle, nbins, xmin, xmax = feature
    canvas = ROOT.TCanvas(f"c_{sanitize(branch)}", branch, 800, 700)
    canvas.SetLogy(True)
    stack = ROOT.THStack(f"stack_{sanitize(branch)}", f";{xtitle};Weighted events")
    legend = ROOT.TLegend(0.56, 0.66, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    stuff = [stack, legend]
    positive_bins = []

    for sample in selected_samples:
        features, nominal = materialized(sample)
        values = features[:, require_feature(sample, branch)]
        weights = target_weights(sample, nominal, ("SM", {}))
        hist = ROOT.TH1F(f"h_{sanitize(branch)}_{sample.name}_SM", "", nbins, xmin, xmax)
        hist.SetFillColor(sample.color)
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        fill_hist(hist, values, weights)
        positive_bins.extend(hist.GetBinContent(ib) for ib in range(1, hist.GetNbinsX() + 1) if hist.GetBinContent(ib) > 0)
        stack.Add(hist)
        legend.AddEntry(hist, f"{sample.tex_name} SM", "f")
        stuff.append(hist)

    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(xtitle)
    stack.GetYaxis().SetTitle("Weighted events")

    for i_point, eft_point in enumerate(eft_points):
        label, values = eft_point
        color = EFT_COLORS[i_point % len(EFT_COLORS)]
        for sample in selected_samples:
            features, nominal = materialized(sample)
            hist = ROOT.TH1F(f"h_{sanitize(branch)}_{sample.name}_{sanitize(label)}", "", nbins, xmin, xmax)
            hist.SetLineColor(color)
            hist.SetLineWidth(2)
            hist.SetFillStyle(0)
            fill_hist(hist, features[:, require_feature(sample, branch)], target_weights(sample, nominal, eft_point))
            positive_bins.extend(hist.GetBinContent(ib) for ib in range(1, hist.GetNbinsX() + 1) if hist.GetBinContent(ib) > 0)
            hist.Draw("HIST SAME")
            legend.AddEntry(hist, f"{sample.tex_name} {label}", "l")
            stuff.append(hist)

    if positive_bins:
        stack.SetMinimum(max(1e-8, 0.5 * min(positive_bins)))
        stack.SetMaximum(2.0 * max(positive_bins))
    legend.Draw()
    base = os.path.join(plot_dir, sanitize(branch))
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    stack.Write("stack")
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


def compute_a4_unrolled(sample, weights):
    features, _ = materialized(sample)
    mll = features[:, require_feature(sample, "dy_born_mll")]
    yll = features[:, require_feature(sample, "dy_born_yll")]
    abs_yll = features[:, require_feature(sample, "dy_born_abs_yll")]
    costheta = features[:, require_feature(sample, "cs_born_costheta")]
    signed_costheta = np.sign(yll) * costheta

    n_y = len(A4_ABSY_BINS) - 1
    n_m = len(A4_MASS_BINS) - 1
    a4 = np.full((n_y, n_m), np.nan, dtype=np.float64)
    a4_err = np.full((n_y, n_m), np.nan, dtype=np.float64)
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
            if sw == 0:
                continue
            mean_c = np.sum(w * c) / sw
            a4[iy, im] = 4.0 * mean_c
            a4_err[iy, im] = 4.0 * math.sqrt(max(0.0, np.sum((w * w) * (c - mean_c) * (c - mean_c)) / (sw * sw)))
    return a4, a4_err


def mass_to_unrolled_x(iy, mass):
    return iy + (mass - A4_MASS_BINS[0]) / (A4_MASS_BINS[-1] - A4_MASS_BINS[0])


def make_a4_graph(values, errors, name, iy, color, marker):
    graph = ROOT.TGraphErrors()
    graph.SetName(f"g_A4_{sanitize(name)}_iy{iy}")
    graph.SetLineColor(color)
    graph.SetMarkerColor(color)
    graph.SetMarkerStyle(marker)
    graph.SetMarkerSize(0.65)
    graph.SetLineWidth(2)
    ip = 0
    for im, (mlo, mhi) in enumerate(zip(A4_MASS_BINS[:-1], A4_MASS_BINS[1:])):
        if not np.isfinite(values[iy, im]):
            continue
        graph.SetPoint(ip, mass_to_unrolled_x(iy, 0.5 * (mlo + mhi)), values[iy, im])
        graph.SetPointError(ip, 0.0, errors[iy, im] if np.isfinite(errors[iy, im]) else 0.0)
        ip += 1
    return graph if ip else None


def draw_unrolled_guides(ymin, ymax):
    stuff = []
    n_y = len(A4_ABSY_BINS) - 1
    for iy in range(1, n_y):
        line = ROOT.TLine(iy, ymin, iy, ymax)
        line.SetLineStyle(3)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
        stuff.append(line)
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.026)
    stuff.append(latex)
    y_text = ymax - 0.11 * (ymax - ymin)
    for iy, (ylo, yhi) in enumerate(zip(A4_ABSY_BINS[:-1], A4_ABSY_BINS[1:])):
        latex.DrawLatex(iy + 0.5, y_text, f"#splitline{{|Y|}}{{{ylo:.1f}-{yhi:.1f}}}")
    return stuff


def make_unrolled_a4_plot(plot_dir, selected_samples, eft_points):
    n_y = len(A4_ABSY_BINS) - 1
    ymin, ymax = -1.25, 2.25
    canvas = ROOT.TCanvas("c_A4_unrolled", "A4 unrolled", 1300, 720)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.10)
    canvas.SetBottomMargin(0.18)
    frame = ROOT.TH2F("frame_A4_unrolled", "; ;A_{4}", n_y, 0.0, float(n_y), 100, ymin, ymax)
    frame.GetXaxis().SetLabelSize(0)
    frame.Draw()
    legend = ROOT.TLegend(0.34, 0.90, 0.84, 0.975)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.024)
    stuff = [canvas, frame, legend] + draw_unrolled_guides(ymin, ymax)

    for i_sample, sample in enumerate(selected_samples):
        features, nominal = materialized(sample)
        graph_specs = [(f"{sample.tex_name} SM", target_weights(sample, nominal, ("SM", {})), sample.color, 20 + i_sample, sample.name + "_SM")]
        for i_point, eft_point in enumerate(eft_points):
            label, values = eft_point
            graph_specs.append(
                (
                    f"{sample.tex_name} {label}",
                    target_weights(sample, nominal, eft_point),
                    EFT_COLORS[i_point % len(EFT_COLORS)],
                    EFT_MARKERS[i_point % len(EFT_MARKERS)],
                    sample.name + "_" + label,
                )
            )
        for legend_label, weights, color, marker, graph_name in graph_specs:
            a4, a4_err = compute_a4_unrolled(sample, weights)
            legend_graph = None
            for iy in range(n_y):
                graph = make_a4_graph(a4, a4_err, graph_name, iy, color, marker)
                if graph is None:
                    continue
                graph.Draw("PZ SAME")
                stuff.append(graph)
                if legend_graph is None:
                    legend_graph = graph
            if legend_graph is not None:
                legend.AddEntry(legend_graph, legend_label, "pe")
    legend.Draw()
    base = os.path.join(plot_dir, "A4_unrolled")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


def make_afb_plot(plot_dir, selected_samples, eft_points):
    canvas = ROOT.TCanvas("c_afb_born", "AFB Born", 800, 700)
    canvas.SetGrid()
    frame = ROOT.TH2F("frame_afb", ";m_{ll}^{Born} [GeV];A_{FB}^{Born}", 10, AFB_MASS_BINS[0], AFB_MASS_BINS[-1], 100, -1, 1)
    frame.Draw()
    legend = ROOT.TLegend(0.52, 0.68, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    stuff = [frame, legend]

    for i_sample, sample in enumerate(selected_samples):
        features, nominal = materialized(sample)
        mll = features[:, require_feature(sample, "dy_born_mll")]
        costheta_trueq = features[:, require_feature(sample, "cs_born_costheta")] * features[:, require_feature(sample, "truth_quark_direction")]
        graph_specs = [(f"{sample.tex_name} SM", target_weights(sample, nominal, ("SM", {})), sample.color, 20 + i_sample)]
        for i_point, eft_point in enumerate(eft_points):
            label, values = eft_point
            graph_specs.append((f"{sample.tex_name} {label}", target_weights(sample, nominal, eft_point), EFT_COLORS[i_point % len(EFT_COLORS)], EFT_MARKERS[i_point % len(EFT_MARKERS)]))

        for legend_label, weights, color, marker in graph_specs:
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
                xvals.append(0.5 * (lo + hi))
                yvals.append(afb)
                yerrs.append(math.sqrt(max(0.0, (1.0 - afb * afb) / neff)) if neff > 0 else 0.0)
            if not xvals:
                continue
            graph = ROOT.TGraphErrors(len(xvals))
            for ip, (x, y, ey) in enumerate(zip(xvals, yvals, yerrs)):
                graph.SetPoint(ip, x, y)
                graph.SetPointError(ip, 0.0, ey)
            graph.SetLineColor(color)
            graph.SetMarkerColor(color)
            graph.SetMarkerStyle(marker)
            graph.SetLineWidth(2)
            graph.Draw("P SAME")
            legend.AddEntry(graph, legend_label, "lp")
            stuff.append(graph)
    legend.Draw()
    base = os.path.join(plot_dir, "AFB_born_trueq_vs_mll")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")


parser = argparse.ArgumentParser()
parser.add_argument("--samples", nargs="+", default=["DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120"], help="Samples from samples_postprocessed.py")
parser.add_argument("--max-files", type=int, default=None, help="Files per sample")
parser.add_argument("--level", choices=["fiducial", "parton"], default="parton", help="Selection level to plot")
parser.add_argument("--small", nargs="?", const=10, type=int, default=None, help="Use one in N files, e.g. --small 10")
parser.add_argument("--eft-point", action="append", default=[], help="Overlay EFT point as label:wc=value,wc2=value. Can be repeated.")
args = parser.parse_args()

eft_points = [eft_reweighting.parse_eft_point(point) for point in args.eft_point]
eft_wc_names = sorted({wc for _, values in eft_points for wc, value in values.items() if value != 0.0})

helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features_eft"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features_eft", args.level))

selected_samples = []
for sample_name in args.samples:
    if sample_name not in samples_postprocessed.samples_by_name:
        raise RuntimeError(f"Unknown sample '{sample_name}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")
    component = samples_postprocessed.samples_by_name[sample_name]
    files = component.files
    if args.small:
        files = files[:: args.small]
    if args.max_files is not None:
        files = files[: args.max_files]
    component = samples_postprocessed.PostProcessedSample(component.name, component.tex_name, files, component.color)
    sample = component.get_loader(selection=args.level)
    sample.setFeatures(extra_branches=["LHEReweightingWeight"])
    selected_samples.append(sample)

label = "_".join(sample.name for sample in selected_samples)
if args.small:
    label += f"_small{args.small}"
if eft_wc_names:
    label += "_WC_" + "_".join(sanitize(wc) for wc in eft_wc_names)

plot_dir = os.path.join(user.plot_directory, "DY", "gen_features_eft", args.level, label)
os.makedirs(plot_dir, exist_ok=True)
helpers.copyIndexPHP(plot_dir)

for feature in FEATURES:
    make_stack_plot(plot_dir, selected_samples, feature, eft_points)
syncer.sync()

make_afb_plot(plot_dir, selected_samples, eft_points)
syncer.sync()

make_unrolled_a4_plot(plot_dir, selected_samples, eft_points)
syncer.sync()
