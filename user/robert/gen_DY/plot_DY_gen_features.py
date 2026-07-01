#!/usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import ROOT
import lhapdf
from tqdm import tqdm

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
BASIS_NAMES = ["g", "Sigma", "T3", "T8", "T15", "V", "V3", "V8"]
BASIS_TEX = {
    "g": "g",
    "Sigma": "#Sigma",
    "T3": "T_{3}",
    "T8": "T_{8}",
    "T15": "T_{15}",
    "V": "V",
    "V3": "V_{3}",
    "V8": "V_{8}",
}
BASIS_REQUIRED_BRANCHES = [
    "dy_born_mll",
    "dy_born_yll",
    "dy_born_abs_yll",
    "cs_born_costheta",
    "gen_id1",
    "gen_id2",
    "gen_x1",
    "gen_x2",
    "gen_scalePDF",
]
BASIS_COLORS = {
    "g": ROOT.kBlack,
    "Sigma": ROOT.kRed + 1,
    "T3": ROOT.kBlue + 1,
    "T8": ROOT.kGreen + 2,
    "T15": ROOT.kOrange + 7,
    "V": ROOT.kMagenta + 1,
    "V3": ROOT.kCyan + 2,
    "V8": ROOT.kViolet + 1,
}
BASIS_MARKERS = {
    "g": 20,
    "Sigma": 21,
    "T3": 22,
    "T8": 23,
    "T15": 33,
    "V": 34,
    "V3": 24,
    "V8": 25,
}
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


def selected_files(component, small=None, max_files=None):
    files = component.files
    if small:
        files = files[::small]
    if max_files is not None:
        files = files[:max_files]
    return files


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


def draw_unrolled_guides(ymin, ymax, rapidity_label="|Y|"):
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
        latex.DrawLatex(iy + 0.5, y_text, f"#splitline{{{rapidity_label}}}{{{ylo:.1f}-{yhi:.1f}}}")

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


def pdf_f(pdf, pid, x, q):
    if not np.isfinite(x) or not np.isfinite(q) or x <= 0.0 or x >= 1.0 or q <= 0.0:
        return np.nan
    return pdf.xfxQ(int(pid), float(x), float(q)) / float(x)


def physical_pdfs(pdf, x, q):
    return {
        21: pdf_f(pdf, 21, x, q),
        2: pdf_f(pdf, 2, x, q),
        -2: pdf_f(pdf, -2, x, q),
        1: pdf_f(pdf, 1, x, q),
        -1: pdf_f(pdf, -1, x, q),
        3: pdf_f(pdf, 3, x, q),
        -3: pdf_f(pdf, -3, x, q),
        4: pdf_f(pdf, 4, x, q),
        -4: pdf_f(pdf, -4, x, q),
        5: pdf_f(pdf, 5, x, q),
        -5: pdf_f(pdf, -5, x, q),
    }


def physical_to_basis(phys):
    uplus = phys[2] + phys[-2]
    dplus = phys[1] + phys[-1]
    splus = phys[3] + phys[-3]
    cplus = phys[4] + phys[-4]

    uminus = phys[2] - phys[-2]
    dminus = phys[1] - phys[-1]
    sminus = phys[3] - phys[-3]

    return {
        "g": phys[21],
        "Sigma": uplus + dplus + splus + cplus,
        "T3": uplus - dplus,
        "T8": uplus + dplus - 2.0 * splus,
        "T15": uplus + dplus + splus - 3.0 * cplus,
        "V": uminus + dminus + sminus,
        "V3": uminus - dminus,
        "V8": uminus + dminus - 2.0 * sminus,
    }


def basis_to_physical(basis, phys_nominal):
    sigma = basis["Sigma"]
    t3 = basis["T3"]
    t8 = basis["T8"]
    t15 = basis["T15"]
    valence = basis["V"]
    v3 = basis["V3"]
    v8 = basis["V8"]

    cplus = (sigma - t15) / 4.0
    splus = (3.0 * sigma + t15 - 4.0 * t8) / 12.0
    uplus = (3.0 * sigma + t15 + 2.0 * t8 + 6.0 * t3) / 12.0
    dplus = (3.0 * sigma + t15 + 2.0 * t8 - 6.0 * t3) / 12.0

    sminus = (valence - v8) / 3.0
    uminus = (2.0 * valence + v8 + 3.0 * v3) / 6.0
    dminus = (2.0 * valence + v8 - 3.0 * v3) / 6.0
    cminus = phys_nominal[4] - phys_nominal[-4]

    return {
        21: basis["g"],
        2: 0.5 * (uplus + uminus),
        -2: 0.5 * (uplus - uminus),
        1: 0.5 * (dplus + dminus),
        -1: 0.5 * (dplus - dminus),
        3: 0.5 * (splus + sminus),
        -3: 0.5 * (splus - sminus),
        4: 0.5 * (cplus + cminus),
        -4: 0.5 * (cplus - cminus),
        5: phys_nominal[5],
        -5: phys_nominal[-5],
    }


def finite_physical_pdfs(phys):
    return all(np.isfinite(phys[pid]) for pid in [21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5])


def basis_varied_value(phys, pid, basis_name, sign, epsilon):
    pid = int(pid)
    if pid not in phys:
        return np.nan, np.nan
    if pid in [5, -5]:
        return phys[pid], phys[pid]
    if not finite_physical_pdfs(phys):
        return np.nan, np.nan
    basis = physical_to_basis(phys)
    basis[basis_name] *= 1.0 + sign * epsilon
    varied = basis_to_physical(basis, phys)
    return phys[pid], varied[pid]


def compute_a4_basis_response(sample, pdf, epsilon, max_events=None):
    for branch in BASIS_REQUIRED_BRANCHES:
        require_feature(sample, branch)

    features, weights = materialized(sample)
    mll = features[:, require_feature(sample, "dy_born_mll")]
    yll = features[:, require_feature(sample, "dy_born_yll")]
    abs_yll = features[:, require_feature(sample, "dy_born_abs_yll")]
    costheta = features[:, require_feature(sample, "cs_born_costheta")]
    gen_id1 = features[:, require_feature(sample, "gen_id1")]
    gen_id2 = features[:, require_feature(sample, "gen_id2")]
    gen_x1 = features[:, require_feature(sample, "gen_x1")]
    gen_x2 = features[:, require_feature(sample, "gen_x2")]
    gen_q = features[:, require_feature(sample, "gen_scalePDF")]

    n_y = len(A4_ABSY_BINS) - 1
    n_m = len(A4_MASS_BINS) - 1
    n_bins = n_y * n_m

    sumw_plus = {name: np.zeros(n_bins, dtype=np.float64) for name in BASIS_NAMES}
    sumwc_plus = {name: np.zeros(n_bins, dtype=np.float64) for name in BASIS_NAMES}
    sumw_minus = {name: np.zeros(n_bins, dtype=np.float64) for name in BASIS_NAMES}
    sumwc_minus = {name: np.zeros(n_bins, dtype=np.float64) for name in BASIS_NAMES}
    invalid = {name: 0 for name in BASIS_NAMES}
    used = {name: 0 for name in BASIS_NAMES}
    b_fraction = 0
    unsupported = 0

    signed_costheta = np.sign(yll) * costheta
    finite = (
        np.isfinite(mll)
        & np.isfinite(yll)
        & np.isfinite(abs_yll)
        & np.isfinite(costheta)
        & np.isfinite(weights)
        & np.isfinite(gen_id1)
        & np.isfinite(gen_id2)
        & np.isfinite(gen_x1)
        & np.isfinite(gen_x2)
        & np.isfinite(gen_q)
        & (mll > -998)
        & (abs_yll > -998)
        & (costheta > -998)
        & (np.sign(yll) != 0)
    )

    mass_bin = np.searchsorted(A4_MASS_BINS, mll, side="right") - 1
    abs_y_bin = np.searchsorted(A4_ABSY_BINS, abs_yll, side="right") - 1
    in_bins = finite & (mass_bin >= 0) & (mass_bin < n_m) & (abs_y_bin >= 0) & (abs_y_bin < n_y)
    indices = np.flatnonzero(in_bins)
    if max_events is not None:
        indices = indices[:max_events]

    for idx in tqdm(indices, desc=f"basis response {sample.name}", unit="evt"):
        pid1 = int(gen_id1[idx])
        pid2 = int(gen_id2[idx])
        if pid1 not in [21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5] or pid2 not in [21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5]:
            unsupported += 1
            continue
        if abs(pid1) == 5 or abs(pid2) == 5:
            b_fraction += 1

        phys1 = physical_pdfs(pdf, gen_x1[idx], gen_q[idx])
        phys2 = physical_pdfs(pdf, gen_x2[idx], gen_q[idx])
        ibin = int(abs_y_bin[idx] * n_m + mass_bin[idx])
        w = float(weights[idx])
        c = float(signed_costheta[idx])

        for basis_name in BASIS_NAMES:
            f1_nom_p, f1_plus = basis_varied_value(phys1, pid1, basis_name, +1, epsilon)
            f2_nom_p, f2_plus = basis_varied_value(phys2, pid2, basis_name, +1, epsilon)
            f1_nom_m, f1_minus = basis_varied_value(phys1, pid1, basis_name, -1, epsilon)
            f2_nom_m, f2_minus = basis_varied_value(phys2, pid2, basis_name, -1, epsilon)

            valid_plus = (
                np.isfinite(f1_nom_p)
                and np.isfinite(f2_nom_p)
                and np.isfinite(f1_plus)
                and np.isfinite(f2_plus)
                and f1_nom_p != 0.0
                and f2_nom_p != 0.0
            )
            valid_minus = (
                np.isfinite(f1_nom_m)
                and np.isfinite(f2_nom_m)
                and np.isfinite(f1_minus)
                and np.isfinite(f2_minus)
                and f1_nom_m != 0.0
                and f2_nom_m != 0.0
            )
            if not valid_plus or not valid_minus:
                invalid[basis_name] += 1
                continue

            r_plus = (f1_plus / f1_nom_p) * (f2_plus / f2_nom_p)
            r_minus = (f1_minus / f1_nom_m) * (f2_minus / f2_nom_m)
            if not np.isfinite(r_plus) or not np.isfinite(r_minus):
                invalid[basis_name] += 1
                continue

            wp = w * r_plus
            wm = w * r_minus
            sumw_plus[basis_name][ibin] += wp
            sumwc_plus[basis_name][ibin] += wp * c
            sumw_minus[basis_name][ibin] += wm
            sumwc_minus[basis_name][ibin] += wm * c
            used[basis_name] += 1

    delta = {}
    for basis_name in BASIS_NAMES:
        a4_plus = np.full(n_bins, np.nan, dtype=np.float64)
        a4_minus = np.full(n_bins, np.nan, dtype=np.float64)
        ok_plus = sumw_plus[basis_name] != 0.0
        ok_minus = sumw_minus[basis_name] != 0.0
        a4_plus[ok_plus] = 4.0 * sumwc_plus[basis_name][ok_plus] / sumw_plus[basis_name][ok_plus]
        a4_minus[ok_minus] = 4.0 * sumwc_minus[basis_name][ok_minus] / sumw_minus[basis_name][ok_minus]
        delta[basis_name] = 0.5 * (a4_plus - a4_minus).reshape((n_y, n_m))
        total = used[basis_name] + invalid[basis_name]
        frac = invalid[basis_name] / total if total else 0.0
        print(f"[basis response] {sample.name} {basis_name}: used={used[basis_name]} invalid={invalid[basis_name]} invalid_fraction={frac:.4g}")

    total_input = len(indices) + unsupported
    b_frac = b_fraction / len(indices) if len(indices) else 0.0
    unsupported_frac = unsupported / total_input if total_input else 0.0
    print(f"[basis response] {sample.name}: b_or_bbar_input_fraction={b_frac:.4g} unsupported_pid_fraction={unsupported_frac:.4g}")
    return delta


def make_graph_for_basis_response(delta_values, basis_name, iy):
    graph = ROOT.TGraph()
    graph.SetName(f"g_A4_unrolled_basis_response_{sanitize(basis_name)}_iy{iy}")
    graph.SetLineColor(BASIS_COLORS[basis_name])
    graph.SetMarkerColor(BASIS_COLORS[basis_name])
    graph.SetMarkerStyle(BASIS_MARKERS[basis_name])
    graph.SetMarkerSize(0.65)
    graph.SetLineWidth(2)

    ip = 0
    for im, (mlo, mhi) in enumerate(zip(A4_MASS_BINS[:-1], A4_MASS_BINS[1:])):
        value = delta_values[iy, im]
        if not np.isfinite(value):
            continue
        graph.SetPoint(ip, mass_to_unrolled_x(iy, 0.5 * (mlo + mhi)), value)
        ip += 1

    return graph if ip > 0 else None


def epsilon_label(epsilon):
    pct = 100.0 * epsilon
    return f"{pct:g}"


def epsilon_filename(epsilon):
    return f"eps{epsilon:.3f}".replace(".", "p").replace("-", "m")


def make_unrolled_a4_basis_response_plot(plot_dir, selected_samples, pdf, pdf_set, pdf_member, epsilon, max_events=None):
    if not selected_samples:
        return

    sample = selected_samples[0]
    delta = compute_a4_basis_response(sample, pdf, epsilon, max_events=max_events)
    finite_chunks = [values[np.isfinite(values)] for values in delta.values() if np.any(np.isfinite(values))]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=np.float64)
    ymax = 1.25 * np.max(np.abs(finite_values)) if len(finite_values) else 0.001
    ymax = max(float(ymax), 0.001)
    ymin = -ymax

    n_y = len(A4_ABSY_BINS) - 1
    canvas = ROOT.TCanvas("c_A4_unrolled_basis_response", "A4 basis response", 1300, 720)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.16)
    canvas.SetBottomMargin(0.18)
    stuff = [canvas]

    frame = ROOT.TH2F(
        "frame_A4_unrolled_basis_response",
        f"; ;#Delta A_{{4}} for #pm{epsilon_label(epsilon)}% basis deformation",
        n_y,
        0.0,
        float(n_y),
        100,
        ymin,
        ymax,
    )
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTickLength(0.02)
    frame.GetYaxis().SetTitleSize(0.046)
    frame.GetYaxis().SetLabelSize(0.040)
    frame.GetYaxis().SetTitleOffset(0.95)
    frame.Draw()
    stuff.append(frame)

    zero = ROOT.TLine(0.0, 0.0, float(n_y), 0.0)
    zero.SetLineColor(ROOT.kGray + 2)
    zero.SetLineStyle(2)
    zero.SetLineWidth(1)
    zero.Draw()
    stuff.append(zero)

    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.037)
    title.SetTextAlign(21)
    title.DrawLatex(0.50, 0.955, "NNPDF evolution-basis response")
    stuff.append(title)

    legend = ROOT.TLegend(0.12, 0.885, 0.94, 0.935)
    legend.SetNColumns(len(BASIS_NAMES))
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.027)
    stuff.append(legend)

    stuff += draw_unrolled_guides(ymin, ymax, rapidity_label="|y_{ll}|")

    for basis_name in BASIS_NAMES:
        legend_graph = None
        for iy in range(n_y):
            graph = make_graph_for_basis_response(delta[basis_name], basis_name, iy)
            if graph is None:
                continue
            graph.Draw("LP SAME")
            stuff.append(graph)
            if legend_graph is None:
                legend_graph = graph
        if legend_graph is not None:
            legend.AddEntry(legend_graph, BASIS_TEX[basis_name], "lp")

    legend.Draw()

    mass_label = ROOT.TLatex()
    mass_label.SetNDC(True)
    mass_label.SetTextFont(42)
    mass_label.SetTextSize(0.038)
    mass_label.SetTextAlign(31)
    mass_label.DrawLatex(0.95, 0.060, "M_{ll} (GeV)")
    stuff.append(mass_label)

    meta_text = (
        "A4_definition=4*weighted_mean(sign(dy_born_yll)*cs_born_costheta); "
        f"pdf_set={pdf_set}; "
        f"pdf_member={pdf_member}; "
        f"epsilon={epsilon}; "
        "basis=g,Sigma,T3,T8,T15,V,V3,V8; "
        "branches=dy_born_mll,dy_born_yll,dy_born_abs_yll,cs_born_costheta,gen_id1,gen_id2,gen_x1,gen_x2,gen_scalePDF; "
        "deltaA4=0.5*(A4_plus-A4_minus); "
        "diagnostic_not_official_pdf_uncertainty"
    )
    meta_named = ROOT.TNamed("metadata", meta_text)
    stuff.append(meta_named)

    base = os.path.join(plot_dir, "A4_unrolled_basis_response_" + epsilon_filename(epsilon))
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

        if not xvals:
            print(
                f"[plot_DY_gen_features] skip AFB graph for {sample.name}: "
                f"no events in {AFB_MASS_BINS[0]:.0f}-{AFB_MASS_BINS[-1]:.0f} GeV mass bins"
            )
            continue

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
parser.add_argument("--max-files", type=int, default=None, help="Files per sample")
parser.add_argument("--level", choices=["both", "fiducial", "parton"], default="both", help="Selection level to plot")
parser.add_argument("--small", nargs="?", const=10, type=int, default=None, help="Use one in N files, e.g. --small 10")
parser.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180", help="LHAPDF set for basis-response diagnostics")
parser.add_argument("--pdf-member", type=int, default=0, help="LHAPDF member for basis-response diagnostics")
parser.add_argument("--basis-epsilon", type=float, default=0.01, help="Relative evolution-basis deformation for response plot")
parser.add_argument("--basis-max-events", type=int, default=None, help="Optional event cap for testing the basis-response calculation")
args = parser.parse_args()

helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features"))

levels = ["fiducial", "parton"] if args.level == "both" else [args.level]
pdf = lhapdf.mkPDF(args.pdf_set, args.pdf_member)

for level in levels:
    selected_samples = []
    for sample_name in args.samples:
        if sample_name not in samples_postprocessed.samples_by_name:
            raise RuntimeError(f"Unknown sample '{sample_name}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")
        component = samples_postprocessed.samples_by_name[sample_name]
        files = selected_files(component, small=args.small, max_files=args.max_files)
        component = samples_postprocessed.PostProcessedSample(
            name=component.name,
            tex_name=component.tex_name,
            files=files,
            color=component.color,
        )
        sample = component.get_loader(selection=level)
        selected_samples.append(sample)

    label = "_".join(sample.name for sample in selected_samples)
    if args.small:
        label += f"_small{args.small}"

    plot_dir = os.path.join(user.plot_directory, "DY", "gen_features", level, label)
    os.makedirs(plot_dir, exist_ok=True)
    helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "gen_features", level))
    helpers.copyIndexPHP(plot_dir)

    for feature in FEATURES:
        make_stack_plot(plot_dir, selected_samples, feature)
    syncer.sync()

    make_afb_plot(plot_dir, selected_samples)
    syncer.sync()
    make_unrolled_a4_plot(plot_dir, selected_samples)
    syncer.sync()
    make_unrolled_a4_basis_response_plot(
        plot_dir,
        selected_samples,
        pdf,
        args.pdf_set,
        args.pdf_member,
        args.basis_epsilon,
        max_events=args.basis_max_events,
    )
    syncer.sync()
