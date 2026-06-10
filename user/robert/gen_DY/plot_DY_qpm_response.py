#!/usr/bin/env python3

import argparse
import os
import sys

import lhapdf
import numpy as np
import ROOT
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

A4_MASS_BINS = np.array([60, 70, 80, 86, 91, 96, 106, 120, 133], dtype=np.float64)
A4_ABSY_BINS = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.7, 3.0, 3.4], dtype=np.float64)

REQUIRED_BRANCHES = [
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

QPM_BASIS = ["uplus", "dplus", "splus", "cplus", "uminus", "dminus", "sminus"]
QPM_TEX = {
    "uplus": "u^{+}=u+#bar{u}",
    "dplus": "d^{+}=d+#bar{d}",
    "splus": "s^{+}=s+#bar{s}",
    "cplus": "c^{+}=c+#bar{c}",
    "uminus": "u^{-}=u-#bar{u}",
    "dminus": "d^{-}=d-#bar{d}",
    "sminus": "s^{-}=s-#bar{s}",
    "g": "g",
}
QPM_COLORS = {
    "uplus": ROOT.kBlue + 1,
    "uminus": ROOT.kBlue + 1,
    "dplus": ROOT.kRed + 1,
    "dminus": ROOT.kRed + 1,
    "splus": ROOT.kGreen + 2,
    "sminus": ROOT.kGreen + 2,
    "cplus": ROOT.kOrange + 7,
    "g": ROOT.kBlack,
}
QPM_STYLES = {
    "uplus": 1,
    "dplus": 1,
    "splus": 1,
    "cplus": 1,
    "uminus": 2,
    "dminus": 2,
    "sminus": 2,
    "g": 3,
}
QPM_MARKERS = {
    "uplus": 20,
    "dplus": 21,
    "splus": 22,
    "cplus": 33,
    "uminus": 24,
    "dminus": 25,
    "sminus": 26,
    "g": 27,
}


def sanitize(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def epsilon_filename(epsilon):
    return f"eps{epsilon:.3f}".replace(".", "p").replace("-", "m")


def mass_to_unrolled_x(iy, mass):
    return iy + (mass - A4_MASS_BINS[0]) / (A4_MASS_BINS[-1] - A4_MASS_BINS[0])


def require_feature(sample, branch):
    if branch not in sample.feature_names:
        raise RuntimeError(f"Required branch '{branch}' not found in sample {sample.name}")
    return sample.feature_names.index(branch)


def pdf_f(pdf, pid, x, q):
    if not np.isfinite(x) or not np.isfinite(q) or x <= 0.0 or x >= 1.0 or q <= 0.0:
        return np.nan
    return pdf.xfxQ(int(pid), float(x), float(q)) / float(x)


def nominal_physical_pdfs(pdf, x, q):
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


def varied_pdf_value_qpm(phys, pid, basis_name, sign, epsilon):
    pid = int(pid)
    if pid not in phys:
        return np.nan, np.nan
    if pid in [5, -5] or (pid == 21 and basis_name != "g"):
        return phys[pid], phys[pid]

    u, ubar = phys[2], phys[-2]
    d, dbar = phys[1], phys[-1]
    s, sbar = phys[3], phys[-3]
    c, cbar = phys[4], phys[-4]
    b, bbar = phys[5], phys[-5]
    gluon = phys[21]

    if not all(np.isfinite(x) for x in [u, ubar, d, dbar, s, sbar, c, cbar, b, bbar, gluon]):
        return np.nan, np.nan

    qpm = {
        "uplus": u + ubar,
        "dplus": d + dbar,
        "splus": s + sbar,
        "cplus": c + cbar,
        "uminus": u - ubar,
        "dminus": d - dbar,
        "sminus": s - sbar,
        "cminus": c - cbar,
        "g": gluon,
    }
    qpm[basis_name] *= 1.0 + sign * epsilon

    varied = {
        21: qpm["g"],
        2: 0.5 * (qpm["uplus"] + qpm["uminus"]),
        -2: 0.5 * (qpm["uplus"] - qpm["uminus"]),
        1: 0.5 * (qpm["dplus"] + qpm["dminus"]),
        -1: 0.5 * (qpm["dplus"] - qpm["dminus"]),
        3: 0.5 * (qpm["splus"] + qpm["sminus"]),
        -3: 0.5 * (qpm["splus"] - qpm["sminus"]),
        4: 0.5 * (qpm["cplus"] + qpm["cminus"]),
        -4: 0.5 * (qpm["cplus"] - qpm["cminus"]),
        5: b,
        -5: bbar,
    }
    return phys[pid], varied[pid]


def compute_qpm_response(sample, pdf, basis_names, epsilon, max_events=None):
    for branch in REQUIRED_BRANCHES:
        require_feature(sample, branch)

    features, weights = sample.materialize(0, "fw")
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

    mass_bin = np.searchsorted(A4_MASS_BINS, mll, side="right") - 1
    abs_y_bin = np.searchsorted(A4_ABSY_BINS, abs_yll, side="right") - 1
    valid_event = (
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
        & (mass_bin >= 0)
        & (mass_bin < n_m)
        & (abs_y_bin >= 0)
        & (abs_y_bin < n_y)
    )
    indices = np.flatnonzero(valid_event)
    if max_events is not None:
        indices = indices[:max_events]

    sumw_plus = {name: np.zeros(n_bins, dtype=np.float64) for name in basis_names}
    sumwc_plus = {name: np.zeros(n_bins, dtype=np.float64) for name in basis_names}
    sumw_minus = {name: np.zeros(n_bins, dtype=np.float64) for name in basis_names}
    sumwc_minus = {name: np.zeros(n_bins, dtype=np.float64) for name in basis_names}
    invalid = {name: 0 for name in basis_names}
    used = {name: 0 for name in basis_names}
    b_or_bbar = 0
    unsupported = 0
    supported_pids = {21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5}

    for idx in tqdm(indices, desc=f"q+/q- response {sample.name}", unit="evt"):
        pid1 = int(gen_id1[idx])
        pid2 = int(gen_id2[idx])
        if pid1 not in supported_pids or pid2 not in supported_pids:
            unsupported += 1
            continue
        if abs(pid1) == 5 or abs(pid2) == 5:
            b_or_bbar += 1

        phys1 = nominal_physical_pdfs(pdf, gen_x1[idx], gen_q[idx])
        phys2 = nominal_physical_pdfs(pdf, gen_x2[idx], gen_q[idx])
        ibin = int(abs_y_bin[idx] * n_m + mass_bin[idx])
        w = float(weights[idx])
        c_obs = float(np.sign(yll[idx]) * costheta[idx])

        for basis_name in basis_names:
            f1_nom_p, f1_plus = varied_pdf_value_qpm(phys1, pid1, basis_name, +1, epsilon)
            f2_nom_p, f2_plus = varied_pdf_value_qpm(phys2, pid2, basis_name, +1, epsilon)
            f1_nom_m, f1_minus = varied_pdf_value_qpm(phys1, pid1, basis_name, -1, epsilon)
            f2_nom_m, f2_minus = varied_pdf_value_qpm(phys2, pid2, basis_name, -1, epsilon)

            valid = (
                np.isfinite(f1_nom_p)
                and np.isfinite(f2_nom_p)
                and np.isfinite(f1_nom_m)
                and np.isfinite(f2_nom_m)
                and np.isfinite(f1_plus)
                and np.isfinite(f2_plus)
                and np.isfinite(f1_minus)
                and np.isfinite(f2_minus)
                and f1_nom_p > 0.0
                and f2_nom_p > 0.0
                and f1_nom_m > 0.0
                and f2_nom_m > 0.0
            )
            if not valid:
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
            sumwc_plus[basis_name][ibin] += wp * c_obs
            sumw_minus[basis_name][ibin] += wm
            sumwc_minus[basis_name][ibin] += wm * c_obs
            used[basis_name] += 1

    delta = {}
    for basis_name in basis_names:
        a4_plus = np.full(n_bins, np.nan, dtype=np.float64)
        a4_minus = np.full(n_bins, np.nan, dtype=np.float64)
        ok_plus = sumw_plus[basis_name] != 0.0
        ok_minus = sumw_minus[basis_name] != 0.0
        a4_plus[ok_plus] = 4.0 * sumwc_plus[basis_name][ok_plus] / sumw_plus[basis_name][ok_plus]
        a4_minus[ok_minus] = 4.0 * sumwc_minus[basis_name][ok_minus] / sumw_minus[basis_name][ok_minus]
        delta[basis_name] = 0.5 * (a4_plus - a4_minus).reshape((n_y, n_m))

        total = used[basis_name] + invalid[basis_name]
        frac = invalid[basis_name] / total if total else 0.0
        print(f"[qpm response] {sample.name} {basis_name}: used={used[basis_name]} invalid={invalid[basis_name]} invalid_fraction={frac:.4g}")

    total_input = len(indices) + unsupported
    print(f"[qpm response] {sample.name}: b_or_bbar_fraction={b_or_bbar / len(indices) if len(indices) else 0.0:.4g} unsupported_pid_fraction={unsupported / total_input if total_input else 0.0:.4g}")
    return delta


def draw_unrolled_guides(ymin, ymax):
    stuff = []
    yrange = ymax - ymin
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
    for iy, (ylo, yhi) in enumerate(zip(A4_ABSY_BINS[:-1], A4_ABSY_BINS[1:])):
        latex.DrawLatex(iy + 0.5, ymax - 0.11 * yrange, f"#splitline{{|y_{{ll}}|}}{{{ylo:.1f}-{yhi:.1f}}}")

    latex.SetTextAngle(90)
    latex.SetTextSize(0.024)
    for iy in range(n_y):
        latex.DrawLatex(iy + 0.10, ymin + 0.09 * yrange, f"{A4_MASS_BINS[0]:.0f}")
        latex.DrawLatex(iy + 0.90, ymin + 0.09 * yrange, f"{A4_MASS_BINS[-1]:.0f}")
    latex.SetTextAngle(0)
    return stuff


def make_graph(delta_values, basis_name, iy):
    graph = ROOT.TGraph()
    graph.SetName(f"g_A4_unrolled_qpm_response_{sanitize(basis_name)}_iy{iy}")
    graph.SetLineColor(QPM_COLORS[basis_name])
    graph.SetMarkerColor(QPM_COLORS[basis_name])
    graph.SetLineStyle(QPM_STYLES[basis_name])
    graph.SetLineWidth(2)
    graph.SetMarkerStyle(QPM_MARKERS[basis_name])
    graph.SetMarkerSize(0.65)

    ip = 0
    for im, (mlo, mhi) in enumerate(zip(A4_MASS_BINS[:-1], A4_MASS_BINS[1:])):
        val = delta_values[iy, im]
        if not np.isfinite(val):
            continue
        graph.SetPoint(ip, mass_to_unrolled_x(iy, 0.5 * (mlo + mhi)), val)
        ip += 1
    return graph if ip else None


def plot_qpm_response(plot_dir, sample, delta, basis_names, pdf_set, pdf_member, epsilon):
    finite_chunks = [x[np.isfinite(x)] for x in delta.values() if np.any(np.isfinite(x))]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=np.float64)
    ymax = max(1.25 * np.max(np.abs(finite_values)) if len(finite_values) else 0.001, 0.001)
    ymin = -ymax
    n_y = len(A4_ABSY_BINS) - 1

    canvas = ROOT.TCanvas("c_A4_unrolled_qpm_response", "A4 q+/q- response", 1300, 720)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.16)
    canvas.SetBottomMargin(0.18)
    stuff = [canvas]

    frame = ROOT.TH2F(
        "frame_A4_unrolled_qpm_response",
        f"; ;#Delta A_{{4}} for #pm{100.0 * epsilon:g}% q^{{+}}/q^{{-}} deformation",
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
    zero.SetLineStyle(2)
    zero.SetLineColor(ROOT.kGray + 2)
    zero.Draw()
    stuff.append(zero)

    title = ROOT.TLatex()
    title.SetNDC(True)
    title.SetTextFont(42)
    title.SetTextSize(0.037)
    title.SetTextAlign(21)
    title.DrawLatex(0.50, 0.955, "DY q^{+}/q^{-} PDF-response diagnostic")
    stuff.append(title)

    legend = ROOT.TLegend(0.09, 0.885, 0.96, 0.935)
    legend.SetNColumns(len(basis_names))
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.023)
    stuff.append(legend)

    stuff += draw_unrolled_guides(ymin, ymax)

    for basis_name in basis_names:
        legend_graph = None
        for iy in range(n_y):
            graph = make_graph(delta[basis_name], basis_name, iy)
            if graph is None:
                continue
            graph.Draw("LP SAME")
            stuff.append(graph)
            if legend_graph is None:
                legend_graph = graph
        if legend_graph is not None:
            legend.AddEntry(legend_graph, QPM_TEX[basis_name], "lp")
    legend.Draw()

    mass_label = ROOT.TLatex()
    mass_label.SetNDC(True)
    mass_label.SetTextFont(42)
    mass_label.SetTextSize(0.038)
    mass_label.SetTextAlign(31)
    mass_label.DrawLatex(0.95, 0.060, "M_{ll} (GeV)")
    stuff.append(mass_label)

    meta = ROOT.TNamed(
        "metadata",
        "A4_definition=4*weighted_mean(sign(dy_born_yll)*cs_born_costheta); "
        f"pdf_set={pdf_set}; pdf_member={pdf_member}; epsilon={epsilon}; "
        f"basis={','.join(basis_names)}; "
        "deltaA4=0.5*(A4_plus-A4_minus); diagnostic_not_official_pdf_uncertainty",
    )
    stuff.append(meta)

    base = os.path.join(plot_dir, "A4_unrolled_qpm_response_" + epsilon_filename(epsilon))
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
    print(f"[qpm response] output: {base}.{{png,pdf,root}}")


parser = argparse.ArgumentParser()
parser.add_argument("--sample", default="DYJetsToLL_M50_LO_UL17", help="Sample from samples_postprocessed.py")
parser.add_argument("--level", choices=["parton", "fiducial"], default="parton")
parser.add_argument("--max-files", type=int, default=None, help="Files to use; default is all complete files")
parser.add_argument("--small", action="store_true", help="Use one file and write to a _small directory")
parser.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
parser.add_argument("--pdf-member", type=int, default=0)
parser.add_argument("--basis-epsilon", type=float, default=0.01)
parser.add_argument("--basis-max-events", type=int, default=None, help="Optional cap for fast tests")
parser.add_argument("--include-gluon-response", action="store_true")
args = parser.parse_args()

if args.small:
    args.max_files = 1
    if args.basis_max_events is None:
        args.basis_max_events = 1000_000

if args.sample not in samples_postprocessed.samples_by_name:
    raise RuntimeError(f"Unknown sample '{args.sample}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")

basis_names = list(QPM_BASIS)
if args.include_gluon_response:
    basis_names = ["g"] + basis_names

sample = samples_postprocessed.samples_by_name[args.sample].get_loader(max_files=args.max_files, selection=args.level)
label = sample.name + ("_small" if args.small else "")
plot_dir = os.path.join(user.plot_directory, "DY", "qpm_response", args.level, label)
os.makedirs(plot_dir, exist_ok=True)
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "qpm_response"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "qpm_response", args.level))
helpers.copyIndexPHP(plot_dir)

pdf = lhapdf.mkPDF(args.pdf_set, args.pdf_member)
delta = compute_qpm_response(sample, pdf, basis_names, args.basis_epsilon, max_events=args.basis_max_events)
plot_qpm_response(plot_dir, sample, delta, basis_names, args.pdf_set, args.pdf_member, args.basis_epsilon)
syncer.sync()
