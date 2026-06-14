import argparse
import gc
import os
import sys

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.user as user
import common.syncer as syncer
from pdf.PODBasis import PODBasis
from data.samples_RunII import tt2l_delphes, tt2l_delphes_RunII
from data.RDataLoader import RDataLoader


MASS_BRANCH = "tr_ttbar_mass"
Y_BRANCH = "tr_ttbar_y"
GEN_BRANCHES = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]
DEFAULT_MASS_THRESHOLDS = [300.0, 400.0, 500.0, 650.0, 1500.0]
DEFAULT_ABS_Y_THRESHOLDS = [0.0, 0.4, 0.8, 1.2, 2.5]
DEFAULT_NUMERATOR_PDF = "gluon_POD_nongluon_ATLASpdf21"
DEFAULT_DENOMINATOR_PDF = "gluon_POD_nongluon_PDF4LHC21"


def parse_edges(values, default_edges):
    if values is None:
        return np.asarray(default_edges, dtype=np.float64)
    edges = np.asarray([float(v) for v in values], dtype=np.float64)
    if len(edges) < 2:
        raise RuntimeError("Need at least two thresholds to define one bin.")
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"Thresholds must be strictly increasing, got {edges.tolist()}.")
    return edges


def unrolled_indices(mass, y, mass_edges, y_edges, fold_flow=True):
    n_mass = len(mass_edges) - 1
    n_y = len(y_edges) - 1

    mass_idx = np.searchsorted(mass_edges, mass, side="right") - 1
    y_idx = np.searchsorted(y_edges, y, side="right") - 1

    finite = np.isfinite(mass) & np.isfinite(y)
    if fold_flow:
        mass_idx = np.clip(mass_idx, 0, n_mass - 1)
        y_idx = np.clip(y_idx, 0, n_y - 1)
        mask = finite
    else:
        mask = finite & (mass_idx >= 0) & (mass_idx < n_mass) & (y_idx >= 0) & (y_idx < n_y)

    return mass_idx[mask] * n_y + y_idx[mask], mask


def set_sparse_bin_labels(hist, mass_edges, y_edges, max_labels=80):
    n_mass = len(mass_edges) - 1
    n_y = len(y_edges) - 1
    nbins = n_mass * n_y
    if nbins > max_labels:
        return
    for im in range(n_mass):
        for iy in range(n_y):
            ib = im * n_y + iy + 1
            hist.GetXaxis().SetBinLabel(
                ib,
                f"{y_edges[iy]:g}-{y_edges[iy + 1]:g}",
            )


def clone_empty_hist(hist, name, title=None):
    out = hist.Clone(name)
    out.Reset("ICESM")
    out.SetTitle(title or name)
    out.SetDirectory(0)
    if out.GetSumw2N() == 0:
        out.Sumw2()
    return out


def make_loader(sample_name, n_split, max_files=None):
    samples = {
        "RunII": tt2l_delphes_RunII,
        "2016": tt2l_delphes,
    }
    if sample_name not in samples:
        raise RuntimeError(f"Unknown sample {sample_name!r}. Choices are {sorted(samples)}.")

    sample = samples[sample_name]
    loader = RDataLoader(
        input_paths=sample.input_paths,
        tree_name=sample.tree_name,
        branches=sample.branches,
        selection=None,
        n_split=1,
        splitting_strategy="files",
        strict_branches=sample.strict_branches,
        max_files=max_files,
        feature_names=sample.feature_names,
        observer_names=sample.observer_names,
        weight_branches=sample.weight_branches,
        weight_rescale=sample.weight_rescale,
    )

    # File shards avoid re-reading the full Delphes directory for every event split.
    n_files = len(loader.files)
    if n_files < 1:
        raise RuntimeError("Delphes sample has no files.")
    loader.set_n_split(min(max(1, int(n_split)), n_files))
    return loader


def central_xfx(pdf, x, pid, q):
    return np.asarray([row.get(int(pid_)) for row, pid_ in zip(pdf.xfxQ(tuple(x), tuple(q)), pid)], dtype=np.float64)


def build_pdf_reweighter(args):
    if not args.pdf_reweight:
        return None

    numerator = PODBasis(variations=[], var_set=args.pdf_numerator, gen_pdf=None, rescale_pod_amplitudes=False)
    denominator = PODBasis(variations=[], var_set=args.pdf_denominator, gen_pdf=None, rescale_pod_amplitudes=False)

    def reweight(ar, loader):
        gen = loader.scalar_branches(ar, GEN_BRANCHES)
        x1 = gen[:, 0].astype(np.float64, copy=False)
        x2 = gen[:, 1].astype(np.float64, copy=False)
        id1 = gen[:, 2].astype(np.int32, copy=False)
        id2 = gen[:, 3].astype(np.int32, copy=False)
        q = gen[:, 4].astype(np.float64, copy=False)

        out = np.ones(len(gen), dtype=np.float64)
        quark_init = (np.abs(id1) != 21) & (np.abs(id2) != 21)
        if not np.any(quark_init):
            return out, quark_init

        num1 = central_xfx(numerator.reference_pdf, x1[quark_init], id1[quark_init], q[quark_init])
        num2 = central_xfx(numerator.reference_pdf, x2[quark_init], id2[quark_init], q[quark_init])
        den1 = central_xfx(denominator.reference_pdf, x1[quark_init], id1[quark_init], q[quark_init])
        den2 = central_xfx(denominator.reference_pdf, x2[quark_init], id2[quark_init], q[quark_init])
        ratio = (num1 * num2) / (den1 * den2)
        valid = np.isfinite(ratio) & np.isfinite(num1) & np.isfinite(num2) & np.isfinite(den1) & np.isfinite(den2) & (den1 != 0) & (den2 != 0)
        quark_indices = np.flatnonzero(quark_init)
        out[quark_indices[valid]] = ratio[valid]
        return out, quark_init

    return reweight


def fill_histograms(loader, hist_nominal, hist_reweighted, hist_pdf_weights, mass_edges, y_edges, args):
    n_shards = len(loader)
    shard_range = range(n_shards)
    if args.small:
        shard_range = range(min(args.small_shards, n_shards))

    total_events = 0
    selected_events = 0
    sumw = 0.0
    sumw_reweighted = 0.0
    quark_events = 0
    reweighter = build_pdf_reweighter(args)

    for shard in tqdm(shard_range, desc=f"{args.sample}:tt2l_delphes", leave=False):
        ar = loader[shard]
        values = loader.scalar_branches(ar, [MASS_BRANCH, Y_BRANCH]).astype(np.float64, copy=False)
        weights = loader.weight_vector(shard).astype(np.float64, copy=False)
        pdf_weights = np.ones(len(weights), dtype=np.float64)
        quark_init = np.zeros(len(weights), dtype=bool)
        if reweighter is not None:
            pdf_weights, quark_init = reweighter(ar, loader)

        if args.max_events is not None:
            remaining = args.max_events - total_events
            if remaining <= 0:
                break
            values = values[:remaining]
            weights = weights[:remaining]
            pdf_weights = pdf_weights[:remaining]
            quark_init = quark_init[:remaining]

        finite_pdf_weights = np.isfinite(pdf_weights)
        if np.any(finite_pdf_weights):
            h_x = np.ascontiguousarray(pdf_weights[finite_pdf_weights], dtype=np.float64)
            h_w = np.ones(len(h_x), dtype=np.float64)
            hist_pdf_weights.FillN(int(len(h_x)), h_x, h_w)
        quark_events += int(np.count_nonzero(quark_init))

        mass = values[:, 0]
        y = np.abs(values[:, 1])
        idx, mask = unrolled_indices(
            mass,
            y,
            mass_edges,
            y_edges,
            fold_flow=not args.drop_out_of_range,
        )
        weights = weights[mask]
        pdf_weights = pdf_weights[mask]
        finite_weight = np.isfinite(weights)
        finite_weight &= np.isfinite(pdf_weights)
        idx = idx[finite_weight]
        weights = weights[finite_weight]
        pdf_weights = pdf_weights[finite_weight]

        if len(idx):
            x = np.ascontiguousarray(idx.astype(np.float64) + 0.5)
            w = np.ascontiguousarray(weights, dtype=np.float64)
            rw = np.ascontiguousarray(weights * pdf_weights, dtype=np.float64)
            hist_nominal.FillN(int(len(x)), x, w)
            hist_reweighted.FillN(int(len(x)), x, rw)
            selected_events += len(idx)
            sumw += float(np.sum(w))
            sumw_reweighted += float(np.sum(rw))

        total_events += len(values)
        del ar, values, weights, pdf_weights
        loader.clear_cache()
        gc.collect()

    return total_events, selected_events, sumw, sumw_reweighted, quark_events


def make_ratio_hist(num, den, name):
    ratio = num.Clone(name)
    ratio.SetDirectory(0)
    ratio.Divide(den)
    ratio.SetTitle("")
    ratio.GetYaxis().SetTitle("rew. / nom.")
    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.095)
    ratio.GetYaxis().SetLabelSize(0.085)
    ratio.GetYaxis().SetTitleOffset(0.42)
    ratio.GetXaxis().SetTitle("")
    ratio.GetXaxis().SetLabelSize(0.070)
    ratio.GetXaxis().SetLabelOffset(0.020)
    ratio.SetMinimum(0.94)
    ratio.SetMaximum(1.06)
    return ratio


def draw_unrolled_guides(hist, mass_edges, y_edges, y_min, y_max, label_mass=False):
    n_mass = len(mass_edges) - 1
    n_y = len(y_edges) - 1
    lines = []
    for im in range(1, n_mass):
        x = float(im * n_y)
        line = ROOT.TLine(x, y_min, x, y_max)
        line.SetLineColor(ROOT.kGray + 1)
        line.SetLineStyle(ROOT.kDotted)
        line.SetLineWidth(1)
        line.Draw("SAME")
        lines.append(line)

    labels = []
    if label_mass:
        pad = ROOT.gPad
        left = pad.GetLeftMargin()
        right = pad.GetRightMargin()
        width = 1.0 - left - right
        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextAlign(22)
        latex.SetTextFont(42)
        latex.SetTextSize(0.070)
        for im in range(n_mass):
            x_center = (im + 0.5) * n_y / float(hist.GetNbinsX())
            x_ndc = left + width * x_center
            latex.DrawLatex(x_ndc, 0.145, f"{mass_edges[im]:g}-{mass_edges[im + 1]:g}")
        latex.SetTextSize(0.075)
        latex.DrawLatex(left + 0.5 * width, 0.050, "m(tt) [GeV]")
        latex.SetTextSize(0.060)
        latex.SetTextAlign(12)
        latex.DrawLatex(left + 0.005, 0.235, "abs(y)")
        labels.append(latex)

    return lines, labels


def draw_overlay(hist_nominal, hist_reweighted, mass_edges, y_edges, outdir, name, nominal_label, reweighted_label, log_y=False):
    canvas = ROOT.TCanvas(f"c_{name}_{'log' if log_y else 'lin'}", "", 1200, 850)
    top = ROOT.TPad("top", "top", 0.0, 0.30, 1.0, 1.0)
    bottom = ROOT.TPad("bottom", "bottom", 0.0, 0.0, 1.0, 0.30)
    top.SetLeftMargin(0.12)
    top.SetRightMargin(0.04)
    top.SetBottomMargin(0.02)
    top.SetTopMargin(0.06)
    top.SetTickx(1)
    top.SetTicky(1)
    top.SetLogy(bool(log_y))
    bottom.SetLeftMargin(0.12)
    bottom.SetRightMargin(0.04)
    bottom.SetTopMargin(0.03)
    bottom.SetBottomMargin(0.40)
    bottom.SetTickx(1)
    bottom.SetTicky(1)
    top.Draw()
    bottom.Draw()

    hist_nominal.SetTitle("")
    hist_reweighted.SetTitle("")
    hist_nominal.SetLineColor(ROOT.kBlack)
    hist_nominal.SetFillStyle(0)
    hist_nominal.SetLineWidth(2)
    hist_reweighted.SetLineColor(ROOT.kAzure + 2)
    hist_reweighted.SetFillStyle(0)
    hist_reweighted.SetLineWidth(2)
    hist_reweighted.SetLineStyle(ROOT.kSolid)

    for hist in (hist_nominal, hist_reweighted):
        hist.GetXaxis().SetTitle("")
        hist.GetYaxis().SetTitle("weighted events")
        hist.GetYaxis().SetTitleOffset(1.05)
        hist.GetXaxis().SetLabelSize(0)
        hist.LabelsOption("v", "X")

    max_y = max(hist_nominal.GetMaximum(), hist_reweighted.GetMaximum())
    if log_y:
        positive = [
            h.GetBinContent(i)
            for h in (hist_nominal, hist_reweighted)
            for i in range(1, h.GetNbinsX() + 1)
            if h.GetBinContent(i) > 0
        ]
        if positive:
            hist_nominal.SetMinimum(max(1e-6, min(positive) * 0.5))
            hist_nominal.SetMaximum(max(positive) * 80.0)
    else:
        hist_nominal.SetMinimum(0.0)
        hist_nominal.SetMaximum(1.70 * max_y if max_y > 0 else 1.0)

    top.cd()
    hist_nominal.Draw("HIST")
    hist_reweighted.Draw("HIST SAME")
    draw_unrolled_guides(hist_nominal, mass_edges, y_edges, hist_nominal.GetMinimum(), hist_nominal.GetMaximum())
    leg = ROOT.TLegend(0.50, 0.78, 0.94, 0.90)
    leg.SetBorderSize(0)
    leg.SetLineColor(0)
    leg.SetLineWidth(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hist_nominal, nominal_label, "l")
    leg.AddEntry(hist_reweighted, reweighted_label, "l")
    leg.Draw()
    top.RedrawAxis()

    bottom.cd()
    ratio = make_ratio_hist(hist_reweighted, hist_nominal, f"{name}_ratio_{'log' if log_y else 'lin'}")
    ratio.Draw("HIST")
    ratio.LabelsOption("v", "X")
    line = ROOT.TLine(0.0, 1.0, float(hist_nominal.GetNbinsX()), 1.0)
    line.SetLineColor(ROOT.kGray + 2)
    line.SetLineStyle(ROOT.kDashed)
    line.Draw("SAME")
    draw_unrolled_guides(ratio, mass_edges, y_edges, ratio.GetMinimum(), ratio.GetMaximum(), label_mass=True)
    bottom.RedrawAxis()

    canvas.RedrawAxis()
    canvas.SaveAs(os.path.join(outdir, f"{name}.png"))
    canvas.SaveAs(os.path.join(outdir, f"{name}.pdf"))


def draw_weight_histogram(hist, outdir):
    canvas = ROOT.TCanvas("c_pdf_reweights", "", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.12)
    canvas.SetLogy(True)
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    hist.SetLineColor(ROOT.kRed + 1)
    hist.SetFillStyle(0)
    hist.SetLineWidth(2)
    hist.GetXaxis().SetTitle("PDF event reweight")
    hist.GetYaxis().SetTitle("events")
    positive = [hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1) if hist.GetBinContent(i) > 0]
    if positive:
        hist.SetMinimum(max(0.5, min(positive) * 0.5))
        hist.SetMaximum(max(positive) * 20.0)
    hist.Draw("HIST")
    canvas.RedrawAxis()
    canvas.SaveAs(os.path.join(outdir, "pdf_reweights.png"))
    canvas.SaveAs(os.path.join(outdir, "pdf_reweights.pdf"))


def main():
    default_mass_edges = np.asarray(DEFAULT_MASS_THRESHOLDS, dtype=np.float64)
    default_y_edges = np.asarray(DEFAULT_ABS_Y_THRESHOLDS, dtype=np.float64)

    parser = argparse.ArgumentParser(description="Unrolled Delphes ttbar M(ttbar) x |y(ttbar)| ROOT histogram.")
    parser.add_argument("--plot_directory", default="v1_delphes_ttbar_unrolled")
    parser.add_argument("--sample", default="RunII", choices=["RunII", "2016"],
                        help="RunII uses tt2l_delphes_RunII; 2016 uses tt2l_delphes.")
    parser.add_argument("--mass-thresholds", nargs="+", type=float, default=None,
                        help="Bin edges for tr_ttbar_mass.")
    parser.add_argument("--y-thresholds", nargs="+", type=float, default=None,
                        help="Bin edges for abs(tr_ttbar_y).")
    parser.add_argument("--n-split", type=int, default=20, help="Number of file shards.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit ROOT files before sharding.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap after sharding.")
    parser.add_argument("--small", action="store_true", help="Run only a small number of shards for testing.")
    parser.add_argument("--small-shards", type=int, default=1, help="Number of shards used with --small.")
    parser.set_defaults(drop_out_of_range=True)
    parser.add_argument("--drop-out-of-range", dest="drop_out_of_range", action="store_true",
                        help="Drop values outside thresholds. This is the default.")
    parser.add_argument("--fold-flow", dest="drop_out_of_range", action="store_false",
                        help="Fold values outside thresholds into the edge bins.")
    parser.set_defaults(pdf_reweight=True)
    parser.add_argument("--pdf-reweight", dest="pdf_reweight", action="store_true",
                        help="Overlay quark-initiated event reweighting. This is the default.")
    parser.add_argument("--no-pdf-reweight", dest="pdf_reweight", action="store_false",
                        help="Disable the PDF reweight overlay.")
    parser.add_argument("--pdf-numerator", default=DEFAULT_NUMERATOR_PDF,
                        help="Numerator POD/LHAPDF set for the quark-only reweight.")
    parser.add_argument("--pdf-denominator", default=DEFAULT_DENOMINATOR_PDF,
                        help="Denominator POD/LHAPDF set for the quark-only reweight.")
    args = parser.parse_args()

    mass_edges = parse_edges(args.mass_thresholds, default_mass_edges)
    y_edges = parse_edges(args.y_thresholds, default_y_edges)
    n_mass = len(mass_edges) - 1
    n_y = len(y_edges) - 1
    n_unrolled = n_mass * n_y

    loader = make_loader(args.sample, args.n_split, args.max_files)
    hist_name = "h_unrolled_tr_ttbar_mass_y"
    hist_nominal = ROOT.TH1D(hist_name, "", n_unrolled, 0.0, float(n_unrolled))
    hist_nominal.SetDirectory(0)
    hist_nominal.Sumw2()
    set_sparse_bin_labels(hist_nominal, mass_edges, y_edges)
    hist_reweighted = clone_empty_hist(hist_nominal, f"{hist_name}_pdf_reweighted")
    hist_pdf_weights = ROOT.TH1D("h_pdf_reweights", "", 80, 0.8, 1.2)
    hist_pdf_weights.SetDirectory(0)
    hist_pdf_weights.Sumw2()

    print(
        f"Filling {hist_name}: sample={args.sample}, files={len(loader.files)}, shards={len(loader)}, "
        f"mass bins={n_mass}, y bins={n_y}, unrolled bins={n_unrolled}",
        flush=True,
    )
    total_events, selected_events, sumw, sumw_reweighted, quark_events = fill_histograms(
        loader, hist_nominal, hist_reweighted, hist_pdf_weights, mass_edges, y_edges, args
    )

    postfix = f"{args.sample}_{n_mass}x{n_y}"
    if args.small:
        postfix += "_small"
    plot_directory = f"{args.plot_directory}_{postfix}"
    outdir = os.path.join(user.plot_directory, "unrolled_ttbar", plot_directory)
    outdir_lin = os.path.join(outdir, "lin")
    outdir_log = os.path.join(outdir, "log")
    os.makedirs(outdir_lin, exist_ok=True)
    os.makedirs(outdir_log, exist_ok=True)
    helpers.copyIndexPHP(outdir)
    helpers.copyIndexPHP(outdir_lin)
    helpers.copyIndexPHP(outdir_log)

    root_path = os.path.join(outdir, f"{hist_name}.root")
    root_file = ROOT.TFile.Open(root_path, "RECREATE")
    hist_nominal.Write()
    hist_reweighted.Write()
    hist_pdf_weights.Write()
    root_file.Close()

    plot_name = "unrolled_tr_ttbar_mass_y"
    nominal_label = "PDF4LHC21 (quarks)"
    reweighted_label = "ATLASpdf21 (quarks)"
    draw_overlay(hist_nominal, hist_reweighted, mass_edges, y_edges, outdir_lin, plot_name, nominal_label, reweighted_label, log_y=False)
    draw_overlay(hist_nominal, hist_reweighted, mass_edges, y_edges, outdir_log, plot_name, nominal_label, reweighted_label, log_y=True)
    draw_weight_histogram(hist_pdf_weights, outdir)

    print(f"Read events: {total_events}", flush=True)
    print(f"Filled events: {selected_events}", flush=True)
    print(f"Quark-initiated events before kinematic binning: {quark_events}", flush=True)
    print(f"Sum of weights: {sumw:.6g}", flush=True)
    print(f"Sum of PDF-reweighted weights: {sumw_reweighted:.6g}", flush=True)
    print(f"Wrote ROOT histogram: {root_path}", flush=True)
    print(f"Wrote plots: {outdir}", flush=True)

    syncer.sync()

if __name__ == "__main__":
    main()
