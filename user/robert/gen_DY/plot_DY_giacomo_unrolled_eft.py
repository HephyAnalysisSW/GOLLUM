#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import sys

import awkward as ak
import numpy as np
import ROOT
import uproot
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
from data.RDataLoader import RDataLoader
import eft_reweighting
import samples_postprocessed


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

GIACOMO_MLL_BINS = np.array(
    [
        50,
        64,
        76,
        82,
        86,
        90,
        98,
        103,
        121,
        127,
        130,
        133,
        148,
        151,
        154,
        157,
        163,
        166,
        172,
        178,
        184,
        205,
        210,
        220,
        235,
        240,
        260,
        265,
        325,
        345,
        500,
        530,
        570,
        618,
        654,
        708,
        3000,
    ],
    dtype=np.float64,
)
GIACOMO_ABSY_BINS = np.array([0.0, 0.8, 1.6, 2.4], dtype=np.float64)
LOW_MASS_MLL_BINS = np.array([60, 70, 80, 86, 91, 96, 106, 120, 133], dtype=np.float64)
LOW_MASS_ABSY_BINS = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], dtype=np.float64)
MLL_BINS = GIACOMO_MLL_BINS
ABSY_BINS = GIACOMO_ABSY_BINS
COSTHETA_BINS = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=np.float64)

SCALAR_BRANCHES = [
    "dy_born_has_candidate",
    "dy_born_mll",
    "dy_born_yll",
    "dy_born_abs_yll",
    "cs_born_costheta",
    "xsec_weight",
]
VECTOR_BRANCHES = ["LHEReweightingWeight"]
WEIGHT_BRANCHES = ["xsec_weight"]
DEFAULT_SELECTION = "(dy_born_has_candidate > 0)"

# Approximate Set B swatches from Fig. 1 of arXiv:2107.02270.
COLORS = [
    ROOT.TColor.GetColor("#b0e0e8"),
    ROOT.TColor.GetColor("#50b8e8"),
    ROOT.TColor.GetColor("#607070"),
    ROOT.TColor.GetColor("#4068e0"),
    ROOT.TColor.GetColor("#a89898"),
    ROOT.TColor.GetColor("#b01008"),
    ROOT.TColor.GetColor("#d86850"),
    ROOT.TColor.GetColor("#f0b030"),
]


def sanitize(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def file_label(eft_points):
    wc_names = sorted({wc for _, values in eft_points for wc, value in values.items() if value != 0.0})
    return "_WC_" + "_".join(sanitize(wc) for wc in wc_names) if wc_names else ""


def selected_files(component, small=None, max_files=None):
    files = component.files
    if small:
        files = files[::small]
    if max_files is not None:
        files = files[:max_files]
    return files


def copy_index_php(directory):
    os.makedirs(directory, exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(helpers.__file__), "scripts/php/index.php"),
        os.path.join(directory, "index.php"),
    )


def make_loader(name, files, selection, files_per_chunk):
    required = list(dict.fromkeys(SCALAR_BRANCHES + VECTOR_BRANCHES))
    if not files:
        raise RuntimeError(f"No complete input files for {name}")
    n_split = max(1, int(math.ceil(len(files) / float(files_per_chunk))))
    loader = RDataLoader(
        input_paths=files,
        tree_name="Events",
        branches=required,
        selection=None,
        n_split=n_split,
        splitting_strategy="files",
        strict_branches=True,
        weight_branches=WEIGHT_BRANCHES,
        feature_names=SCALAR_BRANCHES,
        observer_names=[],
    )
    loader.addSelection(selection, required_branches=SCALAR_BRANCHES)
    loader.name = name
    return loader


def unroll_triple_index(mass_bin, abs_y_bin, costheta_bin):
    n_m = len(MLL_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    return (abs_y_bin * n_c + costheta_bin) * n_m + mass_bin


def unroll_yield_index(mass_bin, abs_y_bin, costheta_bin, yield_mode):
    if yield_mode == "mll_y_costheta":
        return unroll_triple_index(mass_bin, abs_y_bin, costheta_bin)
    if yield_mode == "mll_y":
        return unroll_afb_index(mass_bin, abs_y_bin)
    if yield_mode == "mll":
        return mass_bin
    raise RuntimeError(f"Unknown yield mode '{yield_mode}'")


def unroll_afb_index(mass_bin, abs_y_bin):
    n_m = len(MLL_BINS) - 1
    return abs_y_bin * n_m + mass_bin


def target_weight_matrix(lhe_weights, nominal_weights, eft_points):
    matrix = {"SM": nominal_weights * eft_reweighting.eft_weight(lhe_weights, config="auto")}
    for label, values in eft_points:
        matrix[label] = nominal_weights * eft_reweighting.eft_weight(lhe_weights, config="auto", **values)
    return matrix


def make_accumulators(labels, yield_mode):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    if yield_mode == "mll_y_costheta":
        n_yield = n_y * n_c * n_m
    elif yield_mode == "mll_y":
        n_yield = n_y * n_m
    elif yield_mode == "mll":
        n_yield = n_m
    else:
        raise RuntimeError(f"Unknown yield mode '{yield_mode}'")
    n_afb = n_y * n_m
    yield_sum = {label: np.zeros(n_yield, dtype=np.float64) for label in labels}
    afb_sumw = {label: np.zeros(n_afb, dtype=np.float64) for label in labels}
    afb_sum_sign = {label: np.zeros(n_afb, dtype=np.float64) for label in labels}
    return yield_sum, afb_sumw, afb_sum_sign


def print_shard_diagnostics(loader, ishard, lengths, error):
    files = []
    if hasattr(loader, "_file_splits"):
        try:
            files = loader._file_splits[ishard]
        except Exception:
            files = []
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    length_summary = ", ".join(f"{int(length)}:{int(count)}" for length, count in zip(unique_lengths, counts))
    print("[giacomo] ERROR while evaluating EFT weights")
    print(f"[giacomo] sample: {loader.name}")
    print(f"[giacomo] shard: {ishard}/{len(loader)}")
    print(f"[giacomo] LHEReweightingWeight vector lengths in valid events: {length_summary}")
    print(f"[giacomo] exception: {error}")
    print(f"[giacomo] files in failing shard ({len(files)}):")
    for filename in files:
        print(f"[giacomo]   {filename}")
    print("[giacomo] per-file LHEReweightingWeight vector-length diagnostic:")
    for filename in files:
        try:
            with uproot.open(filename, object_cache=None, array_cache=None) as root_file:
                weights = root_file[loader.tree_name]["LHEReweightingWeight"].array(library="ak")
            file_lengths = ak.to_numpy(ak.num(weights, axis=1))
            file_unique, file_counts = np.unique(file_lengths, return_counts=True)
            file_summary = ", ".join(
                f"{int(length)}:{int(count)}" for length, count in zip(file_unique, file_counts)
            )
            print(f"[giacomo]   {filename}: {file_summary}")
        except Exception as file_error:
            print(f"[giacomo]   {filename}: diagnostic failed: {file_error}")


def accumulate(loader, labels, eft_points, yield_sum, afb_sumw, afb_sum_sign, yield_mode):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    total_selected = 0
    total_binned = 0
    total_empty_lhe_weights = 0
    for ishard in tqdm(range(len(loader)), desc=f"Giacomo unrolled chunks {loader.name}", unit="chunk"):
        ar = loader.load_selection_shard(ishard)
        if len(ar) == 0:
            continue
        mll = ak.to_numpy(ar["dy_born_mll"])
        yll = ak.to_numpy(ar["dy_born_yll"])
        abs_yll = ak.to_numpy(ar["dy_born_abs_yll"])
        costheta = ak.to_numpy(ar["cs_born_costheta"])
        nominal = ak.to_numpy(ar["xsec_weight"])
        lhe_weights = loader.vector_branch(ar, "LHEReweightingWeight")
        n_lhe_weights = ak.to_numpy(ak.num(lhe_weights, axis=1))

        mass_bin = np.searchsorted(MLL_BINS, mll, side="right") - 1
        abs_y_bin = np.searchsorted(ABSY_BINS, abs_yll, side="right") - 1
        costheta_bin = np.searchsorted(COSTHETA_BINS, costheta, side="right") - 1
        signed_costheta = np.sign(yll) * costheta

        valid_common = (
            np.isfinite(mll)
            & np.isfinite(nominal)
            & (mll > -998)
            & (mass_bin >= 0)
            & (mass_bin < n_m)
        )
        valid_yield = valid_common.copy()
        if yield_mode in ("mll_y", "mll_y_costheta"):
            valid_yield = valid_yield & np.isfinite(abs_yll) & (abs_yll > -998) & (abs_y_bin >= 0) & (abs_y_bin < n_y)
        if yield_mode == "mll_y_costheta":
            valid_yield = valid_yield & np.isfinite(costheta) & (costheta > -998) & (costheta_bin >= 0) & (costheta_bin < n_c)

        valid_afb = (
            valid_common
            & np.isfinite(yll)
            & np.isfinite(abs_yll)
            & np.isfinite(costheta)
            & (abs_yll > -998)
            & (costheta > -998)
            & (np.sign(yll) != 0)
            & (abs_y_bin >= 0)
            & (abs_y_bin < n_y)
            & (costheta_bin >= 0)
            & (costheta_bin < n_c)
        )
        valid = valid_yield | valid_afb
        empty_lhe_weights = valid & (n_lhe_weights <= 0)
        total_empty_lhe_weights += int(np.count_nonzero(empty_lhe_weights))
        valid = valid & (n_lhe_weights > 0)
        valid_yield = valid_yield & (n_lhe_weights > 0)
        valid_afb = valid_afb & (n_lhe_weights > 0)
        total_selected += len(mll)
        total_binned += int(np.count_nonzero(valid_yield))
        if not np.any(valid):
            continue

        valid_positions = np.flatnonzero(valid)
        yield_in_valid = np.nonzero(valid_yield[valid])[0]
        afb_in_valid = np.nonzero(valid_afb[valid])[0]
        yield_idx = unroll_yield_index(
            mass_bin[valid_positions[yield_in_valid]],
            abs_y_bin[valid_positions[yield_in_valid]],
            costheta_bin[valid_positions[yield_in_valid]],
            yield_mode,
        )
        afb_idx = unroll_afb_index(mass_bin[valid_positions[afb_in_valid]], abs_y_bin[valid_positions[afb_in_valid]])
        a4_basis = signed_costheta[valid_positions[afb_in_valid]]
        valid_lhe_weights = lhe_weights[valid]
        valid_lhe_lengths = n_lhe_weights[valid]
        try:
            weights_by_label = target_weight_matrix(valid_lhe_weights, nominal[valid], eft_points)
        except Exception as error:
            print_shard_diagnostics(loader, ishard, valid_lhe_lengths, error)
            raise

        for label in labels:
            weights = weights_by_label[label]
            finite_w = np.isfinite(weights)
            finite_yield = finite_w[yield_in_valid]
            if np.any(finite_yield):
                np.add.at(yield_sum[label], yield_idx[finite_yield], weights[yield_in_valid][finite_yield])
            finite_afb = finite_w[afb_in_valid]
            if np.any(finite_afb):
                afb_weights = weights[afb_in_valid][finite_afb]
                np.add.at(afb_sumw[label], afb_idx[finite_afb], afb_weights)
                np.add.at(afb_sum_sign[label], afb_idx[finite_afb], afb_weights * a4_basis[finite_afb])

    print(f"[giacomo] selected events after loader selection: {total_selected}")
    print(f"[giacomo] events inside hard-coded unrolling bins: {total_binned}")
    if total_empty_lhe_weights:
        print(f"[giacomo] skipped events with empty LHEReweightingWeight: {total_empty_lhe_weights}")
    return total_selected, total_binned


def finalize_a4(labels, afb_sumw, afb_sum_sign):
    n_a4 = (len(ABSY_BINS) - 1) * (len(MLL_BINS) - 1)
    a4 = {}
    for label in labels:
        values = np.full(n_a4, np.nan, dtype=np.float64)
        nonzero = afb_sumw[label] != 0.0
        values[nonzero] = 4.0 * afb_sum_sign[label][nonzero] / afb_sumw[label][nonzero]
        a4[label] = values
    return a4


def make_hist(name, title, values):
    hist = ROOT.TH1F(name, title, len(values), 0, len(values))
    for i, value in enumerate(values, start=1):
        hist.SetBinContent(i, float(value) if np.isfinite(value) else 0.0)
    return hist


def make_block_hists(values, name, block, n_m, color, positive_only=True):
    hists = []
    start = block * n_m
    stop = start + n_m
    seg_start = None
    seg_values = []
    for ibin in range(start, stop):
        value = values[ibin]
        valid = np.isfinite(value) and (not positive_only or value > 0)
        if valid:
            if seg_start is None:
                seg_start = ibin
            seg_values.append(float(value))
            continue
        if seg_start is not None:
            hists.append(make_segment_hist(name, block, seg_start, seg_values, color))
            seg_start = None
            seg_values = []
    if seg_start is not None:
        hists.append(make_segment_hist(name, block, seg_start, seg_values, color))
    return hists


def make_segment_hist(name, block, start_bin, values, color):
    hist = ROOT.TH1D(f"{name}_block{block}_{start_bin}", "", len(values), start_bin, start_bin + len(values))
    hist.SetLineColor(color)
    hist.SetLineWidth(2)
    hist.SetFillStyle(0)
    for i, value in enumerate(values, start=1):
        hist.SetBinContent(i, value)
    return hist


def set_pad_ticks(pad):
    pad.SetTickx(1)
    pad.SetTicky(1)


def shorten_ticks(frame, x_length=0.020, y_length=0.010):
    frame.GetXaxis().SetTickLength(x_length)
    frame.GetYaxis().SetTickLength(y_length)


def draw_block_lines(n_blocks, n_m, ymin, ymax):
    stuff = []
    for iblock in range(1, n_blocks):
        line = ROOT.TLine(iblock * n_m, ymin, iblock * n_m, ymax)
        line.SetLineStyle(3)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
        stuff.append(line)
    return stuff


def draw_mass_bin_lines(n_blocks, n_m, ymin, ymax):
    stuff = []
    for iblock in range(n_blocks):
        offset = iblock * n_m
        for im in range(1, n_m):
            line = ROOT.TLine(offset + im, ymin, offset + im, ymax)
            line.SetLineStyle(3)
            line.SetLineColor(ROOT.kGray)
            line.SetLineWidth(1)
            line.Draw()
            stuff.append(line)
    return stuff


def draw_z_peak_bands(n_blocks, n_m, ymin, ymax):
    stuff = []
    ilo = int(np.searchsorted(MLL_BINS, 82.0, side="left"))
    ihi = int(np.searchsorted(MLL_BINS, 103.0, side="left"))
    for iblock in range(n_blocks):
        box = ROOT.TBox(iblock * n_m + ilo, ymin, iblock * n_m + ihi, ymax)
        box.SetFillColor(ROOT.kGray)
        box.SetFillStyle(3005)
        box.SetLineColor(ROOT.kGray)
        box.Draw()
        stuff.append(box)
    return stuff


def draw_mass_edge_labels(n_blocks, n_m, ymin, ymax):
    stuff = []
    label_edges = {50, 82, 103, 3000}
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(12)
    latex.SetTextAngle(90)
    latex.SetTextSize(0.035)
    y_text = ymin + 0.05 * (ymax - ymin)
    for iedge, edge in enumerate(MLL_BINS):
        if int(edge) not in label_edges:
            continue
        obj = latex.DrawLatex(iedge + 0.15, y_text, f"{edge:g}")
        stuff.append(obj)
    latex.SetTextAngle(0)
    latex.SetTextAlign(13)
    latex.SetTextSize(0.034)
    title = latex.DrawLatex(0.5, ymax - 0.06 * (ymax - ymin), "hatched: 82#leq m_{#mu#mu}<103 GeV")
    stuff.append(title)
    stuff.append(latex)
    return stuff


def interval_label(var, lo, hi):
    return f"{lo:g}#leq {var}<{hi:g}"


def draw_triple_labels(n_m, ymin, ymax):
    stuff = []
    n_c = len(COSTHETA_BINS) - 1
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.014)
    y_text = ymax / (ymax / ymin) ** 0.075
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        for ic, (clo, chi) in enumerate(zip(COSTHETA_BINS[:-1], COSTHETA_BINS[1:])):
            block = iy * n_c + ic
            text = (
                f"#splitline{{{interval_label('|y_{#mu#mu}|', ylo, yhi)}}}"
                f"{{{interval_label('cos#theta^{*}', clo, chi)}}}"
            )
            obj = latex.DrawLatex((block + 0.5) * n_m, y_text, text)
            stuff.append(obj)
    stuff.append(latex)
    return stuff


def draw_yield_labels(n_m, ymin, ymax, yield_mode):
    if yield_mode == "mll_y_costheta":
        return draw_triple_labels(n_m, ymin, ymax)
    if yield_mode == "mll":
        return []

    stuff = []
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.030)
    y_text = ymax / (ymax / ymin) ** 0.075
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        obj = latex.DrawLatex((iy + 0.5) * n_m, y_text, interval_label("|y_{#mu#mu}|", ylo, yhi))
        stuff.append(obj)
    stuff.append(latex)
    return stuff


def draw_afb_labels(n_m, ymin, ymax):
    stuff = []
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.032)
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        obj = latex.DrawLatex(
            (iy + 0.5) * n_m,
            ymax - 0.10 * (ymax - ymin),
            interval_label("|y_{#mu#mu}|", ylo, yhi),
        )
        stuff.append(obj)
    stuff.append(latex)
    return stuff


def configure_mll_axis(axis, n_blocks, n_m):
    axis.SetTitle("m_{#mu#mu} bin [GeV], repeated in |y_{#mu#mu}| blocks")
    axis.SetLabelSize(0.050)
    for block in range(n_blocks):
        for im in range(n_m):
            ibin = block * n_m + im + 1
            if im == n_m - 1:
                label = f"{MLL_BINS[im]:g}-{MLL_BINS[im + 1]:g}"
            else:
                label = f"{MLL_BINS[im]:g}"
            axis.SetBinLabel(ibin, label)
    axis.LabelsOption("v")


def plot_yield(plot_dir, labels, yield_sum, yield_mode):
    n_m = len(MLL_BINS) - 1
    if yield_mode == "mll_y_costheta":
        n_blocks = (len(ABSY_BINS) - 1) * (len(COSTHETA_BINS) - 1)
        suffix = "mll_y_costheta"
        x_title = "Triple diff bin"
    elif yield_mode == "mll_y":
        n_blocks = len(ABSY_BINS) - 1
        suffix = "mll_y"
        x_title = "m_{#mu#mu} unrolled in |y_{#mu#mu}|"
    elif yield_mode == "mll":
        n_blocks = 1
        suffix = "mll"
        x_title = "m_{#mu#mu}"
    else:
        raise RuntimeError(f"Unknown yield mode '{yield_mode}'")
    canvas = ROOT.TCanvas(f"c_giacomo_unrolled_{suffix}", f"Giacomo unrolled {suffix}", 1800, 760)
    top = ROOT.TPad("top", "top", 0.0, 0.30, 1.0, 1.0)
    bot = ROOT.TPad("bot", "bot", 0.0, 0.0, 1.0, 0.30)
    top.SetBottomMargin(0.02)
    top.SetLogy(True)
    top.SetRightMargin(0.07)
    bot.SetTopMargin(0.03)
    bot.SetBottomMargin(0.34)
    bot.SetRightMargin(0.07)
    set_pad_ticks(top)
    set_pad_ticks(bot)
    top.Draw()
    bot.Draw()
    stuff = [canvas, top, bot]

    hists = []
    for ilabel, label in enumerate(labels):
        hist = make_hist(f"h_yield_{sanitize(label)}", f";{x_title};Events", yield_sum[label])
        color = ROOT.kBlack if label == "SM" else COLORS[(ilabel - 1) % len(COLORS)]
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetFillStyle(0)
        hists.append(hist)
        stuff.append(hist)

    positive = [h.GetBinContent(i) for h in hists for i in range(1, h.GetNbinsX() + 1) if h.GetBinContent(i) > 0]
    ymax = 2.0 * max(positive) if positive else 1.0
    ymin = max(1e-6, 0.5 * min(positive)) if positive else 1e-6

    top.cd()
    frame_top = ROOT.TH2F("frame_yield", f";{x_title};Events", len(yield_sum["SM"]), 0, len(yield_sum["SM"]), 100, ymin, ymax)
    frame_top.GetXaxis().SetLabelSize(0)
    frame_top.GetYaxis().SetTitleSize(0.055)
    frame_top.GetYaxis().SetLabelSize(0.052)
    frame_top.GetYaxis().SetTitleOffset(0.72)
    shorten_ticks(frame_top, x_length=0.010, y_length=0.006)
    frame_top.Draw()
    stuff.append(frame_top)
    if yield_mode in ("mll_y", "mll"):
        stuff += draw_z_peak_bands(n_blocks, n_m, ymin, ymax)
        stuff += draw_mass_bin_lines(n_blocks, n_m, ymin, ymax)
    stuff += draw_block_lines(n_blocks, n_m, ymin, ymax)
    stuff += draw_yield_labels(n_m, ymin, ymax, yield_mode)
    legend = ROOT.TLegend(0.10, 0.12, 0.30, 0.34)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.045)
    for ilabel, label in enumerate(labels):
        color = ROOT.kBlack if label == "SM" else COLORS[(ilabel - 1) % len(COLORS)]
        legend_hist = None
        for block in range(n_blocks):
            for hist in make_block_hists(yield_sum[label], f"h_yield_{sanitize(label)}", block, n_m, color):
                hist.Draw("L SAME")
                stuff.append(hist)
                if legend_hist is None:
                    legend_hist = hist
        if legend_hist is not None:
            legend.AddEntry(legend_hist, label, "l")
    legend.Draw()
    stuff.append(legend)

    bot.cd()
    ratios = []
    sm = yield_sum["SM"]
    for ilabel, label in enumerate(labels[1:], start=1):
        ratio = np.full_like(sm, np.nan, dtype=np.float64)
        ok = sm != 0.0
        ratio[ok] = yield_sum[label][ok] / sm[ok]
        hist = make_hist(f"h_yield_ratio_{sanitize(label)}", f";{x_title};Ratio to SM", ratio)
        hist.SetLineColor(COLORS[(ilabel - 1) % len(COLORS)])
        hist.SetLineWidth(2)
        hist.SetFillStyle(0)
        ratios.append(hist)
        stuff.append(hist)
    if ratios:
        frame_ratio = ROOT.TH2F("frame_yield_ratio", f";{x_title};Ratio to SM", len(sm), 0, len(sm), 100, 0.75, 1.25)
        frame_ratio.GetXaxis().SetTitleSize(0.13)
        frame_ratio.GetXaxis().SetLabelSize(0.10)
        frame_ratio.GetYaxis().SetTitleSize(0.12)
        frame_ratio.GetYaxis().SetLabelSize(0.10)
        frame_ratio.GetYaxis().SetTitleOffset(0.32)
        frame_ratio.GetYaxis().SetNdivisions(505)
        shorten_ticks(frame_ratio, x_length=0.020, y_length=0.006)
        frame_ratio.Draw()
        stuff.append(frame_ratio)
        for ilabel, label in enumerate(labels[1:], start=1):
            ratio = np.full_like(sm, np.nan, dtype=np.float64)
            ok = sm != 0.0
            ratio[ok] = yield_sum[label][ok] / sm[ok]
            for block in range(n_blocks):
                for hist in make_block_hists(
                    ratio,
                    f"h_yield_ratio_{sanitize(label)}",
                    block,
                    n_m,
                    COLORS[(ilabel - 1) % len(COLORS)],
                ):
                    hist.Draw("L SAME")
                    stuff.append(hist)
    line = ROOT.TLine(0, 1.0, len(sm), 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()
    stuff.append(line)
    if yield_mode in ("mll_y", "mll"):
        stuff += draw_z_peak_bands(n_blocks, n_m, 0.75, 1.25)
        stuff += draw_mass_bin_lines(n_blocks, n_m, 0.75, 1.25)
        stuff += draw_mass_edge_labels(n_blocks, n_m, 0.75, 1.25)
    stuff += draw_block_lines(n_blocks, n_m, 0.75, 1.25)

    base = os.path.join(plot_dir, f"giacomo_unrolled_{suffix}")
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
    print(f"[giacomo] output: {base}.{{png,pdf,root}}")


def plot_a4(plot_dir, labels, a4, delta_a4_y_range=None):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_bins = n_y * n_m

    finite_chunks = [values[np.isfinite(values)] for values in a4.values() if np.any(np.isfinite(values))]
    finite = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=np.float64)
    if len(finite):
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        pad = max(0.05 * (ymax - ymin), 0.02)
        ymin -= pad
        ymax += pad
    else:
        ymin, ymax = -0.1, 0.1

    sm = a4["SM"]
    delta = {}
    for label in labels[1:]:
        delta[label] = a4[label] - sm
    finite_delta_chunks = [values[np.isfinite(values)] for values in delta.values() if np.any(np.isfinite(values))]
    finite_delta = np.concatenate(finite_delta_chunks) if finite_delta_chunks else np.array([], dtype=np.float64)
    if delta_a4_y_range is not None:
        delta_ymin, delta_ymax = delta_a4_y_range
    else:
        delta_ymax = max(0.001, 1.25 * float(np.max(np.abs(finite_delta))) if len(finite_delta) else 0.001)
        delta_ymin = -delta_ymax

    canvas = ROOT.TCanvas("c_giacomo_a4_mll_y", "A4 mll y unrolled", 1700, 860)
    top = ROOT.TPad("a4_top", "a4_top", 0.0, 0.34, 1.0, 1.0)
    bot = ROOT.TPad("a4_bot", "a4_bot", 0.0, 0.0, 1.0, 0.34)
    top.SetBottomMargin(0.02)
    top.SetTopMargin(0.11)
    top.SetLeftMargin(0.10)
    top.SetRightMargin(0.04)
    bot.SetTopMargin(0.03)
    bot.SetBottomMargin(0.42)
    bot.SetLeftMargin(0.10)
    bot.SetRightMargin(0.04)
    set_pad_ticks(top)
    set_pad_ticks(bot)
    top.Draw()
    bot.Draw()
    stuff = [canvas, top, bot]

    top.cd()
    frame = ROOT.TH2F("frame_a4", ";m_{#mu#mu} unrolled in |y_{#mu#mu}|;A_{4}", n_bins, 0, n_bins, 100, ymin, ymax)
    frame.GetXaxis().SetLabelSize(0)
    frame.GetYaxis().SetTitleSize(0.060)
    frame.GetYaxis().SetLabelSize(0.052)
    frame.GetYaxis().SetTitleOffset(0.70)
    shorten_ticks(frame, x_length=0.010, y_length=0.007)
    frame.Draw()
    stuff.append(frame)

    zero = ROOT.TLine(0, 0, n_bins, 0)
    zero.SetLineStyle(2)
    zero.SetLineColor(ROOT.kGray + 2)
    zero.Draw()
    stuff.append(zero)
    stuff += draw_block_lines(n_y, n_m, ymin, ymax)
    stuff += draw_afb_labels(n_m, ymin, ymax)

    legend = ROOT.TLegend(0.10, 0.89, 0.96, 0.97)
    legend.SetNColumns(min(len(labels), 4))
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.030)
    stuff.append(legend)

    for ilabel, label in enumerate(labels):
        color = ROOT.kBlack if label == "SM" else COLORS[(ilabel - 1) % len(COLORS)]
        legend_hist = None
        for iy in range(n_y):
            for hist in make_block_hists(a4[label], f"h_a4_{sanitize(label)}", iy, n_m, color, positive_only=False):
                hist.Draw("L SAME")
                stuff.append(hist)
                if legend_hist is None:
                    legend_hist = hist
        if legend_hist is not None:
            legend.AddEntry(legend_hist, label, "l")
    legend.Draw()

    bot.cd()
    frame_delta = ROOT.TH2F(
        "frame_deltaA4",
        ";m_{#mu#mu} bin [GeV], repeated in |y_{#mu#mu}| blocks;#Delta A_{4}",
        n_bins,
        0,
        n_bins,
        100,
        delta_ymin,
        delta_ymax,
    )
    configure_mll_axis(frame_delta.GetXaxis(), n_y, n_m)
    frame_delta.GetXaxis().SetTitleSize(0.095)
    frame_delta.GetXaxis().SetTitleOffset(2.30)
    frame_delta.GetYaxis().SetTitleSize(0.105)
    frame_delta.GetYaxis().SetLabelSize(0.085)
    frame_delta.GetYaxis().SetTitleOffset(0.40)
    frame_delta.GetYaxis().SetNdivisions(505)
    shorten_ticks(frame_delta, x_length=0.014, y_length=0.007)
    frame_delta.Draw()
    stuff.append(frame_delta)
    zero_delta = ROOT.TLine(0, 0, n_bins, 0)
    zero_delta.SetLineStyle(2)
    zero_delta.SetLineColor(ROOT.kGray + 2)
    zero_delta.Draw()
    stuff.append(zero_delta)
    stuff += draw_block_lines(n_y, n_m, delta_ymin, delta_ymax)
    for ilabel, label in enumerate(labels[1:], start=1):
        color = COLORS[(ilabel - 1) % len(COLORS)]
        for iy in range(n_y):
            for hist in make_block_hists(
                delta[label],
                f"h_deltaA4_{sanitize(label)}",
                iy,
                n_m,
                color,
                positive_only=False,
            ):
                hist.Draw("L SAME")
                stuff.append(hist)

    base = os.path.join(plot_dir, "A4_DeltaA4_unrolled_mll_y")
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
    print(f"[giacomo] output: {base}.{{png,pdf,root}}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    nargs="+",
    default=None,
    help="Samples from samples_postprocessed.py. Defaults to the two EFT weight-config groups.",
)
parser.add_argument(
    "--low-mass",
    action="store_true",
    help="Use the reduced A4/qpm low-mass binning and default to the low-mass EFT sample group.",
)
parser.add_argument("--selection", default=DEFAULT_SELECTION, help="RDataLoader selection string; default has no mass cut")
parser.add_argument("--small", nargs="?", const=10, type=int, default=None, help="Use every Nth file, e.g. --small 10")
parser.add_argument("--max-files", type=int, default=None)
parser.add_argument("--files-per-chunk", type=int, default=200)
parser.add_argument("--eft-point", action="append", default=[], help="label:wc=value,wc2=value. Can be repeated.")
parser.add_argument(
    "--yield-mode",
    choices=["mll_y_costheta", "mll_y", "mll"],
    default="mll_y_costheta",
    help="Yield plot unrolling. Default keeps mll x |y| x cos(theta*); mll_y makes four spectra; mll makes one inclusive spectrum.",
)
parser.add_argument(
    "--afb-ratio-eps",
    type=float,
    default=0.03,
    help="Deprecated; kept for command compatibility. The A4 plot now shows absolute Delta A4, not a ratio.",
)
parser.add_argument("--delta-a4-y-min", type=float, default=None, help="Optional lower y-axis bound for the Delta A4 bottom pad")
parser.add_argument("--delta-a4-y-max", type=float, default=None, help="Optional upper y-axis bound for the Delta A4 bottom pad")
args = parser.parse_args()

if (args.delta_a4_y_min is None) != (args.delta_a4_y_max is None):
    raise RuntimeError("Set both --delta-a4-y-min and --delta-a4-y-max, or neither.")
if args.delta_a4_y_min is not None and args.delta_a4_y_min >= args.delta_a4_y_max:
    raise RuntimeError("--delta-a4-y-min must be smaller than --delta-a4-y-max.")
delta_a4_y_range = None if args.delta_a4_y_min is None else (args.delta_a4_y_min, args.delta_a4_y_max)

default_samples = ["DYMuMu_NLO_EFT_SMEFTatNLO_shortEFT", "DYMuMu_NLO_EFT_SMEFTatNLO_fullEFT"]
low_mass_samples = ["DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT"]
if args.samples is None:
    args.samples = low_mass_samples if args.low_mass else default_samples
if args.low_mass:
    MLL_BINS = LOW_MASS_MLL_BINS
    ABSY_BINS = LOW_MASS_ABSY_BINS

eft_points = [eft_reweighting.parse_eft_point(point) for point in args.eft_point]
labels = ["SM"] + [label for label, _ in eft_points]
copy_index_php(os.path.join(user.plot_directory, "DY"))
copy_index_php(os.path.join(user.plot_directory, "DY", "giacomo_unrolled_eft_2"))

if args.samples == default_samples:
    label = "DYMuMu_NLO_EFT_SMEFTatNLO_allEFT"
elif args.samples == low_mass_samples:
    label = "DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT"
else:
    label = "_".join(args.samples)
if args.small:
    label += f"_small{args.small}"
if args.low_mass:
    label += "_lowMassBins"
if args.yield_mode != "mll_y_costheta":
    label += f"_{args.yield_mode}"
label += file_label(eft_points)

plot_dir = os.path.join(user.plot_directory, "DY", "giacomo_unrolled_eft_2", label)
os.makedirs(plot_dir, exist_ok=True)
copy_index_php(plot_dir)
print(f"[giacomo] output directory: {plot_dir}")
print(f"[giacomo] selection: {args.selection}")
print(f"[giacomo] yield mode: {args.yield_mode}")
print(f"[giacomo] binning: {'low-mass A4/qpm' if args.low_mass else 'Giacomo'}")

yield_sum, afb_sumw, afb_sum_sign = make_accumulators(labels, args.yield_mode)
grand_selected = 0
grand_binned = 0

for sample_name in args.samples:
    if sample_name not in samples_postprocessed.samples_by_name:
        raise RuntimeError(f"Unknown sample '{sample_name}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")

    component = samples_postprocessed.samples_by_name[sample_name]
    files = selected_files(component, small=args.small, max_files=args.max_files)

    print(f"[giacomo] sample: {component.name}")
    print(f"[giacomo] files: {len(files)}")
    print(f"[giacomo] files per shard: {args.files_per_chunk}")

    loader = make_loader(component.name, files, args.selection, args.files_per_chunk)
    selected, binned = accumulate(loader, labels, eft_points, yield_sum, afb_sumw, afb_sum_sign, args.yield_mode)
    grand_selected += selected
    grand_binned += binned

print(f"[giacomo] total selected events after loader selection: {grand_selected}")
print(f"[giacomo] total events inside hard-coded unrolling bins: {grand_binned}")
a4 = finalize_a4(labels, afb_sumw, afb_sum_sign)
plot_yield(plot_dir, labels, yield_sum, args.yield_mode)
plot_a4(plot_dir, labels, a4, delta_a4_y_range=delta_a4_y_range)
syncer.sync()
