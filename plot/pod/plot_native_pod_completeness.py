#!/usr/bin/env python3
"""ROOT plots for native POD-basis PDF reconstruction completeness."""

from __future__ import annotations

import argparse
import os
import sys
from array import array

import lhapdf
import numpy as np
import ROOT

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user

from check_native_pod_completeness import (
    BASIS_SET,
    Q0,
    TARGET_CANDIDATES,
    decompose,
    first_installed_pdf,
    metric_weights,
    pdfset_size,
)
from native_pod_basis_40k import NativePODBasis
from nnpdf.constants import LHAPDF_XGRID


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)


DEFAULT_FLAVORS = (21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5)


def pid_to_tex(pid: int) -> str:
    labels = {
        21: "g",
        2: "u",
        -2: "#bar{u}",
        1: "d",
        -1: "#bar{d}",
        3: "s",
        -3: "#bar{s}",
        4: "c",
        -4: "#bar{c}",
        5: "b",
        -5: "#bar{b}",
        6: "t",
        -6: "#bar{t}",
        22: "#gamma",
    }
    return labels.get(pid, f"pid {pid}")


def pid_to_name(pid: int) -> str:
    labels = {
        21: "g",
        2: "u",
        -2: "ubar",
        1: "d",
        -1: "dbar",
        3: "s",
        -3: "sbar",
        4: "c",
        -4: "cbar",
        5: "b",
        -5: "bbar",
        6: "t",
        -6: "tbar",
        22: "gamma",
    }
    return labels.get(pid, f"pid{pid}")


def parse_flavors(value: str, basis_set: str, target_set: str) -> tuple[int, ...]:
    if value == "qcd5":
        return DEFAULT_FLAVORS
    if value in {"qcd4", "no_b"}:
        return tuple(pid for pid in DEFAULT_FLAVORS if abs(pid) != 5)
    if value == "common":
        basis = set(int(pid) for pid in lhapdf.mkPDF(basis_set, 0).flavors())
        target = set(int(pid) for pid in lhapdf.mkPDF(target_set, 0).flavors())
        return tuple(pid for pid in DEFAULT_FLAVORS if pid in basis and pid in target)
    if value == "lhapdf":
        return tuple(int(pid) for pid in lhapdf.mkPDF(basis_set, 0).flavors())
    return tuple(int(pid.strip()) for pid in value.split(",") if pid.strip())


def make_graph(name: str, x_values: np.ndarray, y_values: np.ndarray, color: int, style: int) -> ROOT.TGraph:
    graph = ROOT.TGraph(len(x_values), array("d", x_values), array("d", y_values))
    graph.SetName(name)
    graph.SetLineColor(color)
    graph.SetLineStyle(style)
    graph.SetLineWidth(2)
    return graph


def draw_latex(x: float, y: float, text: str, size: float = 0.045) -> ROOT.TLatex:
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(size)
    latex.DrawLatex(x, y, text)
    return latex


def style_log_x_axis(frame: ROOT.TH1D) -> None:
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetXaxis().SetLabelSize(0.032)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetLabelSize(0.040)
    frame.GetYaxis().SetTitleOffset(1.25)


def draw_info_block(lines: Sequence[str]) -> None:
    y = 0.84
    for line in lines:
        if line:
            draw_latex(0.08, y, line, 0.045)
            y -= 0.11


def y_range(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))])
    finite = finite[np.abs(finite) > 0]
    if finite.size == 0:
        return -1.0, 1.0

    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if ymin > 0:
        return 0.75 * ymin, 1.25 * ymax
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    return ymin - 0.15 * span, ymax + 0.15 * span


def draw_panel_grid(
    out_dir: str,
    out_base: str,
    x_grid: np.ndarray,
    true_grid: np.ndarray,
    reco_grid: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_pdf_reco", "c_pdf_reco", 1500, 420 * n_rows)
    canvas.Divide(n_cols, n_rows, 0.002, 0.002)

    keepalive = []
    for i_flavor, pid in enumerate(flavors):
        pad = canvas.cd(i_flavor + 1)
        pad.SetLogx(True)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.13)
        pad.SetTicks(1, 1)

        true = true_grid[i_flavor]
        reco = reco_grid[i_flavor]
        ymin, ymax = y_range([true, reco])

        frame = ROOT.TH1D(
            f"frame_{pid_to_name(pid)}",
            f";x;x f_{{{pid_to_tex(pid)}}}(x,Q)",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(ymin)
        frame.SetMaximum(ymax)
        style_log_x_axis(frame)
        frame.Draw()

        g_true = make_graph(f"g_true_{pid_to_name(pid)}", x_grid, true, ROOT.kBlack, ROOT.kSolid)
        g_reco = make_graph(f"g_reco_{pid_to_name(pid)}", x_grid, reco, ROOT.kRed + 1, ROOT.kDashed)
        g_true.Draw("L SAME")
        g_reco.Draw("L SAME")

        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)

        keepalive.extend([frame, g_true, g_reco])

    info_pad_index = n_flavors + 1 if n_flavors < n_cols * n_rows else 1
    pad = canvas.cd(info_pad_index)
    pad.SetLeftMargin(0.08)
    pad.SetRightMargin(0.04)
    pad.SetTopMargin(0.08)
    pad.SetBottomMargin(0.08)
    draw_info_block(info_lines)
    legend = ROOT.TLegend(0.08, 0.54, 0.88, 0.74)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.050)
    legend.AddEntry(keepalive[1], "true target", "l")
    legend.AddEntry(keepalive[2], "POD reconstruction", "l")
    legend.Draw()
    keepalive.append(legend)

    for ext in ("png", "pdf", "root"):
        path = os.path.join(out_dir, f"{out_base}.{ext}")
        canvas.SaveAs(path)
        outputs.append(path)
    return outputs


def draw_ratio_grid(
    out_dir: str,
    out_base: str,
    x_grid: np.ndarray,
    true_grid: np.ndarray,
    reco_grid: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_pdf_reco_ratio", "c_pdf_reco_ratio", 1500, 420 * n_rows)
    canvas.Divide(n_cols, n_rows, 0.002, 0.002)

    keepalive = []
    for i_flavor, pid in enumerate(flavors):
        pad = canvas.cd(i_flavor + 1)
        pad.SetLogx(True)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.13)
        pad.SetTicks(1, 1)

        true = true_grid[i_flavor]
        reco = reco_grid[i_flavor]
        scale = max(float(np.max(np.abs(true))), 1e-14)
        active = np.abs(true) > 1e-10 * scale
        ratio = np.full_like(true, np.nan, dtype=float)
        ratio[active] = reco[active] / true[active]
        finite = ratio[np.isfinite(ratio)]

        if finite.size:
            ymin = max(0.98, float(np.min(finite)) - 0.002)
            ymax = min(1.02, float(np.max(finite)) + 0.002)
            if ymin >= ymax:
                ymin, ymax = 0.995, 1.005
        else:
            ymin, ymax = 0.995, 1.005

        frame = ROOT.TH1D(
            f"frame_ratio_{pid_to_name(pid)}",
            f";x;reco / true",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(ymin)
        frame.SetMaximum(ymax)
        style_log_x_axis(frame)
        frame.GetYaxis().SetNdivisions(505)
        frame.Draw()

        line = ROOT.TLine(float(x_grid[0]), 1.0, float(x_grid[-1]), 1.0)
        line.SetLineColor(ROOT.kGray + 2)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw("SAME")

        if finite.size:
            g_ratio = make_graph(
                f"g_ratio_{pid_to_name(pid)}",
                x_grid[active],
                ratio[active],
                ROOT.kBlue + 1,
                ROOT.kSolid,
            )
            g_ratio.Draw("L SAME")
            keepalive.append(g_ratio)
        else:
            draw_latex(0.25, 0.50, "inactive at this Q", 0.045)

        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)
        keepalive.extend([frame, line])

    if n_flavors < n_cols * n_rows:
        pad = canvas.cd(n_flavors + 1)
        pad.SetLeftMargin(0.08)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.08)
        draw_info_block(info_lines)
        draw_latex(0.08, 0.20, "Ratio shown only where true PDF is nonzero.", 0.040)

    for ext in ("png", "pdf", "root"):
        path = os.path.join(out_dir, f"{out_base}.{ext}")
        canvas.SaveAs(path)
        outputs.append(path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot native POD-basis reconstruction vs target PDF.")
    parser.add_argument("--basis-set", default=BASIS_SET)
    parser.add_argument("--target-set", default="auto")
    parser.add_argument("--target-member", type=int, default=1)
    parser.add_argument("--q", type=float, default=Q0)
    parser.add_argument("--flavors", default="qcd5")
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--x-start", type=int, default=36)
    parser.add_argument("--x-stop", type=int, default=-20)
    parser.add_argument("--metric", default="dist0")
    parser.add_argument("--coeff-threshold", type=float, default=1e-6)
    parser.add_argument("--plot-directory", default="pod_basis_completeness")
    parser.add_argument("--postfix", default="")
    return parser.parse_args()


def main() -> None:
    try:
        lhapdf.setVerbosity(0)
    except AttributeError:
        pass

    args = parse_args()
    target_set = first_installed_pdf(TARGET_CANDIDATES) if args.target_set == "auto" else args.target_set
    target_size = pdfset_size(target_set)
    if args.target_member >= target_size:
        raise ValueError(f"{target_set} has {target_size} members, cannot use {args.target_member}")

    x_grid = np.asarray(LHAPDF_XGRID[args.x_start : args.x_stop], dtype=float)
    flavors = parse_flavors(args.flavors, args.basis_set, target_set)
    variations = tuple(range(1, args.n_basis + 1))

    basis = NativePODBasis.load(args.basis_set, variations=variations, flavors=flavors)
    target_pdf = lhapdf.mkPDF(target_set, args.target_member)

    central_grid = basis.reference_grid(x_grid, args.q)
    shift_grid = basis.native_shift_grid(x_grid, args.q)
    target_grid = basis.xfx_grid(target_pdf, x_grid, args.q)

    x_matrix = shift_grid.reshape(len(variations), -1).T
    y_shift = (target_grid - central_grid).reshape(-1)
    weights = metric_weights(args.metric, x_grid, len(flavors))
    coeffs, residual, _gram, condition_number = decompose(x_matrix, y_shift, weights)
    reco_grid = central_grid + (x_matrix @ coeffs).reshape(len(flavors), len(x_grid))

    residual_norm = float(np.sqrt(residual @ (weights * residual)))
    target_norm = float(np.sqrt(y_shift @ (weights * y_shift)))
    rel_residual = residual_norm / target_norm if target_norm > 0 else float("nan")

    postfix = f"_{args.postfix}" if args.postfix else ""
    out_dir = os.path.join(user.plot_directory, args.plot_directory)
    os.makedirs(out_dir, exist_ok=True)
    helpers.copyIndexPHP(out_dir)

    target_tag = f"{target_set}_m{args.target_member}"
    flavor_tag = args.flavors.replace(",", "_").replace("-", "m")
    q_tag = f"Q{int(round(args.q * 1000)):06d}"
    base = f"{args.basis_set}_to_{target_tag}_{flavor_tag}_{args.metric}_{q_tag}{postfix}"
    nonzero = np.flatnonzero(np.abs(coeffs) > args.coeff_threshold)
    top = np.argsort(np.abs(coeffs))[-3:][::-1]
    top_label = ", ".join(f"c{idx + 1}={coeffs[idx]:+.2g}" for idx in top)
    info_lines = [
        f"{target_set} member {args.target_member}",
        f"{args.n_basis} POD modes",
        f"residual/shift={rel_residual:.2e}",
        f"|c|>{args.coeff_threshold:g}: {len(nonzero)}/{len(coeffs)}",
        f"rms(c)={np.sqrt(np.mean(coeffs**2)):.2g}, max|c|={np.max(np.abs(coeffs)):.2g}",
        f"largest: {top_label}",
    ]

    outputs = []
    outputs.extend(
        draw_panel_grid(
            out_dir,
            base + "_xf",
            x_grid,
            target_grid,
            reco_grid,
            flavors,
            info_lines,
        )
    )
    outputs.extend(
        draw_ratio_grid(
            out_dir,
            base + "_ratio",
            x_grid,
            target_grid,
            reco_grid,
            flavors,
            info_lines,
        )
    )

    print(f"[info] target          : {target_set} member {args.target_member}")
    print(f"[info] basis           : {args.basis_set}, modes 1..{args.n_basis}")
    print(f"[info] flavors         : {list(flavors)}")
    print(f"[info] metric          : {args.metric}")
    print(f"[info] cond(Gram)      : {condition_number:.6e}")
    print(f"[info] residual/shift  : {rel_residual:.6e}")
    print("[info] wrote:")
    for output in outputs:
        print(f"  {output}")

    try:
        syncer.sync()
    except Exception as err:
        print(f"[warning] sync failed: {err}")
        if hasattr(syncer, "file_sync_storage"):
            syncer.file_sync_storage = []


if __name__ == "__main__":
    main()
