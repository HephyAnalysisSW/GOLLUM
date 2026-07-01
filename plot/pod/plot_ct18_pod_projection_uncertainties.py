#!/usr/bin/env python3
"""Compare CT18NNLO Hessian uncertainty bands with their POD projection."""

from __future__ import annotations

import argparse
import os
import sys
from array import array
from typing import Sequence

import lhapdf
import numpy as np
import ROOT

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user

from check_native_pod_completeness import BASIS_SET, Q0, metric_weights, pdfset_size
from native_pod_basis_40k import NativePODBasis
from nnpdf.constants import LHAPDF_XGRID
from plot_native_pod_completeness import (
    draw_info_block,
    draw_latex,
    parse_flavors,
    pid_to_name,
    pid_to_tex,
    style_log_x_axis,
    y_range,
)


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)


TARGET_SET = "CT18NNLO"


def make_graph(name: str, x_values: np.ndarray, y_values: np.ndarray, color: int, style: int) -> ROOT.TGraph:
    graph = ROOT.TGraph(len(x_values), array("d", x_values), array("d", y_values))
    graph.SetName(name)
    graph.SetLineColor(color)
    graph.SetLineStyle(style)
    graph.SetLineWidth(2)
    return graph


def make_band(
    name: str,
    x_values: np.ndarray,
    central: np.ndarray,
    error: np.ndarray,
    color: int,
    alpha: float,
    fill_style: int | None = None,
) -> ROOT.TGraphAsymmErrors:
    graph = ROOT.TGraphAsymmErrors(len(x_values))
    graph.SetName(name)
    if fill_style is None:
        graph.SetFillColorAlpha(color, alpha)
    else:
        graph.SetFillColor(color)
        graph.SetFillStyle(fill_style)
    graph.SetLineColor(color)
    graph.SetLineWidth(1)
    for i, (x, y, err) in enumerate(zip(x_values, central, error)):
        graph.SetPoint(i, float(x), float(y))
        graph.SetPointError(i, 0.0, 0.0, float(err), float(err))
    return graph


def draw_compact_info_block(lines: Sequence[str]) -> None:
    y = 0.86
    for line in lines:
        if line:
            draw_latex(0.08, y, line, 0.034)
            y -= 0.075


def project_member_grid(
    target_grid: np.ndarray,
    central_grid: np.ndarray,
    x_matrix: np.ndarray,
    weights: np.ndarray,
    gram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_shift = (target_grid - central_grid).reshape(-1)
    rhs = x_matrix.T @ (weights * y_shift)
    coeffs = np.linalg.solve(gram, rhs)
    projected_grid = central_grid + (x_matrix @ coeffs).reshape(central_grid.shape)
    return projected_grid, coeffs


def hessian_90_band(member_grids: np.ndarray) -> np.ndarray:
    if member_grids.shape[0] < 3 or (member_grids.shape[0] - 1) % 2:
        raise ValueError("Expected central plus an even number of Hessian eigenvector members")
    diffs = member_grids[1::2] - member_grids[2::2]
    return 0.5 * np.sqrt(np.sum(diffs * diffs, axis=0))


def coefficient_covariance_90(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if coeffs.shape[0] < 3 or (coeffs.shape[0] - 1) % 2:
        raise ValueError("Expected central plus an even number of coefficient eigenvector members")
    displacements = 0.5 * (coeffs[1::2] - coeffs[2::2])
    return displacements.T @ displacements, displacements


def draw_uncertainty_grid(
    out_dir: str,
    out_base: str,
    x_grid: np.ndarray,
    ct_central: np.ndarray,
    ct_error: np.ndarray,
    projected_central: np.ndarray,
    projected_error: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_ct18_pod_unc", "c_ct18_pod_unc", 1500, 420 * n_rows)
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

        ct = ct_central[i_flavor]
        ct_err = ct_error[i_flavor]
        pr = projected_central[i_flavor]
        pr_err = projected_error[i_flavor]
        ymin, ymax = y_range([ct - ct_err, ct + ct_err, pr - pr_err, pr + pr_err])

        frame = ROOT.TH1D(
            f"frame_unc_{pid_to_name(pid)}",
            f";x;x f_{{{pid_to_tex(pid)}}}(x,Q)",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(ymin)
        frame.SetMaximum(ymax)
        style_log_x_axis(frame)
        frame.Draw()

        band_ct = make_band(
            f"band_ct_{pid_to_name(pid)}",
            x_grid,
            ct,
            ct_err,
            ROOT.kGray + 1,
            0.35,
        )
        band_pr = make_band(
            f"band_proj_{pid_to_name(pid)}",
            x_grid,
            pr,
            pr_err,
            ROOT.kOrange + 8,
            1.0,
            fill_style=3354,
        )
        line_ct = make_graph(
            f"line_ct_{pid_to_name(pid)}", x_grid, ct, ROOT.kBlack, ROOT.kSolid
        )
        line_pr = make_graph(
            f"line_proj_{pid_to_name(pid)}", x_grid, pr, ROOT.kRed + 1, ROOT.kDashed
        )

        band_ct.Draw("3 SAME")
        band_pr.Draw("3 SAME")
        line_ct.Draw("L SAME")
        line_pr.Draw("L SAME")
        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)
        keepalive.extend([frame, band_ct, band_pr, line_ct, line_pr])

    info_pad_index = n_flavors + 1 if n_flavors < n_cols * n_rows else 1
    pad = canvas.cd(info_pad_index)
    pad.SetLeftMargin(0.08)
    pad.SetRightMargin(0.04)
    pad.SetTopMargin(0.08)
    pad.SetBottomMargin(0.08)
    draw_compact_info_block(info_lines)

    legend = ROOT.TLegend(0.08, 0.04, 0.88, 0.28)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.045)
    legend.AddEntry(keepalive[1], "CT18 90% Hessian", "f")
    legend.AddEntry(keepalive[2], "projected 90% Hessian", "f")
    legend.AddEntry(keepalive[3], "CT18 central", "l")
    legend.AddEntry(keepalive[4], "projected central", "l")
    legend.Draw()
    keepalive.append(legend)

    for ext in ("png", "pdf", "root"):
        path = os.path.join(out_dir, f"{out_base}.{ext}")
        canvas.SaveAs(path)
        outputs.append(path)
    return outputs


def draw_diagnostic_grid(
    out_dir: str,
    out_base: str,
    x_grid: np.ndarray,
    ct_central: np.ndarray,
    ct_error: np.ndarray,
    projected_central: np.ndarray,
    projected_error: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_ct18_pod_diag", "c_ct18_pod_diag", 1500, 420 * n_rows)
    canvas.Divide(n_cols, n_rows, 0.002, 0.002)
    keepalive = []
    first_bias = None
    first_ratio = None

    for i_flavor, pid in enumerate(flavors):
        pad = canvas.cd(i_flavor + 1)
        pad.SetLogx(True)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.13)
        pad.SetTicks(1, 1)

        err_scale = max(float(np.max(ct_error[i_flavor])), 1e-14)
        active = ct_error[i_flavor] > 1e-8 * err_scale
        bias_pull = np.full_like(ct_error[i_flavor], np.nan)
        error_ratio = np.full_like(ct_error[i_flavor], np.nan)
        bias_pull[active] = (
            projected_central[i_flavor][active] - ct_central[i_flavor][active]
        ) / ct_error[i_flavor][active]
        error_ratio[active] = projected_error[i_flavor][active] / ct_error[i_flavor][active]

        finite_values = np.concatenate(
            [
                bias_pull[np.isfinite(bias_pull)],
                error_ratio[np.isfinite(error_ratio)] - 1.0,
            ]
        )
        if finite_values.size:
            lim = min(2.0, max(0.05, 1.25 * float(np.max(np.abs(finite_values)))))
        else:
            lim = 1.0

        frame = ROOT.TH1D(
            f"frame_diag_{pid_to_name(pid)}",
            ";x;bias/#Delta_{CT},  #Delta_{proj}/#Delta_{CT}-1",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(-lim)
        frame.SetMaximum(lim)
        style_log_x_axis(frame)
        frame.GetYaxis().SetTitleSize(0.035)
        frame.GetYaxis().SetNdivisions(505)
        frame.Draw()

        zero = ROOT.TLine(float(x_grid[0]), 0.0, float(x_grid[-1]), 0.0)
        zero.SetLineColor(ROOT.kGray + 2)
        zero.SetLineStyle(ROOT.kDashed)
        zero.Draw("SAME")
        keepalive.extend([frame, zero])

        if np.any(active):
            line_bias = make_graph(
                f"bias_pull_{pid_to_name(pid)}",
                x_grid[active],
                bias_pull[active],
                ROOT.kRed + 1,
                ROOT.kSolid,
            )
            line_ratio = make_graph(
                f"err_ratio_{pid_to_name(pid)}",
                x_grid[active],
                error_ratio[active] - 1.0,
                ROOT.kBlue + 1,
                ROOT.kDashed,
            )
            line_bias.Draw("L SAME")
            line_ratio.Draw("L SAME")
            keepalive.extend([line_bias, line_ratio])
            if first_bias is None:
                first_bias = line_bias
                first_ratio = line_ratio
        else:
            draw_latex(0.25, 0.50, "inactive uncertainty", 0.045)

        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)

    if n_flavors < n_cols * n_rows:
        pad = canvas.cd(n_flavors + 1)
        pad.SetLeftMargin(0.08)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.08)
        draw_compact_info_block(info_lines)
        legend = ROOT.TLegend(0.08, 0.12, 0.88, 0.32)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.042)
        if first_bias is not None and first_ratio is not None:
            legend.AddEntry(first_bias, "(projected - CT18 central) / CT18 #Delta_{90}", "l")
            legend.AddEntry(first_ratio, "projected #Delta_{90} / CT18 #Delta_{90} - 1", "l")
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
    ct_central: np.ndarray,
    ct_error: np.ndarray,
    projected_central: np.ndarray,
    projected_error: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_ct18_pod_ratio", "c_ct18_pod_ratio", 1500, 420 * n_rows)
    canvas.Divide(n_cols, n_rows, 0.002, 0.002)
    keepalive = []
    first_central_ratio = None
    first_error_ratio = None

    for i_flavor, pid in enumerate(flavors):
        pad = canvas.cd(i_flavor + 1)
        pad.SetLogx(True)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.13)
        pad.SetTicks(1, 1)

        central_scale = max(float(np.max(np.abs(ct_central[i_flavor]))), 1e-14)
        error_scale = max(float(np.max(ct_error[i_flavor])), 1e-14)
        central_active = np.abs(ct_central[i_flavor]) > 1e-8 * central_scale
        error_active = ct_error[i_flavor] > 1e-8 * error_scale

        central_ratio = np.full_like(ct_central[i_flavor], np.nan)
        error_ratio = np.full_like(ct_error[i_flavor], np.nan)
        central_ratio[central_active] = (
            projected_central[i_flavor][central_active] / ct_central[i_flavor][central_active]
        )
        error_ratio[error_active] = projected_error[i_flavor][error_active] / ct_error[i_flavor][error_active]

        finite_values = np.concatenate(
            [
                central_ratio[np.isfinite(central_ratio)],
                error_ratio[np.isfinite(error_ratio)],
            ]
        )
        if finite_values.size:
            ymin = max(0.0, float(np.min(finite_values)) - 0.05)
            ymax = min(2.0, float(np.max(finite_values)) + 0.05)
            if ymin >= ymax:
                ymin, ymax = 0.95, 1.05
        else:
            ymin, ymax = 0.95, 1.05

        frame = ROOT.TH1D(
            f"frame_ratio_{pid_to_name(pid)}",
            ";x;projected / CT18",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(ymin)
        frame.SetMaximum(ymax)
        style_log_x_axis(frame)
        frame.GetYaxis().SetNdivisions(505)
        frame.Draw()

        unity = ROOT.TLine(float(x_grid[0]), 1.0, float(x_grid[-1]), 1.0)
        unity.SetLineColor(ROOT.kGray + 2)
        unity.SetLineStyle(ROOT.kDashed)
        unity.Draw("SAME")
        keepalive.extend([frame, unity])

        if np.any(central_active):
            line_central_ratio = make_graph(
                f"central_ratio_{pid_to_name(pid)}",
                x_grid[central_active],
                central_ratio[central_active],
                ROOT.kRed + 1,
                ROOT.kSolid,
            )
            line_central_ratio.Draw("L SAME")
            keepalive.append(line_central_ratio)
            if first_central_ratio is None:
                first_central_ratio = line_central_ratio

        if np.any(error_active):
            line_error_ratio = make_graph(
                f"error_ratio_{pid_to_name(pid)}",
                x_grid[error_active],
                error_ratio[error_active],
                ROOT.kBlue + 1,
                ROOT.kDashed,
            )
            line_error_ratio.Draw("L SAME")
            keepalive.append(line_error_ratio)
            if first_error_ratio is None:
                first_error_ratio = line_error_ratio

        if not np.any(central_active) and not np.any(error_active):
            draw_latex(0.25, 0.50, "inactive", 0.045)

        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)

    if n_flavors < n_cols * n_rows:
        pad = canvas.cd(n_flavors + 1)
        pad.SetLeftMargin(0.08)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.08)
        draw_compact_info_block(info_lines)
        legend = ROOT.TLegend(0.08, 0.12, 0.88, 0.32)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.042)
        if first_central_ratio is not None:
            legend.AddEntry(first_central_ratio, "projected central / CT18 central", "l")
        if first_error_ratio is not None:
            legend.AddEntry(first_error_ratio, "projected #Delta_{90} / CT18 #Delta_{90}", "l")
        legend.Draw()
        keepalive.append(legend)

    for ext in ("png", "pdf", "root"):
        path = os.path.join(out_dir, f"{out_base}.{ext}")
        canvas.SaveAs(path)
        outputs.append(path)
    return outputs


def draw_ratio_band_grid(
    out_dir: str,
    out_base: str,
    x_grid: np.ndarray,
    ct_central: np.ndarray,
    ct_error: np.ndarray,
    projected_central: np.ndarray,
    projected_error: np.ndarray,
    flavors: tuple[int, ...],
    info_lines: Sequence[str],
) -> list[str]:
    outputs = []
    n_flavors = len(flavors)
    n_cols = 3
    n_rows = int(np.ceil(n_flavors / n_cols))

    canvas = ROOT.TCanvas("c_ct18_pod_ratio_bands", "c_ct18_pod_ratio_bands", 1500, 420 * n_rows)
    canvas.Divide(n_cols, n_rows, 0.002, 0.002)
    keepalive = []
    first_ct_band = None
    first_projected_band = None
    first_ct_line = None
    first_projected_line = None

    for i_flavor, pid in enumerate(flavors):
        pad = canvas.cd(i_flavor + 1)
        pad.SetLogx(True)
        pad.SetLeftMargin(0.13)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.13)
        pad.SetTicks(1, 1)

        denom = ct_central[i_flavor]
        denom_scale = max(float(np.max(np.abs(denom))), 1e-14)
        active = np.abs(denom) > 1e-8 * denom_scale

        if np.any(active):
            x_active = x_grid[active]
            ct_ratio = np.ones_like(x_active)
            ct_ratio_error = ct_error[i_flavor][active] / np.abs(denom[active])
            projected_ratio = projected_central[i_flavor][active] / denom[active]
            projected_ratio_error = projected_error[i_flavor][active] / np.abs(denom[active])

            finite_values = np.concatenate(
                [
                    ct_ratio - ct_ratio_error,
                    ct_ratio + ct_ratio_error,
                    projected_ratio - projected_ratio_error,
                    projected_ratio + projected_ratio_error,
                ]
            )
            finite_values = finite_values[np.isfinite(finite_values)]
            ymin = max(-1.0, float(np.min(finite_values)) - 0.05) if finite_values.size else 0.8
            ymax = min(3.0, float(np.max(finite_values)) + 0.05) if finite_values.size else 1.2
            if ymin >= ymax:
                ymin, ymax = 0.8, 1.2
        else:
            x_active = x_grid
            ct_ratio = np.ones_like(x_grid)
            ct_ratio_error = np.zeros_like(x_grid)
            projected_ratio = np.ones_like(x_grid)
            projected_ratio_error = np.zeros_like(x_grid)
            ymin, ymax = 0.95, 1.05

        frame = ROOT.TH1D(
            f"frame_ratio_band_{pid_to_name(pid)}",
            ";x;PDF / CT18 central",
            100,
            float(x_grid[0]),
            float(x_grid[-1]),
        )
        frame.SetMinimum(ymin)
        frame.SetMaximum(ymax)
        style_log_x_axis(frame)
        frame.GetYaxis().SetNdivisions(505)
        frame.Draw()

        unity = ROOT.TLine(float(x_grid[0]), 1.0, float(x_grid[-1]), 1.0)
        unity.SetLineColor(ROOT.kGray + 2)
        unity.SetLineStyle(ROOT.kDashed)
        unity.Draw("SAME")
        keepalive.extend([frame, unity])

        if np.any(active):
            band_ct = make_band(
                f"ratio_band_ct_{pid_to_name(pid)}",
                x_active,
                ct_ratio,
                ct_ratio_error,
                ROOT.kGray + 1,
                0.35,
            )
            band_projected = make_band(
                f"ratio_band_projected_{pid_to_name(pid)}",
                x_active,
                projected_ratio,
                projected_ratio_error,
                ROOT.kOrange + 8,
                1.0,
                fill_style=3354,
            )
            line_ct = make_graph(
                f"ratio_line_ct_{pid_to_name(pid)}",
                x_active,
                ct_ratio,
                ROOT.kBlack,
                ROOT.kSolid,
            )
            line_projected = make_graph(
                f"ratio_line_projected_{pid_to_name(pid)}",
                x_active,
                projected_ratio,
                ROOT.kRed + 1,
                ROOT.kDashed,
            )
            band_ct.Draw("3 SAME")
            band_projected.Draw("3 SAME")
            line_ct.Draw("L SAME")
            line_projected.Draw("L SAME")
            keepalive.extend([band_ct, band_projected, line_ct, line_projected])
            if first_ct_band is None:
                first_ct_band = band_ct
                first_projected_band = band_projected
                first_ct_line = line_ct
                first_projected_line = line_projected
        else:
            draw_latex(0.25, 0.50, "inactive", 0.045)

        draw_latex(0.18, 0.82, pid_to_tex(pid), 0.060)

    if n_flavors < n_cols * n_rows:
        pad = canvas.cd(n_flavors + 1)
        pad.SetLeftMargin(0.08)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.08)
        draw_compact_info_block(info_lines)
        legend = ROOT.TLegend(0.08, 0.04, 0.88, 0.28)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.045)
        if first_ct_band is not None:
            legend.AddEntry(first_ct_band, "CT18 90% / CT18 central", "f")
            legend.AddEntry(first_projected_band, "projected 90% / CT18 central", "f")
            legend.AddEntry(first_ct_line, "CT18 central", "l")
            legend.AddEntry(first_projected_line, "projected central / CT18 central", "l")
        legend.Draw()
        keepalive.append(legend)

    for ext in ("png", "pdf", "root"):
        path = os.path.join(out_dir, f"{out_base}.{ext}")
        canvas.SaveAs(path)
        outputs.append(path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare CT18NNLO Hessian bands with POD projection.")
    parser.add_argument("--basis-set", default=BASIS_SET)
    parser.add_argument("--target-set", default=TARGET_SET)
    parser.add_argument("--q", type=float, default=Q0)
    parser.add_argument("--flavors", default="qcd5")
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--x-start", type=int, default=36)
    parser.add_argument("--x-stop", type=int, default=-20)
    parser.add_argument("--metric", default="dist0")
    parser.add_argument("--coeff-threshold", type=float, default=1e-6)
    parser.add_argument("--plot-directory", default="pod_basis_uncertainties")
    parser.add_argument("--postfix", default="")
    return parser.parse_args()


def main() -> None:
    try:
        lhapdf.setVerbosity(0)
    except AttributeError:
        pass

    args = parse_args()
    n_members = pdfset_size(args.target_set)
    if n_members != 59:
        print(f"[warning] expected 59 CT18NNLO members, got {n_members}")
    if (n_members - 1) % 2:
        raise ValueError(f"{args.target_set} has no even Hessian member pairing")

    x_grid = np.asarray(LHAPDF_XGRID[args.x_start : args.x_stop], dtype=float)
    flavors = parse_flavors(args.flavors, args.basis_set, args.target_set)
    variations = tuple(range(1, args.n_basis + 1))

    basis = NativePODBasis.load(args.basis_set, variations=variations, flavors=flavors)
    central_grid = basis.reference_grid(x_grid, args.q)
    shift_grid = basis.native_shift_grid(x_grid, args.q)
    x_matrix = shift_grid.reshape(len(variations), -1).T
    weights = metric_weights(args.metric, x_grid, len(flavors))
    gram = x_matrix.T @ (weights[:, np.newaxis] * x_matrix)

    ct_grids = []
    projected_grids = []
    coeffs = []
    for member in range(n_members):
        pdf = lhapdf.mkPDF(args.target_set, member)
        ct_grid = basis.xfx_grid(pdf, x_grid, args.q)
        projected_grid, coeff = project_member_grid(ct_grid, central_grid, x_matrix, weights, gram)
        ct_grids.append(ct_grid)
        projected_grids.append(projected_grid)
        coeffs.append(coeff)

    ct_grids = np.asarray(ct_grids)
    projected_grids = np.asarray(projected_grids)
    coeffs = np.asarray(coeffs)

    ct_error_90 = hessian_90_band(ct_grids)
    projected_error_90 = hessian_90_band(projected_grids)
    cov90, displacements90 = coefficient_covariance_90(coeffs)

    central_shift = (ct_grids[0] - central_grid).reshape(-1)
    central_residual = (ct_grids[0] - projected_grids[0]).reshape(-1)
    central_shift_norm = float(np.sqrt(central_shift @ (weights * central_shift)))
    central_residual_norm = float(np.sqrt(central_residual @ (weights * central_residual)))
    central_rel_residual = (
        central_residual_norm / central_shift_norm if central_shift_norm > 0 else float("nan")
    )

    eigs = np.linalg.eigvalsh(cov90)
    cov_rank = int(np.sum(eigs > 1e-12 * max(float(eigs[-1]), 1.0)))
    coeff_sigma90 = np.sqrt(np.clip(np.diag(cov90), 0.0, None))
    active_coeffs = int(np.sum(coeff_sigma90 > args.coeff_threshold))

    postfix = f"_{args.postfix}" if args.postfix else ""
    flavor_tag = args.flavors.replace(",", "_").replace("-", "m")
    q_tag = f"Q{int(round(args.q * 1000)):06d}"
    out_dir = os.path.join(user.plot_directory, args.plot_directory)
    os.makedirs(out_dir, exist_ok=True)
    helpers.copyIndexPHP(out_dir)

    base = (
        f"{args.basis_set}_to_{args.target_set}_hessian90_"
        f"{flavor_tag}_{args.metric}_{q_tag}{postfix}"
    )
    info_lines = [
        f"{args.target_set}: 29 pairs, 90% CL",
        f"{args.n_basis} POD modes, {args.metric}",
        f"central res/shift {central_rel_residual:.2e}",
        f"rank(C_{{c,90}}) {cov_rank}/{args.n_basis}",
        f"active #sigma_{{c,90}} {active_coeffs}/{args.n_basis}",
        (
            f"rms/max #sigma_{{c,90}} "
            f"{np.sqrt(np.mean(coeff_sigma90**2)):.2g}/{np.max(coeff_sigma90):.2g}"
        ),
    ]

    outputs = []
    outputs.extend(
        draw_uncertainty_grid(
            out_dir,
            base + "_bands",
            x_grid,
            ct_grids[0],
            ct_error_90,
            projected_grids[0],
            projected_error_90,
            flavors,
            info_lines,
        )
    )
    outputs.extend(
        draw_diagnostic_grid(
            out_dir,
            base + "_diagnostics",
            x_grid,
            ct_grids[0],
            ct_error_90,
            projected_grids[0],
            projected_error_90,
            flavors,
            info_lines,
        )
    )
    outputs.extend(
        draw_ratio_grid(
            out_dir,
            base + "_ratios",
            x_grid,
            ct_grids[0],
            ct_error_90,
            projected_grids[0],
            projected_error_90,
            flavors,
            info_lines,
        )
    )
    outputs.extend(
        draw_ratio_band_grid(
            out_dir,
            base + "_ratio_bands",
            x_grid,
            ct_grids[0],
            ct_error_90,
            projected_grids[0],
            projected_error_90,
            flavors,
            info_lines,
        )
    )

    print(f"[info] target                 : {args.target_set}, members 0..{n_members - 1}")
    print(f"[info] basis                  : {args.basis_set}, modes 1..{args.n_basis}")
    print(f"[info] Hessian pairs          : {(n_members - 1) // 2}, native 90% CL")
    print(f"[info] cond(Gram)             : {np.linalg.cond(gram):.6e}")
    print(f"[info] central residual/shift : {central_rel_residual:.6e}")
    print(f"[info] rank(C_c,90)           : {cov_rank}/{args.n_basis}")
    print(f"[info] active coeff sigmas    : {active_coeffs}/{args.n_basis}")
    print(f"[info] displacement shape     : {displacements90.shape}")
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
