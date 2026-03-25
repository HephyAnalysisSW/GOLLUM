#!/usr/bin/env python3
"""
Project a target LHAPDF set onto the current POD basis and compare the PDFs.

This is a PDF-space helper only. It does not retrain or run the likelihood.
It finds the coefficient vector c that best approximates

    f_target(x, Q, pid) ~= f_ref(x, Q, pid) + sum_i c_i * basis_i(x, Q, pid)

using the same scaled basis convention as pdf.PODBasis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common.user as user
from pdf.PDFParametrization import PDFParametrization
from pdf.nnpdf.constants import LHAPDF_XGRID

import lhapdf


FLAVOUR_LABELS = {
    21: "g",
    5: "b",
    4: "c",
    3: "s",
    2: "u",
    1: "d",
    -1: "dbar",
    -2: "ubar",
    -3: "sbar",
    -4: "cbar",
    -5: "bbar",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project a target LHAPDF set onto a POD basis and plot target vs reconstruction."
    )
    p.add_argument(
        "--basis-set",
        default="gluon_POD_nongluon_PDF4LHC21",
        help="LHAPDF set used as the POD basis/reference.",
    )
    p.add_argument(
        "--basis-members",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Variation members that define the active POD coordinates.",
    )
    p.add_argument(
        "--target-set",
        default="PDF4LHC21_mc",
        help="Target LHAPDF set to approximate.",
    )
    p.add_argument(
        "--target-member",
        type=int,
        default=0,
        help="LHAPDF member of the target set. Use 0 for the central member.",
    )
    p.add_argument(
        "--Q",
        type=float,
        default=1.65,
        help="Scale Q at which to compare the PDFs.",
    )
    p.add_argument(
        "--x-min",
        type=float,
        default=0.001,
        help="Minimum x included in the projection and plots.",
    )
    p.add_argument(
        "--x-max",
        type=float,
        default=1.0,
        help="Maximum x included in the projection and plots.",
    )
    p.add_argument(
        "--pids",
        nargs="+",
        default=["all"],
        help='Active PDG ids to include, for example "--pids 21 1 -1". Default: all POD-active flavours.',
    )
    p.add_argument(
        "--weighting",
        choices=["uniform", "relative", "x"],
        default="uniform",
        help="Weighting used in the least-squares projection.",
    )
    p.add_argument(
        "--postfix",
        default="",
        help="Optional string appended to the output file names.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to common.user.output_directory/pdf_projection.",
    )
    return p.parse_args()


def selected_x_grid(x_min: float, x_max: float) -> np.ndarray:
    x = np.asarray(LHAPDF_XGRID, dtype=float)
    mask = (x >= x_min) & (x <= x_max)
    out = x[mask]
    if out.size == 0:
        raise ValueError(f"No x grid points left after applying x_min={x_min} and x_max={x_max}.")
    return out


def active_pids_from_args(raw_pids: list[str], pdf) -> list[int]:
    if raw_pids == ["all"]:
        return list(pdf.active_pids)
    return [int(pid) for pid in raw_pids]


def pdf_values(pdf_obj, x: np.ndarray, q: np.ndarray, pid: np.ndarray) -> np.ndarray:
    values = pdf_obj.xfxQ(tuple(x), tuple(q))
    return np.array([entry.get(int(fl), 0.0) for entry, fl in zip(values, pid)], dtype=float)


def build_projection_grid(x_grid: np.ndarray, pids: list[int], q0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.tile(x_grid, len(pids))
    pid = np.repeat(np.asarray(pids, dtype=int), len(x_grid))
    q = np.full(x.shape, float(q0), dtype=float)
    return x, pid, q


def make_sigma_diag(weighting: str, target_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    if weighting == "uniform":
        return np.ones_like(target_vals)
    if weighting == "relative":
        return 1.0 / np.maximum(np.abs(target_vals), 1e-12)
    if weighting == "x":
        return x.copy()
    raise ValueError(f"Unknown weighting '{weighting}'.")


def solve_projection(pdf, target_pdf, x: np.ndarray, pid: np.ndarray, q: np.ndarray, weighting: str):
    ref = pdf_values(pdf.reference_pdf, x, q, pid)
    target = pdf_values(target_pdf, x, q, pid)

    cols = []
    for i_var, var_pdf in enumerate(pdf.var_pdfs):
        var_vals = pdf_values(var_pdf, x, q, pid)
        col = var_vals - ref
        if pdf.scale_c is not None:
            col = pdf.scale_c[i_var] * col
        cols.append(col)

    X = np.stack(cols, axis=1)
    Y = target - ref

    sigma_diag = make_sigma_diag(weighting, target, x)
    sqrt_w = np.sqrt(sigma_diag)
    Xw = X * sqrt_w[:, np.newaxis]
    Yw = Y * sqrt_w

    coeffs, residuals, rank, singular_vals = np.linalg.lstsq(Xw, Yw, rcond=None)
    reco = ref + X @ coeffs

    diff = reco - target
    abs_rms = math.sqrt(float(np.mean(diff ** 2)))
    rel_rms = math.sqrt(float(np.mean((diff / np.maximum(np.abs(target), 1e-12)) ** 2)))
    chi2_like = float(np.sum(sigma_diag * diff * diff))

    return {
        "coeffs": coeffs,
        "reference": ref,
        "target": target,
        "reco": reco,
        "design_matrix": X,
        "residuals_raw": residuals.tolist(),
        "rank": int(rank),
        "singular_values": singular_vals.tolist(),
        "abs_rms": abs_rms,
        "rel_rms": rel_rms,
        "chi2_like": chi2_like,
    }


def reshape_by_pid(values: np.ndarray, x_grid: np.ndarray, pids: list[int]) -> dict[int, np.ndarray]:
    n_x = len(x_grid)
    return {
        pid: values[i * n_x : (i + 1) * n_x]
        for i, pid in enumerate(pids)
    }


def plot_projection(
    *,
    x_grid: np.ndarray,
    pids: list[int],
    target_by_pid: dict[int, np.ndarray],
    reco_by_pid: dict[int, np.ndarray],
    ref_by_pid: dict[int, np.ndarray],
    q0: float,
    target_label: str,
    basis_label: str,
    out_path: str,
):
    n_panels = len(pids)
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        2 * n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 3.8 * 2 * n_rows),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n_rows, "hspace": 0.08},
    )
    axes = np.atleast_2d(axes)

    for i, pid in enumerate(pids):
        row_block = i // n_cols
        col = i % n_cols
        ax_up = axes[2 * row_block, col]
        ax_dn = axes[2 * row_block + 1, col]

        target_vals = target_by_pid[pid]
        reco_vals = reco_by_pid[pid]
        ref_vals = ref_by_pid[pid]
        ratio = reco_vals / np.maximum(target_vals, 1e-12)

        ax_up.plot(x_grid, target_vals, lw=2.2, color="#1f77b4", label=target_label)
        ax_up.plot(x_grid, reco_vals, lw=2.0, ls="--", color="#d62728", label=f"Projection on {basis_label}")
        ax_up.plot(x_grid, ref_vals, lw=1.3, ls=":", color="#666666", label="Reference")
        ax_up.set_xscale("log")
        ax_up.set_ylabel(r"$x f(x, Q)$")
        ax_up.grid(True, ls="--", alpha=0.35)
        ax_up.set_title(f"pid = {pid} ({FLAVOUR_LABELS.get(pid, str(pid))}), Q = {q0:g}")
        ax_up.legend(frameon=False, fontsize=9)

        ax_dn.plot(x_grid, ratio, lw=1.8, color="#d62728")
        ax_dn.axhline(1.0, color="#666666", lw=1.0, ls="--")
        ax_dn.set_xscale("log")
        ax_dn.set_xlabel("x")
        ax_dn.set_ylabel("Reco/Target")
        ax_dn.grid(True, ls="--", alpha=0.35)

        finite = ratio[np.isfinite(ratio)]
        if finite.size:
            band = max(0.05, min(0.5, 1.2 * np.max(np.abs(finite - 1.0))))
            ax_dn.set_ylim(1.0 - band, 1.0 + band)

    total_axes = axes.shape[0] * axes.shape[1]
    used_axes = 2 * n_rows * n_cols
    for idx in range(n_panels, n_rows * n_cols):
        row_block = idx // n_cols
        col = idx % n_cols
        axes[2 * row_block, col].set_visible(False)
        axes[2 * row_block + 1, col].set_visible(False)

    fig.suptitle(f"{target_label} vs POD projection", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    pdf = PDFParametrization(
        n=args.basis_members,
        typ="PODBasis",
        basis=args.basis_set,
        active_pids="all",
    )
    pids = active_pids_from_args(args.pids, pdf)
    x_grid = selected_x_grid(args.x_min, args.x_max)
    x, pid, q = build_projection_grid(x_grid, pids, args.Q)

    target_pdf = lhapdf.mkPDF(args.target_set, args.target_member)
    result = solve_projection(pdf, target_pdf, x, pid, q, args.weighting)

    target_by_pid = reshape_by_pid(result["target"], x_grid, pids)
    reco_by_pid = reshape_by_pid(result["reco"], x_grid, pids)
    ref_by_pid = reshape_by_pid(result["reference"], x_grid, pids)

    outdir = args.outdir or os.path.join(user.output_directory, "pdf_projection")
    os.makedirs(outdir, exist_ok=True)

    postfix = f"_{args.postfix}" if args.postfix else ""
    stem = (
        f"project_{args.target_set}_m{args.target_member}"
        f"_onto_{args.basis_set}_n{len(args.basis_members)}"
        f"_Q{str(args.Q).replace('.', 'p')}{postfix}"
    )

    plot_path = os.path.join(outdir, f"{stem}.png")
    json_path = os.path.join(outdir, f"{stem}.json")

    target_label = f"{args.target_set} member {args.target_member}"
    basis_label = f"{args.basis_set} ({len(args.basis_members)} modes)"

    plot_projection(
        x_grid=x_grid,
        pids=pids,
        target_by_pid=target_by_pid,
        reco_by_pid=reco_by_pid,
        ref_by_pid=ref_by_pid,
        q0=args.Q,
        target_label=target_label,
        basis_label=basis_label,
        out_path=plot_path,
    )

    coeffs = np.asarray(result["coeffs"], dtype=float)
    coeff_map = {f"c{i}": float(val) for i, val in enumerate(coeffs)}

    payload = {
        "basis_set": args.basis_set,
        "basis_members": list(args.basis_members),
        "target_set": args.target_set,
        "target_member": int(args.target_member),
        "Q": float(args.Q),
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "pids": list(map(int, pids)),
        "weighting": args.weighting,
        "coefficients": coeff_map,
        "abs_rms": float(result["abs_rms"]),
        "rel_rms": float(result["rel_rms"]),
        "chi2_like": float(result["chi2_like"]),
        "rank": int(result["rank"]),
        "singular_values": list(map(float, result["singular_values"])),
        "plot_path": plot_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("\nProjection result")
    print(f"  target     : {args.target_set} member {args.target_member}")
    print(f"  basis      : {args.basis_set}")
    print(f"  members    : {args.basis_members}")
    print(f"  Q          : {args.Q:g}")
    print(f"  x range    : [{args.x_min:g}, {args.x_max:g}]")
    print(f"  pids       : {pids}")
    print(f"  weighting  : {args.weighting}")
    print("")
    for name, val in coeff_map.items():
        print(f"  {name:>3s} = {val: .8e}")
    print("")
    print(f"  abs_rms    = {result['abs_rms']:.8e}")
    print(f"  rel_rms    = {result['rel_rms']:.8e}")
    print(f"  chi2_like  = {result['chi2_like']:.8e}")
    print(f"  plot       = {plot_path}")
    print(f"  json       = {json_path}")


if __name__ == "__main__":
    main()
