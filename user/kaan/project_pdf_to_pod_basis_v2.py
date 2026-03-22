#!/usr/bin/env python3
"""
Project a target LHAPDF set onto a POD basis using only the gluon PDF.

What this script does:
1. Build the current POD basis from a basis set and a list of members.
2. Fit the coefficients c0..cN by matching only the gluon PDF (pid = 21).
3. Plot the target and reconstructed PDFs for any flavours you want to inspect.

Important:
- The minimization uses only the gluon.
- The --pids argument affects plotting only.
- This is a PDF-space diagnostic tool. It does not run the likelihood or retrain anything.
"""

from __future__ import annotations

import argparse
import json
import math
import os

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
        description="Project a target PDF onto the POD basis using gluon-only minimization."
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
        help="Variation members that define the POD coordinates.",
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
        help="Target LHAPDF member. Use 0 for the central member.",
    )
    p.add_argument(
        "--Q",
        type=float,
        default=70.0,
        help="Scale Q for the comparison.",
    )
    p.add_argument(
        "--x-min",
        type=float,
        default=3e-3,
        help="Minimum x shown and used in the fit.",
    )
    p.add_argument(
        "--x-max",
        type=float,
        default=0.6,
        help="Maximum x shown and used in the fit.",
    )
    p.add_argument(
        "--pids",
        nargs="+",
        default=["21", "1", "-1", "2", "-2", "3", "-3", "4", "-4", "5", "-5"],
        help="Flavours to plot. This does not affect the minimization.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to common.user.output_directory/pdf_projection.",
    )
    p.add_argument(
        "--rotate",
        default=None,
        help="Optional Fisher rotation JSON. If given, also store the rotated vector d = D·c.",
    )
    return p.parse_args()


def select_x_grid(x_min: float, x_max: float) -> np.ndarray:
    """Pick the subset of the standard LHAPDF x-grid used in the study."""
    x = np.asarray(LHAPDF_XGRID, dtype=float)
    mask = (x >= x_min) & (x <= x_max)
    out = x[mask]
    if out.size == 0:
        raise ValueError(f"No x points survive the range [{x_min}, {x_max}].")
    return out


def parse_plot_pids(raw_pids: list[str]) -> list[int]:
    """Convert the plotting PID list from strings to ints."""
    return [int(pid) for pid in raw_pids]


def evaluate_pdf(pdf_obj, x: np.ndarray, q: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """
    Evaluate one LHAPDF object on many (x, Q, pid) points.

    LHAPDF returns one dictionary per point, so we extract the requested flavour
    from each dictionary and assemble a plain numpy array.
    """
    values = pdf_obj.xfxQ(tuple(x), tuple(q))
    return np.array([entry.get(int(fl), 0.0) for entry, fl in zip(values, pid)], dtype=float)


def build_grid(x_grid: np.ndarray, pids: list[int], q0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build flat arrays of x, pid, Q for vectorized PDF evaluation."""
    x = np.tile(x_grid, len(pids))
    pid = np.repeat(np.asarray(pids, dtype=int), len(x_grid))
    q = np.full(x.shape, float(q0), dtype=float)
    return x, pid, q


def fit_gluon_coefficients(pdf, target_pdf, x_grid: np.ndarray, q0: float) -> dict:
    """
    Fit the POD coefficients using only the gluon PDF.

    The target is:
        target_gluon(x) - reference_gluon(x)

    The design matrix columns are the POD basis shifts for each active coefficient.
    """
    fit_pids = [21]
    x, pid, q = build_grid(x_grid, fit_pids, q0)

    reference = evaluate_pdf(pdf.reference_pdf, x, q, pid)
    target = evaluate_pdf(target_pdf, x, q, pid)

    basis_columns = []
    for i_var, var_pdf in enumerate(pdf.var_pdfs):
        var_vals = evaluate_pdf(var_pdf, x, q, pid)
        shift = var_vals - reference
        if pdf.scale_c is not None:
            shift = pdf.scale_c[i_var] * shift
        basis_columns.append(shift)

    design = np.stack(basis_columns, axis=1)
    target_shift = target - reference

    coeffs, residuals, rank, singular_values = np.linalg.lstsq(design, target_shift, rcond=None)
    reco = reference + design @ coeffs

    diff = reco - target
    abs_rms = math.sqrt(float(np.mean(diff ** 2)))
    rel_rms = math.sqrt(float(np.mean((diff / np.maximum(np.abs(target), 1e-12)) ** 2)))

    return {
        "coeffs": coeffs,
        "reference": reference,
        "target": target,
        "reco": reco,
        "rank": int(rank),
        "residuals_raw": residuals.tolist(),
        "singular_values": singular_values.tolist(),
        "abs_rms": abs_rms,
        "rel_rms": rel_rms,
    }


def rotate_coefficients(coeffs: np.ndarray, rotate_json: str) -> dict:
    """
    Convert the base coefficient vector c into rotated coordinates d = D · c.

    The rotation JSON is produced by orth/orthogonalize_Fisher.py and contains:
    - poi_order: order of the base coefficients expected by D
    - D: rotation matrix
    - basis_labels: names of the rotated coordinates
    """
    with open(rotate_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    poi_order = list(payload.get("poi_order", []) or [])
    d_names = list(payload.get("basis_labels", []) or [])
    D = np.asarray(payload.get("D", None), dtype=float)

    if not poi_order:
        raise ValueError(f"Rotation file '{rotate_json}' does not contain 'poi_order'.")
    if D.size == 0:
        raise ValueError(f"Rotation file '{rotate_json}' does not contain a valid 'D' matrix.")
    if D.shape[1] != len(poi_order):
        raise ValueError(
            f"Rotation file '{rotate_json}' has inconsistent dimensions: "
            f"D.shape={D.shape}, len(poi_order)={len(poi_order)}."
        )

    base_map = {f"c{i}": float(val) for i, val in enumerate(coeffs)}
    c_vec = np.array([base_map.get(name, 0.0) for name in poi_order], dtype=float)
    d_vec = D @ c_vec

    if not d_names:
        d_names = [f"d{i}" for i in range(len(d_vec))]
    elif len(d_names) != len(d_vec):
        raise ValueError(
            f"Rotation file '{rotate_json}' has inconsistent dimensions: "
            f"len(basis_labels)={len(d_names)}, len(d_vec)={len(d_vec)}."
        )

    return {
        "rotation_file": rotate_json,
        "poi_order": poi_order,
        "rotated_names": d_names,
        "rotated_map": {name: float(val) for name, val in zip(d_names, d_vec)},
    }


def evaluate_many_flavours(pdf, target_pdf, coeffs: np.ndarray, x_grid: np.ndarray, pids: list[int], q0: float) -> dict:
    """
    Evaluate target, reconstruction, and reference for the flavours we want to plot.

    This is separate from the minimization. The fit is always gluon-only.
    """
    x, pid, q = build_grid(x_grid, pids, q0)

    target = evaluate_pdf(target_pdf, x, q, pid)
    reference = evaluate_pdf(pdf.reference_pdf, x, q, pid)
    reco = pdf.evaluate(x=x, id=pid, Q=q, coeffs=coeffs)

    n_x = len(x_grid)
    return {
        "target": {pid_: target[i * n_x : (i + 1) * n_x] for i, pid_ in enumerate(pids)},
        "reference": {pid_: reference[i * n_x : (i + 1) * n_x] for i, pid_ in enumerate(pids)},
        "reco": {pid_: reco[i * n_x : (i + 1) * n_x] for i, pid_ in enumerate(pids)},
    }


def make_plot(
    x_grid: np.ndarray,
    pids: list[int],
    pdf_curves: dict,
    q0: float,
    target_label: str,
    basis_label: str,
    out_path: str,
):
    """Plot the target, reference, and reconstruction for each requested flavour."""
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

        target_vals = pdf_curves["target"][pid]
        reco_vals = pdf_curves["reco"][pid]
        ref_vals = pdf_curves["reference"][pid]
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

    # Build the POD basis object used everywhere else in the repo.
    pdf = PDFParametrization(
        n=args.basis_members,
        typ="PODBasis",
        basis=args.basis_set,
        active_pids="all",
    )

    # The target is one named LHAPDF set/member.
    target_pdf = lhapdf.mkPDF(args.target_set, args.target_member)

    # Use the standard x-grid, restricted to the requested x-range.
    x_grid = select_x_grid(args.x_min, args.x_max)

    # Fit coefficients with gluon only.
    fit_result = fit_gluon_coefficients(pdf, target_pdf, x_grid, args.Q)
    coeffs = np.asarray(fit_result["coeffs"], dtype=float)

    # Plot any flavours you want, using the gluon-fitted coefficients.
    plot_pids = parse_plot_pids(args.pids)
    pdf_curves = evaluate_many_flavours(pdf, target_pdf, coeffs, x_grid, plot_pids, args.Q)

    outdir = args.outdir or os.path.join(user.output_directory, "pdf_projection")
    os.makedirs(outdir, exist_ok=True)

    stem = (
        f"project_gluon_only_{args.target_set}_m{args.target_member}"
        f"_onto_{args.basis_set}_n{len(args.basis_members)}"
        f"_Q{str(args.Q).replace('.', 'p')}"
    )
    plot_path = os.path.join(outdir, f"{stem}.png")
    json_path = os.path.join(outdir, f"{stem}.json")

    target_label = f"{args.target_set} member {args.target_member}"
    basis_label = f"{args.basis_set} ({len(args.basis_members)} modes)"

    make_plot(
        x_grid=x_grid,
        pids=plot_pids,
        pdf_curves=pdf_curves,
        q0=args.Q,
        target_label=target_label,
        basis_label=basis_label,
        out_path=plot_path,
    )

    coeff_map = {f"c{i}": float(val) for i, val in enumerate(coeffs)}
    rotated_info = None
    if args.rotate:
        rotated_info = rotate_coefficients(coeffs, args.rotate)

    payload = {
        "basis_set": args.basis_set,
        "basis_members": list(args.basis_members),
        "target_set": args.target_set,
        "target_member": int(args.target_member),
        "Q": float(args.Q),
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "minimization_pid": 21,
        "plot_pids": plot_pids,
        "coefficients": coeff_map,
        "rotated_coefficients": None if rotated_info is None else rotated_info["rotated_map"],
        "rotation_file": None if rotated_info is None else rotated_info["rotation_file"],
        "rotation_poi_order": None if rotated_info is None else rotated_info["poi_order"],
        "rotation_basis_labels": None if rotated_info is None else rotated_info["rotated_names"],
        "gluon_abs_rms": float(fit_result["abs_rms"]),
        "gluon_rel_rms": float(fit_result["rel_rms"]),
        "rank": int(fit_result["rank"]),
        "singular_values": list(map(float, fit_result["singular_values"])),
        "plot_path": plot_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("\nProjection result")
    print(f"  target            : {args.target_set} member {args.target_member}")
    print(f"  basis             : {args.basis_set}")
    print(f"  basis members     : {args.basis_members}")
    print(f"  Q                 : {args.Q:g}")
    print(f"  x range           : [{args.x_min:g}, {args.x_max:g}]")
    print(f"  minimization pid  : 21 (gluon only)")
    print(f"  plotted pids      : {plot_pids}")
    print("")
    for name, val in coeff_map.items():
        print(f"  {name:>3s} = {val: .8e}")
    if rotated_info is not None:
        print("")
        print(f"  rotation file      : {args.rotate}")
        for name, val in rotated_info["rotated_map"].items():
            print(f"  {name:>3s} = {val: .8e}")
    print("")
    print(f"  gluon abs_rms     = {fit_result['abs_rms']:.8e}")
    print(f"  gluon rel_rms     = {fit_result['rel_rms']:.8e}")
    print(f"  plot              = {plot_path}")
    print(f"  json              = {json_path}")


if __name__ == "__main__":
    main()
