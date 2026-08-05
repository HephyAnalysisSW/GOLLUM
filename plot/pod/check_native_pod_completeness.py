#!/usr/bin/env python3
"""Check native POD-basis completeness by decomposing a target PDF.

This script uses the native LHAPDF wiggles

    X_i = member_i - member_0

with no max_amplitudes and no orthonormalization.  Completeness is assessed by
projecting a target PDF shift onto the span of those wiggles and reporting the
reconstruction residual.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import lhapdf
import numpy as np

from native_pod_basis_40k import NativePODBasis
from nnpdf.constants import LHAPDF_XGRID


BASIS_SET = "250503_pod_basis_40k"
Q0 = 1.65
QCD5_FLAVORS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21)
QCD4_FLAVORS = (-4, -3, -2, -1, 1, 2, 3, 4, 21)
PAPER_FLAVORS = (1, -1, 2, -2, 3, -3, 4, 21)
TARGET_CANDIDATES = (
    "NNPDF40_nnlo_as_01180",
    "CT18NNLO",
    "MSHT20nnlo_as118",
    "NNPDF31_nnlo_as_0118",
)
METRICS = ("dist0", "dist4_x", "trapz_x", "trapz_logx")


@dataclass(frozen=True)
class CompletenessSummary:
    metric: str
    n_rows: int
    n_basis: int
    rank: int
    condition_number: float
    residual_norm: float
    target_shift_norm: float
    relative_residual: float
    rms_residual_over_pdf: float
    max_abs_residual: float
    max_abs_target: float
    coeff_min: float
    coeff_max: float
    coeff_rms: float


def parse_flavors(value: str, basis_set: str, target_set: str) -> tuple[int, ...]:
    if value == "qcd5":
        return QCD5_FLAVORS
    if value in {"qcd4", "no_b"}:
        return QCD4_FLAVORS
    if value == "paper":
        return PAPER_FLAVORS
    if value == "common":
        basis = set(int(pid) for pid in lhapdf.mkPDF(basis_set, 0).flavors())
        target = set(int(pid) for pid in lhapdf.mkPDF(target_set, 0).flavors())
        return tuple(pid for pid in QCD5_FLAVORS if pid in basis and pid in target)
    return tuple(int(pid.strip()) for pid in value.split(",") if pid.strip())


def pdfset_size(pdf_set: str) -> int:
    pdf_info = lhapdf.getPDFSet(pdf_set)
    size = pdf_info.size
    return int(size() if callable(size) else size)


def first_installed_pdf(candidates: Sequence[str]) -> str:
    for pdf_set in candidates:
        try:
            lhapdf.getPDFSet(pdf_set)
            lhapdf.mkPDF(pdf_set, 0)
        except Exception:
            continue
        return pdf_set
    raise RuntimeError(f"None of these target PDFs are installed: {', '.join(candidates)}")


def trapezoid_weights(grid: Sequence[float]) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("trapezoid_weights expects a one-dimensional grid with >=2 points")

    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def metric_weights(metric: str, x_grid: np.ndarray, n_flavors: int) -> np.ndarray:
    if metric == "dist0":
        return np.ones(n_flavors * len(x_grid))
    if metric == "dist4_x":
        return np.tile(np.abs(x_grid), n_flavors)
    if metric == "trapz_x":
        return np.tile(trapezoid_weights(x_grid), n_flavors)
    if metric == "trapz_logx":
        return np.tile(trapezoid_weights(np.log(x_grid)), n_flavors)
    raise ValueError(f"Unknown metric {metric!r}")


def flatten_flavor_grid(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=float).reshape(-1)


def decompose(
    x_matrix: np.ndarray,
    y_shift: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weighted_x = weights[:, np.newaxis] * x_matrix
    gram = x_matrix.T @ weighted_x
    rhs = x_matrix.T @ (weights * y_shift)
    coeffs = np.linalg.solve(gram, rhs)
    residual = y_shift - x_matrix @ coeffs
    condition_number = float(np.linalg.cond(gram))
    return coeffs, residual, gram, condition_number


def summarize(
    metric: str,
    x_matrix: np.ndarray,
    y_shift: np.ndarray,
    target_grid: np.ndarray,
    residual: np.ndarray,
    coeffs: np.ndarray,
    weights: np.ndarray,
    gram: np.ndarray,
) -> CompletenessSummary:
    residual_norm_sq = float(residual @ (weights * residual))
    target_norm_sq = float(y_shift @ (weights * y_shift))
    pdf_norm = float(np.sqrt(np.mean(target_grid**2)))

    return CompletenessSummary(
        metric=metric,
        n_rows=x_matrix.shape[0],
        n_basis=x_matrix.shape[1],
        rank=int(np.linalg.matrix_rank(x_matrix)),
        condition_number=float(np.linalg.cond(gram)),
        residual_norm=float(np.sqrt(max(residual_norm_sq, 0.0))),
        target_shift_norm=float(np.sqrt(max(target_norm_sq, 0.0))),
        relative_residual=float(np.sqrt(residual_norm_sq / target_norm_sq))
        if target_norm_sq > 0
        else float("nan"),
        rms_residual_over_pdf=float(np.sqrt(np.mean(residual**2)) / pdf_norm)
        if pdf_norm > 0
        else float("nan"),
        max_abs_residual=float(np.max(np.abs(residual))),
        max_abs_target=float(np.max(np.abs(target_grid))),
        coeff_min=float(np.min(coeffs)),
        coeff_max=float(np.max(coeffs)),
        coeff_rms=float(np.sqrt(np.mean(coeffs**2))),
    )


def per_flavor_residuals(
    residual: np.ndarray,
    y_shift: np.ndarray,
    target_grid: np.ndarray,
    flavors: Sequence[int],
    x_grid: np.ndarray,
) -> list[tuple[int, float, float, float]]:
    residual_grid = residual.reshape(len(flavors), len(x_grid))
    shift_grid = y_shift.reshape(len(flavors), len(x_grid))
    target_grid = target_grid.reshape(len(flavors), len(x_grid))

    out = []
    for i_flavor, pid in enumerate(flavors):
        shift_norm = float(np.linalg.norm(shift_grid[i_flavor]))
        pdf_norm = float(np.linalg.norm(target_grid[i_flavor]))
        residual_norm = float(np.linalg.norm(residual_grid[i_flavor]))
        rel_shift = residual_norm / shift_norm if shift_norm > 1e-14 else float("nan")
        rel_pdf = residual_norm / pdf_norm if pdf_norm > 1e-14 else float("nan")
        out.append((pid, residual_norm, rel_shift, rel_pdf))
    return out


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.6e}"


def print_summary(summary: CompletenessSummary) -> None:
    print(f"\nmetric: {summary.metric}")
    print(f"  X shape                    : ({summary.n_rows}, {summary.n_basis})")
    print(f"  rank(X)                    : {summary.rank}")
    print(f"  cond(X.T Sigma X)          : {summary.condition_number:.6e}")
    print(f"  ||target shift||_Sigma     : {summary.target_shift_norm:.6e}")
    print(f"  ||residual||_Sigma         : {summary.residual_norm:.6e}")
    print(f"  residual / target shift    : {summary.relative_residual:.6e}")
    print(f"  RMS residual / RMS target  : {summary.rms_residual_over_pdf:.6e}")
    print(f"  max |residual| / max |PDF| : {summary.max_abs_residual / summary.max_abs_target:.6e}")
    print(
        "  coeff min/rms/max          : "
        f"{summary.coeff_min:.6e} / {summary.coeff_rms:.6e} / {summary.coeff_max:.6e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check native 250503_pod_basis_40k completeness by projection."
    )
    parser.add_argument("--basis-set", default=BASIS_SET)
    parser.add_argument(
        "--target-set",
        default="auto",
        help="Target PDF set. 'auto' picks the first installed known target.",
    )
    parser.add_argument("--target-member", type=int, default=1)
    parser.add_argument("--q", type=float, default=Q0)
    parser.add_argument(
        "--flavors",
        default="qcd5",
        help="'qcd5', 'qcd4'/'no_b', 'paper', 'common', or comma-separated PIDs.",
    )
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--x-start", type=int, default=36)
    parser.add_argument("--x-stop", type=int, default=-20)
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x cut applied after the x-grid slice.",
    )
    parser.add_argument(
        "--metrics",
        default="dist0",
        help=f"Comma-separated metrics from: {','.join(METRICS)}",
    )
    parser.add_argument(
        "--save-npz",
        default=None,
        help="Optional path to save X, target, coefficients, and residuals.",
    )
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
    if args.x_min is not None:
        x_grid = x_grid[x_grid >= args.x_min]
    if len(x_grid) == 0:
        raise ValueError("Selected x grid is empty")

    flavors = parse_flavors(args.flavors, args.basis_set, target_set)
    variations = tuple(range(1, args.n_basis + 1))
    basis = NativePODBasis.load(args.basis_set, variations=variations, flavors=flavors)
    target_pdf = lhapdf.mkPDF(target_set, args.target_member)

    central_grid = basis.reference_grid(x_grid, args.q)
    shift_grid = basis.native_shift_grid(x_grid, args.q)
    target_grid = basis.xfx_grid(target_pdf, x_grid, args.q)

    central = flatten_flavor_grid(central_grid)
    target = flatten_flavor_grid(target_grid)
    y_shift = target - central
    x_matrix = shift_grid.reshape(len(variations), -1).T

    print(f"basis_set      : {args.basis_set}")
    print(f"target_set     : {target_set}")
    print(f"target_member  : {args.target_member}")
    print(f"Q              : {args.q}")
    print(f"flavors        : {list(flavors)}")
    print(f"x_grid slice   : LHAPDF_XGRID[{args.x_start}:{args.x_stop}]")
    if args.x_min is not None:
        print(f"x_min cut      : {args.x_min}")
    print(f"x_grid range   : {x_grid[0]:.8g} .. {x_grid[-1]:.8g} ({len(x_grid)} points)")
    print(f"basis members  : {variations[0]} .. {variations[-1]} ({len(variations)} columns)")
    print("decomposition  : solve((X.T Sigma X) c = X.T Sigma y)")

    save_payload = {
        "x_grid": x_grid,
        "flavors": np.asarray(flavors, dtype=int),
        "variations": np.asarray(variations, dtype=int),
        "central": central,
        "target": target,
        "X": x_matrix,
    }

    for metric in [item.strip() for item in args.metrics.split(",") if item.strip()]:
        weights = metric_weights(metric, x_grid, len(flavors))
        coeffs, residual, gram, _condition_number = decompose(x_matrix, y_shift, weights)
        summary = summarize(metric, x_matrix, y_shift, target, residual, coeffs, weights, gram)
        print_summary(summary)

        print("  per-flavor ||residual|| / target-shift / PDF:")
        for pid, abs_residual, rel_shift, rel_pdf in per_flavor_residuals(
            residual, y_shift, target, flavors, x_grid
        ):
            print(
                f"    {pid:>3}: {abs_residual:.6e} / "
                f"{format_float(rel_shift)} / {format_float(rel_pdf)}"
            )

        save_payload[f"weights_{metric}"] = weights
        save_payload[f"coeffs_{metric}"] = coeffs
        save_payload[f"residual_{metric}"] = residual
        save_payload[f"gram_{metric}"] = gram

    if args.save_npz:
        np.savez(args.save_npz, **save_payload)
        print(f"\nsaved decomposition: {args.save_npz}")


if __name__ == "__main__":
    main()
