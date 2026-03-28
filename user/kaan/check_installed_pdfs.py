#!/usr/bin/env python3
"""
Print the installed LHAPDF sets and compare a few gluon PDFs to NNPDF3.1.

This script is intentionally simple:
- always uses the gluon only (pid = 21)
- always divides by NNPDF31_nnlo_as_0118 member 0
- plots only:
  - NNPDF31_nnlo_as_0118
  - gluon_POD_nongluon_PDF4LHC21
  - PDF4LHC21_mc
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lhapdf
import numpy as np

import common.user as user
from pdf.nnpdf.constants import LHAPDF_XGRID


REFERENCE_SET = "NNPDF31_nnlo_as_0118"
PDFS_TO_PLOT = [
    "NNPDF31_nnlo_as_0118",
    "gluon_POD_nongluon_PDF4LHC21",
    "PDF4LHC21_mc",
]
PDF_COLORS = {
    "NNPDF31_nnlo_as_0118": "red",
    "gluon_POD_nongluon_PDF4LHC21": "blue",
    "PDF4LHC21_mc": "gold",
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected gluon PDFs divided by NNPDF31_nnlo_as_0118."
    )
    parser.add_argument("--Q", type=float, default=70.0, help="Scale Q. Default: 70.")
    parser.add_argument("--x-min", type=float, default=3e-3, help="Minimum x. Default: 3e-3.")
    parser.add_argument("--x-max", type=float, default=0.6, help="Maximum x. Default: 0.6.")
    parser.add_argument(
        "--probe-x",
        type=float,
        default=0.1,
        help="x value used for the printed ratio summary. Default: 0.1.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the discovered PDF sets and exit.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to common.user.output_directory/pdf_checks.",
    )
    return parser.parse_args()


def discover_available_sets() -> list[str]:
    return sorted(lhapdf.availablePDFSets())


def selected_x_grid(x_min: float, x_max: float) -> np.ndarray:
    x_vals = np.asarray(LHAPDF_XGRID, dtype=float)
    mask = (x_vals >= x_min) & (x_vals <= x_max)
    return x_vals[mask]


def evaluate_gluon(pdf_obj, x_vals: np.ndarray, q_val: float) -> np.ndarray:
    return np.array([pdf_obj.xfxQ(21, float(x), float(q_val)) for x in x_vals], dtype=float)


def ensure_output_dir(outdir: str | None) -> Path:
    base = Path(outdir) if outdir is not None else Path(user.output_directory) / "pdf_checks"
    base.mkdir(parents=True, exist_ok=True)
    return base


def format_q_value(q_val: float) -> str:
    return f"{q_val:g}"


def load_required_pdfs() -> dict[str, object]:
    return {name: lhapdf.mkPDF(name, 0) for name in PDFS_TO_PLOT}


def plot_gluon_ratios(
    pdfs: dict[str, object],
    x_vals: np.ndarray,
    q_val: float,
    probe_x: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ref_vals = evaluate_gluon(pdfs[REFERENCE_SET], x_vals, q_val)
    probe_idx = int(np.argmin(np.abs(x_vals - probe_x)))

    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        constrained_layout=True,
    )

    print("\n[gluon ratio summary]")
    print(f"probe x={x_vals[probe_idx]:.5f}, Q={format_q_value(q_val)}")

    for set_name in PDFS_TO_PLOT:
        vals = evaluate_gluon(pdfs[set_name], x_vals, q_val)
        ratio = vals / ref_vals
        color = PDF_COLORS[set_name]
        ax_top.plot(x_vals, vals, lw=2.0, label=set_name, color=color)
        ax_ratio.plot(x_vals, ratio, lw=2.0, label=set_name, color=color)
        print(f"{set_name:<32} ratio={ratio[probe_idx]: .6f}")

    ax_top.set_xscale("log")
    ax_top.set_ylabel("g(x,Q)")
    ax_top.set_title(f"Gluon PDFs at Q={format_q_value(q_val)}")
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(frameon=False)

    ax_ratio.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("x")
    ax_ratio.set_ylabel(f"/ {REFERENCE_SET}")
    ax_ratio.set_ylim(0.80, 1.20)
    ax_ratio.grid(True, linestyle="--", alpha=0.35)

    fig.savefig(out_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    available_sets = discover_available_sets()
    print("[available PDF sets]")
    for set_name in available_sets:
        print(set_name)

    if args.list_only:
        return

    x_vals = selected_x_grid(args.x_min, args.x_max)
    pdfs = load_required_pdfs()
    outdir = ensure_output_dir(args.outdir)
    out_path = outdir / f"gluon_ratios_to_{REFERENCE_SET}_Q{format_q_value(args.Q)}.pdf"

    plot_gluon_ratios(
        pdfs=pdfs,
        x_vals=x_vals,
        q_val=args.Q,
        probe_x=args.probe_x,
        out_path=out_path,
    )

    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
