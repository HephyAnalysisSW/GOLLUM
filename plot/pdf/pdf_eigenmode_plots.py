#!/usr/bin/env python3
"""
Evaluate and plot PDF shapes along all eigenvector directions in a single comparison figure.
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from typing import Dict, List

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer
import common.helpers as helpers
from pdf.PDFParametrization import PDFParametrization

logger = logging.getLogger(__name__)

def load_basis_json(filepath: str) -> Dict:
    """Load eigenbasis JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_d_vector(mode_idx: int, n_modes: int, step: float = 1.0) -> np.ndarray:
    """Build d-vector with single component set, rest zero."""
    d = np.zeros(n_modes)
    d[mode_idx] = step
    return d


def coeffs_from_d(d: np.ndarray, V_new: np.ndarray) -> np.ndarray:
    """Convert d-space to c-space via rotation: c = V_new @ d."""
    return V_new @ d


def evaluate_pdf(pdf_obj: PDFParametrization, x_vals: np.ndarray, 
                 Q_val: float, coeffs: np.ndarray, pid: int = 21) -> np.ndarray:
    """Evaluate PDF on x-grid at fixed Q with given coefficients."""
    id_arr = np.full(len(x_vals), pid, dtype=int)
    Q_arr = np.full(len(x_vals), Q_val, dtype=float)
    return np.array(pdf_obj.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs), dtype=float)

linestyles = ["-","--",":","-."]
plt.rcParams['font.size'] = 24
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def plot_unit_modes(x_vals: np.ndarray, pdf_central: np.ndarray, 
                           pdfs_unit: Dict[str, np.ndarray], 
                           basis_labels: List[str], mode_indices: List[int],
                           Q_val: float, output_dir: str) -> None:
    """Plot all d_n=1 variants overlaid on same figure."""
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8],hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    
    ax_top.semilogx(x_vals, pdf_central, 'k-', linewidth=1.5, label=r'$g^{(ref)}$', zorder=10)

    ax_top.text(0.95, 0.45,
        rf"$Q = {Q_val:.2f}\ \mathrm{{GeV}}$",
        transform=ax_top.transAxes,
        ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0),
    )
    
    for idx, mode_idx in enumerate(mode_indices):
        mode_label = basis_labels[mode_idx]
        color = f"C{mode_idx % 10}"
        linestyle = linestyles[mode_idx % 4]
        pdf_variant = pdfs_unit[mode_label]
        ax_top.semilogx(x_vals, pdf_variant, color=color, linestyle=linestyle, linewidth=1.5, label=f'EV({mode_idx+1})')
        
        mask = pdf_central > 1e-15
        ratio = np.ones_like(pdf_central)
        ratio[mask] = pdf_variant[mask] / pdf_central[mask]
        ax_bot.semilogx(x_vals, ratio, color=color, linestyle=linestyle, linewidth=1.5, label=f'EV({mode_idx+1})')
        ax_bot.set_ylim(-1.0,4.0)
    
    ax_top.set_ylabel(r'$g$')
    ax_top.legend(loc='best', ncol=2, fontsize=18)
    ax_top.grid(True, alpha=0.3, which='both')
    
    ax_bot.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.25)
    ax_bot.set_xlabel(r'$x$')
    ax_bot.set_ylabel(r'$g/g^{(ref)}$')
    ax_bot.legend(loc='best', ncol=2, fontsize=18)
    ax_bot.grid(True, alpha=0.3, which='both')
    
    filename = os.path.join(output_dir, f'eigenmodes_d1_Q{int(Q_val*1000)}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace(".png",".pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def plot_sigma_modes(x_vals: np.ndarray, pdf_central: np.ndarray,
                              pdfs_minus: Dict[str, np.ndarray],
                              pdfs_plus: Dict[str, np.ndarray],
                              sigma_d: np.ndarray,
                              basis_labels: List[str], mode_indices: List[int],
                              Q_val: float, output_dir: str) -> None:
    """Plot all modes with ±sigma bands."""
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8], hspace=0.0)
    
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
        
    ax_top.semilogx(x_vals, pdf_central, 'k-', linewidth=1.5, label=r'$g^{(ref)}$', zorder=10)
    
    ax_top.text(0.95, 0.55,
        rf"$Q = {Q_val:.2f}\ \mathrm{{GeV}}$",
        transform=ax_top.transAxes,
        ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none",pad=2.0),
    )
    
    for idx, mode_idx in enumerate(mode_indices):
        mode_label = basis_labels[mode_idx]
        color = f"C{mode_idx%10}"
        linestyle = linestyles[mode_idx % 4]

        ax_top.fill_between(x_vals, pdfs_minus[mode_label], pdfs_plus[mode_label], 
        color=color, alpha=0.08, linewidth=1.5, linestyle=linestyle)
        ax_top.semilogx(x_vals, pdfs_plus[mode_label], linestyle=linestyle, color=color, linewidth=1.5, label=f'EV({mode_idx+1})', zorder=3)
        ax_top.semilogx(x_vals, pdfs_minus[mode_label], linestyle=linestyle, color=color, linewidth=1.5, zorder=3)
        
        mask = pdf_central > 1e-15
        ratio_minus = np.ones_like(pdf_central)
        ratio_plus = np.ones_like(pdf_central)
        
        ratio_minus[mask] = pdfs_minus[mode_label][mask] / pdf_central[mask]
        ratio_plus[mask] = pdfs_plus[mode_label][mask] / pdf_central[mask]
        
        ax_bot.fill_between(x_vals, ratio_minus, ratio_plus, color=color, alpha=0.08)
        ax_bot.semilogx(x_vals, ratio_plus, color=color, linestyle=linestyle, linewidth=1.5, label=f'EV({mode_idx+1})', zorder=3)
        ax_bot.semilogx(x_vals, ratio_minus, color=color, linestyle=linestyle, linewidth=1.5, zorder=3)

        ax_bot.set_xscale("log")
        ax_bot.set_ylim(0.75,1.25)
    
    ax_top.set_ylabel(r'$g$')
    ax_top.legend(loc='best', ncol=2, fontsize=18)
    ax_top.grid(True, alpha=0.3, which='both')
    
    ax_bot.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.25)
    ax_bot.set_xlabel(r'$x$')
    ax_bot.set_ylabel(r'$g/g^{(ref)}$')
    ax_bot.legend(loc='best', ncol=2, fontsize=18)
    ax_bot.grid(True, alpha=0.3, which='both')
    
    filename = os.path.join(output_dir, f'eigenmodes_sigma_d_Q{int(Q_val*1000)}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace(".png",".pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PDF shapes along all eigenvector directions in one figure."
    )
    parser.add_argument("--basis-json", required=True, help="Path to eigen_basis JSON")
    parser.add_argument("--Q", type=float, default=1.65, help="Scale Q for PDF evaluation")
    parser.add_argument("--modes", default="all", help="Comma-separated mode indices or 'all'")
    parser.add_argument("--include-sigma", action="store_true", help="Also plot ±sigma variants")
    parser.add_argument("--pdf-basis", default="gluon_POD_nongluon_PDF4LHC21",
                       help="PDF basis identifier")
    
    args = parser.parse_args()
    
    logger.info(f"Loading basis from: {args.basis_json}")
    basis = load_basis_json(args.basis_json)
    
    V_new: np.ndarray = np.array(basis["V_new"], dtype=float)
    sigma_d: np.ndarray = np.array(basis["sigma_d"], dtype=float)
    basis_labels: List[str] = basis["basis_labels"]
    eigenvalues: List[float] = basis["eigenvalues"]
    n_modes: int = len(basis_labels)
    
    logger.info(f"Basis has {n_modes} modes: {', '.join(basis_labels)}")
    
    if args.modes.lower() == "all":
        mode_indices: List[int] = list(range(n_modes))
    else:
        mode_indices = [int(m.strip()) for m in args.modes.split(",")]
    
    logger.info(f"Evaluating modes: {[basis_labels[i] for i in mode_indices]}")
    
    pdf_n: List[int] = list(range(1, n_modes + 1))
    
    logger.info(f"Instantiating PDFParametrization(n={pdf_n}, typ='PODBasis', basis='{args.pdf_basis}')")
    pdf: PDFParametrization = PDFParametrization(
        n=pdf_n, typ="PODBasis", basis=args.pdf_basis, rescale_pod_amplitudes=True
    )
    
    x_min: float = 1e-3
    x_max: float = 0.8
    n_x: int = 200
    x_vals: np.ndarray = np.logspace(np.log10(x_min), np.log10(x_max), n_x)
    logger.info(f"Built x grid: {n_x} points in [{x_min}, {x_max}]")
    
    output_dir = os.path.join(user.plot_directory, "eigen_basis_plots", 
                              os.path.basename(args.basis_json).removesuffix(".json"))
    os.makedirs(output_dir, exist_ok=True)
    helpers.copyIndexPHP(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    d_central: np.ndarray = np.zeros(n_modes)
    c_central: np.ndarray = coeffs_from_d(d_central, V_new)
    pdf_central: np.ndarray = evaluate_pdf(pdf, x_vals, args.Q, c_central)
    logger.info(f"Evaluated central (d=0) at Q={args.Q}")
    
    pdfs_unit: Dict[str, np.ndarray] = {}
    pdfs_minus: Dict[str, np.ndarray] = {}
    pdfs_plus: Dict[str, np.ndarray] = {}
    
    for mode_idx in mode_indices:
        mode_label: str = basis_labels[mode_idx]

        # eigenvariations at same coefficient value
        # eigenvectors with smaller eigenvalue will have
        # much lower impact    
        d_unit: np.ndarray = build_d_vector(mode_idx, n_modes, step=1.0)
        c_unit: np.ndarray = coeffs_from_d(d_unit, V_new)
        pdfs_unit[mode_label] = evaluate_pdf(pdf, x_vals, args.Q, c_unit)

        # eigenvariations at coefficient value
        # at which they reach one sigma
        # shows different shapes
        sigma_val: float = sigma_d[mode_idx]
        
        d_plus: np.ndarray = build_d_vector(mode_idx, n_modes, step=sigma_val)
        c_plus: np.ndarray = coeffs_from_d(d_plus, V_new)
        pdfs_plus[mode_label] = evaluate_pdf(pdf, x_vals, args.Q, c_plus)
        
        d_minus: np.ndarray = build_d_vector(mode_idx, n_modes, step=-sigma_val)
        c_minus: np.ndarray = coeffs_from_d(d_minus, V_new)
        pdfs_minus[mode_label] = evaluate_pdf(pdf, x_vals, args.Q, c_minus)
    
    plot_sigma_modes(x_vals, pdf_central, pdfs_minus, pdfs_plus,
                                sigma_d, basis_labels, mode_indices, args.Q, output_dir)
    plot_unit_modes(x_vals, pdf_central, pdfs_unit, 
                            basis_labels, mode_indices, args.Q, output_dir)
    
    logger.info("Done")
    syncer.sync()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    main()