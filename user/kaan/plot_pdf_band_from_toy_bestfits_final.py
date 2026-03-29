#!/usr/bin/env python3
"""
Plot PDF curves from toy best fits.

The plot contains:
- the mean toy-fit curve with its 68% band
- the PDF from the config
- NNPDF31_nnlo_as_0118 member 0
- PDF4LHC21_mc member 0

The ratio panel always divides by PDF4LHC21_mc member 0.
"""

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import lhapdf

import common.yaml_loader as yaml_loader
from pdf.PDFParametrization import PDFParametrization

NNPDF31_SET = "NNPDF31_nnlo_as_0118"
NNPDF31_MEMBER = 0
PDF4LHC21_MC_SET = "PDF4LHC21_mc"
PDF4LHC21_MC_MEMBER = 0
X_MIN = 0.003
X_MAX = 0.6
N_X = 200


def load_fit_results_dir(fit_dir, pattern="*.npz"):
    """
    Load all toy-fit result files from a directory and merge them into one dataset.

    Also validates that shared metadata agrees across files and that each
    toy id appears only once.
    """
    paths = sorted(glob.glob(os.path.join(fit_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No files matched {os.path.join(fit_dir, pattern)}")

    poi_names_ref = None
    config_ref = None
    toys_npz_ref = None

    all_ids = []
    all_chat = []
    all_n2ll = []
    seen = set()

    for path in paths:
        z = np.load(path, allow_pickle=False)

        for key in ["config", "toys_npz", "toy_ids", "POI_names", "n2ll_min", "c_hat"]:
            if key not in z:
                raise RuntimeError(f"Missing key '{key}' in {path}")

        poi_names = np.asarray(z["POI_names"])
        toy_ids = np.asarray(z["toy_ids"], dtype=int).reshape(-1)
        chat = np.asarray(z["c_hat"], dtype=float)
        n2ll = np.asarray(z["n2ll_min"], dtype=float).reshape(-1)

        config = str(np.asarray(z["config"]).reshape(-1)[0])
        toys_npz = str(np.asarray(z["toys_npz"]).reshape(-1)[0])


        if poi_names_ref is None:
            poi_names_ref = poi_names
            config_ref = config
            toys_npz_ref = toys_npz
        else:
            if not np.array_equal(poi_names, poi_names_ref):
                raise RuntimeError(f"POI_names mismatch in {path}")
            if config != config_ref:
                raise RuntimeError(f"config mismatch in {path}")
            if toys_npz != toys_npz_ref:
                raise RuntimeError(f"toys_npz mismatch in {path}\n  ref: {toys_npz_ref}\n  this:{toys_npz}")

        for toy_id in toy_ids.tolist():
            if toy_id in seen:
                raise RuntimeError(f"Duplicate toy_id={toy_id} found in {path}")
            seen.add(toy_id)

        all_ids.append(toy_ids)
        all_chat.append(chat)
        all_n2ll.append(n2ll)

    toy_ids = np.concatenate(all_ids, axis=0)
    c_hat = np.concatenate(all_chat, axis=0)
    n2ll_min = np.concatenate(all_n2ll, axis=0)

    meta = {
        "config": config_ref,
        "toys_npz": toys_npz_ref,
        "n_files": len(paths),
        "n_toys": int(toy_ids.shape[0]),
        "n_poi": int(poi_names_ref.shape[0]),
    }
    return poi_names_ref, toy_ids, c_hat, n2ll_min, meta


def find_pdf_job_and_pois(cfg):
    """
    Find the BIT POI block in the config and return its parameter names and PDF config.

    The script uses this to recover the base POI ordering and the 'PDFParametrization'
    settings needed to evaluate the toy-fit curves.
    """
    for region in cfg.get("likelihood", {}).get("regions", []) or []:
        for cls in region.get("classes", []) or []:
            poi = cls.get("POI", {}) or {}
            if poi.get("type") != "bit":
                continue

            job_id = poi.get("job")
            for job in cfg.get("jobs", []) or []:
                if job.get("id") == job_id:
                    return poi.get("parameters", []), job.get("pdf", {}) or {}

    raise RuntimeError("Could not find a BIT POI job in the config.")


def build_rot_to_base(rotate_json, base_poi_names):
    """
    Build a function that converts fitted coefficients into the base POI basis.

    Without a rotation file, this just reorders coefficients by name. With a
    rotation file, it reconstructs the base coefficients using the stored
    rotation matrix and its pseudo-inverse (Moore-Penrose).
    """
    if rotate_json is None:
        def coeffs_to_base(fit_poi_names, fit_values):
            value_map = {name: float(val) for name, val in zip(fit_poi_names, fit_values)}
            return np.array([value_map[name] for name in base_poi_names], dtype=float)

        return coeffs_to_base

    with open(rotate_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    poi_order = list(payload.get("poi_order", []) or [])
    D_raw = np.asarray(payload.get("D"), dtype=float)
    if D_raw.ndim != 2:
        raise RuntimeError("Rotation JSON does not contain a valid 2D matrix 'D'.")

    json_name_to_col = {name: i for i, name in enumerate(poi_order)}

    D_cols = np.zeros((D_raw.shape[0], len(base_poi_names)), dtype=float)
    used = set()
    for i, cname in enumerate(base_poi_names):
        if cname in json_name_to_col:
            D_cols[:, i] = D_raw[:, json_name_to_col[cname]]
            used.add(cname)

    missing = [name for name in base_poi_names if name not in used]
    if missing:
        eye = np.zeros((len(missing), len(base_poi_names)), dtype=float)
        for i, name in enumerate(missing):
            eye[i, base_poi_names.index(name)] = 1.0
        D_full = np.vstack([D_cols, eye])
    else:
        D_full = D_cols

    basis_labels = payload.get("basis_labels", None)
    if basis_labels is not None and len(basis_labels) == D_raw.shape[0]:
        d_names = list(basis_labels) + missing
    else:
        d_names = [f"d{i}" for i in range(D_raw.shape[0])] + missing

    D_pinv = np.linalg.pinv(D_full, rcond=1e-12)

    def coeffs_to_base(fit_poi_names, fit_values):
        value_map = {name: float(val) for name, val in zip(fit_poi_names, fit_values)}
        d_vec = np.array([value_map[name] for name in d_names], dtype=float)
        return D_pinv @ d_vec

    return coeffs_to_base


def evaluate_external_pdf(pdf_set, pdf_member, x_vals, pid, Q):
    """
    Evaluate one LHAPDF member on the requested x grid.
    """
    target_pdf = lhapdf.mkPDF(pdf_set, int(pdf_member))
    return np.array(
        [target_pdf.xfxQ(int(pid), float(x), float(Q)) for x in x_vals],
        dtype=float,
    )


def pretty_pdf_label(pdf_set, pdf_member):
    """
    Build a shorter legend label for a PDF set and member number.
    """
    aliases = {
        "gluon_POD_nongluon_PDF4LHC21": "gluon_POD",
        "NNPDF31_nnlo_as_0118": "NNPDF3.1",
        "PDF4LHC21_mc": "PDF4LHC21_mc",
    }
    display_name = aliases.get(pdf_set, pdf_set)
    return f"{display_name} member {int(pdf_member)}"


def build_pdf_band(config, fit_dir, rotate_json=None, Q=1.65, pid=21):
    """
    Assemble all curves and ratio bands needed for the final two-panel plot.

    Loads the toy fits, maps the fitted coefficients back to the base PDF basis,
    evaluates the toy-sample PDFs, and computes the mean and 68% band.
    """
    fit_poi_names, _, chat, _, fit_meta = load_fit_results_dir(fit_dir)

    print("[fit metadata]\n", fit_meta)

    cfg = yaml_loader.load_yaml(config)
    base_poi_names, pdf_cfg = find_pdf_job_and_pois(cfg)

    pdf = PDFParametrization(
        n=pdf_cfg.get("pdf_n"),
        typ=pdf_cfg.get("pdf_type"),
        basis=pdf_cfg.get("pdf_basis"),
    )

    rot_to_base = build_rot_to_base(rotate_json, base_poi_names)

    x_vals = np.logspace(np.log10(X_MIN), np.log10(X_MAX), int(N_X))
    id_arr = np.full(len(x_vals), int(pid), dtype=int)
    Q_arr = np.full(len(x_vals), float(Q), dtype=float)

    config_pdf_set = getattr(pdf, "reference_pdf_name", None) or pdf_cfg.get("pdf_basis")
    if config_pdf_set is None:
        raise RuntimeError("Could not determine the POD reference PDF set from the config.")
    
    config_pdf_curve = evaluate_external_pdf(config_pdf_set, 0, x_vals, pid, Q)
    nnpdf31_curve = evaluate_external_pdf(NNPDF31_SET, NNPDF31_MEMBER, x_vals, pid, Q)
    pdf4lhc21_mc_curve = evaluate_external_pdf(PDF4LHC21_MC_SET, PDF4LHC21_MC_MEMBER, x_vals, pid, Q)

    pdf_samples_from_toys = np.zeros((chat.shape[0], len(x_vals)), dtype=float)
    for i in range(chat.shape[0]):
        coeffs_base = rot_to_base(fit_poi_names, chat[i])
        pdf_samples_from_toys[i] = pdf.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs_base)

    toy_mean = np.mean(pdf_samples_from_toys, axis=0)
    p16, p84 = np.percentile(pdf_samples_from_toys, [16, 84], axis=0)

    # Avoid division by zero.
    mask = pdf4lhc21_mc_curve != 0.0
    r_config_pdf = np.ones_like(pdf4lhc21_mc_curve)
    r_nnpdf31 = np.ones_like(pdf4lhc21_mc_curve)
    r_toy_mean = np.ones_like(pdf4lhc21_mc_curve)
    r16 = np.ones_like(pdf4lhc21_mc_curve)
    r84 = np.ones_like(pdf4lhc21_mc_curve)
    r_config_pdf[mask] = config_pdf_curve[mask] / pdf4lhc21_mc_curve[mask]
    r_nnpdf31[mask] = nnpdf31_curve[mask] / pdf4lhc21_mc_curve[mask]
    r_toy_mean[mask] = toy_mean[mask] / pdf4lhc21_mc_curve[mask]
    r16[mask] = p16[mask] / pdf4lhc21_mc_curve[mask]
    r84[mask] = p84[mask] / pdf4lhc21_mc_curve[mask]

    config_pdf_label = pretty_pdf_label(config_pdf_set, 0)
    nnpdf31_label = pretty_pdf_label(NNPDF31_SET, NNPDF31_MEMBER)
    pdf4lhc21_mc_label = pretty_pdf_label(PDF4LHC21_MC_SET, PDF4LHC21_MC_MEMBER)
    return (
        x_vals,
        config_pdf_curve,
        nnpdf31_curve,
        pdf4lhc21_mc_curve,
        toy_mean,
        p16,
        p84,
        r_config_pdf,
        r_nnpdf31,
        r_toy_mean,
        r16,
        r84,
        config_pdf_label,
        nnpdf31_label,
        pdf4lhc21_mc_label,
    )


plt.rcParams.update({
    "figure.figsize": (6, 5),
    "axes.linewidth": 1.2,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "cm",
})


import common.user as common_user
OUTDIR = os.path.join(common_user.output_directory, 'toy_plots')
os.makedirs(OUTDIR, exist_ok=True)

CONFIG_PDF_COLOR = "red"
NNPDF31_COLOR = "indigo"
PDF4LHC21_MC_COLOR = "lime"
PDF_LINESTYLE = "--"


def plot_pdf_band(x, config_pdf_curve, nnpdf31_curve, pdf4lhc21_mc_curve, toy_mean, p16, p84,
                  r_config_pdf, r_nnpdf31, r_toy_mean, r16, r84,
                  Q, outname, config_pdf_label, nnpdf31_label, pdf4lhc21_mc_label):
    """
    Draw the main PDF panel and the ratio panel, then save the figure.

    The upper panel shows the toy mean, its 68% band, and the reference PDFs.
    The lower panel shows the same information divided by PDF4LHC21_mc member 0.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    out_pdf = os.path.join(OUTDIR, outname + ".pdf")
    out_png = os.path.join(OUTDIR, outname + ".png")

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.set_xscale("log")
    axr.set_xscale("log")

    locmaj = mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=100)
    fmtmaj = mtick.LogFormatter(base=10.0, labelOnlyBase=False)
    for axis in (ax, axr):
        axis.xaxis.set_major_locator(locmaj)
        axis.xaxis.set_major_formatter(fmtmaj)

    ax.fill_between(x, p16, p84, alpha=0.25, label=r"$68\%$ CI toy band")
    ax.plot(x, toy_mean, color="black", lw=1.8, label="Mean toy curve")
    ax.plot(x, config_pdf_curve, color=CONFIG_PDF_COLOR, lw=1.5, ls=PDF_LINESTYLE, label=config_pdf_label)
    ax.plot(x, nnpdf31_curve, color=NNPDF31_COLOR, lw=1.5, ls=PDF_LINESTYLE, label=nnpdf31_label)
    ax.plot(x, pdf4lhc21_mc_curve, color=PDF4LHC21_MC_COLOR, lw=1.5, ls=PDF_LINESTYLE, label=pdf4lhc21_mc_label)

    axr.fill_between(x, r16, r84, alpha=0.25)
    axr.axhline(1.0, color=PDF4LHC21_MC_COLOR, lw=1.4, ls=PDF_LINESTYLE)
    axr.plot(x, r_toy_mean, color="black", lw=1.6)
    axr.plot(x, r_config_pdf, color=CONFIG_PDF_COLOR, lw=1.6, ls=PDF_LINESTYLE)
    axr.plot(x, r_nnpdf31, color=NNPDF31_COLOR, lw=1.6, ls=PDF_LINESTYLE)

    ax.set_ylabel(rf"$f(x,Q = {float(Q)})$", fontsize=16)
    axr.set_xlabel(r"$x$", fontsize=16)
    axr.set_ylabel(r"$f/f_{\mathrm{target}}$", fontsize=16)
    axr.set_ylim(0.80, 1.20)

    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(5))
    axr.yaxis.set_minor_locator(mtick.AutoMinorLocator(5))
    axr.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))

    # ax.text(0.0, 1.02, "SBI-PDF", transform=ax.transAxes, fontsize=16, weight="bold",
    #         ha="left", va="baseline")
    # ax.text(0.22, 1.02, "Simulation Preliminary", transform=ax.transAxes, fontsize=14,
    #         style="italic", ha="left", va="baseline")

    ax.set_xlim(X_MIN, X_MAX)
    axr.set_xlim(X_MIN, X_MAX)


    ax.legend(frameon=False, loc="upper right")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close(fig)
    print("saved:", out_pdf)
    print("saved:", out_png)


def main():
    """
    Parse command-line arguments, build the plot inputs, and write the figure.
    """
    parser = argparse.ArgumentParser(description="Plot PDF bands from v3 toy best fits.")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--fit-dir", required=True, help="Directory with fit_toys.py outputs")
    parser.add_argument("--rotate", default=None, help="Optional rotation JSON used in the toy fit")
    parser.add_argument("--Q", type=float, default=70.0, help="Q value for the PDF plot")
    parser.add_argument("--pid", type=int, default=21, help="PDG id to plot. Default: 21 (gluon)")
    parser.add_argument("--outname", default=None, help="Output figure stem")
    args = parser.parse_args()

    (
        x_vals,
        config_pdf_curve,
        nnpdf31_curve,
        pdf4lhc21_mc_curve,
        toy_mean,
        p16,
        p84,
        r_config_pdf,
        r_nnpdf31,
        r_toy_mean,
        r16,
        r84,
        config_pdf_label,
        nnpdf31_label,
        pdf4lhc21_mc_label,
    ) = build_pdf_band(
        args.config,
        args.fit_dir,
        rotate_json=args.rotate,
        Q=args.Q,
        pid=args.pid,
    )

    outname = args.outname or f"pdf_band_pid{args.pid}_Q{args.Q:g}"
    plot_pdf_band(
        x_vals,
        config_pdf_curve,
        nnpdf31_curve,
        pdf4lhc21_mc_curve,
        toy_mean,
        p16,
        p84,
        r_config_pdf,
        r_nnpdf31,
        r_toy_mean,
        r16,
        r84,
        args.Q,
        outname=outname,
        config_pdf_label=config_pdf_label,
        nnpdf31_label=nnpdf31_label,
        pdf4lhc21_mc_label=pdf4lhc21_mc_label,
    )


if __name__ == "__main__":
    main()
