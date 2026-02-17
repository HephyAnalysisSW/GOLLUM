'''
plot_pdf_band_from_toy_bestfits.py

This scripts plots the PDF band from the best fit values obtained by the toy fits.

Usage: 
'''

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import common.yaml_loader as yaml_loader
from pdf.PDFParametrization import PDFParametrization
from fit.Modeling import Rotated
from fit.Likelihood import load_likelihood
from fit.Likelihood import build_hypothesis_from_likelihood




def load_fit_results_dir(fit_dir, pattern="*.npz"):
    """
    Read and merge per toy (or per-chunk) fit outputs saved as .npz files.

    Each file contains:
      ['config','toys_npz','toy_ids','region_ids','POI_names','n2ll_min','c_hat']

    Returns:
      POI_names : np.ndarray[str] shape (Npoi,)
      toy_ids   : np.ndarray[int] shape (Ntoys,)
      c_hat     : np.ndarray[float] shape (Ntoys, Npoi)
      n2ll_min  : np.ndarray[float] shape (Ntoys,)
      meta      : dict with 'config' and 'toys_npz' (first file), plus counts
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

    for p in paths:
        z = np.load(p, allow_pickle=False)

        # basic key check (fail fast)
        need = ["config", "toys_npz", "toy_ids", "POI_names", "n2ll_min", "c_hat"]
        for k in need:
            if k not in z:
                raise RuntimeError(f"Missing key '{k}' in {p}")

        poi_names = np.asarray(z["POI_names"])
        chat = np.asarray(z["c_hat"], dtype=float)
        toy_ids = np.asarray(z["toy_ids"]).astype(int).reshape(-1)
        n2ll = np.asarray(z["n2ll_min"], dtype=float).reshape(-1)

        # config / toys_npz are stored as 1-element arrays in your files
        config = str(np.asarray(z["config"]).reshape(-1)[0])
        toys_npz = str(np.asarray(z["toys_npz"]).reshape(-1)[0])

        if poi_names_ref is None:
            poi_names_ref = poi_names
            config_ref = config
            toys_npz_ref = toys_npz
        else:
            if not np.array_equal(poi_names, poi_names_ref):
                raise RuntimeError(
                    f"POI_names mismatch in {p}\n"
                    f"  ref: {poi_names_ref}\n"
                    f"  this:{poi_names}"
                )
            if config != config_ref:
                raise RuntimeError(f"config mismatch in {p}\n  ref: {config_ref}\n  this:{config}")
            if toys_npz != toys_npz_ref:
                raise RuntimeError(f"toys_npz mismatch in {p}\n  ref: {toys_npz_ref}\n  this:{toys_npz}")

        # shape sanity
        if chat.ndim != 2:
            raise RuntimeError(f"c_hat not 2D in {p}: shape={chat.shape}")
        if chat.shape[1] != poi_names_ref.shape[0]:
            raise RuntimeError(
                f"c_hat columns != Npoi in {p}: c_hat={chat.shape}, Npoi={poi_names_ref.shape[0]}"
            )
        if chat.shape[0] != toy_ids.shape[0]:
            raise RuntimeError(f"len(toy_ids) != c_hat rows in {p}: {toy_ids.shape[0]} vs {chat.shape[0]}")
        if n2ll.shape[0] != toy_ids.shape[0]:
            raise RuntimeError(f"len(n2ll_min) != len(toy_ids) in {p}: {n2ll.shape[0]} vs {toy_ids.shape[0]}")

        # duplicates check
        for tid in toy_ids.tolist():
            if tid in seen:
                raise RuntimeError(f"Duplicate toy_id={tid} found (e.g. in {p}).")
            seen.add(tid)

        all_ids.append(toy_ids)
        all_chat.append(chat)
        all_n2ll.append(n2ll)

    toy_ids = np.concatenate(all_ids, axis=0)
    c_hat = np.concatenate(all_chat, axis=0)
    n2ll_min = np.concatenate(all_n2ll, axis=0)

    # sort by toy id (useful for reproducibility)
    order = np.argsort(toy_ids)
    toy_ids = toy_ids[order]
    c_hat = c_hat[order, :]
    n2ll_min = n2ll_min[order]

    meta = {
        "config": config_ref,
        "toys_npz": toys_npz_ref,
        "n_files": len(paths),
        "n_toys": int(toy_ids.shape[0]),
        "n_poi": int(poi_names_ref.shape[0]),
    }
    return poi_names_ref, toy_ids, c_hat, n2ll_min, meta



def pdf_band_from_toy_bestfits(config, fit_dir, Q=1.65, pid=21, x_min=0.015, x_max=0.3, n_x=200, rotate_json=None):
    """
    Returns:
      x_vals: (Nx,)
      central: (Nx,)      # PDF at central point (median POIs)
      p16,p50,p84: (Nx,)  # toy quantiles
      ratio_p16,p50,p84: (Nx,) w.r.t. central
    """

    # Load toy best fits
    POI_names, toy_ids, chat, n2ll_min, meta = load_fit_results_dir(fit_dir, pattern="*.npz")
    print("[fit metadata]\n", meta)
    toy_names = [n for n in POI_names]

    # Build hypo.
    cfg = yaml_loader.load_yaml(config)
    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info)
    hyp_rot = Rotated(hyp, rotate_json, name="Fisher-basis") if rotate_json else hyp
    poi_names = [p.name for p in hyp_rot.POIs]


    # rotated -> base conversion
    def rot_to_base(poi_vec_rot):
        if rotate_json is None:
            return np.array(poi_vec_rot, float)
        hyp_tmp = hyp_rot.cloneModify(**dict(zip(poi_names, poi_vec_rot)))
        return np.array([p.val for p in hyp_tmp.base().POIs], float)


    
    
    # ----------------------------------------------------------------------
    # Find first POI-dependent BIT entry in likelihood and corresponding job
    # ----------------------------------------------------------------------
    like_params = None
    poi_job_id = None

    for region in like_info.get("regions", []):
        for cls in region.get("classes", []):
            poi = cls.get("POI", None)
            if poi and poi.get("type") == "bit" and poi.get("parameters"):
                like_params = poi["parameters"]
                poi_job_id = poi["job"]
                break
        if like_params is not None:
            break

    if like_params is None or poi_job_id is None:
        print("[error] Could not find a POI-dependent BIT term in the likelihood.")
        sys.exit(1)

    # find corresponding job J in cfg['jobs']
    J = None
    for job in cfg.get("jobs", []):
        if job.get("id") == poi_job_id:
            J = job
            break

    if J is None:
        print(f"[error] No job with id '{poi_job_id}' found in cfg['jobs'].")
        sys.exit(1)

    pdf_cfg   = J.get("pdf", {})
    pdf_n     = pdf_cfg.get("pdf_n", None)
    pdf_type  = pdf_cfg.get("pdf_type", None)
    pdf = PDFParametrization(n=pdf_n, typ=pdf_type)

    # --- x grid ---
    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), n_x)
    id_arr = np.full(n_x, int(pid), dtype=int)
    Q_arr  = np.full(n_x, float(Q), dtype=float)

    # --- evaluate pdf for each toy ---
    n_toys = chat.shape[0]
    pdf_samples = np.zeros((n_toys, n_x), float)
    for itoy in range(n_toys):
        coeffs_base = rot_to_base(chat[itoy])
        pdf_samples[itoy] = pdf.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs_base)

    # toy quantiles per x
    p16, p50, p84 = np.percentile(pdf_samples, [16, 50, 84], axis=0)

    # define “central” as the toy median curve
    central = p50

    # --- expected curve: rotated basis with shape_2 = 1, others 0 ---
    poi_exp_rot = np.zeros(len(poi_names), float)
    poi_exp_rot[poi_names.index("shape_2")] = 1.0
    expected = pdf.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=rot_to_base(poi_exp_rot))

    # ratio expected / central (toy median)
    rexp = np.ones_like(central)
    mask = central != 0
    rexp[mask] = expected[mask] / central[mask]


    # ratios
    mask = central != 0
    r16 = np.ones_like(central); r50 = np.ones_like(central); r84 = np.ones_like(central)
    r16[mask] = p16[mask] / central[mask]
    r50[mask] = p50[mask] / central[mask]
    r84[mask] = p84[mask] / central[mask]

    return x_vals, central, p16, p50, p84, r16, r50, r84, expected, rexp





# HEP-ish global style
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

OUTDIR = "/users/alikaan.gueven/sbi-pdf/GOLLUM/user/kaan/figs"

def plot_pdf_band(x, p16, p50, p84, r16, r50, r84, Q, outname,
                  header="B-simov", body=r"Toys median $\pm 1\sigma$",
                  expected=None, rexp=None):

    os.makedirs(OUTDIR, exist_ok=True)
    out_pdf = os.path.join(OUTDIR, outname + ".pdf")
    out_png = os.path.join(OUTDIR, outname + ".png")

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax  = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.set_xscale("log")
    axr.set_xscale("log")

    # label more x ticks on log axis (choose subs)
    subs = (1.0, 2.0, 5.0)          # nice + readable
    # subs = tuple(np.arange(1,10)) # VERY dense (labels 1..9 each decade)

    locmaj = mtick.LogLocator(base=10.0, subs=subs, numticks=100)
    fmtmaj = mtick.LogFormatter(base=10.0, labelOnlyBase=False)

    for a in (ax, axr):
        a.xaxis.set_major_locator(locmaj)
        a.xaxis.set_major_formatter(fmtmaj)
        a.tick_params(axis="x", which="major", labelsize=11)


    ax.fill_between(x, p16, p84, alpha=0.25, label=r"$68\%$ CI")
    ax.plot(x, p50, color="black", lw=1.8, label="Toy median")

    axr.fill_between(x, r16, r84, alpha=0.25)
    axr.axhline(1.0, color="black", lw=1.2)

    if expected is not None:
        ax.plot(x, expected, ls="--", lw=1.6, label=r"Expected ($\hat{s}_2=1$)")
    if rexp is not None:
        axr.plot(x, rexp, ls="--", lw=1.4)

    ax.set_ylabel(rf"$f(x,Q = {float(Q)})$")
    ax.set_ylim(-0.2, 4.8)

    axr.set_xlabel(r"$x$")
    axr.set_ylabel("variation / central", fontsize=12)
    axr.set_ylim(0.80, 1.25)

    # more minor ticks on y (no minor labels)
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(5))
    axr.yaxis.set_minor_locator(mtick.AutoMinorLocator(5))
    for a in (ax, axr):
        a.yaxis.set_minor_formatter(mtick.NullFormatter())
        a.tick_params(axis="y", which="minor", length=3)




    axr.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))

    # Text box
    ax.text(0.03, 0.95, header, transform=ax.transAxes, fontsize=12, weight="bold", va="top")
    ax.text(0.03, 0.89, body,   transform=ax.transAxes, fontsize=11, va="top")

    # Custom title
    ax.text(0.0, 1.02, "SBI-PDF", transform=ax.transAxes, fontsize=16, weight="bold",
            ha="left", va="baseline")
    ax.text(0.22, 1.02, "Simulation Preliminary", transform=ax.transAxes, fontsize=14,
            style="italic", ha="left", va="baseline")

    ax.legend(frameon=False, loc="upper right", alignment="left")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close(fig)
    print("saved:", out_pdf)
    print("saved:", out_png)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Global fit on toys using Likelihood.py cached evaluator.")
    ap.add_argument("--config", help="YAML config")
    ap.add_argument("--fit-dir", help="Directory containing toy fit results")
    ap.add_argument("--rotate", default=None, help="Directory containing toy fit results")
    ap.add_argument("--Q", default=1.65, help="Directory containing toy fit results")
    args = ap.parse_args()

    if not args.rotate:
        print("No rotation json is passed. Are you sure about this choice?")

    # x_vals, central, p16, p50, p84, r16, r50, r84 = pdf_band_from_toy_bestfits(args.config, args.fit_dir, Q=1.65, pid=21, x_min=0.001, x_max=0.9, n_x=200, rotate_json=args.rotate)
    x_vals, central, p16, p50, p84, r16, r50, r84, expected, rexp = pdf_band_from_toy_bestfits(args.config, args.fit_dir, Q=args.Q, pid=21, x_min=0.001, x_max=0.9, n_x=200, rotate_json=args.rotate)


    plot_pdf_band(
        x_vals, p16, p50, p84, r16, r50, r84, args.Q,
        outname="pdf_band_gluon_Q165",
        body="Toys generated under\n" + r"$\hat{s}_{2}=1$",
        expected=expected,
        rexp=rexp
    )

    
    print('x_vals: ', x_vals)
    print('central: ', central)
    print('p16: ', p16)
    print('p50: ', p50)
    print('p84: ', p84)
    print('r16: ', r16)
    print('r50: ', r50)
    print('r84: ', r84)