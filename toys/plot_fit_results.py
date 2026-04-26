import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import common.user as common_user


FIT_DIRECTORY = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_PDF4LHC21_mc_m0_rw_N1000"
TARGET_PDF_TEXT = "PDF4LHC21_mc"
# HEADER_TEXT = "B-simov"
HEADER_TEXT = ""
OUTPUT_SUBDIR = "toy_plots"
N_BINS = 50

STYLE = {
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
}

# Addition after Robert's comments
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"


def configure_style():
    plt.rcParams.update(STYLE)


def output_dir():
    path = os.path.join(common_user.output_directory, OUTPUT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def load_fit_results(fit_directory):
    fit_files = sorted(glob.glob(os.path.join(fit_directory, "*.npz")))
    if not fit_files:
        raise FileNotFoundError(f"No .npz files found in {fit_directory}")

    c_hats = []
    n2ll_mins = []
    for fit_file in fit_files:
        with np.load(fit_file) as data:
            c_hats.append(data["c_hat"])
            n2ll_mins.append(data["n2ll_min"].item())

    return np.concatenate(c_hats, axis=0), np.array(n2ll_mins)


def make_bins(values, n_bins=N_BINS):
    left = np.min(values)
    right = np.max(values)
    span = right - left
    padding = span / 2 if span else 0.5
    return np.linspace(left - padding, right + padding, n_bins)


def add_target_pdf_text(ax):
    # ax.text(
    #     0.03,
    #     0.95,
    #     HEADER_TEXT,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     weight="bold",
    #     va="top",
    # )
    ax.text(
        0.03,
        0.95,
        "Toys generated under\n"
        "the alternative hypo:\n"
        f"{TARGET_PDF_TEXT}",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
    )


def add_legend(ax, n_toys):
    legend = ax.legend(
        title=f"N toys = {n_toys}",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.963),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        alignment="left",
        title_fontproperties={"size": 12},
    )

    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if text.get_text() == "68% CI":
            handle.set_linewidth(2.0)


def save_figure(fig, basename, outdir):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{basename}.pdf"))
    fig.savefig(os.path.join(outdir, f"{basename}.png"))
    plt.close(fig)


def format_stats_text(values):
    mean = np.mean(values)
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    width_label = 12
    width_value = 7
    return (
        f"{'mean':<{width_label}}{mean:>{width_value}.2f}\n"
        f"{'median':<{width_label}}{q50:>{width_value}.2f}\n"
        f"{'16% quantile':<{width_label}}{q16:>{width_value}.2f}\n"
        f"{'84% quantile':<{width_label}}{q84:>{width_value}.2f}\n"
    )


def add_stats_box(ax, values):
    ax.text(
        0.97,
        0.65,
        format_stats_text(values),
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        fontfamily="DejaVu Sans Mono",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )


# ----- MAIN STARTS HERE ---------

configure_style()
outdir = output_dir()
# c_hats are the EV_0,1...5
c_hats, n2ll_mins = load_fit_results(FIT_DIRECTORY)
n_toys = len(n2ll_mins)
q16, q84 = np.percentile(n2ll_mins, [16, 84])

fig, ax = plt.subplots()
ax.hist(
    n2ll_mins,
    bins=make_bins(n2ll_mins),
    histtype="step",
    linewidth=1.5,
    color="black",
    label="Data",
)
ax.axvline(q16, color="red", linestyle="--", linewidth=1.2, label="68% CI")
ax.axvline(q84, color="red", linestyle="--", linewidth=1.2)

add_target_pdf_text(ax)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# ax.set_xlabel(r"$\text{negative log likelihood}$", fontsize=16)
ax.set_xlabel("negative log likelihood")
ax.set_ylabel("Counts")
add_legend(ax, n_toys)
save_figure(fig, "n2ll_min_edited", outdir)

n_toys, n_cols = c_hats.shape
for index in range(n_cols):
    values = c_hats[:, index]
    q16, q84 = np.percentile(values, [16, 84])

    fig, ax = plt.subplots()
    ax.hist(
        values,
        bins=make_bins(values),
        histtype="step",
        linewidth=1.5,
        color="black",
        label="Data",
    )
    ax.axvline(q16, color="red", linestyle="--", linewidth=1.2, label="68% CI")
    ax.axvline(q84, color="red", linestyle="--", linewidth=1.2)

    # add_stats_box(ax, values)
    add_target_pdf_text(ax)
    ax.set_xlabel(rf"$v^{{({index})}}$")
    ax.set_ylabel("Counts")
    add_legend(ax, n_toys)
    save_figure(fig, f"c_{index}_edited", outdir)

print("Saved plots to ", outdir)
