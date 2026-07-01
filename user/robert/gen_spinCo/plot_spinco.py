#!/usr/bin/env python3

import argparse
import os
import sys
from array import array

import numpy as np
import ROOT

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
from samples_gen import (
    tt2l,
    tt2l_noSpinCo,
    parton_features,
    observers,
)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetNumberContours(255)
ROOT.gStyle.SetPaintTextFormat(".2f")
ROOT.TColor.CreateGradientColorTable(
    5,
    np.array([0.00, 0.45, 0.50, 0.55, 1.00], dtype=float),
    np.array([0.10, 0.65, 1.00, 0.85, 0.45], dtype=float),
    np.array([0.20, 0.85, 1.00, 0.70, 0.00], dtype=float),
    np.array([0.55, 0.95, 1.00, 0.25, 0.10], dtype=float),
    255,
)


SPIN_CORRELATION_VARS = [
    "parton_xi_nn",
    "parton_xi_rr",
    "parton_xi_kk",
    "parton_xi_nr_plus",
    "parton_xi_nr_minus",
    "parton_xi_rk_plus",
    "parton_xi_rk_minus",
    "parton_xi_nk_plus",
    "parton_xi_nk_minus",
    "parton_xi_r_star_k",
    "parton_xi_k_r_star",
    "parton_xi_kk_star",
]

POLARIZATION_VARS = [
    "parton_cosThetaPlus_n",
    "parton_cosThetaMinus_n",
    "parton_cosThetaPlus_r",
    "parton_cosThetaMinus_r",
    "parton_cosThetaPlus_k",
    "parton_cosThetaMinus_k",
    "parton_cosThetaPlus_r_star",
    "parton_cosThetaMinus_r_star",
    "parton_cosThetaPlus_k_star",
    "parton_cosThetaMinus_k_star",
]

DERIVED_VARS = [
    "parton_xi_trace",
]

DEFAULT_VARIABLES = SPIN_CORRELATION_VARS + POLARIZATION_VARS + DERIVED_VARS

PRETTY = {
    "parton_xi_nn": "#xi_{nn}",
    "parton_xi_rr": "#xi_{rr}",
    "parton_xi_kk": "#xi_{kk}",
    "parton_xi_nr_plus": "#xi_{nr}^{+}",
    "parton_xi_nr_minus": "#xi_{nr}^{-}",
    "parton_xi_rk_plus": "#xi_{rk}^{+}",
    "parton_xi_rk_minus": "#xi_{rk}^{-}",
    "parton_xi_nk_plus": "#xi_{nk}^{+}",
    "parton_xi_nk_minus": "#xi_{nk}^{-}",
    "parton_xi_r_star_k": "#xi_{r*k}",
    "parton_xi_k_r_star": "#xi_{kr*}",
    "parton_xi_kk_star": "#xi_{kk*}",
    "parton_xi_trace": "#xi_{nn}+#xi_{rr}+#xi_{kk}",
    "parton_cosThetaPlus_n": "cos#theta_{+}^{n}",
    "parton_cosThetaMinus_n": "cos#theta_{-}^{n}",
    "parton_cosThetaPlus_r": "cos#theta_{+}^{r}",
    "parton_cosThetaMinus_r": "cos#theta_{-}^{r}",
    "parton_cosThetaPlus_k": "cos#theta_{+}^{k}",
    "parton_cosThetaMinus_k": "cos#theta_{-}^{k}",
    "parton_cosThetaPlus_r_star": "cos#theta_{+}^{r*}",
    "parton_cosThetaMinus_r_star": "cos#theta_{-}^{r*}",
    "parton_cosThetaPlus_k_star": "cos#theta_{+}^{k*}",
    "parton_cosThetaMinus_k_star": "cos#theta_{-}^{k*}",
}

SAMPLES = {
    "tt2l": tt2l,
    "tt2l_noSpinCo": tt2l_noSpinCo,
    "noSpinCo": tt2l_noSpinCo,
    "nosc": tt2l_noSpinCo,
}

plot_directory = os.path.join(user.plot_directory, "spinCo", "spin_maps")


def sanitize(name):
    return (
        name.replace("parton_", "")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("*", "star")
        .replace("/", "_")
    )


def make_hist(name, z_title, values, x_edges, y_edges, min_den, symmetric_z=True):
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    hist = ROOT.TH2D(
        name,
        f";cos#theta_{{t}};M_{{t#bar{{t}}}} [GeV];{z_title}",
        nx,
        array("d", x_edges),
        ny,
        array("d", y_edges),
    )

    filled_values = []
    for ix in range(nx):
        for iy in range(ny):
            num, den = values[ix, iy]
            if abs(den) < min_den:
                hist.SetBinContent(ix + 1, iy + 1, 0.0)
                continue

            mean = num / den
            hist.SetBinContent(ix + 1, iy + 1, mean)
            filled_values.append(mean)

    if symmetric_z and filled_values:
        zmax = max(0.1, max(abs(value) for value in filled_values))
        hist.SetMinimum(-zmax)
        hist.SetMaximum(+zmax)

    hist.GetXaxis().SetTitleSize(0.045)
    hist.GetYaxis().SetTitleSize(0.042)
    hist.GetZaxis().SetTitleSize(0.040)
    hist.GetXaxis().SetLabelSize(0.040)
    hist.GetYaxis().SetLabelSize(0.040)
    hist.GetZaxis().SetLabelSize(0.035)
    hist.GetXaxis().SetTitleOffset(1.05)
    hist.GetYaxis().SetTitleOffset(1.75)
    hist.GetZaxis().SetTitleOffset(1.45)
    return hist


def draw_hist(hist, out_base, draw_text=True):
    canvas = ROOT.TCanvas("c", "c", 1000, 850)
    canvas.SetLeftMargin(0.18)
    canvas.SetRightMargin(0.23)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.06)

    hist.SetMarkerSize(0.9)
    hist.Draw("COLZ TEXT" if draw_text else "COLZ")
    canvas.SaveAs(out_base + ".png")
    canvas.SaveAs(out_base + ".pdf")


parser = argparse.ArgumentParser(
    description="Plot spin-correlation and polarization variables in the cos(theta_t) x Mtt plane."
)
parser.add_argument(
    "--sample",
    default="tt2l",
    choices=sorted(SAMPLES),
    help="Which sample object from samples_gen.py to use.",
)
parser.add_argument(
    "--plot-directory",
    default=plot_directory,
    help="Base output directory. Defaults to common.user.plot_directory/spinCo/spin_maps.",
)
parser.add_argument(
    "--outdir",
    default=None,
    help="Exact output directory. Overrides --plot-directory when set.",
)
parser.add_argument(
    "--small",
    action="store_true",
    help="Use only one input file and one loader split.",
)
parser.add_argument(
    "--n-shards",
    type=int,
    default=100,
    help="Maximum number of RDataLoader shards to process.",
)
parser.add_argument(
    "--max-events-per-shard",
    type=int,
    default=None,
    help="Optional event cap per shard.",
)
parser.add_argument(
    "--weight",
    default="genWeight",
    choices=["genWeight", "loader", "unit"],
    help="Event weight choice.",
)
parser.add_argument(
    "--cos-bins",
    type=int,
    default=12,
    help="Number of cos(theta_t) bins from -1 to 1.",
)
parser.add_argument(
    "--mtt-edges",
    default="340,380,420,460,500,550,600,700,800,1000,1200,1600",
    help="Comma-separated Mtt bin edges in GeV.",
)
parser.add_argument(
    "--variables",
    default=None,
    help="Comma-separated variable list. Default: all xi and cosTheta variables plus trace, filled in one data loop.",
)
parser.add_argument(
    "--no-derived",
    action="store_true",
    help="Do not include derived variables such as xi trace.",
)
parser.add_argument(
    "--no-text",
    action="store_true",
    help="Draw COLZ only, without bin numbers.",
)
parser.add_argument(
    "--no-symmetric-z",
    action="store_true",
    help="Do not force the z-axis range to be symmetric around zero.",
)
parser.add_argument(
    "--min-den",
    type=float,
    default=1e-12,
    help="Minimum absolute denominator for filling a plotted bin.",
)
parser.add_argument(
    "--require-gen-tops",
    action="store_true",
    default=True,
    help="Require parton_hasGenTops > 0.5.",
)
parser.add_argument(
    "--require-gen-spin",
    action="store_true",
    default=True,
    help="Require parton_hasGenSpin > 0.5.",
)
args = parser.parse_args()

if args.n_shards < 1:
    raise RuntimeError("--n-shards must be positive.")

sample = SAMPLES[args.sample]
if args.small:
    sample.set_max_files(1)
    sample.set_n_split(1)
elif hasattr(sample, "_all_files"):
    sample.set_n_split(min(args.n_shards, len(sample._all_files)))

feature_index = {name: i for i, name in enumerate(parton_features)}
variables = (
    [var.strip() for var in args.variables.split(",") if var.strip()]
    if args.variables is not None
    else DEFAULT_VARIABLES if not args.no_derived else SPIN_CORRELATION_VARS + POLARIZATION_VARS
)

for var in variables:
    if var not in DERIVED_VARS and var not in feature_index:
        raise RuntimeError(f"Variable {var} not found in parton_features.")

x_edges = np.linspace(-1.0, 1.0, args.cos_bins + 1, dtype=float)
y_edges = np.array([float(edge) for edge in args.mtt_edges.split(",")], dtype=float)
nx = len(x_edges) - 1
ny = len(y_edges) - 1

acc = {var: np.zeros((nx, ny, 2), dtype=float) for var in variables}
i_cos = feature_index["parton_cosTheta_t"]
i_mtt = feature_index["parton_Mtt"]

n_shards = min(args.n_shards, len(sample))
for shard in range(n_shards):
    print(f"[spinCo] materialize shard {shard}/{n_shards - 1}")

    X, O, w_loader = sample.materialize(
        shard=shard,
        what="fow",
        n=args.max_events_per_shard,
    )

    X = np.asarray(X)
    O = np.asarray(O)
    if len(X) == 0:
        continue

    if args.weight == "unit":
        weights = np.ones(len(O), dtype=float)
    elif args.weight == "genWeight":
        weights = np.asarray(O[:, observers.index("genWeight")], dtype=float)
    elif w_loader is None:
        raise RuntimeError("Requested --weight loader, but RDataLoader returned w=None.")
    else:
        weights = np.asarray(w_loader, dtype=float).reshape(-1)

    x = X[:, i_cos]
    y = X[:, i_mtt]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    mask &= x >= x_edges[0]
    mask &= x <= x_edges[-1]
    mask &= y >= y_edges[0]
    mask &= y < y_edges[-1]

    if args.require_gen_tops and "parton_hasGenTops" in feature_index:
        mask &= X[:, feature_index["parton_hasGenTops"]] > 0.5

    if args.require_gen_spin and "parton_hasGenSpin" in feature_index:
        mask &= X[:, feature_index["parton_hasGenSpin"]] > 0.5

    if not np.any(mask):
        continue

    ix = np.searchsorted(x_edges, x[mask], side="right") - 1
    iy = np.searchsorted(y_edges, y[mask], side="right") - 1
    ix[ix == nx] = nx - 1
    iy[iy == ny] = ny - 1
    bin_index = ix * ny + iy
    w_m = weights[mask]

    for var in variables:
        if var == "parton_xi_trace":
            values = (
                X[:, feature_index["parton_xi_nn"]]
                + X[:, feature_index["parton_xi_rr"]]
                + X[:, feature_index["parton_xi_kk"]]
            )
        else:
            values = X[:, feature_index[var]]

        v_m = values[mask]
        good = np.isfinite(v_m)
        if not np.any(good):
            continue

        acc[var][:, :, 0] += np.bincount(
            bin_index[good],
            weights=w_m[good] * v_m[good],
            minlength=nx * ny,
        ).reshape(nx, ny)
        acc[var][:, :, 1] += np.bincount(
            bin_index[good],
            weights=w_m[good],
            minlength=nx * ny,
        ).reshape(nx, ny)

label = sanitize(args.sample)
if args.small:
    label += "_small"

plot_dir = args.outdir or os.path.join(args.plot_directory, label)
helpers.copyIndexPHP(os.path.join(user.plot_directory, "spinCo"))
helpers.copyIndexPHP(args.plot_directory)
helpers.copyIndexPHP(plot_dir)
print(f"[spinCo] output directory: {plot_dir}")

root_path = os.path.join(plot_dir, f"spin_maps_{label}.root")
fout = ROOT.TFile(root_path, "RECREATE")

for var in variables:
    pretty = PRETTY.get(var, var.replace("parton_", ""))
    hist = make_hist(
        name=f"h2_{sanitize(var)}",
        z_title=f"#LT {pretty} #GT",
        values=acc[var],
        x_edges=x_edges,
        y_edges=y_edges,
        min_den=args.min_den,
        symmetric_z=not args.no_symmetric_z,
    )

    hist.Write()
    draw_hist(
        hist,
        out_base=os.path.join(plot_dir, f"{label}_{sanitize(var)}"),
        draw_text=not args.no_text,
    )

fout.Close()
print(f"[spinCo] wrote {root_path}")
syncer.sync()
