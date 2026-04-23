#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import math
import argparse
import importlib
import itertools

import numpy as np
import ROOT
from tqdm import tqdm

# project roots
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.user as user
import common.yaml_loader as yaml_loader

from pdf.PDFParametrization import PDFParametrization
from ML.BIT.NumbaBIT import MultiBoostedInformationTree
from data.UIDSplitter import UIDSplitter
from propaganda_plot_options import plot_options as PLOT_OPTS
import common.syncer as syncer
import common.helpers as helpers

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    _predict_classifier,
    predict_pnn_deltaA,
)

# --------------------------------------------------------------------------------
# hard-coded knobs
# --------------------------------------------------------------------------------

# Show all available derivatives by default.
# To restrict the output, uncomment and edit, e.g.
# SHOW_ONLY = [("c1",), ("c2",), ("c1", "c1"), ("c1", "c2")]
SHOW_ONLY = [("c0",), ("c1",), ("c2",), ("c3",), ("c4",), ("c5",)]

# --small will stop after this many selected events
SMALL_MAX_EVENTS = 500000

# nuisance toy sampling (prefit)
NUISANCE_N_TOYS = 1000
NUISANCE_TOY_RNG_SEED = 42
NUISANCE_TOY_MEAN = 0.0
NUISANCE_TOY_SIGMA = 1.0

# More discernible hard-coded colors (Wong/Okabe-Ito style + a few extras)
COLOR_HEX = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#882255",  # wine
    "#44AA99",  # teal
    "#AA4499",  # magenta
    "#117733",  # dark green
    "#332288",  # dark blue
]

# Keep the plot logic in the main scope; only tiny utilities live here.
def canonical_derivative(der):
    der = tuple(der)
    if len(der) <= 1:
        return der
    return tuple(sorted(der))

def derivative_label(der):
    if len(der) == 1:
        return der[0]
    if len(der) == 2 and der[0] == der[1]:
        return f"{der[0]}^{{2}}"
    if len(der) == 2:
        return f"{der[0]} #times {der[1]}"
    return str(der)

def make_root_hist(name, title, edges, values):
    h = ROOT.TH1D(name, title, len(edges) - 1, edges.astype(np.float64))
    for i_bin, val in enumerate(values, start=1):
        h.SetBinContent(i_bin, float(val))
    return h

def map_right_to_left(values, right_min, right_max, left_max):
    out = np.zeros_like(values, dtype=np.float64)
    if right_max <= right_min:
        return out
    out[:] = left_max * (values - right_min) / (right_max - right_min)
    return out


# --------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------

p = argparse.ArgumentParser(description="BIT plotting from a trained model (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--small", action="store_true", help=f"Only use the first {SMALL_MAX_EVENTS} selected events")
p.add_argument("--truth_only", action="store_true", help="Only plot truth prediction")
p.add_argument("--uncertainty", action="store_true", help=f"Plot uncertainties?")
args = p.parse_args()


# --------------------------------------------------------------------------------
# ROOT style
# --------------------------------------------------------------------------------

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleBorderSize(0)
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)
ROOT.gStyle.SetLegendBorderSize(0)
ROOT.TGaxis.SetMaxDigits(3)

# --------------------------------------------------------------------------------
# cfg / job selection
# --------------------------------------------------------------------------------

cfg_path = os.path.expanduser(os.path.expandvars(args.config))
CFG = yaml_loader.load_yaml(cfg_path)
D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "bit"]
    if not jobs:
        print("No BIT jobs found.")
        sys.exit(0)

    script = os.path.basename(__file__)
    flags = []
    if args.small:
        flags.append("--small")

    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}".strip())
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "bit":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'bit'.")


# --------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------

samples_mod = importlib.import_module(module_samples)
loader_name = J.get("process")
if not hasattr(samples_mod, loader_name):
    raise RuntimeError(f"Loader/view '{loader_name}' not found in module {module_samples}.")
L = getattr(samples_mod, loader_name)

L.set_n_split(100)

sel = J.get("selection", None)
sel_f = J.get("selection_features", [])
if sel:
    L.addSelection(sel, sel_f)
    print(f"Added selection to loader: {sel} with selection_features {sel_f}")

L.setFeatures(J["features"])

feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")

GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

plot_feats = [f for f in feat_names if f in PLOT_OPTS]
if not plot_feats:
    print("No plottable features found in data.plot_options. Nothing to do.")
    sys.exit(0)

print(f"Plottable features: {plot_feats}")

# --------------------------------------------------------------------------------
# Uncertainty helpers 
# --------------------------------------------------------------------------------
if args.uncertainty:
    def make_column_mask(all_features, wanted_features):
        pos = {f: i for i, f in enumerate(all_features)}
        mask = np.zeros(len(all_features), dtype=bool)
        for f in wanted_features:
            mask[pos[f]] = True
        return mask

    def nuis_to_A_matrix(combinations, nuisance_toys, nuisance_name_to_idx):
        if not combinations:
            return np.zeros((nuisance_toys.shape[0], 0), dtype=np.float64)

        out = np.empty((nuisance_toys.shape[0], len(combinations)), dtype=np.float64)
        for i_comb, comb in enumerate(combinations):
            v = np.ones(nuisance_toys.shape[0], dtype=np.float64)
            for p in comb:
                v *= nuisance_toys[:, nuisance_name_to_idx[p]]
            out[:, i_comb] = v
        return out

# --------------------------------------------------------------------------------
# UID split config
# --------------------------------------------------------------------------------

UID_CFG = (J.get("splitting") or {})
uid_enabled = bool(UID_CFG.get("enabled", False))
uid_fields = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme = (UID_CFG.get("scheme") or {})

uid_splitter = None
train_interval = None

if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    keys = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)

    bit_train_key = "pnn_train"
    if bit_train_key not in uid_intervals:
        raise RuntimeError(f"UID split enabled, but '{bit_train_key}' was not found in the splitting scheme.")

    train_interval = uid_intervals[bit_train_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] plotting split '{bit_train_key}' -> {train_interval}")
else:
    print("[UID] disabled -> plotting all selected events")


# --------------------------------------------------------------------------------
# PDF parametrization
# --------------------------------------------------------------------------------

pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
pdf_rescale_pod_amplitudes = J.get("pdf", {}).get("rescale_pod_amplitudes", True)

pdf = PDFParametrization(
    n=pdf_n,
    typ=pdf_type,
    basis=pdf_basis,
    rescale_pod_amplitudes=pdf_rescale_pod_amplitudes,
)

combos = [canonical_derivative(c) for c in pdf.combinations]
combo_to_idx = {c: i for i, c in enumerate(combos)}

if () not in combo_to_idx:
    raise RuntimeError("Could not find the nominal coefficient () in PDF combinations.")

nominal_idx = combo_to_idx[()]


# --------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------

cfg_base = os.path.join(CFG.get("version", "default"), J["region"])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))

print(f"Trying to load BIT from {model_path}")
try:
    bit = MultiBoostedInformationTree.load(model_path)
except Exception as e:
    print(f"Could not load BIT model from '{model_path}'.")
    print(f"Reason: {e}")
    sys.exit(1)

n_loaded_trees = len(getattr(bit, "trees", []) or [])
print(f"Loaded BIT model successfully: {n_loaded_trees} trees")

model_derivatives_raw = list(getattr(bit, "derivatives", []) or [])
if not model_derivatives_raw:
    print("Loaded BIT model, but bit.derivatives is empty. Nothing to plot.")
    sys.exit(1)

model_derivatives = [canonical_derivative(d) for d in model_derivatives_raw]
pred_derivative_to_idx = {}
for i_der, der in enumerate(model_derivatives):
    pred_derivative_to_idx[der] = i_der

if SHOW_ONLY is None:
    plot_derivatives = [d for d in model_derivatives if d in combo_to_idx]
else:
    plot_derivatives = []
    for der in SHOW_ONLY:
        cder = canonical_derivative(der)
        if cder not in combo_to_idx:
            print(f"Requested derivative {der} not available in PDF combinations -> skipping")
            continue
        if cder not in pred_derivative_to_idx:
            print(f"Requested derivative {der} not available in loaded model -> skipping")
            continue
        if cder not in plot_derivatives:
            plot_derivatives.append(cder)

if not plot_derivatives:
    print("No derivatives selected for plotting.")
    sys.exit(1)

print("Plotting derivatives:")
for der in plot_derivatives:
    print(f"  {der}")


# --------------------------------------------------------------------------------
# output directory
# --------------------------------------------------------------------------------

out_dir = os.path.join(user.plot_directory, "BIT-plot"+("_unc" if args.uncertainty else ""), cfg_base, J["id"])

if args.small:
    out_dir = os.path.join(out_dir, "small")

if args.truth_only:
    out_dir += "_truth_only"    
os.makedirs(out_dir, exist_ok=True)

print(f"Output directory: {out_dir}")

# --------------------------------------------------------------------------------
# uncertainty setup (prefit)
# --------------------------------------------------------------------------------
if args.uncertainty:
    yaml_loader.load_surrogates(CFG, cfg_path, overwrite=False)
    uncertainty_like_info = load_likelihood(CFG)
    uncertainty_hyp = build_hypothesis_from_likelihood(uncertainty_like_info, name=J["region"])

    uncertainty_region = next(r for r in uncertainty_like_info["regions"] if r["id"] == J["region"])
    uncertainty_classes = list(uncertainty_region.get("classes", []) or [])
    uncertainty_classifier = (uncertainty_region.get("classifier") or {}).get("predictor", None)

    uncertainty_nuisances = [par for par in uncertainty_hyp.nuisances if not par.isFrozen]
    uncertainty_nuisance_names = [par.name for par in uncertainty_nuisances]
    uncertainty_nuisance_name_to_idx = {name: i for i, name in enumerate(uncertainty_nuisance_names)}
    uncertainty_n_nuisances = len(uncertainty_nuisance_names)

    uncertainty_mean = np.full(uncertainty_n_nuisances, NUISANCE_TOY_MEAN, dtype=np.float64)
    uncertainty_cov = np.eye(uncertainty_n_nuisances, dtype=np.float64) * (NUISANCE_TOY_SIGMA ** 2)

    np.random.seed(NUISANCE_TOY_RNG_SEED)
    uncertainty_toys = np.random.multivariate_normal(
        mean=uncertainty_mean,
        cov=uncertainty_cov,
        size=NUISANCE_N_TOYS,
    )

    if uncertainty_classifier is not None:
        uncertainty_classifier_column_mask = make_column_mask(feat_names, uncertainty_classifier.feature_names)
    else:
        uncertainty_classifier_column_mask = None

    uncertainty_class_infos = []
    for i_class, C in enumerate(uncertainty_classes):
        class_info = {
            "id": C["id"],
            "index": i_class,
            "pnn": [],
            "lnN": [],
        }

        for S in (C.get("systematics") or []):
#            if S.get("type") == "pnn":
#                column_mask = make_column_mask(feat_names, S["predictor"].feature_names)
#                nuA_toys = nuis_to_A_matrix(
#                    [tuple(c) for c in (S.get("combinations") or [])],
#                    uncertainty_toys,
#                    uncertainty_nuisance_name_to_idx,
#                )
#                class_info["pnn"].append({
#                    "id": S["id"],
#                    "predictor": S["predictor"],
#                    "column_mask": column_mask,
#                    "nuA_toys": nuA_toys,
#                })
            # clipping for uncertainty evaluation
            if S.get("type") == "pnn":
                pnn_feature_names = list(S["predictor"].feature_names)
                column_mask = make_column_mask(feat_names, pnn_feature_names)

                clip_low = np.array(
                    [PLOT_OPTS[name]["binning"][1] for name in pnn_feature_names],
                    dtype=np.float64,
                )
                clip_high = np.array(
                    [PLOT_OPTS[name]["binning"][2] for name in pnn_feature_names],
                    dtype=np.float64,
                )

                nuA_toys = nuis_to_A_matrix(
                    [tuple(c) for c in (S.get("combinations") or [])],
                    uncertainty_toys,
                    uncertainty_nuisance_name_to_idx,
                )

                class_info["pnn"].append({
                    "id": S["id"],
                    "predictor": S["predictor"],
                    "column_mask": column_mask,
                    "clip_low": clip_low,
                    "clip_high": clip_high,
                    "nuA_toys": nuA_toys,
                })
            elif S.get("type") == "lnN":
                class_info["lnN"].append(
                    (S["parameters"][0], math.log1p(float(S.get("value", 0.0))))
                )

        uncertainty_class_infos.append(class_info)

    uncertainty_n_toys = uncertainty_toys.shape[0]
    print("sampling done")

else:
    uncertainty_like_info = None
    uncertainty_hyp = None
    uncertainty_region = None
    uncertainty_classes = []
    uncertainty_classifier = None
    uncertainty_classifier_column_mask = None
    uncertainty_nuisances = []
    uncertainty_nuisance_names = []
    uncertainty_nuisance_name_to_idx = {}
    uncertainty_toys = np.empty((0, 0), dtype=np.float64)
    uncertainty_class_infos = []
    uncertainty_n_toys = 0


# --------------------------------------------------------------------------------
# histogram containers
# --------------------------------------------------------------------------------

feature_columns = {f: feat_names.index(f) for f in plot_feats}
feature_edges = {}
nominal_hists = {}
nominal_toy_hists = {}
truth_num_hists = {}
pred_num_hists = {}

for feat in plot_feats:
    n_bins, x_lo, x_hi = PLOT_OPTS[feat]["binning"]
    edges = np.linspace(x_lo, x_hi, n_bins + 1, dtype=np.float64)
    feature_edges[feat] = edges
    nominal_hists[feat] = np.zeros(n_bins, dtype=np.float64)
    nominal_toy_hists[feat] = np.zeros((n_bins, uncertainty_n_toys), dtype=np.float64)
    truth_num_hists[feat] = {der: np.zeros(n_bins, dtype=np.float64) for der in plot_derivatives}
    pred_num_hists[feat] = {der: np.zeros(n_bins, dtype=np.float64) for der in plot_derivatives}

root_colors = {}
for i_der, der in enumerate(plot_derivatives):
    root_colors[der] = ROOT.TColor.GetColor(COLOR_HEX[i_der % len(COLOR_HEX)])


# --------------------------------------------------------------------------------
# event loop
# --------------------------------------------------------------------------------

on2idx = {n: i for i, n in enumerate(obs_names)}

i_Q = on2idx["Generator_scalePDF"]
i_x1 = on2idx["Generator_x1"]
i_x2 = on2idx["Generator_x2"]
i_id1 = on2idx["Generator_id1"]
i_id2 = on2idx["Generator_id2"]

if uid_enabled:
    uid_idx = [on2idx[f] for f in uid_fields]
    lo_tr, hi_tr = train_interval

selected_events = 0
n_shards = len(L)

for shard in tqdm(range(n_shards), desc="Shards", unit="shard"):
    X, G, w = L.materialize(shard=shard, what="fow")

    if len(X) == 0:
        continue

    if uid_enabled:
        O_uid = G[:, uid_idx]
        mask = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_tr, hi_tr)
    else:
        mask = np.ones(len(X), dtype=bool)

    if not np.any(mask):
        continue

    X = X[mask].astype(np.float32, copy=False)
    G = G[mask]
    w = w[mask].astype(np.float32, copy=False)

    if args.small:
        remaining = SMALL_MAX_EVENTS - selected_events
        if remaining <= 0:
            break
        if len(X) > remaining:
            X = X[:remaining]
            G = G[:remaining]
            w = w[:remaining]

    if len(X) == 0:
        continue

    if args.uncertainty:
        if uncertainty_classifier is None or len(uncertainty_class_infos) <= 1:
            uncertainty_g = np.ones((len(X), len(uncertainty_class_infos)), dtype=np.float64)
        else:
            uncertainty_g = _predict_classifier(
                uncertainty_classifier,
                X[:, uncertainty_classifier_column_mask],
            )

        uncertainty_T = np.zeros((len(X), uncertainty_n_toys), dtype=np.float64)

        for class_info in uncertainty_class_infos:
            g_cls = uncertainty_g[:, class_info["index"]].astype(np.float64, copy=False)
            expo = np.zeros((len(X), uncertainty_n_toys), dtype=np.float64)

            for syst_info in class_info["pnn"]:
                X_syst = X[:, syst_info["column_mask"]].astype(np.float64, copy=True)
                np.clip(X_syst, syst_info["clip_low"], syst_info["clip_high"], out=X_syst)

                dA = predict_pnn_deltaA(
                    syst_info["predictor"],
                    X_syst,
                )
                expo += dA @ syst_info["nuA_toys"].T

            ln_bias = np.zeros(uncertainty_n_toys, dtype=np.float64)
            for pname, log1p_alpha in class_info["lnN"]:
                ln_bias += log1p_alpha * uncertainty_toys[:, uncertainty_nuisance_name_to_idx[pname]]

            exp_expo = np.exp(expo + ln_bias.reshape(1, -1))
            uncertainty_T += g_cls.reshape(-1, 1) * (exp_expo - 1.0)

        uncertainty_R = 1.0 + uncertainty_T

    else:
        uncertainty_R = np.empty((len(X), 0), dtype=np.float64)

    Q = G[:, i_Q].astype(np.float32, copy=False)
    x1 = G[:, i_x1].astype(np.float32, copy=False)
    x2 = G[:, i_x2].astype(np.float32, copy=False)
    id1 = G[:, i_id1].astype(np.int32, copy=False)
    id2 = G[:, i_id2].astype(np.int32, copy=False)

    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q).astype(np.float32, copy=False)
    deriv_w = deriv * w.reshape(-1, 1)

    nominal_w = deriv_w[:, nominal_idx]

    if uncertainty_R.shape[1] > 0:
        toy_weights = nominal_w.reshape(-1, 1) * uncertainty_R

        bad_mask = (~np.isfinite(toy_weights)) | (np.abs(toy_weights) > 1e4)
        if np.any(bad_mask):
            bad_idx = np.argwhere(bad_mask)
            print("outlier toy weights:")
            for i_evt, i_toy in bad_idx[:10]:
                feat_str = ", ".join(
                    f"{name}={X[i_evt, i_feat]:.6e}"
                    for i_feat, name in enumerate(feat_names)
                )

                print(
                    f"  shard={shard} evt={i_evt} toy={i_toy} "
                    f"w0={nominal_w[i_evt]:.6e} "
                    f"R={uncertainty_R[i_evt, i_toy]:.6e} "
                    f"w={toy_weights[i_evt, i_toy]:.6e}"
                )
                print(f"    features: {feat_str}")

                for class_info in uncertainty_class_infos:
                    g_val = uncertainty_g[i_evt, class_info["index"]]

                    ln_bias_terms = []
                    ln_bias_val = 0.0
                    for pname, log1p_alpha in class_info["lnN"]:
                        term = log1p_alpha * uncertainty_toys[i_toy, uncertainty_nuisance_name_to_idx[pname]]
                        ln_bias_terms.append((pname, term))
                        ln_bias_val += term

                    syst_terms = []
                    expo_val = 0.0
                    for syst_info in class_info["pnn"]:
                        X_evt_raw = X[i_evt:i_evt+1, syst_info["column_mask"]].astype(np.float64, copy=True)
                        X_evt = X_evt_raw.copy()
                        np.clip(X_evt, syst_info["clip_low"], syst_info["clip_high"], out=X_evt)

                        dA_evt = predict_pnn_deltaA(
                            syst_info["predictor"],
                            X_evt,
                        )[0]
                        nuA_toy = syst_info["nuA_toys"][i_toy]
                        term = float(np.dot(dA_evt, nuA_toy))
                        syst_terms.append((syst_info["id"], term))
                        expo_val += term

                        if np.any(X_evt != X_evt_raw):
                            used_names = list(syst_info["predictor"].feature_names)
                            raw_str = ", ".join(
                                f"{name}={X_evt_raw[0, j]:.6e}"
                                for j, name in enumerate(used_names)
                            )
                            clip_str = ", ".join(
                                f"{name}={X_evt[0, j]:.6e}"
                                for j, name in enumerate(used_names)
                            )
                            print(f"      syst {syst_info['id']} input clipped")
                            print(f"        raw : {raw_str}")
                            print(f"        clip: {clip_str}")

                    arg_val = expo_val + ln_bias_val
                    exp_val = np.exp(arg_val) if np.isfinite(arg_val) else np.nan
                    class_contribution = g_val * (exp_val - 1.0) if np.isfinite(exp_val) else np.nan

                    print(
                        f"    class={class_info['id']} "
                        f"g={g_val:.6e} "
                        f"expo={expo_val:.6e} "
                        f"ln_bias={ln_bias_val:.6e} "
                        f"arg={arg_val:.6e} "
                        f"exp(arg)={exp_val:.6e} "
                        f"class_term={class_contribution:.6e}"
                    )

                    syst_terms = sorted(syst_terms, key=lambda x: abs(x[1]), reverse=True)
                    for sid, term in syst_terms[:5]:
                        print(f"      syst {sid}: {term:.6e}")

                    ln_bias_terms = sorted(ln_bias_terms, key=lambda x: abs(x[1]), reverse=True)
                    for pname, term in ln_bias_terms[:5]:
                        print(f"      lnN  {pname}: {term:.6e}")
    else:
        toy_weights = np.empty((len(X), 0), dtype=np.float64)

    pred = np.asarray(bit.predict(X, max_n_tree=n_loaded_trees))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    for feat in plot_feats:
        xvals = X[:, feature_columns[feat]].astype(np.float64, copy=False)
        edges = feature_edges[feat]

        nominal_hists[feat] += np.histogram(xvals, bins=edges, weights=nominal_w)[0]

        if toy_weights.shape[1] > 0:
            for i_toy in range(uncertainty_n_toys):
                nominal_toy_hists[feat][:, i_toy] += np.histogram(
                    xvals,
                    bins=edges,
                    weights=toy_weights[:, i_toy],
                )[0]

        for der in plot_derivatives:
            truth_col = combo_to_idx[der]
            pred_col = pred_derivative_to_idx[der]

            truth_num_hists[feat][der] += np.histogram(
                xvals,
                bins=edges,
                weights=deriv_w[:, truth_col],
            )[0]

            pred_num_hists[feat][der] += np.histogram(
                xvals,
                bins=edges,
                weights=nominal_w * pred[:, pred_col],
            )[0]

    selected_events += len(X)

if selected_events == 0:
    print("No events passed the selection / split. Nothing to plot.")
    sys.exit(1)

print(f"Processed {selected_events} selected events")


# --------------------------------------------------------------------------------
# plots: one feature per canvas
# --------------------------------------------------------------------------------

for feat in plot_feats:
    edges = feature_edges[feat]
    n_bins = len(edges) - 1
    x_lo = edges[0]
    x_hi = edges[-1]
    x_title = PLOT_OPTS[feat]["tex"]

    nominal = nominal_hists[feat]
    nominal_toys = nominal_toy_hists[feat]
    have_uncertainty = nominal_toys.shape[1] > 0

    if have_uncertainty:
        nominal_q_low = np.quantile(nominal_toys, 0.16, axis=1)
        nominal_q_high = np.quantile(nominal_toys, 0.84, axis=1)
    else:
        nominal_q_low = None
        nominal_q_high = None

    coeff_truth = {}
    coeff_pred = {}
    coeff_values = []

    nz = nominal != 0.0

    for der in plot_derivatives:
        ct = np.zeros_like(nominal, dtype=np.float64)
        cp = np.zeros_like(nominal, dtype=np.float64)

        if np.any(nz):
            ct[nz] = truth_num_hists[feat][der][nz] / nominal[nz]
            cp[nz] = pred_num_hists[feat][der][nz] / nominal[nz]

        coeff_truth[der] = ct
        coeff_pred[der] = cp

        coeff_values.append(ct[np.isfinite(ct)])
        coeff_values.append(cp[np.isfinite(cp)])

    if coeff_values:
        coeff_values = np.concatenate([v for v in coeff_values if len(v) > 0], axis=0) if any(len(v) > 0 for v in coeff_values) else np.array([0.0, 1.0])
    else:
        coeff_values = np.array([0.0, 1.0])

    finite = np.isfinite(coeff_values)
    if finite.any():
        right_min = float(np.min(coeff_values[finite]))
        right_max = float(np.max(coeff_values[finite]))
    else:
        right_min, right_max = -1.0, 1.0

    if right_max <= right_min:
        width = 1.0 if right_max == right_min else abs(right_max - right_min)
        right_min -= 0.5 * width
        right_max += 0.5 * width

    right_pad_lo = 0.18 * (right_max - right_min)
    right_pad_hi = 0.28 * (right_max - right_min)
    right_min -= right_pad_lo
    right_max += right_pad_hi

    left_max = float(np.max(nominal)) if len(nominal) else 0.0
    if left_max <= 0.0:
        left_max = 1.0
    left_max *= 1.35

    canvas = ROOT.TCanvas(f"c_{feat}", feat, 900, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.14)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.08)
    canvas.SetTicks(1, 0)

    frame = ROOT.TH1D(f"frame_{feat}", "", n_bins, edges.astype(np.float64))
    frame.SetMinimum(0.0)
    frame.SetMaximum(left_max)
    frame.GetXaxis().SetTitle(x_title)
    frame.GetYaxis().SetTitle("Number of events")
    frame.GetXaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetTitleOffset(1.20)
    frame.Draw("axis")
    frame.GetYaxis().SetTicks("-")

    drawn_objects = [frame]

    if have_uncertainty:
        uncertainty_boxes = []
        for i_bin in range(n_bins):
            box = ROOT.TBox(
                float(edges[i_bin]),
                float(nominal_q_low[i_bin]),
                float(edges[i_bin + 1]),
                float(nominal_q_high[i_bin]),
            )
            box.SetLineWidth(0)
            box.SetLineColor(0)
            box.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
            uncertainty_boxes.append(box)

        for box in uncertainty_boxes:
            box.Draw("same")

        drawn_objects.extend(uncertainty_boxes)

    h_nominal = make_root_hist(f"h_nominal_{feat}", "", edges, nominal)
    h_nominal.SetLineColor(ROOT.kGray + 2)
    h_nominal.SetLineWidth(2)

    if have_uncertainty:
        h_nominal.SetFillStyle(0)
        h_nominal.SetFillColor(0)
    else:
        h_nominal.SetFillColorAlpha(ROOT.kGray + 1, 0.35)

    h_nominal.Draw("hist same")
    drawn_objects.append(h_nominal)

    for der in plot_derivatives:
        y_truth_left = map_right_to_left(coeff_truth[der], right_min, right_max, left_max)
        y_pred_left = map_right_to_left(coeff_pred[der], right_min, right_max, left_max)

        h_truth = make_root_hist(f"h_truth_{feat}_{str(der)}", "", edges, y_truth_left)
        h_pred = make_root_hist(f"h_pred_{feat}_{str(der)}", "", edges, y_pred_left)

        color = root_colors[der]

        h_truth.SetLineColor(color)
        h_truth.SetLineStyle(2)
        h_truth.SetLineWidth(3)
        h_truth.SetMarkerStyle(0)

        h_pred.SetLineColor(color)
        h_pred.SetLineStyle(1)
        h_pred.SetLineWidth(3)
        h_pred.SetMarkerStyle(0)

        h_truth.Draw("hist same")
        if not args.truth_only:
            h_pred.Draw("hist same")

        drawn_objects.extend([h_truth, h_pred])

    right_axis = ROOT.TGaxis(x_hi, 0.0, x_hi, left_max, right_min, right_max, 510, "+L")
    right_axis.SetTitle("Polynomial coefficient")
    right_axis.SetLabelFont(frame.GetYaxis().GetLabelFont())
    right_axis.SetTitleFont(frame.GetYaxis().GetTitleFont())
    right_axis.SetLabelSize(frame.GetYaxis().GetLabelSize())
    right_axis.SetTitleSize(frame.GetYaxis().GetTitleSize())
    right_axis.SetTitleOffset(1.15)
    right_axis.SetLineWidth(1)
    right_axis.Draw()

    drawn_objects.append(right_axis)

    canvas.RedrawAxis()
    canvas.Modified()
    canvas.Update()

    file_stub = os.path.join(out_dir, feat)
    canvas.SaveAs(file_stub + ".png")
    canvas.SaveAs(file_stub + ".pdf")
    canvas.Close()


# --------------------------------------------------------------------------------
# separate legend plot
# --------------------------------------------------------------------------------

legend_canvas = ROOT.TCanvas("c_legend", "legend", 1400, 900)
legend_canvas.SetLeftMargin(0.03)
legend_canvas.SetRightMargin(0.03)
legend_canvas.SetBottomMargin(0.03)
legend_canvas.SetTopMargin(0.10)

legend_frame = ROOT.TH1D("legend_frame", "", 1, 0.0, 1.0)
legend_frame.SetMinimum(0.0)
legend_frame.SetMaximum(1.0)
legend_frame.GetXaxis().SetLabelSize(0.0)
legend_frame.GetYaxis().SetLabelSize(0.0)
legend_frame.GetXaxis().SetTickLength(0.0)
legend_frame.GetYaxis().SetTickLength(0.0)
legend_frame.Draw("axis")

legend_entries = 2 * len(plot_derivatives)
if uncertainty_n_toys > 0:
    legend_entries += 2
else:
    legend_entries += 1

if legend_entries > 30:
    n_cols = 4
elif legend_entries > 18:
    n_cols = 3
elif legend_entries > 8:
    n_cols = 2
else:
    n_cols = 1

legend = ROOT.TLegend(0.03, 0.08, 0.97, 0.90)
legend.SetNColumns(n_cols)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.030)

legend_objects = [legend_frame, legend]

dummy_nominal = ROOT.TH1D("dummy_nominal", "", 1, 0.0, 1.0)
dummy_nominal.SetLineColor(ROOT.kGray + 2)
dummy_nominal.SetLineWidth(2)

if uncertainty_n_toys > 0:
    dummy_nominal.SetFillStyle(0)
    dummy_nominal.SetFillColor(0)
    legend.AddEntry(dummy_nominal, "SM nominal", "l")

    dummy_unc = ROOT.TH1D("dummy_unc", "", 1, 0.0, 1.0)
    dummy_unc.SetLineWidth(0)
    dummy_unc.SetLineColor(0)
    dummy_unc.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
    legend.AddEntry(dummy_unc, "68% nuisance interval", "f")
    legend_objects.extend([dummy_nominal, dummy_unc])

else:
    dummy_nominal.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
    legend.AddEntry(dummy_nominal, "SM distribution", "lf")
    legend_objects.append(dummy_nominal)

for der in plot_derivatives:
    label = derivative_label(der)
    color = root_colors[der]

    dummy_truth = ROOT.TH1D(f"dummy_truth_{str(der)}", "", 1, 0.0, 1.0)
    dummy_pred = ROOT.TH1D(f"dummy_pred_{str(der)}", "", 1, 0.0, 1.0)

    dummy_truth.SetLineColor(color)
    dummy_truth.SetLineStyle(2)
    dummy_truth.SetLineWidth(3)

    dummy_pred.SetLineColor(color)
    dummy_pred.SetLineStyle(1)
    dummy_pred.SetLineWidth(3)

    legend.AddEntry(dummy_truth, f"truth  {label}", "l")
    if not args.truth_only:
        legend.AddEntry(dummy_pred,  f"BIT  {label}",   "l")

    legend_objects.extend([dummy_truth, dummy_pred])

legend.Draw()

legend_canvas.Modified()
legend_canvas.Update()

legend_stub = os.path.join(out_dir, "legend")
legend_canvas.SaveAs(legend_stub + ".png")
legend_canvas.SaveAs(legend_stub + ".pdf")
legend_canvas.Close()

helpers.copyIndexPHP(out_dir)
syncer.sync()
print("Done.")
