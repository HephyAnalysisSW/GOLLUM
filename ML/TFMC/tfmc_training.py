#!/usr/bin/env python

# YAML-driven TFMC trainer with optional plotting handled in the training loop.

from __future__ import annotations
import os, sys, time, argparse, importlib, yaml, numpy as np, math
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from typing import List, Tuple

import tensorflow as tf

import common.user as user
import common.yaml_loader as yaml_loader
from data.plot_options import plot_options

from ML.TFMC.TFMC import TFMC
from ML.Scaler.Scaler import Scaler
from data.UIDSplitter import UIDSplitter

from tqdm import trange, tqdm
import math

# ---------------- args ----------------
p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="Classifier job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--small", action="store_true", help="Debug: only first shard")
p.add_argument("--epochs", type=int, default=None, help="Override epochs")
p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
# plotting
p.add_argument("--norm_plot", action="store_true", help="Only plot shapes.")
p.add_argument("--plot_probability", action="store_true", help="Plot probabilities instead of DCR.")
p.add_argument("--every", type=int, default=5, help="Plot every N epochs (default 5)")
args = p.parse_args()

# ---------------- load cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
cfg = yaml_loader.load_yaml(cfg_path)

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")
default_batch = defaults.get("batch_size", 65536)

def list_jobs_and_exit():
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "classifier" and j.get("framework") == "tfmc"]
    if not jobs:
        print("No TFMC classifier jobs found in YAML.")
        sys.exit(0)
    flags = []
    if args.overwrite: flags.append("--overwrite")
    if args.small: flags.append("--small")
    if args.epochs is not None: flags.append(f"--epochs {args.epochs}")
    if args.batch_size is not None: flags.append(f"--batch_size {args.batch_size}")
    if args.every != 5: flags.append(f"--every {args.every}")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}".strip())
    sys.exit(0)

if args.job is None:
    list_jobs_and_exit()

# ---------------- resolve job ----------------
job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
if job is None:
    raise RuntimeError(f"Job id '{args.job}' not found.")
if job.get("type") != "classifier" or job.get("framework") != "tfmc":
    raise RuntimeError(f"Job '{args.job}' is not a TFMC classifier job.")

J = job  # shorthand

# Data / model params from YAML
classes_names = list(J["data"]["classes"])
activation = J["model"].get("activation", "relu")
hidden_layers = J["model"].get("hidden_layers", [64,64,64])
dropout_rate = float(J["model"].get("dropout_rate", 0.0))
l1 = float(J["model"].get("l1", 0.0))
l2 = float(J["model"].get("l2", 0.0))
epochs = args.epochs if args.epochs is not None else int(J["optim"].get("epochs", 300))
phaseout_epochs = int(J["optim"].get("phaseout_epochs", 0))
lr = float(J["optim"].get("learning_rate", 1e-2))
batch_size = args.batch_size if args.batch_size is not None else int(J.get("runtime", {}).get("batch_size", default_batch))
# batches with same N for each class, oversampling underrepresented events 
stratified = bool(J.get("runtime", {}).get("stratified", False))
use_ic = bool(J.get("extras", {}).get("use_ic", True))
use_scaler = bool(J.get("extras", {}).get("use_scaler", True))
reweighting=bool(J.get("reweighting",True))
set_logit_priors = bool(J.get("set_logit_priors", False))
g_prior_l2_reg = float(J["model"].get("g_prior_l2", 0.0)) if set_logit_priors else 0.0

if not use_ic:
    if set_logit_priors:
        raise ValueError("Asking to set logit priors without asking to calculate event counts via use_ic. Confirm and run again.")
    if reweighting:
        raise ValueError("Asking to set reweight the event losses based on class weights without asking to them via use_ic. Confirm and run again.")

# ---------------- dirs ----------------

cfg_base = os.path.join( cfg.get("version", "default"), J['region'] )

model_dir = os.path.join(user.model_directory, cfg_base, "TFMC", J["id"])
plot_dir  = os.path.join(user.plot_directory,  "TFMC", cfg_base, J["id"])

from common.helpers import copyIndexPHP
copyIndexPHP( plot_dir )
if args.small:
    model_dir += "_small"
    plot_dir  += "_small"
if args.norm_plot:
    plot_dir += "_norm"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---------------- resolve loaders ----------------
samples_mod = importlib.import_module(module_samples)
loaders = []
for name in classes_names:
    if not hasattr(samples_mod, name):
        raise RuntimeError(f"Class loader '{name}' not found in {module_samples}.")
    loader = getattr(samples_mod, name)
    loader.setFeatures( J['features'] )
    loaders.append(loader)

sel  = job.get("selection", None)
sel_f= job.get("selection_features", [])
if sel:
    for loader in loaders:
        loader.addSelection( sel, sel_f)
        print("Added selection to loader: {sel} and selection_features {sel_f}")
    print(loader)

# Consistency: same feature_names across classes
feat_names = getattr(loaders[0], "feature_names", None)
if not feat_names:
    raise RuntimeError("First loader has no feature_names set.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != list(feat_names):
        raise RuntimeError("Feature mismatch across class loaders.")
input_dim = len(feat_names)

# Sample split, useful for training larger samples, e.g. full Run 2
n_split = int(J.get("runtime", {}).get("n_split", 1))
if n_split:
    for loader in loaders:
        loader.set_n_split(n_split)

# ---------------- UID splitting (YAML-driven, implemented in data/UIDSplitter.py) ----------------
UID_CFG = (J.get("splitting") or {})
uid_enabled   = bool(UID_CFG.get("enabled", False))
uid_fields    = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed      = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme    = (UID_CFG.get("scheme") or {})

uid_intervals = None
uid_splitter = None
# default for running without training-validation splitting
val_train_fraction_ratio = 1.0
if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    # build bucket intervals (inline; no extra helper, no extra checks)
    keys  = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)
    
    # same intervals as PNN (original use of splitting)
    # won't change key names for now
    train_key = "pnn_train"
    val_key = "pnn_val"
    
    # to take into account different batch sizes in training
    # to rescale training loss to same units as validation loss for plotting
    val_train_fraction_ratio = float(uid_scheme[val_key]["fraction"])/uid_scheme[train_key]["fraction"]
    train_interval = uid_intervals[train_key]
    val_interval   = uid_intervals[val_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] TFMC train split '{train_key}' -> {train_interval}")
    print(f"[UID] TFMC val   split '{val_key}' -> {val_interval}")

# ---------------- data iterator ----------------
def iterate_epoch(shard_limit: int | None = None, do_train_val_split: bool = True):
    """Yield one mixed batch (X, y1hot, w) by concatenating per-class shards."""
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        Xs, Ys, Ws = [], [], []
        Xs_tr, Ys_tr, Ws_tr = [], [], []
        Xs_val, Ys_val, Ws_val = [], [], []
        for ci, (name, L) in enumerate(zip(classes_names, loaders)):
            # Use materialize to fetch features and weights; views apply masks internally.
            X, O, w = L.materialize(shard=shard, what="fow", n=None)
            y = np.zeros((len(X), len(classes_names)), dtype=np.float32)
            y[:, ci] = 1.0

            if uid_enabled and do_train_val_split:
                # training and validation events from UID-based splitting mask
                obs_names = L.observer_names
                uid_idx = [obs_names.index(f) for f in uid_fields]
                O_uid = O[:, uid_idx]
                lo, hi = train_interval
                m_tr = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)

                lo, hi = val_interval
                m_val = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)

                Xs_tr.append(X[m_tr]), Ys_tr.append(y[m_tr]), Ws_tr.append(w[m_tr])
                Xs_val.append(X[m_val]), Ys_val.append(y[m_val]), Ws_val.append(w[m_val])
            else:
                Xs.append(X); Ys.append(y); Ws.append(w)

        if uid_enabled and do_train_val_split:
            X_tr = np.concatenate(Xs_tr, axis=0) if Xs_tr else np.empty((0, input_dim))
            y_tr = np.concatenate(Ys_tr, axis=0) if Ys_tr else np.empty((0, len(classes_names)))
            w_tr = np.concatenate(Ws_tr, axis=0) if Ws_tr else np.empty((0,))
        
            X_val = np.concatenate(Xs_val, axis=0) if Xs_val else np.empty((0, input_dim))
            y_val = np.concatenate(Ys_val, axis=0) if Ys_val else np.empty((0, len(classes_names)))
            w_val = np.concatenate(Ws_val, axis=0) if Ws_val else np.empty((0,))
            
            # shuffling input vectors
            idx_tr = np.random.permutation(len(X_tr))
            idx_val = np.random.permutation(len(X_val))

            yield X_tr[idx_tr], y_tr[idx_tr], w_tr[idx_tr], X_val[idx_val], y_val[idx_val], w_val[idx_val]
        
        else:
            X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, input_dim))
            y = np.concatenate(Ys, axis=0) if Ys else np.empty((0, len(classes_names)))
            w = np.concatenate(Ws, axis=0) if Ws else np.empty((0,))
            
            #yield X, y, w
            #idx = rng.permutation(len(X))
            #yield X[idx], y[idx], w[idx]
            idx = np.random.permutation(len(X))
            # "equal training and validation"
            # same output format for both cases
            yield X[idx], y[idx], w[idx], X[idx], y[idx], w[idx]

# (No explicit observer weight index anymore — weights come from materialize("w"))

# ---------------- load Scaler (by job id; set means/variances) ----------------
scaler_id = J["extras"].get("use_scaler", None)
if scaler_id:
    sj    = next(jj for jj in (cfg.get("jobs") or []) if jj.get("id") == scaler_id)
    sname = sj["output"]["filename"]
    spath = os.path.join(user.model_directory, cfg_base, "Scaler", sname)
    sc    = Scaler.load(spath)
    feature_means     = sc.feature_means
    feature_variances = sc.feature_variances
    print(f"Loaded Scaler {spath}")
else:
    feature_means     = np.zeros(input_dim, dtype=np.float64)
    feature_variances = np.ones(input_dim,  dtype=np.float64)
    print("No Scaler used (identity).")

# ---------------- IC: precompute sums from data once ----------------
if J["extras"].get("use_ic", True):
    print("Pre-compute inclusive cross sections.")
    weight_sums = np.zeros(len(classes_names), dtype=np.float64)
    # when do_train_val_split == False, loops over the entire sample
    # and "training" and "validation" partitions are the same
    # keeping output with 6 members to be able to split training and validation
    event_sums = np.zeros(len(classes_names), dtype=np.int64)
    for _, y, w, _, _, _ in iterate_epoch(shard_limit=None, do_train_val_split=False):
        if len(w) == 0 :
            continue
        w1 = np.asarray(w).reshape(-1)
        weight_sums += y.T @ w1
        event_sums += np.sum(y,axis=0,dtype=np.int64)
    weight_sum_dict = {name: float(s) for name, s in zip(classes_names, weight_sums)}
    event_sum_dict = {name: float(s) for name, s in zip(classes_names, event_sums)}
    print(f"Computed sum of weights per class: {weight_sum_dict}")
    print(f"Computed unweighted event numbers sum of 1 per class: {event_sum_dict}")
else:
    weight_sum_dict = {name: 1.0 for name in classes_names}
    print("No IC used (all weights = 1.0).")

# ---------------- plotting utils (in training loop only) ----------------
# Only plot features that are actually in the training feature list.
plot_feats = [f for f in feat_names if f in plot_options]
feat2col   = {f: i for i, f in enumerate(feat_names)}

def init_histograms(plot_features):
    h_true, h_pred, bins = {}, {}, {}
    for feat in plot_features:
        n, lo, hi = plot_options[feat]['binning']
        h_true[feat] = np.zeros((n, len(classes_names)), dtype=np.float64)
        h_pred[feat] = np.zeros((n, len(classes_names)), dtype=np.float64)
        bins[feat] = np.linspace(lo, hi, n+1)
    return h_true, h_pred, bins

def append_dcr_summary(h_true, h_pred, epoch, out_dir, label=None):
    """
    Append per-(feature, class) DCR shape MAD to a running text file.
    Format: epoch  label  feature  class  mad
    Directly readable without Python — use tail/Read to monitor convergence.
    MAD < 0.01 is well-converged; MAD > 0.05 after 50 epochs means shapes not learned.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "dcr_summary.txt")
    write_header = not os.path.exists(summary_path)
    tag = label if label else "all"
    with open(summary_path, "a") as f:
        if write_header:
            f.write("# epoch  label  feature  class  mad\n")
        for feat in h_true:
            true = h_true[feat]  # (n_bins, n_classes)
            pred = h_pred[feat]
            tn = true / (true.sum(axis=0, keepdims=True) + 1e-12)
            pn = pred / (pred.sum(axis=0, keepdims=True) + 1e-12)
            mad = np.abs(tn - pn).mean(axis=0)  # (n_classes,)
            for ci, cls in enumerate(classes_names):
                f.write(f"{epoch}  {tag}  {feat}  {cls}  {mad[ci]:.6f}\n")

def accumulate_histograms(h_true, h_pred, bins, X_raw, y_onehot, pred_dcrs, weights, plot_features, f2c):
    # Use column indices from feat_names
    for feat in plot_features:
        col = f2c[feat]
        vals = X_raw[:, col]
        edges = bins[feat]
        for c in range(len(classes_names)):
            ht, _ = np.histogram(vals, bins=edges, weights=weights * y_onehot[:, c])
            h_true[feat][:, c] += ht
    
            hp, _ = np.histogram(vals, bins=edges, weights=weights * pred_dcrs[:, c])
            h_pred[feat][:, c] += hp

def plot_convergence_root(true_h, pred_h, epoch, out_dir, feature_names, classes, label=None, probability=False):
    import common.syncer as syncer
    import ROOT, os
    ROOT.gStyle.SetOptStat(0)
    # Load TDR if available; ignore if missing
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    num_features = len(feature_names)
    num_classes = len(classes)
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta]

    for normalized in [False, True]:
        # work on copies to avoid in-place modifications across epochs
        th = {k: v.copy() for k, v in true_h.items()}
        ph = {k: v.copy() for k, v in pred_h.items()}

        # normalizes each class to unit area
        if args.norm_plot:
            for k,v in th.items():
                th[k] = th[k]/th[k].sum(axis=0)
                ph[k] = ph[k]/ph[k].sum(axis=0)

        # gives per-class fractions summed to 1
        if normalized:
            for feat in feature_names:
                tot_t = th[feat].sum(axis=1, keepdims=True)   # per-bin truth total over classes
                tot_p = ph[feat].sum(axis=1, keepdims=True)   # per-bin pred  total over classes
                tot_t = np.where(tot_t == 0, 1, tot_t)
                tot_p = np.where(tot_p == 0, 1, tot_p)
                th[feat] /= tot_t
                ph[feat] /= tot_p

        total_pads = num_features + 1
        gx = int(math.ceil(math.sqrt(total_pads)))
        gy = int(math.ceil(total_pads / gx))
        canvas = ROOT.TCanvas("c_convergence", "Convergence Plot", 500*gx, 500*gy)
        canvas.Divide(gx, gy)

        stuff = []
        for i, feat in enumerate(feature_names):
            pad = canvas.cd(i + 1)
            pad.SetTicks(1, 1)
            pad.SetBottomMargin(0.15)
            pad.SetLeftMargin(0.15)

            n_bins, x_min, x_max = plot_options[feat]["binning"]
            x_axis_title = plot_options[feat]["tex"]
            max_y = 0
            max_y = max(max_y, th[feat].max(), ph[feat].max())
            min_y = max(0,min(th[feat].min(), ph[feat].min()))
            pad.SetLogy((not normalized and plot_options[feat]['logY']) or normalized)

            legend_title = "DCR"
            if probability:
                legend_title = "Probability"
            hframe = ROOT.TH2F(f"hframe_{feat}", f";{x_axis_title};{legend_title}", n_bins, x_min, x_max, 100, 0, 1.2*max_y if max_y>0 else 1.)
            hframe.GetYaxis().SetTitleOffset(1.3)
            hframe.Draw()
            stuff.append(hframe)

            for c in range(num_classes):
                # true dashed
                ht = ROOT.TH1F(f"t_{feat}_{c}", "", n_bins, x_min, x_max)
                for b, y in enumerate(th[feat][:, c]): ht.SetBinContent(b+1, y)
                ht.SetLineColor(colors[c % len(colors)]); ht.SetLineStyle(2); ht.SetLineWidth(2); ht.Draw("HIST SAME")
                stuff.append(ht)
                # pred solid
                hp = ROOT.TH1F(f"p_{feat}_{c}", "", n_bins, x_min, x_max)
                for b, y in enumerate(ph[feat][:, c]): hp.SetBinContent(b+1, y)
                hp.SetLineColor(colors[c % len(colors)]); hp.SetLineStyle(1); hp.SetLineWidth(2); hp.Draw("HIST SAME")
                stuff.append(hp)

        # legend pad
        canvas.cd(num_features + 1)
        leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9); leg.SetBorderSize(0); leg.SetShadowColor(0)
        dtrue, dpred = [], []
        for c, name in enumerate(classes):
            ht = ROOT.TH1F(f"dt_{c}", "", 1, 0, 1); ht.SetLineColor(colors[c % len(colors)]); ht.SetLineStyle(2); ht.SetLineWidth(2)
            hp = ROOT.TH1F(f"dp_{c}", "", 1, 0, 1); hp.SetLineColor(colors[c % len(colors)]); hp.SetLineStyle(1); hp.SetLineWidth(2)
            dtrue.append(ht); dpred.append(hp)
            leg.AddEntry(ht, f"{name} (true)", "l"); leg.AddEntry(hp, f"{name} (pred)", "l")
        leg.Draw()
        tex = ROOT.TLatex(); tex.SetNDC(); tex.SetTextSize(0.07); tex.SetTextAlign(11)
        tex.DrawLatex(0.3, 0.95, f"Epoch = {epoch:5d}")

        fname = os.path.join(out_dir, f"{'norm_' if normalized else ''}epoch_{epoch:04d}.png")
        if label:
            fname = fname.replace(".png",f"_{label}.png")
        if probability:
            fname = fname.replace(".png","_prob.png")
        canvas.SaveAs(fname)
        canvas.SaveAs(fname.replace(".png",".pdf"))
    syncer.sync()

# ---------------- resume if available ----------------
model = None
if not args.overwrite:
    if os.path.exists(os.path.join(model_dir,"done")):
        raise Exception("Training finished properly and rerunning without --overwrite. Will stop here.")
    else:
        print(f"Trying to load TFMC from {model_dir}")
        try:
            model = TFMC.load(model_dir, latest_filename="last_checkpoint")
            print("Found model with unfinished training! Will continue training from the latest epoch.")
        except Exception as e:
            print("Did not find any model! Gonna train from scratch.")

# Build fresh model if not resumed
if model is None:
    model = TFMC(
        input_dim=input_dim,
        classes=classes_names,
        activation=activation,
        hidden_layers=hidden_layers,
        l1_reg=l1,
        l2_reg=l2,
        dropout_rate=dropout_rate,
        learning_rate=lr,
        n_epochs=epochs,
        n_epochs_phaseout=phaseout_epochs,
        reweighting=reweighting,
        set_logit_priors=set_logit_priors,
        g_prior_l2_reg = g_prior_l2_reg
    )
    model.set_scaler(feature_means, feature_variances)
    if use_ic:
        model.set_ic_weights_from_sums(classes_names, weight_sum_dict)

start_epoch = 0
last_epoch_txt = os.path.join(model_dir, "last_epoch.txt")

if not args.overwrite and os.path.exists(last_epoch_txt):
    with open(last_epoch_txt, "r") as f:
        last_epoch = int(f.read().strip())
    start_epoch = last_epoch + 1
    print(f"[RESUME] last_epoch={last_epoch} (from {last_epoch_txt}), start_epoch={start_epoch}")

loss_txt = os.path.join(model_dir,"loss_curve.txt")
if start_epoch == 0:
    with open(loss_txt, "w") as f:
        f.write("# epoch  lr  train_loss_rescaled  val_loss\n")

# A small pointer file for BEST (epoch + val_loss)
best_txt = os.path.join(model_dir, "best_checkpoint.txt")

# ---------------- EARLY STOPPING + BEST tracking (YAML-driven) ----------------
stopping = J.get("early_stopping", None)

if stopping is None:
    raise RuntimeError(f"TFMC job {J.get('id')} missing early_stopping config")

es_enabled = stopping.get("enabled", True)
patience = stopping.get("patience", 20)
min_delta = stopping.get("min_delta", 0.0)
mode = stopping.get("mode", "min").lower()
if mode not in ("min", "max"):
    raise ValueError(f"Invalid early_stopping mode: {mode}")
warmup_epochs = stopping.get("warmup_epochs", 0)

print(f"[EarlyStopping] enabled={es_enabled}, mode={mode}, patience={patience}, min_delta={min_delta}, warmup_epochs={warmup_epochs}")

def _is_improved(current: float, best: float) -> bool:
    # min: want current < best - min_delta
    # max: want current > best + min_delta
    if mode == "min":
        return current < (best - min_delta)
    else:
        return current > (best + min_delta)
    
best_val = float("inf") if mode == "min" else -float("inf")
best_epoch = -1
bad_epochs = 0

# resume historical BEST if exists, so continuing training keeps the old best
if not args.overwrite and os.path.exists(best_txt):
    try:
        with open(best_txt, "r") as f:
            line = f.read().strip()
        # TODO: in next round of trainings, implement access bad_epochs in this file
        # not having it will lead to at max `patience` epochs of unnecessary training
        if line:
            e, v = line.split()[:2]
            best_epoch = int(e)
            best_val = float(v)
            print(f"Found previous BEST: epoch={best_epoch}, val_loss={best_val} (from {best_txt})")
    except Exception:
        pass

# ---------------- stratified batch helper ----------------
def _stratified_batches(y_onehot: np.ndarray, per_class_bs: int) -> Tuple[List[int], List[int]]:
    
    """
    Returns a list with index arrays for each batch.
    Each batch contains `per_class_bs` events from every class,
    cycling (oversampling) minority classes as needed.
    Number of batches is set by the largest class so majority data is fully covered.
    """
    n_classes = y_onehot.shape[1]

    # ensuring deterministic stratified batch composition
    rng = np.random.default_rng(int(uid_seed))
    cls_idx = [rng.permutation(np.where(y_onehot[:, ci] == 1)[0])
               for ci in range(n_classes)]
    
    max_n = max((len(idx) for idx in cls_idx), default=0)
    if max_n == 0:
        return [], []
    n_batches = math.ceil(max_n / per_class_bs)

    batches = []
    inverse_oversampling_factors = []
    for b in range(n_batches):
        parts = []
        inv_factors = []
        
        for idx in cls_idx:

            n_c = len(idx)
            if n_c == 0:
                continue

            # minority classes are oversampled via 'wrap'
            tiled_batch = idx.take(np.arange(b*per_class_bs, (b+1)*per_class_bs), mode='wrap')
            parts.append(tiled_batch)

            inv_factors_cls = np.ones_like(tiled_batch) * n_c/(n_batches * per_class_bs)
            inv_factors.append(inv_factors_cls)

        if parts:
            
            combined = np.concatenate(parts)
            combined_inv_factors = np.concatenate(inv_factors)

            permutation = np.random.permutation(len(combined))

            batches.append(combined[permutation])
            inverse_oversampling_factors.append(combined_inv_factors[permutation])

    return batches, inverse_oversampling_factors

# ---------------- train ----------------
for epoch in trange(start_epoch, epochs, desc="Epoch", position=0):
    lr_now = float(model.lr_schedule(epoch).numpy())
    model.optimizer.learning_rate.assign(lr_now)

    # hist accumulation (only when plotting this epoch)
    do_plot = (epoch % args.every == 0)
    if do_plot:
        true_h_tr, pred_h_tr, bins = init_histograms(plot_feats)
        true_h_val, pred_h_val, _ = init_histograms(plot_feats)

    shard_limit = 1 if args.small else None
    seen = 0
    losses_train = []
    losses_val = []

    # iterate shards; for each mixed shard make a per-batch bar
    for X_tr, y_tr, w_tr, X_val, y_val, w_val in iterate_epoch(shard_limit=shard_limit):
        N = len(X_tr)
        N_val = len(X_val)
        if N == 0:
            continue
        
        eff_bs = N if batch_size == -1 else batch_size

        if stratified and batch_size != -1:
            per_class_bs = max(1, eff_bs // len(classes_names))
            # batching with equal number of events for each class
            # oversamples minority classes, applies inverse_oversample_factors
            # to downweight the oversampled events 
            batch_list_tr, inverse_oversample_factors_tr  = _stratified_batches(y_tr,  per_class_bs)
            batch_list_val, inverse_oversample_factors_val = _stratified_batches(y_val, per_class_bs)
            num_batches     = len(batch_list_tr)
            num_batches_val = max(1, len(batch_list_val))
        else:
            num_batches = math.ceil(N / eff_bs)

        with tqdm(total=num_batches, desc=f"e{epoch} batches", leave=False) as pbar:
            for b in range(num_batches):
                if stratified and batch_size != -1:
                    idx_tr  = batch_list_tr[b]
                    idx_val = batch_list_val[b % num_batches_val]
                    Xb_tr,  yb_tr,  wb_tr  = X_tr[idx_tr],  y_tr[idx_tr],  w_tr[idx_tr]*inverse_oversample_factors_tr[b]
                    Xb_val, yb_val, wb_val = X_val[idx_val], y_val[idx_val], w_val[idx_val]*inverse_oversample_factors_val[b % num_batches_val]
                else:
                    start = b * eff_bs
                    stop  = min(start + eff_bs, N)
                    Xb_tr,  yb_tr,  wb_tr  = X_tr[start:stop],  y_tr[start:stop],  w_tr[start:stop]
                    start_val = math.ceil(start * val_train_fraction_ratio)
                    stop_val  = min(math.ceil(stop * val_train_fraction_ratio), N_val)
                    Xb_val, yb_val, wb_val = X_val[start_val:stop_val], y_val[start_val:stop_val], w_val[start_val:stop_val]



                # train_on_batch performs weight updates
                # putting this here ensures epoch 0 plot
                # gives the state of the network at initialization
                # NB: it will mix the plot from the initialization, done in batch 0
                # with the plots after the gradient updates in epoch 0
                # done with further batches
                if do_plot:
                    values_tr  = model.predict(Xb_tr,  probability=args.plot_probability)
                    accumulate_histograms(true_h_tr,  pred_h_tr,  bins, Xb_tr,  yb_tr,  values_tr,  wb_tr,  plot_feats, feat2col)

                loss_train = model.train_on_batch(Xb_tr, yb_tr, wb_tr)
                losses_train.append(loss_train)

                # compute_loss computes loss without gradient updates
                # NB: train loss is obtained before model updates
                # and validation loss is obtained after model updates
                loss_val = model.compute_loss(Xb_val, yb_val, wb_val)
                losses_val.append(loss_val)

                # NB: plots for "epoch 0" done after the first model upgrade
                # does not match what's on the training loss curves (before update),
                # but matches what's on the validation loss curves (after update)
                if do_plot:
                    values_val = model.predict(Xb_val, probability=args.plot_probability)
                    accumulate_histograms(true_h_val, pred_h_val, bins, Xb_val, yb_val, values_val, wb_val, plot_feats, feat2col)

                pbar.set_postfix(loss=float(loss_train))
                pbar.update(1)

        seen += N


    mean_train_loss = np.mean(losses_train, dtype=float) if losses_train else float('nan')
    mean_val_loss = np.mean(losses_val, dtype=float) if losses_val else float('nan')
    tqdm.write(f"Epoch {epoch}/{epochs-1} - LR {lr_now:.6f}. Seen {seen} events, mean train loss {mean_train_loss:.4f}, mean train loss (rescaled) {mean_train_loss * val_train_fraction_ratio:.4f}  mean validation loss {mean_val_loss:.4f}")
    with open(loss_txt, "a") as f:
        # rescaling training loss to similar units as validation loss for plotting
        f.write(f"{epoch} {float(lr_now):.8g} {mean_train_loss * val_train_fraction_ratio:.8g} {mean_val_loss:.8g}\n")

    model.save(model_dir, epoch=epoch)
    with open(last_epoch_txt, "w") as f:
        f.write(f"{epoch}\n")    

    # ---------------- save/update BEST + early stopping ----------------
    improved = _is_improved(mean_val_loss, best_val)
    if improved:
        best_val = mean_val_loss
        best_epoch = epoch
        bad_epochs = 0

        # Update BEST pointer (checkpoint) in the SAME model_dir
        model.save(model_dir, epoch=epoch, is_best=True)

        # Write pointer file for humans (epoch + val_loss)
        with open(best_txt, "w") as f:
            f.write(f"{best_epoch} {best_val:.12g}\n")

        tqdm.write(f"[BEST] val_loss={best_val:.6g} @ epoch {best_epoch} -> updated checkpoint in {model_dir}")
    else:   
        bad_epochs += 1
        tqdm.write(f"[BEST] no improvement ({bad_epochs}/{patience}), best={best_val:.6g} @ {best_epoch}")

        if es_enabled and epoch >= warmup_epochs and bad_epochs >= patience:
            tqdm.write(f"[EarlyStopping] stop. best val_loss={best_val:.6g} @ epoch {best_epoch}")
            break

    if do_plot:
        plot_convergence_root(true_h_tr, pred_h_tr, epoch, plot_dir, list(plot_feats), classes_names, label="train", probability=args.plot_probability)
        plot_convergence_root(true_h_val, pred_h_val, epoch, plot_dir, list(plot_feats), classes_names, label="val", probability=args.plot_probability)
        append_dcr_summary(true_h_tr, pred_h_tr, epoch, plot_dir, label="train")
        append_dcr_summary(true_h_val, pred_h_val, epoch, plot_dir, label="val")

copyIndexPHP(plot_dir)
print(f"Done. Model stored in {model_dir}")
open(f"{model_dir}/done","w")

# Plot loss curve at the end of training
# Still updated consistently in the text file,
# in case one wants to monitor loss during the training
import subprocess
subprocess.run(
    ["python", "ML/TFMC/plot_loss_curve.py", "-i", loss_txt],
    cwd="/users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM",
    check=False,
)

# this will redo the dataloader materializing, which in theory is unnecessary to have in this script
# given that it is not so heavy, we keep it here for now
subprocess.run(
    ["python", "ML/TFMC/tfmc_training_closure_mpl.py", args.config, "--job", J["id"]],
    cwd="/users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM",
    check=False,
)

subprocess.run(
    ["python", "ML/TFMC/tfmc_training_closure_mpl.py", args.config, "--job", J["id"], "--norm_plot"],
    cwd="/users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM",
    check=False,
)
