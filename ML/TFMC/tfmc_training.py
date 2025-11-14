#!/usr/bin/env python

# YAML-driven TFMC trainer with optional plotting handled in the training loop.

from __future__ import annotations
import os, sys, time, argparse, importlib, yaml, numpy as np, math
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import tensorflow as tf

import common.user as user
import common.yaml_loader as yaml_loader
from data.plot_options import plot_options

from ML.TFMC.TFMC import TFMC
from ML.Scaler.Scaler import Scaler
from ML.IC.IC import InclusiveCrosssection

from tqdm import trange, tqdm
import math

# ---------------- args ----------------
p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="Classifier job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--small", action="store_true", help="Debug: only first shard")
p.add_argument("--epochs", type=int, default=None, help="Override epochs")
p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
# plotting
p.add_argument("--plot", action="store_true", help="Enable convergence plots")
p.add_argument("--plot-every", type=int, default=5, help="Plot every N epochs (default 5)")
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
    if args.batch_size is not None: flags.append(f"--batch-size {args.batch_size}")
    if args.plot: flags.append("--plot")
    if args.plot_every != 5: flags.append(f"--plot-every {args.plot_every}")
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
l1 = float(J["model"].get("regularization", {}).get("l1", 0.0))
l2 = float(J["model"].get("regularization", {}).get("l2", 0.0))
epochs = args.epochs if args.epochs is not None else int(J["optim"].get("epochs", 300))
phaseout_epochs = int(J["optim"].get("phaseout_epochs", 0))
lr = float(J["optim"].get("learning_rate", 1e-2))
batch_size = args.batch_size if args.batch_size is not None else int(J.get("runtime", {}).get("batch_size", default_batch))
use_ic = bool(J.get("extras", {}).get("use_ic", True))
use_scaler = bool(J.get("extras", {}).get("use_scaler", True))

# ---------------- dirs ----------------

cfg_base = os.path.join( cfg.get("version", "default"), job['region'] )

model_dir = os.path.join(user.model_directory, cfg_base, "TFMC", J["id"])
plot_dir  = os.path.join(user.plot_directory,  cfg_base, "TFMC", J["id"])

from common.helpers import copyIndexPHP
copyIndexPHP( plot_dir )
if args.small:
    model_dir += "_small"
    plot_dir  += "_small"
os.makedirs(model_dir, exist_ok=True)
if args.plot:
    os.makedirs(plot_dir, exist_ok=True)

# ---------------- resolve loaders ----------------
samples_mod = importlib.import_module(module_samples)
loaders = []
for name in classes_names:
    if not hasattr(samples_mod, name):
        raise RuntimeError(f"Class loader '{name}' not found in {module_samples}.")
    loaders.append(getattr(samples_mod, name))

# Consistency: same feature_names across classes
feat_names = getattr(loaders[0], "feature_names", None)
if not feat_names:
    raise RuntimeError("First loader has no feature_names set.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != list(feat_names):
        raise RuntimeError("Feature mismatch across class loaders.")
input_dim = len(feat_names)

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

# ---------------- load IC (by job id; broadcast to all classes) ----------------
ic_id = J["extras"].get("use_ic", None)
if ic_id:
    ij    = next(jj for jj in (cfg.get("jobs") or []) if jj.get("id") == ic_id)
    iname = ij["output"]["filename"]
    ipath = os.path.join(user.model_directory, cfg_base, "IC", iname)
    ic    = InclusiveCrosssection.load(ipath)
    w     = float(ic.total_weight)
    ic_weights = {name: w for name in classes_names}
    print(f"Loaded IC {ipath} (broadcast w={w})")
else:
    ic_weights = {name: 1.0 for name in classes_names}
    print("No IC used (all weights = 1.0).")

# ---------------- build model ----------------
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
    reweighting=True,
)
model.set_scaler(feature_means, feature_variances)
model.set_ic_weights_from_sums(classes_names, ic_weights)

# ---------------- data iterator ----------------
def iterate_epoch(shard_limit: int | None = None):
    """Yield one mixed batch (X, y1hot, w) by concatenating per-class shards."""
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        Xs, Ys, Ws = [], [], []
        for ci, (name, L) in enumerate(zip(classes_names, loaders)):
            # Use materialize to fetch features and weights; views apply masks internally.
            X, w = L.materialize(shard=shard, what="fw", n=None)
            y = np.zeros((len(X), len(classes_names)), dtype=np.float32)
            y[:, ci] = 1.0
            Xs.append(X); Ys.append(y); Ws.append(w)
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, input_dim))
        y = np.concatenate(Ys, axis=0) if Ys else np.empty((0, len(classes_names)))
        w = np.concatenate(Ws, axis=0) if Ws else np.empty((0,))
        idx = rng.permutation(len(X))
        yield X[idx], y[idx], w[idx]

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

def plot_convergence_root(true_h, pred_h, epoch, out_dir, feature_names, classes):
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
        if normalized:
            for feat in feature_names:
                tot = th[feat].sum(axis=1, keepdims=True)
                tot = np.where(tot == 0, 1, tot)
                th[feat] /= tot
                ph[feat] /= tot

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
            pad.SetLogy(not normalized and plot_options[feat]['logY'])

            n_bins, x_min, x_max = plot_options[feat]["binning"]
            x_axis_title = plot_options[feat]["tex"]
            max_y = 0
            max_y = max(max_y, th[feat].max(), ph[feat].max())
            hframe = ROOT.TH2F(f"hframe_{feat}", f";{x_axis_title};Probability", n_bins, x_min, x_max, 100, 0, 1.2*max_y if max_y>0 else 1.)
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
        canvas.SaveAs(fname)
    syncer.sync()

# ---------------- resume if available ----------------
start_epoch = 0
if not args.overwrite:
    try:
        latest = tf.train.latest_checkpoint(model_dir)
        if latest:
            start_epoch = int(os.path.basename(latest)) + 1
            model = TFMC.load(model_dir)
            print(f"Resuming from epoch {start_epoch}.")
    except Exception:
        print("Failed!")
        pass

# Build fresh model if not resumed
if 'model' not in locals():
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
        reweighting=True,
    )
    model.set_scaler(feature_means, feature_variances)
    if use_ic:
        model.set_ic_weights_from_sums(classes_names, ic_weights)

# ---------------- train ----------------
for epoch in trange(start_epoch, epochs, desc="Epoch", position=0):
    lr_now = float(model.lr_schedule(epoch).numpy())
    model.optimizer.learning_rate.assign(lr_now)

    # hist accumulation (only when plotting this epoch)
    do_plot = args.plot and (epoch % args.plot_every == 0)
    if do_plot:
        true_h, pred_h, bins = init_histograms(plot_feats)

    shard_limit = 1 if args.small else None
    seen = 0
    losses = []

    # iterate shards; for each mixed shard make a per-batch bar
    for X, y, w in iterate_epoch(shard_limit=shard_limit):
        N = len(X)
        if N == 0:
            continue

        eff_bs = N if batch_size == -1 else batch_size
        num_batches = math.ceil(N / eff_bs)

        with tqdm(total=num_batches, desc=f"e{epoch} batches", leave=False) as pbar:
            for start in range(0, N, eff_bs):
                stop = min(start + eff_bs, N)
                Xb, yb, wb = X[start:stop], y[start:stop], w[start:stop]
                loss = model.train_on_batch(Xb, yb, wb)
                losses.append(loss)

                if do_plot:
                    dcrs = model.predict(Xb, probability=False)
                    accumulate_histograms(true_h, pred_h, bins, Xb, yb, dcrs, wb, plot_feats, feat2col)

                pbar.set_postfix(loss=float(loss))
                pbar.update(1)

        seen += N

    tqdm.write(f"Epoch {epoch}/{epochs-1} - LR {lr_now:.6f}. Seen {seen} events, mean loss {np.mean(losses) if losses else float('nan'):.4f}")
    model.save(model_dir, epoch=epoch)

    if do_plot:
        plot_convergence_root(true_h, pred_h, epoch, plot_dir, list(plot_feats), classes_names)

print(f"Done. Model stored in {model_dir}")

