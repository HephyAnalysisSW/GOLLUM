#!/usr/bin/env python

# YAML-driven TFMC trainer:
# - If --job omitted: list runnable classifier jobs and exit(0)
# - Else: run selected job id, loop data here (not in TFMC), no plotting inside TFMC

from __future__ import annotations
import os, sys, time, argparse, importlib, yaml, numpy as np, math, random

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from ML.TFMC.TFMC import TFMC
from ML.Scaler.Scaler import Scaler
from ML.IC.IC import InclusiveCrosssection

# -------- args --------
p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="Classifier job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--small", action="store_true", help="Debug: only first shard")
p.add_argument("--epochs", type=int, default=None, help="Override epochs")
p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
p.add_argument("--seed", type=int, default=1, help="Random seed")
args = p.parse_args()

rng = np.random.default_rng(args.seed)
random.seed(args.seed)

# -------- load cfg --------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")
observer_weight = defaults.get("observer_weight", "weight")
default_batch = defaults.get("batch_size", 65536)
default_seed = defaults.get("seed", 1)

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
    flags.append(f"--seed {args.seed}")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}".strip())
    sys.exit(0)

if args.job is None:
    list_jobs_and_exit()

# -------- resolve job --------
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

# Output directory: user.model_directory / "TFMC" / <job id>
model_dir = os.path.join(user.model_directory, "TFMC", J["id"])
if args.small:
    model_dir += "_small"
os.makedirs(model_dir, exist_ok=True)

# -------- resolve loaders --------
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

# Weight column index (must exist)
observer_names = getattr(loaders[0], "observer_names", None)
if not observer_names or observer_weight not in observer_names:
    raise RuntimeError(f"Observer '{observer_weight}' not found in first loader.observer_names.")
w_idx = observer_names.index(observer_weight)

# -------- load IC + Scaler artifacts (from common.user.model_directory) --------
if use_scaler:
    # pick a scaler per class? your current setup uses one per process; here we allow per-class scaler
    # resolve from dependencies if present; else fallback by name convention
    dep_ids = list(J.get("extras", {}).get("depends_on", []))
    dep_scalers = [d for d in dep_ids if d.startswith("scaler_")]
    scalers = {}
    if dep_scalers:
        # need filenames to load; read from YAML
        for dep in dep_scalers:
            dep_job = next((jj for jj in cfg.get("jobs", []) if jj.get("id") == dep), None)
            if not dep_job:
                raise RuntimeError(f"Dependency '{dep}' not found in YAML.")
            fname = dep_job.get("output", {}).get("filename")
            if not fname:
                raise RuntimeError(f"Dependency '{dep}' missing output.filename.")
            proc = dep_job["process"]
            path = os.path.join(user.model_directory, "Scaler", fname)
            scalers[proc] = Scaler.load(path)
    else:
        # single shared scaler per class name by convention
        scalers = {}
        for name in classes_names:
            fname = f"Scaler_{name}.pkl"
            path = os.path.join(user.model_directory, "Scaler", fname)
            scalers[name] = Scaler.load(path)

    # Check same feature order
    s0 = scalers[classes_names[0]]
    if list(s0.feature_names) != list(feat_names):
        raise RuntimeError("Scaler feature_names do not match loader feature_names.")
    feature_means = s0.feature_means
    feature_variances = s0.feature_variances
else:
    feature_means = np.zeros(input_dim, dtype=np.float64)
    feature_variances = np.ones(input_dim, dtype=np.float64)

if use_ic:
    dep_ids = list(J.get("extras", {}).get("depends_on", []))
    dep_ics = [d for d in dep_ids if d.startswith("ic_")]
    ic_weights = {}
    if dep_ics:
        for dep in dep_ics:
            dep_job = next((jj for jj in cfg.get("jobs", []) if jj.get("id") == dep), None)
            if not dep_job:
                raise RuntimeError(f"Dependency '{dep}' not found in YAML.")
            fname = dep_job.get("output", {}).get("filename")
            if not fname:
                raise RuntimeError(f"Dependency '{dep}' missing output.filename.")
            proc = dep_job["process"]
            path = os.path.join(user.model_directory, "IC", fname)
            ic = InclusiveCrosssection.load(path)
            ic_weights[proc] = ic.total_weight
    else:
        for name in classes_names:
            fname = f"IC_{name}.pkl"
            path = os.path.join(user.model_directory, "IC", fname)
            ic = InclusiveCrosssection.load(path)
            ic_weights[name] = ic.total_weight
else:
    ic_weights = {name: 1.0 for name in classes_names}

# -------- build model --------
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

# -------- training loop (iterate over shards; batch inside) --------
def iterate_epoch(shard_limit: int | None = None):
    """Yield one mixed batch (X, y1hot, w) by concatenating per-class shards."""
    # assume all classes have same shard count (same base), else take min
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        Xs, Ys, Ws = [], [], []
        for ci, (name, L) in enumerate(zip(classes_names, loaders)):
            if hasattr(L, "features_and_observers"):
                X, G = L.features_and_observers(shard=shard, n=None)
            else:
                base = getattr(L, "base", L)
                X, G = base.features_and_observers(shard=shard, n=None)
                if hasattr(L, "mask"):
                    m = np.asarray(L.mask(shard))
                    if m.dtype != bool or m.ndim != 1 or len(m) != len(X):
                        raise RuntimeError("View mask must be 1D bool and match rows.")
                    X, G = X[m], G[m]
            w = G[:, w_idx].astype(np.float64, copy=False)
            y = np.zeros((len(X), len(classes_names)), dtype=np.float32)
            y[:, ci] = 1.0
            Xs.append(X); Ys.append(y); Ws.append(w)
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, input_dim))
        y = np.concatenate(Ys, axis=0) if Ys else np.empty((0, len(classes_names)))
        w = np.concatenate(Ws, axis=0) if Ws else np.empty((0,))
        # shuffle
        idx = rng.permutation(len(X))
        yield X[idx], y[idx], w[idx]

# resume?
start_epoch = 0
if not args.overwrite:
    try:
        latest = tf.train.latest_checkpoint(model_dir)
        if latest:
            start_epoch = int(os.path.basename(latest)) + 1
            model = TFMC.load(model_dir)  # reload full state
            print(f"Resuming from epoch {start_epoch}.")
    except Exception:
        pass

for epoch in range(start_epoch, epochs):
    # update LR
    lr_now = float(model.lr_schedule(epoch).numpy())
    model.optimizer.learning_rate.assign(lr_now)
    print(f"Epoch {epoch}/{epochs} - LR {lr_now:.6f}")

    shard_limit = 1 if args.small else None
    seen = 0
    losses = []
    for X, y, w in iterate_epoch(shard_limit=shard_limit):
        # batch within shard
        N = len(X)
        if N == 0:
            continue
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            loss = model.train_on_batch(X[start:stop], y[start:stop], w[start:stop])
            losses.append(loss)
        seen += N
    print(f"  seen {seen} events, mean loss {np.mean(losses) if losses else float('nan'):.4f}")

    # save each epoch
    model.save(model_dir, epoch=epoch)

print("Done.")

