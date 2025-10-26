#!/usr/bin/env python

# ic_training_cfg.py
# YAML-driven IC trainer.
# - Runs selected IC job via --job ID
# - If --job is omitted, prints runnable commands and exits(0)
# - Output path = common.user.model_directory / "IC" / <filename from YAML>

import os
import sys
import time
import argparse
import importlib
import numpy as np
import yaml

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from ML.IC.IC import InclusiveCrosssection

# ---------------------- args ----------------------
p = argparse.ArgumentParser(description="Inclusive cross section (IC) training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="Job id to run (omit to list IC jobs)")
p.add_argument("--overwrite", action="store_true", help="Overwrite saved IC result?")
p.add_argument("--small", action="store_true", help="Only first shard, for debugging")
args = p.parse_args()

# ---------------------- load cfg ----------------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")
observer_weight = defaults.get("observer_weight", "weight")

# ---------------------- list mode ----------------------
if args.job is None:
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "ic"]
    if not jobs:
        print("No IC jobs found in YAML.")
        sys.exit(0)
    same_flags = []
    if args.overwrite:
        same_flags.append("--overwrite")
    if args.small:
        same_flags.append("--small")
    for j in jobs:
        print(f"python ic_training_cfg.py {args.config} {' '.join(same_flags)} --job {j['id']}".strip())
    sys.exit(0)

# ---------------------- run selected IC job ----------------------
job_id = args.job
jobs = cfg.get("jobs", [])
job = next((j for j in jobs if j.get("id") == job_id), None)
if job is None:
    raise RuntimeError(f"Job id '{job_id}' not found in YAML.")
if job.get("type") != "ic":
    raise RuntimeError(f"Job '{job_id}' is type '{job.get('type')}', but this script runs only 'ic' jobs.")

process_name = job.get("process")
if not process_name:
    raise RuntimeError("IC job is missing 'process' field.")

selection_name = job.get("selection", None)

# Output path: common.user.model_directory / "IC" / filename
out = job.get("output", {}) or {}
cfg_base = os.path.splitext(os.path.basename(cfg_path))[0]  # e.g. "../configs/no_reg.yaml" -> "no_reg"
filename = out.get("filename", f"IC_{process_name}.pkl")
model_directory = os.path.join(user.model_directory, cfg_base, "IC")
os.makedirs(model_directory, exist_ok=True)
out_path = os.path.join(model_directory, filename)

print(f"IC training for process \033[1m{process_name}\033[0m with extra selection \033[1m{selection_name or 'None'}\033[0m")

# Resolve loader (sample or view)
samples_mod = importlib.import_module(module_samples)
if not hasattr(samples_mod, process_name):
    raise RuntimeError(f"Process/view '{process_name}' not found in module '{module_samples}'.")
loader = getattr(samples_mod, process_name)
base = getattr(loader, "base", loader)

# Strict: require weight observer
observer_names = list(getattr(loader, "observer_names", None) or getattr(base, "observer_names", []) or [])
if observer_weight not in observer_names:
    raise RuntimeError(f"Observer '{observer_weight}' required but not found in loader/base.observer_names.")
w_idx = observer_names.index(observer_weight)

# Optional extra selection on top (boolean mask on features)
sel_fn = None
if selection_name is not None:
    sel_mod = importlib.import_module("common.selections")
    sel_fn = sel_mod.selections.get(selection_name, None)
    if sel_fn is None:
        raise RuntimeError(f"Selection '{selection_name}' not found in common/selections.py")

# Try load existing
ic = None
if not args.overwrite:
    try:
        print(f"Trying to load {job_id} from {out_path}")
        ic = InclusiveCrosssection.load(out_path)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        ic = None

if ic is None or args.overwrite:
    print("Training.")
    t0 = time.time()

    ic = InclusiveCrosssection()
    ic.process = process_name
    ic.selection = selection_name or ""

    n_shards = len(base)
    for shard in range(n_shards):
        # Views implement features_and_observers with mask applied
        if hasattr(loader, "features_and_observers"):
            X, G = loader.features_and_observers(shard=shard, n=None)
        else:
            X, G = base.features_and_observers(shard=shard, n=None)
            if hasattr(loader, "mask"):
                m = loader.mask(shard)
                m = np.asarray(m)
                if m.dtype != bool or m.ndim != 1 or len(m) != len(X):
                    raise RuntimeError("View mask must be a 1D boolean array matching X length.")
                X = X[m]
                G = G[m]

        w = G[:, w_idx].astype(np.float64, copy=False)

        if sel_fn is not None:
            mask = sel_fn(X)
            mask = np.asarray(mask) if mask is not None else None
            if mask is not None:
                if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(X):
                    raise RuntimeError("Top-level selection must return a 1D boolean mask matching X length.")
                w = w[mask]

        ic.accumulate(w)

        if args.small:
            break

    ic.finalize()
    ic.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(ic)

