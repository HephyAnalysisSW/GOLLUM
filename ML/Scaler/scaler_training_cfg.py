#!/usr/bin/env python

# scaler_training_cfg.py
# YAML-driven Scaler trainer.
# - Runs selected scaler job via --job ID
# - If --job is omitted, prints runnable commands and exits(0)
# - Output path = common.user.model_directory / <yaml-basename> / "Scaler" / <filename from YAML>

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
import common.yaml_loader as yaml_loader
from ML.Scaler.Scaler import Scaler

# ---------------------- args ----------------------
p = argparse.ArgumentParser(description="Scaler training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="Job id to run (omit to list scaler jobs)")
p.add_argument("--overwrite", action="store_true", help="Overwrite saved scaler?")
p.add_argument("--small", action="store_true", help="Only first shard, for debugging")
args = p.parse_args()

# ---------------------- load cfg ----------------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
cfg = yaml_loader.load_yaml_recursive(cfg_path)

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")

# ---------------------- list mode ----------------------
if args.job is None:
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "scaler"]
    if not jobs:
        print("No scaler jobs found in YAML.")
        sys.exit(0)
    same_flags = []
    if args.overwrite:
        same_flags.append("--overwrite")
    if args.small:
        same_flags.append("--small")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(same_flags)} --job {j['id']}".strip())
    sys.exit(0)

# ---------------------- run selected scaler job ----------------------
job_id = args.job
jobs = cfg.get("jobs", [])
job = next((j for j in jobs if j.get("id") == job_id), None)
if job is None:
    raise RuntimeError(f"Job id '{job_id}' not found in YAML.")
if job.get("type") != "scaler":
    raise RuntimeError(f"Job '{job_id}' is type '{job.get('type')}', but this script runs only 'scaler' jobs.")

process_name = job.get("process")
if not process_name:
    raise RuntimeError("Scaler job is missing 'process' field.")

selection_name = job.get("selection", None)

# Output path: common.user.model_directory / <yaml-basename> / "Scaler" / filename
out = job.get("output", {}) or {}
filename = out.get("filename", f"Scaler_{process_name}.pkl")
cfg_base = os.path.join( cfg.get("version", "default"), job['region'] )
model_directory = os.path.join(user.model_directory, cfg_base, "Scaler")
os.makedirs(model_directory, exist_ok=True)
out_path = os.path.join(model_directory, filename)

print(f"Scaler training for process \033[1m{process_name}\033[0m with extra selection \033[1m{selection_name or 'None'}\033[0m")

# Resolve loader (sample or view)
samples_mod = importlib.import_module(module_samples)
if not hasattr(samples_mod, process_name):
    raise RuntimeError(f"Process/view '{process_name}' not found in module '{module_samples}'.")
loader = getattr(samples_mod, process_name)
base = getattr(loader, "base", loader)

# Optional extra selection on top (boolean mask on features)
sel_fn = None
if selection_name is not None:
    sel_mod = importlib.import_module("common.selections")
    sel_fn = sel_mod.selections.get(selection_name, None)
    if sel_fn is None:
        raise RuntimeError(f"Selection '{selection_name}' not found in common/selections.py")

# Try load existing
scaler = None
if not args.overwrite:
    try:
        print(f"Trying to load {job_id} from {out_path}")
        scaler = Scaler.load(out_path)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        scaler = None

if scaler is None or args.overwrite:
    print("Training.")
    t0 = time.time()

    # Prepare scaler metadata
    scaler = Scaler()
    scaler.process = process_name
    scaler.selection = selection_name or ""
    feature_names_meta = getattr(loader, "feature_names", None) or getattr(base, "feature_names", None)
    if not feature_names_meta:
        raise RuntimeError("Loader has no feature_names set; set feature_names in RDataLoader constructor.")
    scaler.feature_names = list(feature_names_meta)

    n_events = 0
    n_shards = len(base)
    for shard in range(n_shards):
        # Fetch features and (functional/default) weights in one go.
        # NOTE: materialize signature is assumed (shard, order="fw", n=None, ...)
        X, w = loader.materialize(shard=shard, what="fw", n=None)

        # Optional extra selection on top (mask on features)
        if sel_fn is not None:
            mask_top = sel_fn(X)
            mask_top = np.asarray(mask_top) if mask_top is not None else None
            if mask_top is not None:
                if mask_top.dtype != bool or mask_top.ndim != 1 or len(mask_top) != len(X):
                    raise RuntimeError("Top-level selection must return a 1D boolean mask matching X length.")
                X = X[mask_top]
                w = w[mask_top]

        scaler.accumulate(X, w)
        n_events += len(X)
        if args.small:
            break

    print(f"Used {n_events} Events.")
    scaler.finalize()
    scaler.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(scaler)

