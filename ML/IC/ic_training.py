#!/usr/bin/env python

# ic_training.py
# YAML-driven IC trainer.
# - Runs selected IC job via --job ID
# - If --job is omitted, prints runnable commands and exits(0)
# - Output path = common.user.model_directory / <cfg_base> / "IC" / <filename from YAML>

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
cfg = yaml_loader.load_yaml(cfg_path)

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")

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
        print(f"python ic_training.py {args.config} {' '.join(same_flags)} --job {j['id']}".strip())
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

# Output path: common.user.model_directory / <cfg_base> / "IC" / filename
out = job.get("output", {}) or {}
cfg_base = os.path.join( cfg.get("version", "default"), job['region'] )
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

sel  = job.get("selection", None)
sel_f= job.get("selection_features", [])
if sel:
    loader.addSelection( sel, sel_f)
    print("Added selection to loader: {sel} and selection_features {sel_f}")

print(loader)

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

    n_shards = len(loader if hasattr(loader, "__len__") else getattr(loader, "base", loader))
    for shard in range(n_shards):
        (w,) = loader.materialize(shard=shard, what="w", n=None)
        ic.accumulate(w)
        if args.small:
            break

    ic.finalize()
    ic.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(ic)

