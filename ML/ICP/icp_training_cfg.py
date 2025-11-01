#!/usr/bin/env python
from __future__ import annotations
import os, sys, time, argparse, importlib, yaml
import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.yaml_loader as yaml_loader
from ML.ICP.ICP import InclusiveCrosssectionParametrization

# ---------------- args ----------------
p = argparse.ArgumentParser(description="ICP training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="ICP job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite saved ICP?")
p.add_argument("--small", action="store_true", help="Only first shard, for debugging")
args = p.parse_args()

# ---------------- load cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
cfg = yaml_loader.load_yaml_recursive(cfg_path)

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")

# ---------------- list mode ----------------
if args.job is None:
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "icp"]
    if not jobs:
        print("No ICP jobs found in YAML.")
        sys.exit(0)
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} --job {j['id']}".strip())
    sys.exit(0)

# ---------------- resolve job ----------------
job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
if job is None:
    raise RuntimeError(f"Job id '{args.job}' not found.")
if job.get("type") != "icp":
    raise RuntimeError(f"Job '{args.job}' is not an ICP job.")

J = job  # shorthand

# Required config
params        = list(J["parameters"])
combinations  = [tuple(c) for c in J["combinations"]]  # list of tuples
bp_dict       = {int(k): tuple(v) for k, v in (J["base_points"] or {}).items()}
bp_loaders    = {int(k): str(v)   for k, v in (J["base_point_loaders"] or {}).items()}
nominal_index = int(J["nominal_index"])
train_ratio   = bool(J.get("train_ratio", True))

# Sort base points by key for consistent order
bp_keys_sorted = sorted(bp_dict.keys())
base_points    = [bp_dict[k] for k in bp_keys_sorted]

if nominal_index not in bp_dict:
    raise RuntimeError(f"nominal_index={nominal_index} not in base_points keys {sorted(bp_dict)}.")

# ---------------- output path ----------------
out = J.get("output", {}) or {}
cfg_base = os.path.join( cfg.get("version", "default"), job['region'] )
filename = out.get("filename", f"ICP_{J['id']}.pkl")
model_directory = os.path.join(user.model_directory, cfg_base, "ICP")
os.makedirs(model_directory, exist_ok=True)
out_path = os.path.join(model_directory, filename)

print(f"ICP training \033[1m{J['id']}\033[0m "
      f"params={params} base_points={base_points} nominal={bp_dict[nominal_index]}")

# ---------------- try load ----------------
icp = None
if not args.overwrite:
    try:
        print(f"Trying to load from {out_path}")
        icp = InclusiveCrosssectionParametrization.load(out_path)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        icp = None

# ---------------- run training ----------------
if icp is None or args.overwrite:
    t0 = time.time()

    # Resolve samples module
    samples_mod = importlib.import_module(module_samples)

    # Materialize total weights per base point (respecting each view’s weight override)
    yields = {}
    for k in bp_keys_sorted:
        loader_name = bp_loaders.get(k, None)
        if loader_name is None:
            raise RuntimeError(f"Missing base_point_loaders entry for key {k}.")
        if not hasattr(samples_mod, loader_name):
            raise RuntimeError(f"Loader/view '{loader_name}' not found in module '{module_samples}'.")
        loader = getattr(samples_mod, loader_name)

        # Sum weights across shards; ‘materialize(what="w")’ uses base/view weight logic
        total_w = 0.0
        n_shards = len(getattr(loader, "base", loader))
        for shard in range(n_shards):
            (w,) = loader.materialize(shard=shard, what="w")
            if w.size:
                total_w += float(np.sum(w, dtype=np.float64))
            if args.small:
                break

        yields[bp_dict[k]] = total_w
        print(f"  base_point {bp_dict[k]}  loader={loader_name:>30s}  yield={total_w:.6e}")

    # Build ICP object from YAML config (no external HDF5/data loader coupling)
    icp = InclusiveCrosssectionParametrization(
        combinations=combinations,
        nominal_base_point=bp_dict[nominal_index],
        base_points=base_points,
        parameters=params,
    )

    icp.train(
        small=args.small,
        train_ratio=train_ratio,
        yields=yields,
        selection=None,  # top-level feature selection (not used here)
    )

    icp.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(icp)

