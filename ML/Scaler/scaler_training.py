#!/usr/bin/env python

import os
import sys
import time
import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import argparse
import common.user
from ML.Scaler.Scaler import Scaler

# selections and samples
from common import selections as selections_mod
import data.samples as samples  # contains e.g. tt2l RDataLoader

argParser = argparse.ArgumentParser(description="Scaler training")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite saved scaler?")
argParser.add_argument('--small', action='store_true', help="Only first shard, for debugging")
argParser.add_argument("--selection", default="inclusive", help="Selection name (default: inclusive)")
argParser.add_argument("--process", default="tt2l", help="Process/loader name in data/samples.py (default: tt2l)")
args = argParser.parse_args()

print("Scaler training for selection " + '\033[1m' + f"{args.selection}" + '\033[0m' +
      " and process " + '\033[1m' + f"{args.process}" + '\033[0m')

# Output path
model_directory = os.path.join(common.user.model_directory, "Scaler")
os.makedirs(model_directory, exist_ok=True)

scaler_name = "Scaler_" + "_".join([x for x in [args.process, args.selection] if x])
filename = os.path.join(model_directory, ('small_' if args.small else '') + scaler_name) + '.pkl'

# Try load existing
scaler = None
if not args.overwrite:
    try:
        print(f"Trying to load {scaler_name} from {filename}")
        scaler = Scaler.load(filename)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        scaler = None

if scaler is None or args.overwrite:
    print("Training.")
    t0 = time.time()

    # Resolve selection function (defaults to identity)
    sel_fn = selections_mod.selections.get(args.selection, selections_mod.selections["inclusive"])

    # Resolve RDataLoader by name from data/samples.py
    if not hasattr(samples, args.process):
        raise RuntimeError(f"Process '{args.process}' not found in data/samples.py")
    loader = getattr(samples, args.process)

    # Prepare scaler
    scaler = Scaler()
    scaler.selection = args.selection
    scaler.process = args.process
    # Record feature names for pretty __str__
    if not loader.feature_names:
        raise RuntimeError("Loader has no feature_names set; set feature_names in RDataLoader constructor.")
    scaler.feature_names = list(loader.feature_names)

    # Find weight column among observers (if present)
    weight_name = "weight"
    observer_names = list(loader.observer_names or [])
    w_idx = observer_names.index(weight_name) if weight_name in observer_names else None

    # Loop shards and accumulate
    n_shards = len(loader)
    for shard in range(n_shards):
        # Prefer getting both X and G at once if we have weights
        if w_idx is not None:
            X, G = loader.features_and_observers(shard=shard, n=None)
            w = G[:, w_idx].astype(np.float64, copy=False)
        else:
            X = loader.features(shard=shard, n=None)
            w = None

        # Apply selection: expect either a boolean mask, or pass-through
        try:
            sel = sel_fn(X)
            if isinstance(sel, np.ndarray) and sel.dtype == bool and sel.ndim == 1 and len(sel) == len(X):
                X = X[sel]
                if w is not None:
                    w = w[sel]
            # If selection returns X unchanged (inclusive), nothing to do
        except Exception:
            # Be permissive; inclusive returns input, others may be added later
            pass

        scaler.accumulate(X, w)

        if args.small:
            break

    # Finalize statistics and save
    scaler.finalize()
    scaler.save(filename)
    print(f"Written {filename}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(scaler)

