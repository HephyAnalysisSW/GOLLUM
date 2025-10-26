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
import data.samples as samples  # sample or view by name

argParser = argparse.ArgumentParser(description="Scaler training")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite saved scaler?")
argParser.add_argument('--small', action='store_true', help="Only first shard, for debugging")
argParser.add_argument("--process", default="tt2l", help="Process/view name in data/samples.py (default: tt2l)")
argParser.add_argument("--selection", default=None, help="Optional extra selection applied on top (default: None)")
args = argParser.parse_args()

sel_descr = args.selection if args.selection is not None else "None"
print("Scaler training for process " + '\033[1m' + f"{args.process}" + '\033[0m' +
      " with extra selection " + '\033[1m' + f"{sel_descr}" + '\033[0m')

# Output path
model_directory = os.path.join(common.user.model_directory, "Scaler")
os.makedirs(model_directory, exist_ok=True)

name_parts = [args.process]
if args.selection is not None:
    name_parts.append(args.selection)
scaler_name = "Scaler_" + "_".join(name_parts)
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

    # Resolve loader (sample or view) by name from data/samples.py
    if not hasattr(samples, args.process):
        raise RuntimeError(f"Process/view '{args.process}' not found in data/samples.py")
    loader_obj = getattr(samples, args.process)

    # Distinguish between a base RDataLoader and a SelectionView (from data/samples.py)
    is_view = hasattr(loader_obj, "base") and hasattr(loader_obj, "selection_fn")
    base_loader = loader_obj.base if is_view else loader_obj

    # Prepare scaler metadata
    scaler = Scaler()
    scaler.process = args.process
    scaler.selection = args.selection if args.selection is not None else ""
    feature_names_meta = getattr(loader_obj, "feature_names", None) or getattr(base_loader, "feature_names", None)
    if not feature_names_meta:
        raise RuntimeError("Loader has no feature_names set; set feature_names in RDataLoader constructor.")
    scaler.feature_names = list(feature_names_meta)

    # Strict: require 'weight' in observers
    observer_names_meta = list(getattr(loader_obj, "observer_names", None) or getattr(base_loader, "observer_names", None) or [])
    if "weight" not in observer_names_meta:
        raise RuntimeError("Observer 'weight' is required but not present in loader.observer_names.")
    w_idx = observer_names_meta.index("weight")

    # Optional extra selection on top (expects boolean mask on X). Default: None (no extra cut)
    sel_fn = None
    if args.selection is not None:
        from common import selections as selections_mod
        sel_fn = selections_mod.selections.get(args.selection, None)
        if sel_fn is None:
            raise RuntimeError(f"Selection '{args.selection}' not found in common/selections.py")

    # Loop shards and accumulate
    n_events = 0
    n_shards = len(base_loader)
    for shard in range(n_shards):
        if is_view:
            # Compute (and cache) mask for the view on this shard using the base loader
            mask = base_loader.compute_mask(
                selection_name=getattr(loader_obj, "name", args.process),
                selection_fn=loader_obj.selection_fn,
                shard=shard,
                observer_names=observer_names_meta,
            )
            # Materialize masked features & observers from the cached shard
            X = base_loader.features_from_mask(
                shard=shard, mask=mask, feature_names=feature_names_meta
            )
            G = base_loader.observers_from_mask(
                shard=shard, mask=mask, observer_names=observer_names_meta
            )
        else:
            # Base sample: fetch both in one call
            X, G = base_loader.features_and_observers(shard=shard, n=None)

        # Extract weights
        w = G[:, w_idx].astype(np.float64, copy=False)

        # Apply optional extra selection on top (if provided) — selection is a mask on X
        if sel_fn is not None:
            mask_top = sel_fn(X)
            if isinstance(mask_top, np.ndarray) and mask_top.dtype == bool and mask_top.ndim == 1 and len(mask_top) == len(X):
                X = X[mask_top]
                w = w[mask_top]
            else:
                # If a malformed selection is returned, raise explicitly
                raise RuntimeError("Extra selection did not return a valid 1D boolean mask.")

        scaler.accumulate(X, w)
        n_events += len(X)
        if args.small:
            break
    print(f"Used {n_events} Events.")
    scaler.finalize()
    scaler.save(filename)
    print(f"Written {filename}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(scaler)

