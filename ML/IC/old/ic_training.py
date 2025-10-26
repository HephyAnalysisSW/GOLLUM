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
from ML.IC.IC import InclusiveCrosssection
import data.samples as samples  # sample or view by name

argParser = argparse.ArgumentParser(description="Inclusive cross section (IC) training")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite saved IC result?")
argParser.add_argument('--small', action='store_true', help="Only first shard, for debugging")
argParser.add_argument("--process", default="tt2l", help="Process/view name in data/samples.py (default: tt2l)")
argParser.add_argument("--selection", default=None, help="Optional extra selection applied on top (default: None)")
args = argParser.parse_args()

sel_descr = args.selection if args.selection is not None else "None"
print("IC training for process " + '\033[1m' + f"{args.process}" + '\033[0m' +
      " with extra selection " + '\033[1m' + f"{sel_descr}" + '\033[0m')

# Output path
model_directory = os.path.join(common.user.model_directory, "IC")
os.makedirs(model_directory, exist_ok=True)

name_parts = [args.process]
if args.selection is not None:
    name_parts.append(args.selection)
ic_name = "IC_" + "_".join(name_parts)
filename = os.path.join(model_directory, ('small_' if args.small else '') + ic_name) + '.pkl'

# Try load existing
ic = None
if not args.overwrite:
    try:
        print(f"Trying to load {ic_name} from {filename}")
        ic = InclusiveCrosssection.load(filename)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        ic = None

if ic is None or args.overwrite:
    print("Training.")
    t0 = time.time()

    # Resolve loader (sample or view) by name from data/samples.py
    if not hasattr(samples, args.process):
        raise RuntimeError(f"Process/view '{args.process}' not found in data/samples.py")
    loader = getattr(samples, args.process)

    # Determine base (for shard count) — views should expose .base
    base = getattr(loader, "base", loader)

    # Strict: require 'weight' in observers (on view or fallback to base)
    observer_names = list(getattr(loader, "observer_names", None) or getattr(base, "observer_names", []) or [])
    if "weight" not in observer_names:
        raise RuntimeError("Observer 'weight' is required but not present in loader/base.observer_names.")
    w_idx = observer_names.index("weight")

    # Optional extra selection on top (expects boolean mask on X). Default: None
    sel_fn = None
    if args.selection is not None:
        from common import selections as selections_mod
        sel_fn = selections_mod.selections.get(args.selection, None)
        if sel_fn is None:
            raise RuntimeError(f"Selection '{args.selection}' not found in common/selections.py")

    # Prepare IC object
    ic = InclusiveCrosssection()
    ic.process = args.process
    ic.selection = args.selection if args.selection is not None else ""

    # Loop shards and accumulate weight sums
    n_shards = len(base)
    for shard in range(n_shards):
        # Always fetch features and observers to allow optional selection
        X, G = (loader.features_and_observers(shard=shard, n=None)
                if hasattr(loader, "features_and_observers")
                else base.features_and_observers(shard=shard, n=None))

        # If this is a view without convenience method, apply its mask
        if not hasattr(loader, "features_and_observers") and hasattr(loader, "mask"):
            m = loader.mask(shard)
            m = np.asarray(m)
            if m.dtype != bool or m.ndim != 1 or len(m) != len(X):
                raise RuntimeError("View mask must be a 1D boolean array matching X length.")
            X = X[m]
            G = G[m]

        # Extract weights
        w = G[:, w_idx].astype(np.float64, copy=False)

        # Optional extra selection on top (boolean mask on X)
        if sel_fn is not None:
            mask = sel_fn(X)
            mask = np.asarray(mask) if mask is not None else None
            if mask is not None:
                if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(X):
                    raise RuntimeError("Top-level selection must return a 1D boolean mask matching X length.")
                X = X[mask]
                w = w[mask]

        ic.accumulate(w)

        if args.small:
            break

    ic.finalize()
    ic.save(filename)
    print(f"Written {filename}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(ic)

