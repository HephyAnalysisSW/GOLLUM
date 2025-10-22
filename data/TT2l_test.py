#!/usr/bin/env python3
"""
TT2l_reco.py — tiny, standalone loader using RDataLoader + grouped observables.
Runs without args by defaulting to user.training_data_dir + original subpath.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import awkward as ak

from RDataLoader import RDataLoader
import observables as obs
import sys
sys.path.insert(0, '..')
import common.user as user

DEFAULT_SUBPATH =  "training-ntuples-v7/MVA-training/PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal ROOT data loader for TT2l features")
    p.add_argument("--input", "-i", action="append", default=None,
                   help="Input path(s): file.root or directory with ROOT files. Can be passed multiple times.")
    p.add_argument("--tree", default="Events", help="TTree name (default: Events)")
    p.add_argument("--groups", default="all",
                   help=("Comma-separated groups: top_kinematics, lepton_kinematics, asymmetry, spin_correlation, basic_event; "
                         "use 'all' for all groups"))
    p.add_argument("--n", type=int, default=10000, help="Events to preview")
    p.add_argument("--n-split", type=int, default=1, help="Number of shards")
    p.add_argument("--split-strategy", choices=["files", "events"], default="events")
    p.add_argument("--strict-branches", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    return p.parse_args()


def resolve_feature_names(groups_arg: str) -> list[str]:
    if groups_arg.strip().lower() == "all":
        return obs.all_features()
    groups = [g.strip() for g in groups_arg.split(",") if g.strip()]
    return obs.resolve_groups(*groups)


def main() -> None:
    args = parse_args()

    inputs = args.input or [os.path.join(user.training_data_dir, DEFAULT_SUBPATH)]
    feature_names = resolve_feature_names(args.groups)
    observer_names = obs.observers()
    branches = observer_names + feature_names

    print("[info] inputs:", inputs)
    print("[info] groups:", args.groups)

    ldr = RDataLoader(
        input_paths=inputs,
        tree_name=args.tree,
        branches=branches,
        selection=None,
        n_split=args.n_split,
        splitting_strategy=args.split_strategy,
        strict_branches=args.strict_branches,
    )

    arr = ldr[args.shard]
    # If RDataLoader returns a dict of columns, wrap into a record array
    if not hasattr(arr, "fields"):
        arr = ak.Array(arr)

    n_total = len(arr)
    n = min(args.n, n_total)
    print(f"[info] shard {args.shard}: {n_total} events (showing first {n})")

    X = ldr.scalar_branches(arr, feature_names)[:n]
    G = ldr.scalar_branches(arr, ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"])[:n]

    print("[ok] features shape:", X.shape)
    print("[ok] generator shape:", G.shape)

    np.set_printoptions(suppress=True, linewidth=140, edgeitems=3)
    print("[head] features (first 5 rows):", X[:5])
    print("[head] generator (first 5 rows):", G[:5])

    if n_total > 0:
        i0 = 0
        show_k = min(12, len(feature_names))
        print("[event 0] observers:")
        for name in observer_names:
            print(f"  {name:>20}: {arr[name][i0] if name in arr.fields else None}")
        print(f"[event 0] first {show_k} features:")
        for name in feature_names[:show_k]:
            print(f"  {name:>20}: {arr[name][i0] if name in arr.fields else None}")
        if len(feature_names) > show_k:
            print(f"  ... ({len(feature_names) - show_k} more features not shown)")
        print("[check] NaNs in X:", int(np.isnan(X).sum()))


if __name__ == "__main__":
    main()

