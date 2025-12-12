#!/usr/bin/env python
from __future__ import annotations
import os, sys, time, argparse, importlib
import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.yaml_loader as yaml_loader
from ML.ICPH.ICPH import InclusiveCrosssectionParametrizationHistogram
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

# ---------------- args ----------------
p = argparse.ArgumentParser(description="ICPH training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="ICPH job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite saved ICPH?")
p.add_argument("--small", action="store_true", help="Only first shard, for debugging")
args = p.parse_args()

# ---------------- load cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
cfg = yaml_loader.load_yaml(cfg_path)

defaults = cfg.get("defaults", {}) or {}
module_samples = defaults.get("module_samples", "data.samples")

# ---------------- list mode ----------------
if args.job is None:
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "icph"]
    if not jobs:
        print("No ICPH jobs found in YAML.")
        sys.exit(0)
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} --job {j['id']}".strip())
    sys.exit(0)

# ---------------- resolve job ----------------
job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
if job is None:
    raise RuntimeError(f"Job id '{args.job}' not found.")
if job.get("type") != "icph":
    raise RuntimeError(f"Job '{args.job}' is not an ICPH job.")

J = job

params       = list(J["parameters"])
combinations = [tuple(c) for c in J["combinations"]]
bp_specs     = list(J["base_points"] or [])
if not bp_specs:
    raise RuntimeError(f"ICPH job '{J['id']}' has no base_points defined.")

# binning
binning_spec = J.get("binning", [])
if len(binning_spec) not in (1, 2):
    raise RuntimeError("ICPH: only 1D or 2D binning supported")
axis_names = [b[0] for b in binning_spec]
bin_edges  = [np.asarray(b[1], dtype=float) for b in binning_spec]

# Extract base point coordinates and find nominal
base_points = []
nominal_base_point = None
for i, spec in enumerate(bp_specs):
    coords = tuple(spec["coords"])
    base_points.append(coords)
    if spec.get("nominal", False):
        if nominal_base_point is not None:
            raise RuntimeError(f"ICPH job '{J['id']}' has multiple nominal base points.")
        nominal_base_point = coords
if nominal_base_point is None:
    raise RuntimeError(f"ICPH job '{J['id']}' has no nominal base point.")

# ---------------- output path ----------------
out = J.get("output", {}) or {}
cfg_base = os.path.join(cfg.get("version", "default"), job['region'])
filename = out.get("filename", f"ICPH_{J['id']}.pkl")
model_directory = os.path.join(user.model_directory, cfg_base, "ICPH")
os.makedirs(model_directory, exist_ok=True)
out_path = os.path.join(model_directory, filename)

print(
    f"ICPH training \033[1m{J['id']}\033[0m "
    f"params={params} base_points={base_points} nominal={nominal_base_point}"
)

# ---------------- try load ----------------
icph = None
if not args.overwrite:
    try:
        print(f"Trying to load from {out_path}")
        icph = InclusiveCrosssectionParametrizationHistogram.load(out_path)
    except (IOError, EOFError, ValueError, FileNotFoundError):
        icph = None

# ---------------- run training ----------------
if icph is None or args.overwrite:
    t0 = time.time()

    # ---------------- resolve loaders with possible weight variations ----------------
    samples_mod = importlib.import_module(module_samples)

    loaders: list[object] = []

    for i, spec in enumerate(bp_specs):
        loader_name = spec.get("loader", None)
        if loader_name is None:
            raise RuntimeError(f"ICPH job '{J['id']}', base_points[{i}] has no 'loader'.")
        if not hasattr(samples_mod, loader_name):
            raise RuntimeError(
                f"Loader/view '{loader_name}' not found in module '{module_samples}'."
            )

        base = getattr(samples_mod, loader_name)
        remove = list(spec.get("removeweights", []) or [])
        add    = list(spec.get("addweights", []) or [])

        # No weight modifications for this base point -> keep loader/view as-is
        if not remove and not add:
            loaders.append(base)
            continue

        # 1) Get starting weight list depending on loader type
        if isinstance(base, RDataLoader):
            # Start from the loader's own weight_branches
            base_weights = list(base.weight_branches or [])
            root_loader = base

        elif isinstance(base, SelectionView):
            # For a SelectionView:
            #   - if it has an override -> start from that
            #   - else inherit from its base loader's weight_branches
            if base._w_override is not None:
                base_weights = list(base._w_override)
            else:
                if not isinstance(base.base, RDataLoader):
                    raise RuntimeError(
                        f"SelectionView '{base.name}' has a non-RDataLoader base. "
                        "Layered views are not supported in this ICPH job logic."
                    )
                base_weights = list(base.base.weight_branches or [])
            if not isinstance(base.base, RDataLoader):
                raise RuntimeError(
                    f"Could not find underlying RDataLoader for SelectionView '{base.name}'."
                )
            root_loader = base.base

        else:
            raise RuntimeError(
                f"Loader/view '{loader_name}' has unsupported type {type(base)} "
                "for automatic weight variations."
            )

        new_weights = list(base_weights)

        # 2) Remove requested weights (warn if not present)
        for w in remove:
            if w in new_weights:
                new_weights.remove(w)
            else:
                import warnings
                warnings.warn(
                    f"[ICPH job {J.get('id', '<unknown>')}] weight '{w}' requested for removal "
                    f"but not found in loader '{loader_name}' (current weights: {base_weights})."
                )

        # 3) Add requested weights (avoid duplicates)
        for w in add:
            if w not in new_weights:
                new_weights.append(w)

        # 4) Ensure the underlying RDataLoader reads any new weight branches
        if hasattr(root_loader, "_requested_branches"):
            for b in add:
                if b not in root_loader._requested_branches:
                    root_loader._requested_branches.append(b)

        # Also make sure they appear in observer_names (so they can be materialized if needed)
        if root_loader.observer_names is None:
            root_loader.observer_names = list(add)
        else:
            for b in add:
                if b not in root_loader.observer_names:
                    root_loader.observer_names.append(b)

        # 5) Construct the effective loader for this base point
        if isinstance(base, RDataLoader):
            # Base is a loader -> make a simple view that only changes weights
            vname = f"{loader_name}_wvar{i}"
            eff_loader = SelectionView(
                base=base,
                name=vname,
                selection_fn=None,                    # no extra selection here
                feature_names=base.feature_names,
                observer_names=base.observer_names,
                selection_feature_names=None,
                weight=new_weights,
            )
        else:  # isinstance(base, SelectionView)
            # Base is a view -> copy its behavior, but adjust weights
            vname = f"{base.name}_wvar{i}"
            eff_loader = SelectionView(
                base=base.base,                       # directly use the loader as base
                name=vname,
                selection_fn=base._selection_fns,     # keep existing selections
                feature_names=base._feature_names,
                observer_names=base._observer_names,
                selection_feature_names=base._sel_feats,
                weight=new_weights,
            )

        loaders.append(eff_loader)

    # ---------------- debug print of resolved loaders/views ----------------
    print(f"\nResolved loaders for ICPH job '{J.get('id', '<unknown>')}':")
    for i, (spec, L) in enumerate(zip(bp_specs, loaders)):
        coords = spec["coords"]
        loader_name = spec["loader"]
        is_nominal = bool(spec.get("nominal", False))
        print(
            f"  base point {i}, coords={coords}, loader='{loader_name}', "
            f"nominal={is_nominal}"
        )
        print(L)
        print("-" * 60)

    # ---------------- materialize histograms per base point ----------------
    yields = {}
    for spec, loader in zip(bp_specs, loaders):
        coords = tuple(spec["coords"])
        axis1_name = axis_names[0]
        axis2_name = axis_names[1] if len(axis_names) == 2 else None

        n_shards = len(getattr(loader, "base", loader))
        hist_accum = None
        for i_shard, shard in enumerate(range(n_shards)):
            data = loader.materialize(shard=shard, what="fow")
            X, G, w = data
            vals1 = X[:, loader.feature_names.index(axis1_name)] if axis1_name in loader.feature_names else G[:, loader.observer_names.index(axis1_name)]

            n_nans =  len(np.unique(np.where(np.isnan(vals1))))
            if n_nans>0:
                print( f"Found {n_nans} NaN entries out of {len(X)} events!" )
                #raise RuntimeError( f"Found {n_nans} NaN entries out of {len(X)} events!" )

            vals2 = None
            if axis2_name is not None:
                vals2 = X[:, loader.feature_names.index(axis2_name)] if axis2_name in loader.feature_names else G[:, loader.observer_names.index(axis2_name)]
                n_nans =  len(np.unique(np.where(np.isnan(vals2))))
                if n_nans>0:
                    print ( f"Found {n_nans} NaN entries out of {len(X)} events!" )
                    #raise RuntimeError( f"Found {n_nans} NaN entries out of {len(X)} events!" )
            if len(axis_names) == 1:
                hist, _ = np.histogram(vals1, bins=bin_edges[0], weights=w)
            else:
                hist, _, _ = np.histogram2d(vals1, vals2, bins=(bin_edges[0], bin_edges[1]), weights=w)
            if hist_accum is None:
                hist_accum = hist
            else:
                hist_accum += hist
            if args.small:
                break
        yields[coords] = hist_accum

    icph = InclusiveCrosssectionParametrizationHistogram(
        combinations=combinations,
        nominal_base_point=nominal_base_point,
        base_points=base_points,
        parameters=params,
        axis_names=axis_names,
        bin_edges=bin_edges,
    )

    icph.train(yields=yields, small=args.small)
    icph.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(icph)

