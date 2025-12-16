#!/usr/bin/env python
from __future__ import annotations
import os, sys, time, argparse, importlib
import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.yaml_loader as yaml_loader
from ML.ICP.ICP import InclusiveCrosssectionParametrization
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

# ---------------- args ----------------
p = argparse.ArgumentParser(description="ICP training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="ICP job id to run")
p.add_argument("--overwrite", action="store_true", help="Overwrite saved ICP?")
p.add_argument("--small", action="store_true", help="Only first shard, for debugging")
args = p.parse_args()

# ---------------- load cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
cfg = yaml_loader.load_yaml(cfg_path)

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

# Required config (new style)
params       = list(J["parameters"])
combinations = [tuple(c) for c in J["combinations"]]  # list of tuples
bp_specs     = list(J["base_points"] or [])
if not bp_specs:
    raise RuntimeError(f"ICP job '{J['id']}' has no base_points defined.")

train_ratio  = bool(J.get("train_ratio", True))

# Extract base point coordinates and find nominal
base_points = []
nominal_base_point = None
for i, spec in enumerate(bp_specs):
    if "coords" not in spec:
        raise RuntimeError(f"ICP job '{J['id']}', base_points[{i}] has no 'coords'.")
    coords = tuple(spec["coords"])
    base_points.append(coords)
    if spec.get("nominal", False):
        if nominal_base_point is not None:
            raise RuntimeError(f"ICP job '{J['id']}' has multiple nominal base points.")
        nominal_base_point = coords

if nominal_base_point is None:
    raise RuntimeError(f"ICP job '{J['id']}' has no nominal base point (no 'nominal: true').")

# ---------------- output path ----------------
out = J.get("output", {}) or {}
cfg_base = os.path.join(cfg.get("version", "default"), job['region'])
filename = out.get("filename", f"ICP_{J['id']}.pkl")
model_directory = os.path.join(user.model_directory, cfg_base, "ICP")
os.makedirs(model_directory, exist_ok=True)
out_path = os.path.join(model_directory, filename)

print(
    f"ICP training \033[1m{J['id']}\033[0m "
    f"params={params} base_points={base_points} nominal={nominal_base_point}"
)

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

    # ---------------- resolve loaders with possible weight variations ----------------
    loaders: list[object] = []

    for i, spec in enumerate(bp_specs):
        loader_name = spec.get("loader", None)
        if loader_name is None:
            raise RuntimeError(f"ICP job '{J['id']}', base_points[{i}] has no 'loader'.")
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
            base_weights = list(base.weight_branches or [])
            root_loader = base

        elif isinstance(base, SelectionView):
            # start from override if present, else from its base loader
            if base._w_override is not None:
                base_weights = list(base._w_override)
            else:
                if not isinstance(base.base, RDataLoader):
                    raise RuntimeError(
                        f"SelectionView '{base.name}' has a non-RDataLoader base. "
                        "Layered views are not supported in this ICP job logic."
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
                    f"[ICP job {J.get('id', '<unknown>')}] weight '{w}' requested for removal "
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

        if add:
            if root_loader.observer_names is None:
                root_loader.observer_names = list(add)
            else:
                for b in add:
                    if b not in root_loader.observer_names:
                        root_loader.observer_names.append(b)

        # 5) Construct the effective loader for this base point
        if isinstance(base, RDataLoader):
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
            vname = f"{base.name}_wvar{i}"
            eff_loader = SelectionView(
                base=base.base,
                name=vname,
                selection_fn=base._selection_fns,     # keep existing selections
                feature_names=base._feature_names,
                observer_names=base._observer_names,
                selection_feature_names=base._sel_feats,
                weight=new_weights,
            )

        loaders.append(eff_loader)

    # Debug print of resolved loaders/views
    print(f"\nResolved loaders for ICP job '{J.get('id', '<unknown>')}':")
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
    
    # ---------------- materialize total weights per base point ----------------
    yields = {}
    for spec, loader in zip(bp_specs, loaders):
        coords = tuple(spec["coords"])
        loader_name = spec["loader"]

        total_w = 0.0
        # For views, len(loader) delegates to base loader; for loaders it's direct
        n_shards = len(getattr(loader, "base", loader))
        for shard in range(n_shards):
            (w,) = loader.materialize(shard=shard, what="w")
            if w.size:
                total_w += float(np.sum(w, dtype=np.float64))
            if args.small:
                break

        yields[coords] = total_w
        print(f"  base_point {coords}  loader={loader_name:>30s}  yield={total_w:.6e}")

    # Build ICP object from YAML config (no external HDF5/data loader coupling)
    icp = InclusiveCrosssectionParametrization(
        combinations=combinations,
        nominal_base_point=nominal_base_point,
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

