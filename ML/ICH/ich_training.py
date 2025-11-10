#!/usr/bin/env python
from __future__ import annotations

"""
ich_training.py
YAML-driven training for Inclusive Cross Section (Histogram), ICH.

- Runs selected ICH job via --job ID
- If --job is omitted, prints runnable commands and exits(0)
- Output path:
    user.model_directory / <version>[/<region>] / "ICH" / <pdf.filename>
"""

import os
import sys
import time
import argparse
import importlib
from typing import Any, Dict, List, Tuple

import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.yaml_loader as yaml_loader
from pdf.PDFParametrization import PDFParametrization
from ML.ICH.ICH import InclusiveCrosssectionHistogram


# ---------------- args ----------------
p = argparse.ArgumentParser(description="Inclusive Cross Section (Histogram) training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="ICH job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model file?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
CFG = yaml_loader.load_yaml(cfg_path)
D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")


def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "ich"]
    if not jobs:
        print("No ICH jobs found.")
        sys.exit(0)
    flags: List[str] = []
    if args.overwrite:
        flags.append("--overwrite")
    if args.small:
        flags.append("--small")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}")
    sys.exit(0)


if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "ich":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'ich'.")

# ---------------- resolve loader ----------------
samples_mod = importlib.import_module(module_samples)
loader_name = J.get("process")
if not hasattr(samples_mod, loader_name):
    raise RuntimeError(f"Loader/view '{loader_name}' not found in module {module_samples}.")
L = getattr(samples_mod, loader_name)

# feature & observer names
feat_names = list(getattr(L, "feature_names", []) or [])
obs_names  = list(getattr(L, "observer_names", []) or [])

if not feat_names and not obs_names:
    raise RuntimeError(f"Loader '{loader_name}' has neither feature_names nor observer_names.")

# Generator observables must be present (same as BIT)
GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"]
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

# Map observable name -> (source, index)
# source is "feature" or "observer"
name2loc: Dict[str, Tuple[str, int]] = {}
for i, n in enumerate(feat_names):
    name2loc.setdefault(n, ("feature", i))
for i, n in enumerate(obs_names):
    # don't override features if overlap
    name2loc.setdefault(n, ("observer", i))

# ---------------- parse binning ----------------

# ---- read binning from YAML (supports 1D or 2D) ----
binning_spec = J.get("binning", [])
if not isinstance(binning_spec, list) or len(binning_spec) == 0:
    raise RuntimeError("ICH: 'binning' must be a non-empty list (of [name, edges])")

# Normalize to at most 2 axes
if len(binning_spec) > 2:
    raise RuntimeError("ICH: only 1D or 2D binning is supported")

axis_names: list[str] = []
bin_edges: list[np.ndarray] = []
for item in binning_spec:
    if not (isinstance(item, (list, tuple)) and len(item) == 2):
        raise RuntimeError("ICH: each binning entry must be [axis_name, edges]")
    ax_name, edges = item
    axis_names.append(str(ax_name))
    arr = np.asarray(edges, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise RuntimeError(f"ICH: edges for '{ax_name}' must be 1D with >=2 entries")
    if not np.all(np.diff(arr) > 0):
        raise RuntimeError(f"ICH: edges for '{ax_name}' must be strictly increasing")
    bin_edges.append(arr)

dims = len(bin_edges)  # 1 or 2
if dims == 1:
    hist_shape = (bin_edges[0].size - 1,)
else:
    hist_shape = (bin_edges[0].size - 1, bin_edges[1].size - 1)

for name in axis_names:
    if name not in name2loc:
        raise RuntimeError(f"Axis variable '{name}' not found in features nor observers of loader '{loader_name}'.")

# ---------------- PDF & combinations ----------------

pdf_cfg = J.get("pdf", {}) or {}
pdf_n   = int(pdf_cfg.get("cheb_n", 5))
pdf     = PDFParametrization(n=pdf_n)

variables   = list(pdf.variables)     # ['c0', ..., 'cN']
combinations = list(pdf.combinations) # [(), ('c0',),..., ('ci','cj'),...]

# ---------------- build ICH object ----------------

# version and optional region
cfg_base = CFG.get("version", "default")
region = J.get("region", None)
if region:
    cfg_base = os.path.join(cfg_base, region)

filename = pdf_cfg.get("filename", f"ICH_{loader_name}.pkl")
model_dir = os.path.join(user.model_directory, cfg_base, "ICH")
os.makedirs(model_dir, exist_ok=True)
out_path = os.path.join(model_dir, filename)

ich = None
if not args.overwrite and os.path.exists(out_path):
    try:
        print(f"Attempt to load ICH from {out_path}")
        ich = InclusiveCrosssectionHistogram.load(out_path)
        print("Loaded existing ICH.")
    except Exception as e:
        print(f"Failed to load existing ICH ({e}), retraining.")

if ich is None or args.overwrite:
    print(f"Training ICH for process \033[1m{loader_name}\033[0m.")
    t0 = time.time()

    ich = InclusiveCrosssectionHistogram(
        variables=variables,
        combinations=combinations,
        axis_names=axis_names,
        bin_edges=bin_edges,
        process=loader_name,
        selection=J.get("selection", None),
        note=None,
    )

    # index of generator observers in observer matrix
    on2idx = {n: i for i, n in enumerate(obs_names)}
    gx1_idx  = on2idx["Generator_x1"]
    gx2_idx  = on2idx["Generator_x2"]
    gid1_idx = on2idx["Generator_id1"]
    gid2_idx = on2idx["Generator_id2"]

    # Top-level extra selection (on features/observers) if configured
    selection_name = J.get("selection", None)
    sel_fn = None
    if selection_name is not None:
        sel_mod = importlib.import_module("common.selections")
        sel_fn = sel_mod.selections.get(selection_name, None)
        if sel_fn is None:
            raise RuntimeError(f"Selection '{selection_name}' not found in common/selections.py")

    n_shards = len(getattr(L, "base", L)) if hasattr(L, "base") else len(L)
    if args.small:
        n_shards = min(n_shards, 1)

    for shard in range(n_shards):
        # Always need features, observers, and weights
        X, G, w = L.materialize(shard=shard, what="fow", n=None)
        if X.size == 0:
            continue

        w = np.asarray(w, dtype=np.float64)

        # optional top-level mask
        if sel_fn is not None:
            mask = sel_fn(X)
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype != bool or mask.ndim != 1 or mask.shape[0] != X.shape[0]:
                    raise RuntimeError("Top-level selection must return a 1D boolean mask matching X length.")
                X = X[mask]
                G = G[mask]
                w = w[mask]
                if X.size == 0:
                    continue

        # axis values
        def _get_axis_vals(name: str):
            source, idx = name2loc[name]
            if source == "feature":
                return X[:, idx]
            else:
                return G[:, idx]

        axis1_vals = _get_axis_vals(axis_names[0])
        axis2_vals = _get_axis_vals(axis_names[1]) if len(axis_names) == 2 else None

        # generator inputs for derivatives
        x1  = G[:, gx1_idx]
        x2  = G[:, gx2_idx]
        id1 = G[:, gid1_idx]
        id2 = G[:, gid2_idx]

        # derivatives aligned with pdf.combinations (M columns)
        deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2)       # shape (N, M)
        deriv = np.asarray(deriv, dtype=np.float64)

        # treat derivatives as reweights and multiply by event weight
        weights_per_comb = deriv * w.reshape(-1, 1)                  # (N, M)

        # accumulate into ICH
        ich.accumulate(axis1_vals, weights_per_comb, axis2_vals)

    ich.finalize()
    ich.save(out_path)
    print(f"Written {out_path}")
    print(f"Training time: {time.time() - t0:.2f} s")

print(ich)

