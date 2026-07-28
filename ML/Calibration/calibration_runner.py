#!/usr/bin/env python
"""Shared pipeline for BIT calibration checks (YAML-driven).

Holds the backend-agnostic boilerplate: CLI definition, config/job loading, UID-based
train/valid materialization of truth-weight matrices via a *provider*, loading the
trained BIT, and saving prediction/truth/labels to .npy files for downstream
calibration plotting.

The derivative-specific bits live in the entry scripts (``pdf_calibration.py`` /
``eft_calibration.py``). Each builds a provider (see
``common/derivative_providers.py``) exposing:

  - ``combinations``       : list of canonical derivative tuples, native column
                             order, including the nominal ``()``.
  - ``required_observers`` : observer branch names the provider needs.
  - ``truth_weight_matrix(G, w, observer_names)`` : returns an ``(N, M)`` matrix
                             of truth weights aligned to ``combinations``
                             (column 0 == nominal weight).
"""

from __future__ import annotations

import os
import sys
import argparse
import importlib
import logging
import math
from collections import Counter

import numpy as np
import pandas as pd

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common.user as user
import common.yaml_loader as yaml_loader
from data.UIDSplitter import UIDSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# --------------------------------------------------------------------------------
# CLI + config/job loading
# --------------------------------------------------------------------------------

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Shared argument parser for the calibration entry scripts."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
    p.add_argument("--small", action="store_true", help="Only first shard for debugging")
    return p


def load_cfg_and_job(args):
    """Load the YAML config and select the requested bit job.

    Returns ``(cfg, job, samples_mod)``. Lists jobs and exits if ``--job`` is omitted.
    """
    cfg_path = os.path.expanduser(os.path.expandvars(args.config))
    cfg = yaml_loader.load_yaml(cfg_path)
    defaults = cfg.get("defaults", {}) or {}
    module_samples = defaults.get("module_samples", "data.samples")

    if args.job is None:
        _list_jobs_and_exit(cfg, args)

    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
    if job is None or job.get("type") != "bit":
        raise RuntimeError(f"Job '{args.job}' not found or not type 'bit'.")

    samples_mod = importlib.import_module(module_samples)
    return cfg, job, samples_mod


def _list_jobs_and_exit(cfg, args):
    jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "bit"]
    if not jobs:
        print("No BIT jobs found.")
        sys.exit(0)
    flags = ["--small"] if args.small else []
    script = os.path.basename(sys.argv[0])
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}")
    sys.exit(0)


# --------------------------------------------------------------------------------
# UID splitting
# --------------------------------------------------------------------------------

def _uid_c2st_intervals(job):
    """Build the UID splitter and the 'c2st'/'c2st_val' bucket intervals for a job.

    Calibration always evaluates the split held out from BIT training (the
    'c2st_train'/'c2st_val' buckets), matching the convention used for C2ST checks.
    """
    split_cfg = job.get("splitting") or {}
    if not bool(split_cfg.get("enabled", False)):
        raise RuntimeError("Calibration requires job.splitting.enabled=True (UID splitting) to avoid data leakage.")

    uid_fields = split_cfg.get("uid_fields", ["run", "luminosityBlock", "event"])
    uid_seed = int(split_cfg.get("seed", 0))
    uid_n_buckets = int(split_cfg.get("n_buckets", 10000))
    uid_scheme = split_cfg.get("scheme") or {}

    uid_splitter = UIDSplitter(uid_fields=tuple(uid_fields), seed=uid_seed, n_buckets=uid_n_buckets)

    keys = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]
    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)

    if "c2st_train" not in uid_intervals or "c2st_val" not in uid_intervals:
        raise RuntimeError("splitting.scheme must define 'c2st_train' and 'c2st_val'.")
    
    # assumes they're always in the order c2st_train, c2st_valid
    c2st_interval = (uid_intervals["c2st_train"][0], uid_intervals["c2st_val"][1])

    logger.info("[UID] fields=%s seed=%d n_buckets=%d", uid_fields, uid_seed, uid_n_buckets)
    logger.info("[UID] scheme intervals: %s", uid_intervals)
    logger.info("[UID] BIT train split 'c2st_train'+'c2st_val' -> %s", c2st_interval)

    return uid_splitter, list(uid_fields), c2st_interval


# --------------------------------------------------------------------------------
# derivative labels
# --------------------------------------------------------------------------------

def _format_derivative(der) -> str:
    """Pretty label for coefficient combinations, e.g. ('c0','c0','c1') -> c0^2 * c1"""
    if len(der) == 0:
        return "()"
    counts = Counter(der)
    parts = [(v if counts[v] == 1 else f"{v}^{counts[v]}") for v in sorted(counts.keys())]
    return " * ".join(parts)


# --------------------------------------------------------------------------------
# main entry: materialize + save
# --------------------------------------------------------------------------------

def run_calibration(cfg, job, samples_mod, args, provider):
    """Materialize truth/prediction matrices for the trained BIT and save them as .npy.

    Always draws truth from ``provider``; the trained BIT ('BIT_best.pkl') supplies the
    prediction. This is the whole shared pipeline; the entry scripts only build
    ``provider``.
    """
    loader_name = job.get("process")
    if not hasattr(samples_mod, loader_name):
        raise RuntimeError(f"Loader/view '{loader_name}' not found in module {samples_mod.__name__}.")
    loader = getattr(samples_mod, loader_name)

    uid_splitter, uid_fields, c2st_interval = _uid_c2st_intervals(job)

    required_observers = list(provider.required_observers)
    required_observers += [f for f in uid_fields if f not in required_observers]

    loader.setFeatures(job["features"], observer_names=required_observers)
    feat_names = list(getattr(loader, "feature_names", []) or [])
    if not feat_names:
        raise RuntimeError("Loader has no feature_names.")
    obs_names = list(getattr(loader, "observer_names", []) or [])

    combos = list(provider.combinations)
    if () not in combos:
        raise RuntimeError("provider.combinations missing the nominal '()' entry.")

    runtime_cfg = job.get("runtime", {}) or {}
    n_split = int(runtime_cfg.get("n_split", 1))
    if n_split > 1 and len(loader._all_files) > 1:
        before_split = len(loader)
        loader.set_n_split(n_split)
        print(
            f"Using {len(loader)} file shards for training-data materialization "
            f"(was {before_split}, files={len(loader._all_files)})"
        )

    # ---------------- collect all data (single pass) ----------------
    def iterate_all():
        n_shards = 1 if args.small else len(loader)
        on2idx = {n: i for i, n in enumerate(obs_names)}
        uid_idx = [on2idx[f] for f in uid_fields]
        lo, hi = c2st_interval

        for shard in range(n_shards):
            X, G, w = loader.materialize(shard=shard, what="fow")
            O_uid = G[:, uid_idx]
            m_keep = uid_splitter.mask_from_np(O_uid, uid_fields, lo, hi)
            yield (
                X.astype(np.float32, copy=False),
                G.astype(np.float32, copy=False),
                w.astype(np.float32, copy=False),
                m_keep,
            )

    Xs, targets = [], []

    for X, G, w, m_keep in iterate_all():
        deriv_w = provider.truth_weight_matrix(G, w, obs_names)  # (N, M), aligned to combos

        if np.any(m_keep):
            Xs.append(X[m_keep])
            targets.append(deriv_w[m_keep])

    if not Xs:
        raise RuntimeError("No events selected set (empty UID c2st split).")

    X = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
    DER = np.concatenate(targets, axis=0) if len(targets) > 1 else targets[0]

    weights = {combos[i]: DER[:, i] for i in range(len(combos))}

    if args.small:
        n_max = len(X) // 30
        X = X[:n_max]
        weights = {k: v[:n_max] for k, v in weights.items()}

    # ---------------- load BIT model (no training) ----------------
    from ML.BIT.NumbaBIT import MultiBoostedInformationTree

    cfg_base = os.path.join(cfg.get("version", "default"), job["region"])
    model_dir = os.path.join(user.model_directory, cfg_base, "BIT", job["id"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "BIT_best.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BIT model file not found: {model_path}")

    logger.info("Loading BIT from %s", model_path)
    bit = MultiBoostedInformationTree.load(model_path)
    logger.info("Loaded BIT.")

    ders_all = list(getattr(bit, "derivatives", []))
    ders = [d for d in ders_all if len(d) > 0]
    if not ders:
        raise RuntimeError("BIT has no non-trivial derivatives to evaluate.")

    der_labels = np.array([_format_derivative(d) for d in ders], dtype=object)

    def build_and_save(split_name: str, X: np.ndarray, w: dict):
        if X is None or len(X) == 0:
            logger.info("[CALIB] split=%s: empty, skip.", split_name)
            return

        if () not in w:
            raise RuntimeError(f"[CALIB] split={split_name}: truth_weights missing key ()")

        w0 = w[()]  # (N,)

        # ---- build truth in ratio space ----
        truth_cols = []
        for der in ders:
            col = w.get(der)
            if col is None:
                col = w.get(tuple(reversed(der)))
            if col is None:
                raise KeyError(f"[CALIB] split={split_name}: missing truth_weights for derivative {der}")
            truth_cols.append(col / w0)

        truth = np.stack(truth_cols, axis=1)  # (N, K)

        # ---- prediction ----
        pred = bit.predict(X)  # (N, K)

        if truth.shape != pred.shape:
            raise RuntimeError(
                f"[CALIB] shape mismatch truth {truth.shape} vs pred {pred.shape}"
            )

        logger.info("[CALIB] events: %d", truth.shape[0])
        logger.info("[CALIB] derivatives: %d", truth.shape[1])

        # ---- save as npy ----
        pred_f = pred.astype(np.float32, copy=False)
        truth_f = truth.astype(np.float32, copy=False)
        w0_f = w0.astype(np.float32, copy=False)

        pred_path = os.path.join(model_dir, f"calib_pred.npy")
        truth_path = os.path.join(model_dir, f"calib_truth.npy")
        w0_path = os.path.join(model_dir, f"calib_w0.npy")
        label_path = os.path.join(model_dir, "calib_der_labels.npy")  # shared

        np.save(pred_path, pred_f)
        np.save(truth_path, truth_f)
        np.save(w0_path, w0_f)
        np.save(label_path, der_labels)

        logger.info("[CALIB]")
        logger.info("  pred  -> %s", pred_path)
        logger.info("  truth -> %s", truth_path)
        logger.info("  w0    -> %s", w0_path)
        logger.info("  labels-> %s", label_path)

        # ---- save as csv (one <label>_truth / <label>_pred column pair per derivative, plus weight) ----
        csv_labels, csv_data = [], []
        for j, label in enumerate(der_labels):
            csv_labels.append(f"{label}_truth")
            csv_data.append(truth_f[:, j])
            csv_labels.append(f"{label}_pred")
            csv_data.append(pred_f[:, j])
        csv_labels.append("weight")
        csv_data.append(w0_f)

        csv_path = os.path.join(model_dir, f"calib_prediction.csv")
        pd.DataFrame(csv_data, index=csv_labels).T.to_csv(csv_path, index=True)
        logger.info("  csv   -> %s", csv_path)

    # build + save
    build_and_save("c2st", X, weights)

    logger.info("Done.")
