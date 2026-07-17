#!/usr/bin/env python
# SUPERSEDED: this script has been split into ML/Calibration/calibration_runner.py
# (shared pipeline) plus ML/Calibration/pdf_calibration.py and eft_calibration.py
# (thin PDF/EFT entry points), so calibration checks work for both PDF and EFT jobs.
# Use pdf_calibration.py in place of this file; kept here for reference only.
from __future__ import annotations
import os, sys, argparse, importlib
import numpy as np

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.yaml_loader as yaml_loader

from pdf.PDFParametrization import PDFParametrization

# ---------------- args ----------------
p = argparse.ArgumentParser(description="BIT evaluation (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
args = p.parse_args()

from ML.BIT.NumbaBIT import MultiBoostedInformationTree

from data.UIDSplitter import UIDSplitter
import math

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
CFG = yaml_loader.load_yaml(cfg_path)
D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "bit"]
    if not jobs:
        print("No BIT jobs found.")
        sys.exit(0)
    flags = []
    if args.small: flags.append("--small")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}")
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "bit":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'bit'.")

# ---------------- resolve loader ----------------
samples_mod = importlib.import_module(module_samples)
loader_name = J.get("process")
if not hasattr(samples_mod, loader_name):
    raise RuntimeError(f"Loader/view '{loader_name}' not found in module {module_samples}.")
L = getattr(samples_mod, loader_name)

# features
L.setFeatures(J["features"])
feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")
input_dim = len(feat_names)

# observers: must contain generator columns in this order
GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(
        f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'."
    )

# -------------------------- UID Splitting --------------------------
UID_CFG = (J.get("splitting") or {})
uid_enabled   = bool(UID_CFG.get("enabled", False))
uid_fields    = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed      = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme    = (UID_CFG.get("scheme") or {})

uid_intervals = None
uid_splitter = None
train_interval = None
val_interval   = None

if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    # build bucket intervals EXACTLY like PNN
    keys  = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)
    calibration_train_key = "c2st_train"
    calibration_val_key   = "c2st_val"
    train_interval = uid_intervals[calibration_train_key]
    val_interval   = uid_intervals[calibration_val_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] BIT train split '{calibration_train_key}' -> {train_interval}")
    print(f"[UID] BIT val   split '{calibration_val_key}' -> {val_interval}")

# ---------------- PDF parametrization & combinations ----------------
pdf_n     = J.get("pdf", {}).get("pdf_n", None)
pdf_type  = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
pdf_rescale_pod_amplitudes = J.get("pdf", {}).get("rescale_pod_amplitudes", True)
pdf = PDFParametrization(n=pdf_n, typ=pdf_type, basis=pdf_basis, rescale_pod_amplitudes=pdf_rescale_pod_amplitudes)

combos = list(pdf.combinations)                       # (), ('c0',), ..., ('ci','cj')
# Build base_points like the legacy script (order up to 2)
base_points = []
vars_ = pdf.variables
import itertools
for comb in itertools.combinations_with_replacement(vars_, 1):
    base_points.append({v: comb.count(v) for v in vars_})
for comb in itertools.combinations_with_replacement(vars_, 2):
    base_points.append({v: comb.count(v) for v in vars_})


# ---------------- collect all data (single pass) ----------------
def iterate_all(shard_limit=None):
    """
    Yields per-shard arrays PLUS (optionally) UID masks for train/val.
    - Always yields X,Q,x1,x2,id1,id2,w
    - If uid_enabled: also yields (m_tr, m_va) boolean masks with same length as X
    """
    n_shards = len(L)
    if args.small: n_shards = 1
    if shard_limit is not None: n_shards = min(n_shards, shard_limit)
    on2idx = {n: i for i, n in enumerate(obs_names)}

    # indices for generator obs (existing behavior) with shape (N,)
    i_Q   = on2idx["Generator_scalePDF"]
    i_x1  = on2idx["Generator_x1"]
    i_x2  = on2idx["Generator_x2"]
    i_id1 = on2idx["Generator_id1"]
    i_id2 = on2idx["Generator_id2"]

    uid_idx = [on2idx[f] for f in uid_fields]
    lo_tr, hi_tr = train_interval
    lo_va, hi_va = val_interval

    for shard in range(n_shards):
        X, G, w = L.materialize(shard=shard, what="fow")
        gQ   = G[:, i_Q]
        gx1  = G[:, i_x1]
        gx2  = G[:, i_x2]
        gid1 = G[:, i_id1]
        gid2 = G[:, i_id2]

        O_uid = G[:, uid_idx]  # shape (N, len(uid_fields))
        m_tr = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_tr, hi_tr)
        m_va = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_va, hi_va)
        yield (
            X.astype(np.float32, copy=False),
            gQ.astype(np.float32, copy=False),
            gx1.astype(np.float32, copy=False),
            gx2.astype(np.float32, copy=False),
            gid1.astype(np.int32,  copy=False),
            gid2.astype(np.int32,  copy=False),
            w.astype(np.float32,   copy=False),
            m_tr,
            m_va,
        )

Xs_tr, targets_tr = [], []
Xs_va, targets_va = [], []


for X, Q, x1, x2, id1, id2, w, m_tr, m_va in iterate_all():
    deriv   = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q)      # (N, M)
    deriv_w = deriv * w.reshape(-1, 1).astype(np.float32, copy=False)   # (N, M)

    # append train slice
    if np.any(m_tr):
        Xs_tr.append(X[m_tr])
        targets_tr.append(deriv_w[m_tr])

    # append valid slice
    if np.any(m_va):
        Xs_va.append(X[m_va])
        targets_va.append(deriv_w[m_va])

# concatenate
X_train   = np.concatenate(Xs_tr, axis=0) if len(Xs_tr) > 1 else Xs_tr[0]
DER_train = np.concatenate(targets_tr, axis=0) if len(targets_tr) > 1 else targets_tr[0]

X_valid   = np.concatenate(Xs_va, axis=0) if len(Xs_va) > 1 else Xs_va[0]
DER_valid = np.concatenate(targets_va, axis=0) if len(targets_va) > 1 else targets_va[0]

# Truth weights (fixed; used for plotting)
training_weights_train = {tuple(sorted(combos[i])): DER_train[:, i] for i in range(len(combos))}
training_weights_valid = {tuple(sorted(combos[i])): DER_valid[:, i] for i in range(len(combos))}

if args.small:
    n_max = len(X_train)//30
    X_train   = X_train[:n_max]
    DER_train = DER_train[:n_max]
    training_weights_train = {k: v[:n_max] for k, v in training_weights_train.items()}
    n_max_v = len(X_valid)//30
    X_valid   = X_valid[:n_max_v]
    DER_valid = DER_valid[:n_max_v]
    training_weights_valid = {k: v[:n_max_v] for k, v in training_weights_valid.items()}

# ---------------- load BIT model (no training) ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J['region'])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "BIT_best.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"BIT model file not found: {model_path}")

print(f"Loading BIT from {model_path}")
bit = MultiBoostedInformationTree.load(model_path)
print("Loaded BIT.")

# ---------------- build truth & prediction (train/valid) ----------------
def format_derivative(der):
    """Pretty label for coefficient combinations, e.g. ('c0','c0','c1') -> c0^2 * c1"""
    if len(der) == 0:
        return "()"
    from collections import Counter
    c = Counter(der)
    parts = []
    for v in sorted(c.keys()):
        power = c[v]
        if power == 1:
            parts.append(v)
        else:
            parts.append(f"{v}^{power}")
    return " * ".join(parts)

ders_all = list(getattr(bit, "derivatives", []))
ders = [d for d in ders_all if len(d) > 0]
if not ders:
    raise RuntimeError("BIT has no non-trivial derivatives to evaluate.")

# labels are identical for train/valid (same BIT outputs)
der_labels = np.array([format_derivative(d) for d in ders], dtype=object)

def build_and_save(split_name: str, X: np.ndarray, tw: dict):
    if X is None or len(X) == 0:
        print(f"[CALIB] split={split_name}: empty, skip.")
        return

    if () not in tw:
        raise RuntimeError(f"[CALIB] split={split_name}: truth_weights missing key ()")

    w0 = tw[()]  # (N,)

    # ---- build truth in ratio space ----
    truth_cols = []
    for der in ders:
        col = tw.get(der)
        if col is None:
            col = tw.get(tuple(reversed(der)))
        if col is None:
            raise KeyError(f"[CALIB] split={split_name}: missing truth_weights for derivative {der}")
        truth_cols.append(col / w0)

    truth = np.stack(truth_cols, axis=1)  # (N, K)

    # ---- prediction ----
    pred = bit.predict(X)  # (N, K)

    if truth.shape != pred.shape:
        raise RuntimeError(
            f"[CALIB] split={split_name}: shape mismatch truth {truth.shape} vs pred {pred.shape}"
        )

    print(f"\n[CALIB] split={split_name} events: {truth.shape[0]}")
    print(f"[CALIB] split={split_name} derivatives: {truth.shape[1]}")

    # ---- save as npy ----
    pred_f  = pred.astype(np.float32, copy=False)
    truth_f = truth.astype(np.float32, copy=False)
    w0_f    = w0.astype(np.float32, copy=False)

    pred_path  = os.path.join(model_dir, f"calib_pred_{split_name}.npy")
    truth_path = os.path.join(model_dir, f"calib_truth_{split_name}.npy")
    w0_path    = os.path.join(model_dir, f"calib_w0_{split_name}.npy")
    label_path = os.path.join(model_dir, "calib_der_labels.npy")  # shared

    np.save(pred_path,  pred_f)
    np.save(truth_path, truth_f)
    np.save(w0_path,    w0_f)
    np.save(label_path, der_labels)

    print(f"[CALIB] saved split={split_name}:")
    print(f"  pred  -> {pred_path}")
    print(f"  truth -> {truth_path}")
    print(f"  w0    -> {w0_path}")
    print(f"  labels-> {label_path}")

# build + save both splits
build_and_save("train", X_train, training_weights_train)
build_and_save("valid", X_valid, training_weights_valid)

print("Done.")
