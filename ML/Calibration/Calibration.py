#!/usr/bin/env python
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
p.add_argument("--numba", action="store_true", help="Use the numba implementation")
args = p.parse_args()

if args.numba:
    from ML.BIT.NumbaBIT import MultiBoostedInformationTree
else:
    from ML.BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

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
    if args.numba: flags.append("--numba")
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

# ---------------- PDF parametrization & combinations ----------------
pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", None)
pdf = PDFParametrization(n=pdf_n, typ=pdf_type)                     # defines variables: ['c0',..,'cN']

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
    n_shards = len(L)
    if args.small: n_shards = 1
    if shard_limit is not None: n_shards = min(n_shards, shard_limit)
    on2idx = {n: i for i, n in enumerate(obs_names)}
    for shard in range(n_shards):
        # pull features, observers, and weights in one go
        X, G, w = L.materialize(shard=shard, what="fow")
        gQ   = G[:, on2idx["Generator_scalePDF"]]
        gx1  = G[:, on2idx["Generator_x1"]]
        gx2  = G[:, on2idx["Generator_x2"]]
        gid1 = G[:, on2idx["Generator_id1"]]
        gid2 = G[:, on2idx["Generator_id2"]]
        yield (X.astype(np.float32, copy=False),
               gQ.astype(np.float32, copy=False),
               gx1.astype(np.float32, copy=False),
               gx2.astype(np.float32, copy=False),
               gid1.astype(np.int32,  copy=False),
               gid2.astype(np.int32,  copy=False),
               w.astype(np.float32,   copy=False))

Xs = []
targets_acc = []  # list of (N_i, len(combos)) arrays
for X, Q, x1, x2, id1, id2, w in iterate_all():
    # Unweighted derivatives aligned with pdf.combinations
    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q)            # (N_i, M)
    # Multiply each column by the event weight (treating derivatives as reweights)
    deriv_w = deriv * w.reshape(-1, 1).astype(np.float32, copy=False)   # (N_i, M)
    Xs.append(X)
    targets_acc.append(deriv_w)

X_all   = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
DER_all = np.concatenate(targets_acc, axis=0) if len(targets_acc) > 1 else targets_acc[0]

# Build the dict that BIT expects: {combination: vector}; () term is the nominal weight
training_weights = {combos[i]: DER_all[:, i] for i in range(len(combos))}

if args.small:
    n_max = len(X_all) // 100
    X_all   = X_all[:n_max]
    DER_all = DER_all[:n_max]
    training_weights = {key: val[:n_max] for key, val in training_weights.items()}

# ---------------- load BIT model (no training) ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J['region'])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))
#if args.small:
#    model_path = model_path[:-4] + "_small.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"BIT model file not found: {model_path}")

print(f"Loading BIT from {model_path}")
bit = MultiBoostedInformationTree.load(model_path)
print("Loaded BIT.")

# ---------------- build truth & prediction ----------------
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

w0 = training_weights[()]  # nominal event weight
ders_all = list(getattr(bit, "derivatives", []))

# Just in case, restrict to non-empty derivatives (actual coefficients)
ders = [d for d in ders_all if len(d) > 0]
if not ders:
    raise RuntimeError("BIT has no non-trivial derivatives to evaluate.")

# Truth: ratio of derivative weight to nominal weight
truth_cols = []
for der in ders:
    col = training_weights.get(der)
    if col is None:
        # allow for symmetric combinations, e.g. ('c0','c1') vs ('c1','c0')
        col = training_weights.get(tuple(reversed(der)))
    if col is None:
        raise KeyError(f"No training weights found for derivative {der}")
    truth_cols.append(col / w0)

truth = np.stack(truth_cols, axis=1)  # shape (N, n_der)

# Prediction: final ensemble (no max_n_tree argument)
pred = bit.predict(X_all)  # expected shape (N, n_der)

if truth.shape != pred.shape:
    raise RuntimeError(
        f"Shape mismatch: truth {truth.shape} vs pred {pred.shape}"
    )

print(f"\nNumber of events: {truth.shape[0]}")
print(f"Number of derivatives (coefficients): {truth.shape[1]}")

# ---------------- print first few events ----------------
n_show = min(5, truth.shape[0])
print(f"\nFirst {n_show} events (truth vs prediction):")
for i in range(n_show):
    print(f"\nEvent {i}:")
    for j, der in enumerate(ders):
        label = format_derivative(der)
        t_val = truth[i, j]
        p_val = pred[i, j]
        print(f"  {label:>20s} : truth = {t_val: .5e},  pred = {p_val: .5e}")

print("\nDone.")

