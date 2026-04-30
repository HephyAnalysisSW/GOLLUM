#!/usr/bin/env python
from __future__ import annotations
import os, sys, argparse, importlib, time, pickle, io, contextlib
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader

from eft.EFTWeightInterface import EFTWeightInterface

from tqdm import trange, tqdm

from data.UIDSplitter import UIDSplitter
import math

# ---------------- args ----------------
p = argparse.ArgumentParser(description="EFT BIT training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--postfix", default=None, help="Plot postfix")
p.add_argument("--overwrite", action="store_true", help="Overwrite model file?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--max_n_files", action="store",type=int, default=None, help="Only this numbe of files.")
p.add_argument("--profile", action="store_true", help="Do CPU profiling?")
p.add_argument("--gpu", action="store_true", help="Use GPU-accelerated binned split training backend.")
p.add_argument("--every", default=5, type=int, help="When to plot (plot if tree_index % every == 0). Set <=0 to disable.")
args = p.parse_args()


# Always NUMBA
import numba as nb
if args.gpu:
    from ML.BIT.GpuBIT import MultiBoostedInformationTree
    import ML.BIT.GpuMultiNode as NumbaMultiNode
else:
    from ML.BIT.NumbaBIT import MultiBoostedInformationTree
    import ML.BIT.NumbaMultiNode as NumbaMultiNode

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
    if args.overwrite: flags.append("--overwrite")
    if args.small:     flags.append("--small")
    if args.every is not None: flags.append(f"--every {args.every}")
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

if args.max_n_files is not None:
    before = len(L._all_files)
    L.set_max_files( args.max_n_files )
    print(f"Reduced number of training files from {before} to {len(L._all_files)}")
elif args.small:
    before = len(L._all_files)
    L.set_max_files(3)
    print(f"Reduced number of training files from {before} to {len(L._all_files)} for --small")

sel  = J.get("selection", None)
sel_f= J.get("selection_features", [])
if sel:
    L.addSelection( sel, sel_f)
    print("Added selection to loader: {sel} and selection_features {sel_f}")

print("Using NUMBA")
print("Numba threads:", nb.get_num_threads())
if args.gpu:
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("GPU training requested with --gpu, but no CUDA devices are visible.")
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as e:
        raise RuntimeError(
            "GPU training requested with --gpu, but CuPy/CUDA initialization failed. "
            "Ensure CuPy is installed and a CUDA device is available and accessible."
        ) from e
    print("Training backend: GPU")
    print("GPU device:", device_name)
else:
    print("Training backend: CPU")

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
    missing_uid = [f for f in uid_fields if f not in obs_names]
    if missing_uid:
        raise RuntimeError(
            f"UID splitting requested, but observer_names are missing {missing_uid} in loader '{loader_name}'."
        )

    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    keys  = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)
    bit_train_key = "pnn_train"
    bit_val_key   = "pnn_val"
    train_interval = uid_intervals[bit_train_key]
    val_interval   = uid_intervals[bit_val_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] BIT train split '{bit_train_key}' -> {train_interval}")
    print(f"[UID] BIT val   split '{bit_val_key}' -> {val_interval}")

# ---------------- EFT target interface ----------------
eft = EFTWeightInterface(J.get("eft", {}).get("parameters", []))
combos = list(eft.combinations)
base_points = list(eft.base_points)

# features / observers
needed_observers = list(eft.required_observers)
if uid_enabled:
    for field in uid_fields:
        if field not in needed_observers:
            needed_observers.append(field)

L.setFeatures(J["features"], observer_names=needed_observers)
feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")

obs_names = list(getattr(L, "observer_names", []) or [])
print(L)

# ---------------- collect all data (single pass) ----------------
def iterate_all(shard_limit=None):
    """
    Yields per-shard arrays PLUS (optionally) UID masks for train/val.
    - Always yields X,G,w
    - If uid_enabled: also yields (m_tr, m_va) boolean masks with same length as X
    """
    n_shards = len(L)
    if args.small: n_shards = 1
    if shard_limit is not None: n_shards = min(n_shards, shard_limit)

    if uid_enabled:
        on2idx = {n: i for i, n in enumerate(obs_names)}
        uid_idx = [on2idx[f] for f in uid_fields]
        lo_tr, hi_tr = train_interval
        lo_va, hi_va = val_interval

    for shard in range(n_shards):
        X, G, w = L.materialize(shard=shard, what="fow")
        if uid_enabled:
            O_uid = G[:, uid_idx]
            m_tr = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_tr, hi_tr)
            m_va = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo_va, hi_va)
            yield (
                X.astype(np.float32, copy=False),
                G.astype(np.float32, copy=False),
                w.astype(np.float32, copy=False),
                m_tr,
                m_va,
            )
        else:
            yield (
                X.astype(np.float32, copy=False),
                G.astype(np.float32, copy=False),
                w.astype(np.float32, copy=False),
            )

Xs_tr, targets_tr = [], []
Xs_va, targets_va = [], []

if uid_enabled:
    for X, G, w, m_tr, m_va in iterate_all():
        deriv_w = eft.make_weight_matrix(G, obs_names, w)
        if np.any(m_tr):
            Xs_tr.append(X[m_tr])
            targets_tr.append(deriv_w[m_tr])
        if np.any(m_va):
            Xs_va.append(X[m_va])
            targets_va.append(deriv_w[m_va])

    X_train   = np.concatenate(Xs_tr, axis=0) if len(Xs_tr) > 1 else Xs_tr[0]
    DER_train = np.concatenate(targets_tr, axis=0) if len(targets_tr) > 1 else targets_tr[0]
    X_valid   = np.concatenate(Xs_va, axis=0) if len(Xs_va) > 1 else Xs_va[0]
    DER_valid = np.concatenate(targets_va, axis=0) if len(targets_va) > 1 else targets_va[0]
else:
    Xs, targets_acc = [], []
    for X, G, w in iterate_all():
        deriv_w = eft.make_weight_matrix(G, obs_names, w)
        Xs.append(X)
        targets_acc.append(deriv_w)

    X_train   = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
    DER_train = np.concatenate(targets_acc, axis=0) if len(targets_acc) > 1 else targets_acc[0]
    X_valid, DER_valid = None, None

# Truth weights (fixed; used for plotting)
training_weights_train = {tuple(sorted(combos[i])): DER_train[:, i] for i in range(len(combos))}
training_weights_valid = None
if DER_valid is not None:
    training_weights_valid = {tuple(sorted(combos[i])): DER_valid[:, i] for i in range(len(combos))}

if args.small:
    n_max = len(X_train)//30
    X_train   = X_train[:n_max]
    DER_train = DER_train[:n_max]
    training_weights_train = {k: v[:n_max] for k, v in training_weights_train.items()}

    if X_valid is not None:
        n_max_v = len(X_valid)//30
        X_valid   = X_valid[:n_max_v]
        DER_valid = DER_valid[:n_max_v]
        training_weights_valid = {k: v[:n_max_v] for k, v in training_weights_valid.items()}

# ---------------- plotting function ----------------
def _build_plot_context(X_train, training_weights_train, feat_names, cfg_base, J):
    import ROOT, math
    from data.plot_options import plot_options as DEFAULT_PLOT_OPTS

    PLOT_OPTS = getattr(samples_mod, "plot_options", DEFAULT_PLOT_OPTS)

    plot_feats = [f for f in feat_names if f in PLOT_OPTS]
    if not plot_feats:
        return None

    if args.max_n_files is not None:
        train = f"train_maxFiles{args.max_n_files}"
    else:
        train = "train"
    if args.postfix is not None:
        train += ("_" + args.postfix)

    out_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"], train)
    os.makedirs(out_dir, exist_ok=True)

    feat_cfgs = []
    for feat in plot_feats:
        n, lo, hi = PLOT_OPTS[feat]['binning']
        feat_cfgs.append({
            "name": feat,
            "tex": PLOT_OPTS[feat]['tex'],
            "n": n,
            "lo": lo,
            "hi": hi,
            "edges": np.linspace(lo, hi, n + 1),
            "column": feat_names.index(feat),
        })

    total_pads = len(plot_feats) + 1
    gx = int(math.ceil(math.sqrt(total_pads)))
    gy = int(math.ceil(total_pads / gx))

    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

    return {
        "out_dir": out_dir,
        "feat_cfgs": feat_cfgs,
        "gx": gx,
        "gy": gy,
        "w0": training_weights_train[()],
    }


def plot_bit_training_root(bit, t, X_train, training_weights_train, feat_names, cfg_base, J, plot_ctx=None):
    """
    Plot truth vs prediction ratios after t trees.
    """
    import ROOT

    if plot_ctx is None:
        plot_ctx = _build_plot_context(X_train, training_weights_train, feat_names, cfg_base, J)
    if plot_ctx is None:
        tqdm.write("No plotable features found in PLOT_OPTS; skipping plots.")
        return False

    ders = bit.derivatives

    colors = {}
    i_lin, i_diag, i_mix = 0, 0, 0
    for der in ders:
        if len(der) == 1:
            colors[der] = ROOT.kAzure + i_lin; i_lin += 1
        elif len(der) == 2 and len(set(der)) == 1:
            colors[der] = ROOT.kRed + i_diag; i_diag += 1
        else:
            colors[der] = ROOT.kGreen + i_mix; i_mix += 1

    pred = bit.predict(X_train, max_n_tree=t)  # (N, M-1), aligned to ders[1:]
    w0 = plot_ctx["w0"]

    truth_mat = np.stack([
        training_weights_train.get(der, training_weights_train.get(tuple(reversed(der))))
        for der in ders
    ], axis=1)

    c = ROOT.TCanvas(f"c_iter_{t}", f"BIT iter {t}", 500*plot_ctx["gx"], 500*plot_ctx["gy"])
    c.Divide(plot_ctx["gx"], plot_ctx["gy"])
    keep = []

    leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    leg.SetNColumns(min(3, 1 + len(ders)//10))
    keep.append(leg)

    def _safe_ratio(numer, denom):
        denom2 = denom.copy()
        denom2[denom2 == 0] = 1.0
        return numer / denom2

    for i, feat_cfg in enumerate(plot_ctx["feat_cfgs"]):
        pad = c.cd(i + 1)
        pad.SetTicks(1, 1)
        pad.SetBottomMargin(0.15)
        pad.SetLeftMargin(0.15)

        feat = feat_cfg["name"]
        n = feat_cfg["n"]
        lo = feat_cfg["lo"]
        hi = feat_cfg["hi"]
        edges = feat_cfg["edges"]
        col = feat_cfg["column"]
        x = X_train[:, col]

        h_w0, _ = np.histogram(x, bins=edges, weights=w0)

        ratios_truth = {}
        ratios_pred  = {}
        for i_der, der in enumerate(ders):
            if len(der) == 0:
                continue
            ht, _ = np.histogram(x, bins=edges, weights=truth_mat[:, i_der])
            hp, _ = np.histogram(x, bins=edges, weights=w0 * pred[:, i_der])
            ratios_truth[der] = _safe_ratio(ht, h_w0)
            ratios_pred[der]  = _safe_ratio(hp, h_w0)

        vals = np.concatenate(list(ratios_truth.values())) if ratios_truth else np.array([0.0, 1.0])
        finite = np.isfinite(vals)
        if finite.any():
            y_min = float(np.min(vals[finite])); y_max = float(np.max(vals[finite]))
        else:
            y_min, y_max = 0.0, 1.0
        if y_max <= y_min: y_max = y_min + 1.0
        pad_frac = 0.20
        y_low = y_min - pad_frac * (y_max - y_min)
        y_hi  = y_max + pad_frac * (y_max - y_min)

        hframe = ROOT.TH2F(f"hf_{feat}_{t}", f";{feat_cfg['tex']};ratio",
                           n, lo, hi, 100, y_low, y_hi)
        hframe.GetYaxis().SetTitleOffset(1.3)
        hframe.Draw()
        keep.append(hframe)

        hY = ROOT.TH1F(f"hY_{feat}_{t}", "", n, lo, hi)
        for b in range(1, n+1):
            hY.SetBinContent(b, float(h_w0[b-1]))
        y_max0 = float(np.max(h_w0) if len(h_w0) else 0.0)
        if y_max0 > 0:
            for b in range(1, n+1):
                v = hY.GetBinContent(b)
                scaled = y_low + 0.92*(y_hi - y_low) * (v / max(1e-12, y_max0))
                hY.SetBinContent(b, scaled)
        hY.SetLineColor(ROOT.kGray + 2)
        hY.SetLineWidth(2)
        hY.SetMarkerStyle(0)
        hY.Draw("hist same")
        keep.append(hY)

        for der in ders:
            if len(der) == 0:
                continue
            colr = colors.get(der, ROOT.kBlue)

            hT = ROOT.TH1F(f"hT_{feat}_{t}_{str(der)}", "", n, lo, hi)
            hP = ROOT.TH1F(f"hP_{feat}_{t}_{str(der)}", "", n, lo, hi)
            for b in range(1, n+1):
                hT.SetBinContent(b, float(ratios_truth[der][b-1]))
                hP.SetBinContent(b, float(ratios_pred[der][b-1]))
            for h, sty in ((hT, 2), (hP, 1)):
                h.SetLineColor(colr)
                h.SetLineStyle(sty)
                h.SetLineWidth(2)
                h.SetMarkerStyle(0)
                h.Draw("hist same")
                keep.append(h)

    pad = c.cd(len(plot_ctx["feat_cfgs"]) + 1)
    pad.SetTicks(1, 1)
    pad.SetBottomMargin(0.15)
    pad.SetLeftMargin(0.15)

    added = set()
    for der in ders:
        if len(der) == 0 or der in added:
            continue
        dt = ROOT.TH1F(f"dt_{t}_{str(der)}", "", 1, 0, 1)
        dp = ROOT.TH1F(f"dp_{t}_{str(der)}", "", 1, 0, 1)
        dt.SetLineColor(colors.get(der, ROOT.kBlue)); dt.SetLineStyle(2); dt.SetLineWidth(2)
        dp.SetLineColor(colors.get(der, ROOT.kBlue)); dp.SetLineStyle(1); dp.SetLineWidth(2)
        leg.AddEntry(dt,  "R" + str(der), "l")
        leg.AddEntry(dp, "#hat{R}" + str(der), "l")
        keep.extend([dt, dp])
        added.add(der)

    dy = ROOT.TH1F(f"dy_{t}", "", 1, 0, 1)
    dy.SetLineColor(ROOT.kGray + 2); dy.SetLineWidth(2)
    leg.AddEntry(dy, "yield (SM, scaled)", "l")
    keep.append(dy)

    leg.Draw()
    keep.append(leg)

    tl = ROOT.TLatex()
    tl.SetNDC()
    tl.SetTextSize(0.07)
    tl.SetTextAlign(11)
    tl.DrawLatex(0.30, 0.95, f"Trees = {t:04d}")
    keep.append(tl)

    c.Print(os.path.join(plot_ctx["out_dir"], f"iter_{t:04d}.png"))
    c.Close()
    return True

# ---------------- build & train BIT ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J['region'])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))
if args.small:
    model_path = model_path[:-4] + "_small.pkl"
if args.max_n_files is not None:
    model_path = model_path[:-4] + f"_maxFiles{args.max_n_files}.pkl"

# weights snapshot stays separate (required to resume boosting correctly)
weights_path = model_path[:-4] + ".weights.pkl"

bit = None
start_tree = 0
boost_weights = None

def bit_ratio_mse_loss(bit, X, truth_weights, max_n_tree: int) -> float:
    """
    Loss in ratio space:
      r_true(der) = w_der / w0
      r_pred(der) = bit.predict(...)

    Return: average over derivatives of weighted MSE in ratio space.
    Event weight for averaging: |w0|.
    Uses truth_weights keys to define the derivative list, then aligns to bit.predict columns.
    """
    if X is None:
        return float("nan")

    # nominal weights
    if () not in truth_weights:
        raise RuntimeError("truth_weights must contain key () for nominal weights.")
    w0 = truth_weights[()]  # (N,)

    pred = bit.predict(X, max_n_tree=max_n_tree)  # (N, K)  K = number of non-empty derivatives model predicts

    # ---- use bit.derivatives ordering ----
    ders = getattr(bit, "derivatives", None)
    if ders is None or len(ders) == 0:
        raise RuntimeError("bit.derivatives not initialized yet.")

    # predict columns correspond to non-empty derivatives, in this order
    ders_eval = [d for d in ders if len(d) != 0]

    if pred.shape[1] != len(ders_eval):
        raise RuntimeError(
            f"Mismatch: bit.predict returns {pred.shape[1]} columns, "
            f"but bit.derivatives has {len(ders_eval)} non-empty derivatives."
        )
    # -------------------------------------

    mask = (w0 != 0)
    if not np.any(mask):
        return float("inf")

    w_abs = np.abs(w0[mask])

    losses = []
    for i, der in enumerate(ders_eval):
        wt = truth_weights.get(der, truth_weights.get(tuple(reversed(der))))
        if wt is None:
            raise RuntimeError(f"Missing truth_weights for derivative {der} (or reversed)")

        r_true = wt[mask] / w0[mask]
        r_pred = pred[mask, i]

        mse = np.average((r_pred - r_true) ** 2, weights=w_abs)
        losses.append(mse)

    return float(np.mean(losses))


def _tree_summary(root, feat_names):
    split_feature = feat_names[root.split_i_feature] if feat_names and root.split_i_feature < len(feat_names) else f"X{root.split_i_feature}"
    split_gain = getattr(root, "split_gain", float("nan"))
    left_size = int(getattr(root.left, "size", 0))
    right_size = int(getattr(root.right, "size", 0))
    return (
        f"[TREE] root_feature={split_feature} "
        f"threshold={root.split_value:.6g} "
        f"gain={split_gain:.6g} "
        f"left={left_size} right={right_size}"
    )


model_cfg = J.get("model", {}) or {}
global_cuts = None
bins_train = None
if model_cfg.get("split_mode") == "binned":
    cut_sample_rows = int(model_cfg.get("cut_sample_rows", len(X_train)))
    cut_sample_rows = max(1, min(cut_sample_rows, len(X_train)))
    global_cuts = NumbaMultiNode._compute_global_quantile_cuts(
        X_train,
        int(model_cfg.get("n_bins", 256)),
        cut_sample_rows=cut_sample_rows,
    )
    bins_train = NumbaMultiNode._quantize_feature_matrix(X_train, global_cuts)
    tqdm.write(
        f"[BINS] built global cuts once: rows={cut_sample_rows} "
        f"features={global_cuts.shape[0]} bins={global_cuts.shape[1] + 1}"
    )

# ---- load / resume from model_path directly ----
if not args.overwrite and os.path.exists(model_path):
    try:
        tqdm.write(f"Trying to load BIT from {model_path}")
        bit = MultiBoostedInformationTree.load(model_path)
        start_tree = len(getattr(bit, "trees", []) or [])
        szm = os.path.getsize(model_path) / (1024.0 * 1024.0)
        tqdm.write(f"Loaded: trees={start_tree}/{bit.n_trees} | model size={szm:.1f} MB")
        if start_tree < bit.n_trees:
            if os.path.exists(weights_path):
                with open(weights_path, "rb") as f:
                    boost_weights = pickle.load(f)
                szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
                tqdm.write(f"Loaded boosting weights: {weights_path} ({szz:.1f} MB)")
            else:
                tqdm.write(f"Missing weights snapshot {weights_path}. Cannot resume safely; training new.")
                bit = None
                start_tree = 0
    except Exception:
        bit = None
        start_tree = 0
        tqdm.write("Failed to load model. Training new.")

# ---- fresh init ----
if bit is None:
    bit = MultiBoostedInformationTree(**model_cfg)
    boost_weights = {k: v.copy() for k, v in training_weights_train.items()}

# If training needed but weights missing, start from truth
if boost_weights is None and len(bit.trees) < bit.n_trees:
    boost_weights = {k: v.copy() for k, v in training_weights_train.items()}

# ---------------- external training loop ----------------
rt = J.get("runtime", {}) or {}
enable_plots = bool(rt.get("training_plots", False))
plot_ctx = _build_plot_context(X_train, training_weights_train, feat_names, cfg_base, J) if enable_plots else None

# ---------------- loss history ----------------
loss_trees = []
train_losses = []
valid_losses = []  # if no valid -> store np.nan
best_valid_loss = float("inf")
best_tree = -1
best_model_path = os.path.join(model_dir, "BIT_best.pkl")
best_weights_path = os.path.join(model_dir, "BIT_best.weights.pkl")  # 可选

if len(bit.trees) < bit.n_trees:

    if global_cuts is not None:
        bit.node_cfg["precomputed_cuts"] = global_cuts

    weak_learner_time = 0.0
    update_time = 0.0

    pbar = trange(start_tree, bit.n_trees, desc="Trees", unit="tree", dynamic_ncols=True)
    for n_tree in pbar:

        _get_only_score = ((n_tree == 0) and bool(getattr(bit, "learn_global_score", False)))
        bit.node_cfg["_get_only_score"] = _get_only_score

        # ---------------- CPU profiling: BOOSTING ONLY ----------------
        if args.profile:
            prof = cProfile.Profile()
            cpu_t0  = time.process_time()
            wall_t0 = time.perf_counter()
            prof.enable()
        # --------------------------------------------------------------

        # fit tree (root needs base_points / feature_names)
        t1 = time.process_time()
        root = NumbaMultiNode.MultiNode(
            None if global_cuts is not None else X_train,
            training_weights = boost_weights,
            base_points      = base_points,
            feature_names    = feat_names,
            binned_features  = bins_train,
            **bit.node_cfg
        )
        t2 = time.process_time()
        weak_learner_time += (t2 - t1)
        tqdm.write(_tree_summary(root, feat_names))

        # ---------------- end profiling ----------------
        if args.profile:
            prof.disable()
            cpu_t1  = time.process_time()
            wall_t1 = time.perf_counter()

            tqdm.write("weak learner time: %.2f" % weak_learner_time)
            tqdm.write("update time: %.2f" % update_time)
            tqdm.write(f"Boosting CPU time:  {cpu_t1 - cpu_t0:.2f} s")
            tqdm.write(f"Boosting wall time: {wall_t1 - wall_t0:.2f} s")

            # Print profile summary to shell (no files)
            # (Use a buffer to keep tqdm output clean.)
            buf = io.StringIO()

            buf.write("\n================= cProfile (sorted by cumtime) =================\n")
            st = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("cumtime")
            st.print_stats(60)

            buf.write("\n================= cProfile (sorted by tottime) =================\n")
            st = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("tottime")
            st.print_stats(60)

            # Flush buffer via tqdm.write line-by-line so the bar survives
            for line in buf.getvalue().splitlines():
                tqdm.write(line)


        if (n_tree == 0) and (getattr(bit, "derivatives", None) is None):
            bit.derivatives = root.derivatives[1:]

        bit.trees.append(root)

        # ---------------- TRAIN LOSS (metric) ----------------
        # only valid after bit.derivatives is set (i.e. after first tree finished)
        train_loss = bit_ratio_mse_loss(
        	bit=bit,
        	X=X_train,
        	truth_weights=training_weights_train,
        	max_n_tree=len(bit.trees),   # == n_tree + 1
        )
        tqdm.write(f"[LOSS] tree={len(bit.trees):04d} train_loss={train_loss:.6g}")

        # ---------------- VALID LOSS (metric) ----------------
        if X_valid is not None:
            valid_loss = bit_ratio_mse_loss(
                bit=bit,
                X=X_valid,
                truth_weights=training_weights_valid,
                max_n_tree=len(bit.trees),   # == n_tree + 1
            )
            tqdm.write(f"[LOSS] tree={len(bit.trees):04d} valid_loss={valid_loss:.6g}")

            # ---------------- save BEST on valid ----------------
            if np.isfinite(valid_loss) and (valid_loss < best_valid_loss):
                best_valid_loss = float(valid_loss)
                best_tree = len(bit.trees)
                _trees = bit.trees
                bit.trees = _trees[:best_tree]
                tmp = best_model_path + ".tmp"
                bit.save(tmp)
                os.replace(tmp, best_model_path)
                bit.trees = _trees
                tmpw = best_weights_path + ".tmp"
                with open(tmpw, "wb") as f:
                    pickle.dump(boost_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmpw, best_weights_path)
                tqdm.write(f"[BEST] tree={best_tree:04d} valid_loss={best_valid_loss:.6g} -> {best_model_path}")

        # ---------------- append loss history ----------------
        tree_now = len(bit.trees)
        loss_trees.append(tree_now)
        train_losses.append(float(train_loss))

        if X_valid is not None:
            valid_losses.append(float(valid_loss))
        else:
            valid_losses.append(float("nan"))

        # update weights
        t1 = time.process_time()
        prediction = root.vectorized_predict(X_train)
        len_ = len(prediction)

        delta_weight = boost_weights[tuple()].reshape(len_, -1) * prediction[:, 1:] / prediction[:, 0].reshape(len_, -1)
        lr_eff = 1.0 if _get_only_score else float(bit.learning_rate)

        for i_der, der in enumerate(root.derivatives[1:]):
            boost_weights[der] += -lr_eff * delta_weight[:, i_der]

        t2 = time.process_time()
        update_time += (t2 - t1)

        # checkpoint to model_path (atomic)
        tmp_m = model_path + ".tmp"
        bit.save(tmp_m)
        os.replace(tmp_m, model_path)

        tmp_w = weights_path + ".tmp"
        with open(tmp_w, "wb") as f:
            pickle.dump(boost_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_w, weights_path)

        szm = os.path.getsize(model_path) / (1024.0 * 1024.0)
        szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
        pbar.set_postfix({"model_MB": f"{szm:.1f}", "w_MB": f"{szz:.1f}"})

        do_plot = enable_plots and (args.every is not None) and (args.every > 0) and ((n_tree % args.every) == 0)
        if do_plot:
            tqdm.write(f"Plotting at tree {n_tree+1:04d} ...")
            plot_bit_training_root(
                bit,
                t=n_tree+1,
                X_train=X_train,
                training_weights_train=training_weights_train,
                feat_names=feat_names,
                cfg_base=cfg_base,
                J=J,
                plot_ctx=plot_ctx,
            )

    # ---------------- save loss history ----------------
    loss_txt = os.path.join(model_dir, "loss_history.txt")
    with open(loss_txt, "w") as f:
        f.write("# tree\ttrain_loss\tvalid_loss\n")
        for t, tr, va in zip(loss_trees, train_losses, valid_losses):
            f.write(f"{t}\t{tr:.8e}\t{va:.8e}\n")
    tqdm.write(f"[LOSS] wrote {loss_txt}")

    # ---------------- plot loss curves ----------------
    plt.figure()
    plt.plot(loss_trees, train_losses, label="train")
    # only plot valid curve if at least one finite value exists
    va_arr = np.array(valid_losses, dtype=float)
    if np.isfinite(va_arr).any():
        plt.plot(loss_trees, valid_losses, label="valid")
    plt.xlabel("n_trees")
    plt.ylabel("ratio_mse_loss")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    loss_pdf = os.path.join(model_dir, "loss_history.pdf")
    plt.tight_layout()
    plt.savefig(loss_pdf, dpi=500)
    plt.close()
    tqdm.write(f"[LOSS] wrote {loss_pdf}")

    # After last training: keep weights file, print filename
    szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
    tqdm.write(f"Kept boosting weights snapshot -> {weights_path} ({szz:.1f} MB)")

else:
    if os.path.exists(weights_path):
        szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
        tqdm.write(f"Kept boosting weights snapshot -> {weights_path} ({szz:.1f} MB)")

print("Done.")
