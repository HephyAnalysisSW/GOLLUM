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

from data.RandomSplitter import RandomSplitter
from data.UIDSplitter import UIDSplitter
import math
from eft.EFTWeightInterface import EFTWeightInterface

from tqdm import trange, tqdm

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
    for j in jobs:
        print(f"python {__file__} {args.config} {' '.join(flags)} --job {j['id']}")
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

runtime_cfg = J.get("runtime", {}) or {}
n_split = int(runtime_cfg.get("n_split", 1))
if n_split > 1 and len(L._all_files) > 1:
    before_split = len(L)
    L.set_n_split(n_split)
    print(
        f"Using {len(L)} file shards for training-data materialization "
        f"(was {before_split}, files={len(L._all_files)})"
    )

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

# ---------------- deterministic random splitting ----------------
SPLIT_CFG = (J.get("splitting") or {})
split_enabled  = bool(SPLIT_CFG.get("enabled", False))
split_type     = SPLIT_CFG.get("type", "random")
split_seed     = int(SPLIT_CFG.get("seed", 0))
splitter = None

if split_enabled:

    if split_type == "random":
        split_fraction = float(SPLIT_CFG.get("fraction", 0.5))
        splitter = RandomSplitter(fraction=split_fraction, seed=split_seed)
        print(f"[SPLIT] enabled=True type={split_type} seed={split_seed} fraction={split_fraction}")

    elif split_type == "uid":

        uid_fields    = SPLIT_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
        uid_seed      = split_seed
        uid_n_buckets = int(SPLIT_CFG.get("n_buckets", 10000))
        uid_scheme    = (SPLIT_CFG.get("scheme") or {})

        splitter = UIDSplitter(
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

        bit_train_key = "pnn_train"
        bit_val_key   = "pnn_val"
        train_interval = uid_intervals[bit_train_key]
        val_interval   = uid_intervals[bit_val_key]

        print(f"[SPLIT] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
        print(f"[SPLIT] scheme intervals: {uid_intervals}")
        print(f"[SPLIT] BIT train split '{bit_train_key}' -> {train_interval}")
        print(f"[SPLIT] BIT val   split '{bit_val_key}' -> {val_interval}")
    else:
        raise RuntimeError(f"Unsupported splitting.type='{split_type}'. Only 'random' and 'uid' is implemented.")
    

# ---------------- EFT target interface ----------------
eft = EFTWeightInterface(J.get("eft", {}).get("parameters", []))
combos = list(eft.combinations)
base_points = list(eft.base_points)
combo_to_col = {tuple(sorted(comb)): i for i, comb in enumerate(combos)}


def weight_views(weight_matrix):
    return {tuple(sorted(combos[i])): weight_matrix[:, i] for i in range(len(combos))}

# features / observers
needed_observers = list(eft.required_observers)

if split_enabled and split_type=='uid':
    # keeping only needed derivatives + variables used for UID spliting
    L.setFeatures(J["features"], observer_names=needed_observers+list(uid_fields))
else:
    L.setFeatures(J["features"], observer_names=needed_observers)

feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")

obs_names = list(getattr(L, "observer_names", []) or [])
print(L)

# ---------------- collect all data (single pass) ----------------
def iterate_all(shard_limit=None):
    """
    Yields per-shard arrays PLUS optional deterministic masks.
    - Always yields X,G,w
    - If split_enabled and split_type 'random': also yields m_keep with the same length as X
    - If split_enabled and split_type 'uid': also yields (m_tr, m_va) boolean masks with same length as X
    """
    n_shards = len(L)
    if args.small: n_shards = 1
    if shard_limit is not None: n_shards = min(n_shards, shard_limit)

    on2idx = {n: i for i, n in enumerate(obs_names)}

    if split_enabled and split_type=='uid':
        
        uid_idx = [on2idx[f] for f in uid_fields]
        lo_tr, hi_tr = train_interval
        lo_va, hi_va = val_interval

    for shard in range(n_shards):
        X, G, w = L.materialize(shard=shard, what="fow")
        if split_enabled:
            if split_type=='random':
                m_keep = splitter.mask(len(X), shard=shard)
                yield (
                    X.astype(np.float32, copy=False),
                    G.astype(np.float32, copy=False),
                    w.astype(np.float32, copy=False),
                    m_keep,
                )
            if split_type=='uid':
                O_uid = G[:, uid_idx]  # shape (N, len(uid_fields))
                m_tr = splitter.mask_from_np(O_uid, list(uid_fields), lo_tr, hi_tr)
                m_va = splitter.mask_from_np(O_uid, list(uid_fields), lo_va, hi_va)
                yield (
                        X.astype(np.float32, copy=False),
                        G.astype(np.float32, copy=False),
                        w.astype(np.float32,   copy=False),
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
materialize_pbar = tqdm(
    iterate_all(),
    total=len(L),
    desc="Materialize",
    unit="shard",
    dynamic_ncols=True,
)

if split_enabled:

    if split_type == 'random':
        for X, G, w, m_keep in materialize_pbar:
            deriv_w = eft.make_weight_matrix(G, obs_names, w)
            if np.any(m_keep):
                Xs_tr.append(X[m_keep])
                targets_tr.append(deriv_w[m_keep])

        if not Xs_tr:
            raise RuntimeError("Random splitting removed all training events. Increase splitting.fraction.")

        X_train   = np.concatenate(Xs_tr, axis=0) if len(Xs_tr) > 1 else Xs_tr[0]
        DER_train = np.concatenate(targets_tr, axis=0) if len(targets_tr) > 1 else targets_tr[0]
        X_valid, DER_valid = None, None
    
    elif split_type=='uid':
        for X, G, w, m_tr, m_va in materialize_pbar:
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
    for X, G, w in materialize_pbar:
        deriv_w = eft.make_weight_matrix(G, obs_names, w)
        Xs.append(X)
        targets_acc.append(deriv_w)

    X_train   = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
    DER_train = np.concatenate(targets_acc, axis=0) if len(targets_acc) > 1 else targets_acc[0]
    X_valid, DER_valid = None, None

if args.small:
    n_max = len(X_train)//30
    X_train   = X_train[:n_max]
    DER_train = DER_train[:n_max]

    if X_valid is not None:
        n_max_v = len(X_valid)//30
        X_valid   = X_valid[:n_max_v]
        DER_valid = DER_valid[:n_max_v]

training_weights_train = DER_train
training_weights_valid = DER_valid

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
        "w0": training_weights_train[:, 0],
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
        training_weights_train[:, combo_to_col[tuple(sorted(der))]]
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
    leg.AddEntry(dy, "yield (generation point, scaled)", "l")
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
    syncer.sync()
    return True

# ---------------- build & train BIT ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J['region'])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)

# loss history and its plot go to the plot directory, not the model directory
plot_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"])
os.makedirs(plot_dir, exist_ok=True)

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
    w0 = truth_weights[:, 0]  # (N,)

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
        key = tuple(sorted(der))
        if key not in combo_to_col:
            raise RuntimeError(f"Missing truth_weights for derivative {der}")
        wt = truth_weights[:, combo_to_col[key]]

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
    if model_cfg.get("quantile_bins", False):
        global_cuts = NumbaMultiNode._compute_global_quantile_cuts(
            X_train,
            int(model_cfg.get("n_bins", 256)),
            cut_sample_rows=cut_sample_rows,
        )
    else:
        global_cuts = NumbaMultiNode._compute_global_uniform_cuts(
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
        cfg_n_trees = model_cfg.get("n_trees")
        if cfg_n_trees is not None and int(cfg_n_trees) != int(bit.n_trees):
            old_n_trees = int(bit.n_trees)
            bit.n_trees = int(cfg_n_trees)
            if bit.n_trees > old_n_trees:
                tqdm.write(f"[RESUME] Extending target tree count from {old_n_trees} to {bit.n_trees}.")
            else:
                tqdm.write(f"[RESUME] Config requests fewer trees ({bit.n_trees}) than saved model target ({old_n_trees}); using {bit.n_trees}.")
        start_tree = len(getattr(bit, "trees", []) or [])
        szm = os.path.getsize(model_path) / (1024.0 * 1024.0)
        tqdm.write(f"Loaded: trees={start_tree}/{bit.n_trees} | model size={szm:.1f} MB")
        if start_tree < bit.n_trees:
            if os.path.exists(weights_path):
                with open(weights_path, "rb") as f:
                    boost_weights = pickle.load(f)
                if isinstance(boost_weights, dict):
                    boost_weights = np.column_stack(
                        [boost_weights[tuple(sorted(comb))].astype(np.float32, copy=False) for comb in combos]
                    )
                else:
                    boost_weights = np.asarray(boost_weights, dtype=np.float32)
                expected_shape = (len(X_train), len(combos))
                if boost_weights.shape != expected_shape:
                    tqdm.write(
                        "[RESUME] Weights snapshot shape mismatch: "
                        f"loaded={boost_weights.shape} expected={expected_shape}. "
                        "Training data assembly changed; refusing to resume from stale snapshot."
                    )
                    bit = None
                    boost_weights = None
                    start_tree = 0
                else:
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
    boost_weights = training_weights_train.copy()

# store the point the interface expanded around, so the fit reads the point the
# BIT was actually trained with rather than re-deriving it from GENERATION_POINT
bit.expansion_point = eft.reference_point

# If training needed but weights missing, start from truth
if boost_weights is None and len(bit.trees) < bit.n_trees:
    boost_weights = training_weights_train.copy()

# ---------------- external training loop ----------------
rt = J.get("runtime", {}) or {}
enable_plots = bool(rt.get("training_plots", False))
plot_ctx = _build_plot_context(X_train, training_weights_train, feat_names, cfg_base, J) if enable_plots else None

# ---------------- loss history ----------------
# loss_history.txt is appended to per tree (not batched in memory), so a crash
# mid-run never loses history, and a resumed run continues the same file
# instead of overwriting it.
loss_txt = os.path.join(model_dir, "loss_history.txt")
best_valid_loss = float("inf")
best_tree = -1
best_model_path = os.path.join(model_dir, "BIT_best.pkl")
best_weights_path = os.path.join(model_dir, "BIT_best.weights.pkl")  # 可选
best_txt = os.path.join(model_dir, "best_checkpoint.txt")

resuming = start_tree > 0

# resume the historical BEST if one was recorded, so a resumed run does not
# treat its first (already-trained) tree as automatically the best
if resuming and os.path.exists(best_txt):
    with open(best_txt, "r") as f:
        line = f.read().strip()
    if line:
        t, v = line.split()[:2]
        best_tree = int(t)
        best_valid_loss = float(v)
        tqdm.write(f"[RESUME] Found previous BEST: tree={best_tree}, valid_loss={best_valid_loss:.6g} (from {best_txt})")

# start a fresh loss_history.txt only on a fresh run; a resumed run appends
if not resuming or not os.path.exists(loss_txt):
    with open(loss_txt, "w") as f:
        f.write("# tree\ttrain_loss\tvalid_loss\n")

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
            training_weights = weight_views(boost_weights),
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
        # Disabled to avoid a full extra predict(X_train) pass every epoch.
        train_loss = float("nan")

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
                with open(best_txt, "w") as f:
                    f.write(f"{best_tree} {best_valid_loss:.12g}\n")
                tqdm.write(f"[BEST] tree={best_tree:04d} valid_loss={best_valid_loss:.6g} -> {best_model_path}")

        # ---------------- append loss history (write immediately: survives a crash, and a
        # resumed run continues the same file instead of losing the earlier trees) ----------------
        tree_now = len(bit.trees)
        valid_loss_to_write = float(valid_loss) if X_valid is not None else float("nan")
        with open(loss_txt, "a") as f:
            f.write(f"{tree_now}\t{float(train_loss):.8e}\t{valid_loss_to_write:.8e}\n")

        # update weights
        t1 = time.process_time()
        prediction = root.vectorized_predict(X_train)
        lr_eff = 1.0 if _get_only_score else float(bit.learning_rate)
        denom = prediction[:, 0]
        w0 = boost_weights[:, 0]
        for i_der, der in enumerate(root.derivatives[1:]):
            np.divide(
                prediction[:, i_der + 1],
                denom,
                out=prediction[:, i_der + 1],
                where=(denom != 0),
            )
            boost_weights[:, combo_to_col[tuple(sorted(der))]] += -lr_eff * w0 * prediction[:, i_der + 1]

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

    # ---------------- plot loss curves ----------------
    # loss_txt already holds the full history (this run plus any resumed runs before
    # it, since it is appended to per tree, not overwritten). Read it back rather than
    # keeping a parallel in-memory copy, so the plot always matches the file on disk.
    loss_trees, train_losses, valid_losses = [], [], []
    with open(loss_txt, "r") as f:
        next(f)  # header
        for line in f:
            t, tr, va = line.split()
            loss_trees.append(int(t))
            train_losses.append(float(tr))
            valid_losses.append(float(va))

    plt.figure()
    plt.plot(loss_trees, train_losses, label="train")
    # only plot valid curve if at least one finite value exists
    va_arr = np.array(valid_losses, dtype=float)
    if np.isfinite(va_arr).any():
        plt.plot(loss_trees, valid_losses, label="valid")
    plt.xlabel("n_trees")
    plt.ylabel("ratio_mse_loss")
    plt.axvline(best_tree, color='r', label="best epoch")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    loss_pdf = os.path.join(plot_dir, "loss_history.pdf")
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

open(f"{model_dir}/done","w")
print("Done.")
