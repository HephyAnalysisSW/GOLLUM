#!/usr/bin/env python
from __future__ import annotations
import os, sys, time, argparse, importlib, warnings, yaml, math
import numpy as np
import tensorflow as tf

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer

from ML.PNN.PNN import PNN
from tqdm import trange, tqdm

# Plot options (binning, labels, optional y_ratio_range)
from data.plot_options import plot_options as PLOT_OPTS

# ---------------- args ----------------
p = argparse.ArgumentParser(description="PNN training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="PNN job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--for_debug", action="store_true", help="Fit, but don't overwrite the nominal version")
p.add_argument("--n_split", default=None, help="Set sample split")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
import common.yaml_loader as yaml_loader
CFG = yaml_loader.load_yaml(cfg_path)

D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")

def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "pnn"]
    if not jobs:
        print("No PNN jobs found.")
        sys.exit(0)
    flags = []
    if args.overwrite: flags.append("--overwrite")
    if args.small:     flags.append("--small")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}")
    sys.exit(0)

if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "pnn":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'pnn'.")
# ---------------- resolve loaders ----------------
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

samples_mod = importlib.import_module(module_samples)

bp_specs = J["base_points"]  # list of {coords: [...], loader: "name", optional removeweights/addweights}
base_points = [spec["coords"] for spec in bp_specs]

loaders = []

for i, spec in enumerate(bp_specs):
    nm = spec["loader"]
    if not hasattr(samples_mod, nm):
        raise RuntimeError(f"Loader/view '{nm}' not found in module {module_samples}.")
    base = getattr(samples_mod, nm)

    base.setFeatures( J["features"] )

    remove = list(spec.get("removeweights", []) or [])
    add    = list(spec.get("addweights", []) or [])

    # No weight modifications for this base point -> keep as-is
    if not remove and not add:
        loaders.append(base)
        continue

    # --------------------------------------------------------
    # 1) Get starting weight list depending on loader type
    # --------------------------------------------------------
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
                    "Layered views are not supported in this job logic."
                )
            base_weights = list(base.base.weight_branches or [])
        root_loader = base.base if isinstance(base.base, RDataLoader) else None
        if root_loader is None:
            raise RuntimeError(
                f"Could not find underlying RDataLoader for SelectionView '{base.name}'."
            )
    else:
        raise RuntimeError(
            f"Loader/view '{nm}' has unsupported type {type(base)} for automatic weight variations."
        )

    new_weights = list(base_weights)

    # --------------------------------------------------------
    # 2) Remove requested weights (warn if not present)
    # --------------------------------------------------------
    for w in remove:
        if w in new_weights:
            new_weights.remove(w)
        else:
            warnings.warn(
                f"[job {J.get('id', '<unknown>')}] weight '{w}' requested for removal "
                f"but not found in loader '{nm}' (current weights: {base_weights})."
            )

    # --------------------------------------------------------
    # 3) Add requested weights (avoid duplicates)
    # --------------------------------------------------------
    for w in add:
        if w not in new_weights:
            new_weights.append(w)

    # --------------------------------------------------------
    # 4) Ensure the underlying RDataLoader reads any new branches
    # --------------------------------------------------------
    # (we are guaranteed to have root_loader here)
    if hasattr(root_loader, "_requested_branches"):
        for b in add:
            if b not in root_loader._requested_branches:
                root_loader._requested_branches.append(b)

    if root_loader.observer_names is None:
        # If there were no observer names set, start with the added ones
        root_loader.observer_names = list(add)
    else:
        for b in add:
            if b not in root_loader.observer_names:
                root_loader.observer_names.append(b)

    # --------------------------------------------------------
    # 5) Construct the effective loader for this base point
    # --------------------------------------------------------
    if isinstance(base, RDataLoader):
        # Base is a loader -> make a simple view that only changes weights
        vname = f"{nm}_wvar{i}"
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

    # Reset n_split
    if args.n_split:
        for l in loaders:
            if isinstance( l, SelectionView): 
                l.base.split = args.n_split
            else:
                l.split = args.n_split

            
# ---------------- sanity: same features across loaders ----------------
feat_names = list(getattr(loaders[0], "feature_names", []))
if not feat_names:
    raise RuntimeError("First loader has no feature_names.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != feat_names:
        raise RuntimeError("Feature mismatch across base-point loaders.")

# ---------------- debug printout ----------------
print(f"\nResolved loaders for job '{J.get('id', '<unknown>')}':")
for idx, (spec, L) in enumerate(zip(bp_specs, loaders)):
    print(f"  base point {idx}, coords={spec['coords']}, loader spec='{spec['loader']}':")
    print(L)  # uses __str__ of RDataLoader / SelectionView
    print("-" * 60)

input_dim = len(feat_names)
feat2col = {f: i for i, f in enumerate(feat_names)}

## ---------------- resolve loaders ----------------
#samples_mod = importlib.import_module(module_samples)
#
#bp_specs = J["base_points"]  # list of {coords: [...], loader: "name"}
#base_points = [spec["coords"] for spec in bp_specs]
#loader_names = [spec["loader"] for spec in bp_specs]
#loaders = []
#for nm in loader_names:
#    if not hasattr(samples_mod, nm):
#        raise RuntimeError(f"Loader/view '{nm}' not found in module {module_samples}.")
#    loaders.append(getattr(samples_mod, nm))
#
## sanity: same features across loaders
#feat_names = list(getattr(loaders[0], "feature_names", []))
#if not feat_names:
#    raise RuntimeError("First loader has no feature_names.")
#for L in loaders[1:]:
#    if list(getattr(L, "feature_names", [])) != feat_names:
#        raise RuntimeError("Feature mismatch across base-point loaders.")

# ---------------- artifacts: scaler & ICP (IDs in YAML) ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J["region"])

# Scaler (by job id)
from ML.Scaler.Scaler import Scaler
scaler_id = J["extras"].get("use_scaler", None)
if scaler_id:
    sj     = next(jj for jj in (CFG.get("jobs") or []) if jj.get("id") == scaler_id)
    sname  = sj["output"]["filename"]
    spath  = os.path.join(user.model_directory, cfg_base, "Scaler", sname)
    sc     = Scaler.load(spath)
    scaler_means, scaler_vars = sc.feature_means, sc.feature_variances
    print(f"Loaded Scaler: {spath}")
else:
    scaler_means = np.zeros(input_dim, dtype=np.float64)
    scaler_vars  = np.ones(input_dim,  dtype=np.float64)
    print("No Scaler configured; using identity.")

# ---------------- build model ----------------
parameters      = list(J["parameters"])
combinations    = [tuple(c) for c in J["combinations"]]
hidden_layers   = J.get("model", {}).get("hidden_layers", [128, 128])
activation      = J.get("model", {}).get("activation", "relu")
initialize_zero = bool(J.get("model", {}).get("initialize_zero", False))
epochs          = int(J.get("optim", {}).get("epochs", 200))
phaseout        = int(J.get("optim", {}).get("phaseout_epochs", 0))
lr              = float(J.get("optim", {}).get("learning_rate", 1e-3))

pnn = None
model_dir = os.path.join(user.model_directory, cfg_base+("_for_debug" if args.for_debug else ""), "PNN", J["id"])
plot_dir  = os.path.join(user.plot_directory,  cfg_base+("_for_debug" if args.for_debug else ""), "PNN", J["id"])
os.makedirs(model_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

if not args.overwrite:
    try:
        print(f"Trying to load PNN from {model_dir}")
        pnn = PNN.load(model_dir)
        print("Success!")
    except Exception as e:
        pnn = None
        print("Failed! Gonna train.")
        #raise e

if pnn is None:
    pnn = PNN(parameters=parameters,
              combinations=combinations,
              base_points=base_points,
              input_dim=input_dim,
              hidden_layers=hidden_layers,
              activation=activation,
              learning_rate=lr,
              n_epochs=epochs,
              n_epochs_phaseout=phaseout,
              initialize_zero=initialize_zero)

pnn.set_scaler(scaler_means, scaler_vars)

# ICP (by job id) -> inject ΔA bias into the PNN (so epoch-0 == ICP-only if initialize_zero)
icp_id = J["extras"].get("use_icp", None)
if icp_id:
    from ML.ICP.ICP import InclusiveCrosssectionParametrization
    ij       = next(jj for jj in (CFG.get("jobs") or []) if jj.get("id") == icp_id)
    icp_fn   = ij["output"]["filename"]
    icp_path = os.path.join(user.model_directory, cfg_base, "ICP", icp_fn)
    icp      = InclusiveCrosssectionParametrization.load(icp_path)
    print(f"Loaded ICP: {icp_path}")

    _params = list(icp.parameters)
    _combs  = [tuple(c) for c in icp.combinations]
    _DeltaA = np.asarray(icp.DeltaA, dtype=np.float64)
    # Consistency with YAML is already enforced upstream; set bias:
    pnn.set_icp(parameters=_params, combinations=_combs, DeltaA=_DeltaA)

# ---------------- training utils ----------------
def lr_schedule(epoch):
    if phaseout <= 0: return lr
    if epoch < epochs - phaseout: return lr
    start = epochs - phaseout
    # linear phaseout down to ~0
    return lr * (1.0 - (epoch - start + 1) / phaseout)

def iterate_epoch(shard_limit=None):
    # ensure same number of shards
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)
    for shard in range(n_shards):
        # materialize for each base point: (X, w)
        Xs, Ws = [], []
        for L in loaders:
            X, w = L.materialize(shard=shard, what="fw")
            Xs.append(X); Ws.append(w.astype(np.float32, copy=False))
        yield Xs, Ws

# ---- plotting helpers (ROOT) ----
def init_histograms(plot_features, n_bp, rebin=1):
    h_true, h_pred, bins = {}, {}, {}
    for feat in plot_features:
        n, lo, hi = PLOT_OPTS[feat]['binning']
        n = max(1, n // max(1, int(rebin)))
        h_true[feat] = np.zeros((n, n_bp), dtype=np.float64)
        h_pred[feat] = np.zeros((n, n_bp), dtype=np.float64)
        bins[feat]   = np.linspace(lo, hi, n+1)
    return h_true, h_pred, bins

def accumulate_histograms(h_true, h_pred, bins, Xs, Ws, pnn, VkA, base_points, nom_idx, plot_features, feat2col):
    # true: from Xi, wi; pred: from X0, w0 * exp((ΔA_net+ΔA_icp)·V_k)
    X0, w0 = Xs[nom_idx], Ws[nom_idx]
    dA0 = pnn.deltaA(X0)  # (N0, C) includes ICP bias if set

    # --- fill NOMINAL column so normalization has a real reference ---
    for feat in plot_features:
        col   = feat2col[feat]
        edges = bins[feat]
        # true nominal = histogram of nominal sample
        ht0, _ = np.histogram(X0[:, col], bins=edges, weights=w0)
        h_true[feat][:, nom_idx] += ht0
        # pred nominal equals true nominal (bias=1, exp(ΔA·0)=1)
        h_pred[feat][:, nom_idx] += ht0
    # -----------------------------------------------------------------

    for i_bp, (Xi, wi) in enumerate(zip(Xs, Ws)):
        if i_bp == nom_idx:
            continue
        vk = VkA[i_bp]  # (C,)
        pred_w = w0 * np.exp(dA0 @ vk)

        for feat in plot_features:
            col   = feat2col[feat]
            edges = bins[feat]
            # true (Xi, wi)
            ht, _ = np.histogram(Xi[:, col], bins=edges, weights=wi)
            h_true[feat][:, i_bp] += ht
            # pred (X0, pred_w)
            hp, _ = np.histogram(X0[:, col], bins=edges, weights=pred_w)
            h_pred[feat][:, i_bp] += hp

def plot_convergence_root(true_h, pred_h, epoch, out_dir, feature_names, base_points, nom_idx, rebin=1):
    import ROOT, os, math
    import numpy as np
    try:
        ROOT.gStyle.SetOptStat(0)
        # try to set TDR if present, ignore otherwise
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    n_feat = len(feature_names)
    n_bp   = len(base_points)

    # color palette
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange+1, ROOT.kMagenta+1, ROOT.kCyan+2,
              ROOT.kViolet+1, ROOT.kAzure+1, ROOT.kPink+7, ROOT.kTeal+3]
    if n_bp > len(colors):
        colors = (colors * (n_bp // len(colors) + 1))[:n_bp]
    colors[nom_idx] = ROOT.kBlack

    for normalized in (False, True):
        # work on copies (avoid altering input)
        th = {k: v.copy() for k, v in true_h.items()}
        ph = {k: v.copy() for k, v in pred_h.items()}

        if normalized:
            # divide each bp by the true nominal spectrum (bin-by-bin)
            for feat in feature_names:
                ref = th[feat][:, nom_idx].copy()
                ref[ref == 0] = 1.0
                th[feat] = th[feat] / ref[:, None]
                ph[feat] = ph[feat] / ref[:, None]

        total_pads = n_feat + 1
        gx = int(math.ceil(math.sqrt(total_pads)))
        gy = int(math.ceil(total_pads / gx))
        canvas = ROOT.TCanvas("c_convergence", "PNN Convergence", 500*gx, 500*gy)
        canvas.Divide(gx, gy)

        keep = []

        for i, feat in enumerate(feature_names):
            pad = canvas.cd(i + 1)
            pad.SetTicks(1, 1)
            pad.SetBottomMargin(0.15)
            pad.SetLeftMargin(0.15)
            # logY on raw only if requested
            pad.SetLogy((not normalized) and PLOT_OPTS[feat].get('logY', False))

            n_bins, x_min, x_max = PLOT_OPTS[feat]['binning']
            n_bins = max(1, n_bins // max(1, int(rebin)))

            if normalized:
                # data-driven axis from truth only
                arr = th[feat]
                finite = np.isfinite(arr)
                if not finite.any():
                    tmin, tmax = 0.95, 1.05
                else:
                    tmin = float(np.min(arr[finite])); tmax = float(np.max(arr[finite]))
                    if not np.isfinite(tmin) or not np.isfinite(tmax) or abs(tmax - tmin) < 1e-9:
                        tmin, tmax = 0.95, 1.05
                span = max(1e-9, tmax - tmin)
                pad_frac = 0.10
                y_min = max(0.0, tmin - pad_frac * span)
                y_max = tmax + pad_frac * span
            else:
                max_y = 0.0
                for k in range(n_bp):
                    max_y = max(max_y, th[feat][:, k].max(), ph[feat][:, k].max())
                y_min, y_max = 0.0, (1.2*max_y if max_y > 0 else 1.0)

            y_title = "Probability" if not normalized else "Ratio to nominal"

            hframe = ROOT.TH2F(f"h_{feat}_{'norm' if normalized else 'raw'}",
                               f";{PLOT_OPTS[feat]['tex']};{y_title}",
                               n_bins, x_min, x_max, 100, y_min, y_max)
            hframe.GetYaxis().SetTitleOffset(1.3)
            hframe.Draw()
            keep.append(hframe)

            for k, nu in enumerate(base_points):
                # true dashed
                ht = ROOT.TH1F(f"t_{feat}_{k}_{'n' if normalized else 'r'}", "", n_bins, x_min, x_max)
                for b, y in enumerate(th[feat][:, k]): ht.SetBinContent(b+1, y)
                ht.SetLineColor(colors[k]); ht.SetLineStyle(2); ht.SetLineWidth(2); ht.Draw("HIST SAME")
                keep.append(ht)

                # pred solid
                hp = ROOT.TH1F(f"p_{feat}_{k}_{'n' if normalized else 'r'}", "", n_bins, x_min, x_max)
                for b, y in enumerate(ph[feat][:, k]): hp.SetBinContent(b+1, y)
                hp.SetLineColor(colors[k]); hp.SetLineStyle(1); hp.SetLineWidth(2); hp.Draw("HIST SAME")
                keep.append(hp)

        # legend
        pad = canvas.cd(n_feat + 1)
        leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9); leg.SetBorderSize(0); leg.SetShadowColor(0)
        leg.SetNColumns(1 + n_bp//20)
        dtrue, dpred = [], []
        for k, nu in enumerate(base_points):
            t = ROOT.TH1F(f"dt_{k}_{'n' if normalized else 'r'}", "", 1, 0, 1); t.SetLineColor(colors[k]); t.SetLineStyle(2); t.SetLineWidth(2)
            p = ROOT.TH1F(f"dp_{k}_{'n' if normalized else 'r'}", "", 1, 0, 1); p.SetLineColor(colors[k]); p.SetLineStyle(1); p.SetLineWidth(2)
            dtrue.append(t); dpred.append(p)
            leg.AddEntry(t, f"{tuple(nu)} (true)", "l")
            leg.AddEntry(p, f"{tuple(nu)} (pred)", "l")
        leg.Draw(); keep.extend(dtrue + dpred)

        tex = ROOT.TLatex(); tex.SetNDC(); tex.SetTextSize(0.07); tex.SetTextAlign(11)
        tex.DrawLatex(0.30, 0.95, f"Epoch = {epoch:04d}")
        keep.append(tex)

        fname = os.path.join(out_dir, f"{'norm_' if normalized else ''}epoch_{epoch:04d}.png")
        for fmt in ["png"]:
            canvas.SaveAs(fname.replace(".png", f".{fmt}"))

# ---------------- train ----------------
start_epoch = 0
if not args.overwrite:
    latest = tf.train.latest_checkpoint(model_dir)
    if latest:
        try:
            start_epoch = int(os.path.basename(latest)) + 1
        except Exception:
            pass

shard_limit = 1 if args.small else None
plot_every  = int(J.get("runtime", {}).get("plot_every", 5))
rebin       = int(J.get("runtime", {}).get("rebin", 1))

VkA = pnn.VkA
nom_idx = pnn.nominal_base_point_index

for epoch in trange(start_epoch, epochs, desc="Epoch"):
    # LR
    new_lr = lr_schedule(epoch)
    pnn.optimizer.learning_rate.assign(float(new_lr))

    total_loss = 0.0
    n_batches = 0

    # gradient accumulation buffers (same shape as model params)
    grad_sums = [tf.zeros_like(v) for v in pnn.model.trainable_variables]

    # plotting accumulation
    do_plot = (epoch % plot_every == 0)
    plot_feats = [f for f in feat_names if f in PLOT_OPTS]
    if do_plot:
        true_h, pred_h, bins = init_histograms(plot_feats, n_bp=len(base_points), rebin=rebin)

    #for Xs, Ws in iterate_epoch(shard_limit=shard_limit):
    for Xs, Ws in tqdm(  iterate_epoch(shard_limit=shard_limit), desc="Epoch",  unit="batch" ):

        n_batches += 1

        X0, w0 = Xs[nom_idx], Ws[nom_idx]
        X0n = pnn._normalize(X0)

        with tf.GradientTape() as tape:
            DeltaA0 = pnn.deltaA_tf(tf.convert_to_tensor(X0n, dtype=tf.float32), training=True)

            loss = 0.0
            for i_bp, (Xi, wi) in enumerate(zip(Xs, Ws)):
                if i_bp == nom_idx:
                    continue

                Xin = pnn._normalize(Xi)
                DeltaAi = pnn.deltaA_tf(tf.convert_to_tensor(Xin, dtype=tf.float32), training=True)

                v = tf.convert_to_tensor(VkA[i_bp], dtype=tf.float32)  # (C,)

                term0 = tf.reduce_sum(
                    tf.convert_to_tensor(w0, dtype=tf.float32)
                    * tf.nn.softplus(tf.linalg.matvec(DeltaA0, v))
                )
                termi = tf.reduce_sum(
                    tf.convert_to_tensor(wi, dtype=tf.float32)
                    * tf.nn.softplus(-tf.linalg.matvec(DeltaAi, v))
                )

                const = (np.sum(w0) + np.sum(wi)) * math.log(2.0)
                loss += term0 + termi - const

        # compute grads for this shard
        grads = tape.gradient(loss, pnn.model.trainable_variables)

        # accumulate gradients
        for j, (g, gsum) in enumerate(zip(grads, grad_sums)):
            if g is not None:
                grad_sums[j] = gsum + g

        # accumulate scalar loss for logging
        total_loss += float(loss.numpy())

        # accumulate plots from this shard
        if do_plot and len(X0) and all(len(Xi) for Xi in Xs):
            accumulate_histograms(
                true_h, pred_h, bins, Xs, Ws, pnn, VkA,
                base_points, nom_idx, plot_feats, feat2col
            )

    # ---- apply accumulated gradient once per epoch ----
    if n_batches > 0:
        # you can average or just sum; average is more robust w.r.t. batch count
        avg_grads = [g / float(n_batches) for g in grad_sums]
        # filter out None, just in case
        pnn.optimizer.apply_gradients(
            (g, v) for g, v in zip(avg_grads, pnn.model.trainable_variables) if g is not None
        )

    # total_loss now is the sum over all batches in the epoch
    # (you can divide by n_batches if you want an average loss)

    tqdm.write(f"Epoch {epoch}/{epochs-1} - LR {float(new_lr):.6f} - loss {total_loss:.4f}")
    pnn.save(model_dir, epoch=epoch)

    if do_plot:
        plot_convergence_root(true_h, pred_h, epoch, plot_dir, plot_feats, base_points, nom_idx, rebin=rebin)
        #syncer.makeRemoteGif(plot_dir, pattern="epoch_*.png",      name="epoch")
        #syncer.makeRemoteGif(plot_dir, pattern="norm_epoch_*.png", name="norm_epoch")
        syncer.sync()
    elif not args.small:
        syncer.sync()

print(f"Done. Model stored in {model_dir}")

