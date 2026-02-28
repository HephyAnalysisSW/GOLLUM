#!/usr/bin/env python
from __future__ import annotations
import os, sys, argparse, importlib, time, pickle, io, contextlib
import numpy as np

import cProfile, pstats

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader

from pdf.PDFParametrization import PDFParametrization

# Always NUMBA
import numba as nb
from ML.BIT.NumbaBIT import MultiBoostedInformationTree
import ML.BIT.NumbaMultiNode as NumbaMultiNode

from tqdm import trange, tqdm

# ---------------- args ----------------
p = argparse.ArgumentParser(description="BIT training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model file?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--profile", action="store_true", help="Do CPU profiling?")
p.add_argument("--every", default=5, type=int, help="When to plot (plot if tree_index % every == 0). Set <=0 to disable.")
args = p.parse_args()

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

sel  = J.get("selection", None)
sel_f= J.get("selection_features", [])
if sel:
    L.addSelection( sel, sel_f)
    print("Added selection to loader: {sel} and selection_features {sel_f}")

print(L)

print("Using NUMBA")
print("Numba threads:", nb.get_num_threads())

# features
L.setFeatures(J["features"])
feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")

# observers: must contain generator columns in this order
GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

# ---------------- PDF parametrization & combinations ----------------
pdf_n     = J.get("pdf", {}).get("pdf_n", None)
pdf_type  = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
pdf = PDFParametrization(n=pdf_n, typ=pdf_type, basis=pdf_basis)

combos = list(pdf.combinations)  # (), ('c0',), ..., ('ci','cj')

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
targets_acc = []
for X, Q, x1, x2, id1, id2, w in iterate_all():
    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q)  # (N_i, M)
    deriv_w = deriv * w.reshape(-1, 1).astype(np.float32, copy=False)
    Xs.append(X)
    targets_acc.append(deriv_w)

X_all   = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
DER_all = np.concatenate(targets_acc, axis=0) if len(targets_acc) > 1 else targets_acc[0]

# Truth weights (fixed; used for plotting)
training_weights = {tuple(sorted(combos[i])): DER_all[:, i] for i in range(len(combos))}

if args.small:
    n_max = len(X_all)//30
    X_all   = X_all[:n_max]
    DER_all = DER_all[:n_max]
    training_weights = {key: val[:n_max] for key, val in training_weights.items()}

# ---------------- plotting function ----------------
def plot_bit_training_root(bit, t, X_all, training_weights, feat_names, cfg_base, J):
    """
    Plot truth vs prediction ratios after t trees.
    Syncer output is captured so tqdm bars remain usable.
    """
    import ROOT, math
    from data.plot_options import plot_options as PLOT_OPTS

    plot_feats = [f for f in feat_names if f in PLOT_OPTS]
    if not plot_feats:
        tqdm.write("No plotable features found in PLOT_OPTS; skipping plots.")
        return

    out_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"], "train")
    os.makedirs(out_dir, exist_ok=True)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

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

    pred = bit.predict(X_all, max_n_tree=t)  # (N, M-1), aligned to ders[1:]
    w0 = training_weights[()]

    truth_mat = np.stack([
        training_weights.get(der, training_weights.get(tuple(reversed(der))))
        for der in ders
    ], axis=1)  # (N, M)

    total_pads = len(plot_feats) + 1
    gx = int(math.ceil(math.sqrt(total_pads)))
    gy = int(math.ceil(total_pads / gx))
    c = ROOT.TCanvas(f"c_iter_{t}", f"BIT iter {t}", 500*gx, 500*gy)
    c.Divide(gx, gy)
    keep = []

    leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
    leg.SetBorderSize(0); leg.SetFillStyle(0)
    leg.SetNColumns(min(3, 1 + len(ders)//10))
    keep.append(leg)

    def _safe_ratio(numer, denom):
        denom2 = denom.copy()
        denom2[denom2 == 0] = 1.0
        return numer / denom2

    for i, feat in enumerate(plot_feats):
        pad = c.cd(i + 1)
        pad.SetTicks(1, 1)
        pad.SetBottomMargin(0.15)
        pad.SetLeftMargin(0.15)

        n, lo, hi = PLOT_OPTS[feat]['binning']
        edges = np.linspace(lo, hi, n+1)
        col = feat_names.index(feat)
        x = X_all[:, col]

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

        hframe = ROOT.TH2F(f"hf_{feat}_{t}", f";{PLOT_OPTS[feat]['tex']};ratio",
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

    pad = c.cd(len(plot_feats) + 1)
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

    c.Print(os.path.join(out_dir, f"iter_{t:04d}.png"))
    c.Close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        syncer.sync()
    out = buf.getvalue().strip()
    if out:
        tqdm.write(out)

# ---------------- build & train BIT ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J['region'])
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))
if args.small:
    model_path = model_path[:-4] + "_small.pkl"

# weights snapshot stays separate (required to resume boosting correctly)
weights_path = model_path[:-4] + ".weights.pkl"

bit = None
start_tree = 0
boost_weights = None

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
    mcfg = J.get("model", {}) or {}
    bit = MultiBoostedInformationTree(**mcfg)
    boost_weights = {k: v.copy() for k, v in training_weights.items()}

# If training needed but weights missing, start from truth
if boost_weights is None and len(bit.trees) < bit.n_trees:
    boost_weights = {k: v.copy() for k, v in training_weights.items()}

# ---------------- external training loop ----------------
rt = J.get("runtime", {}) or {}
enable_plots = bool(rt.get("training_plots", False))


if len(bit.trees) < bit.n_trees:

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
            X_all,
            training_weights = boost_weights,
            base_points      = base_points,
            feature_names    = feat_names,
            **bit.node_cfg
        )
        t2 = time.process_time()
        weak_learner_time += (t2 - t1)

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

        # update weights
        t1 = time.process_time()
        prediction = root.vectorized_predict(X_all)
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
            plot_bit_training_root(bit, t=n_tree+1, X_all=X_all, training_weights=training_weights,
                                   feat_names=feat_names, cfg_base=cfg_base, J=J)

    # After last training: keep weights file, print filename
    szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
    tqdm.write(f"Kept boosting weights snapshot -> {weights_path} ({szz:.1f} MB)")

else:
    if os.path.exists(weights_path):
        szz = os.path.getsize(weights_path) / (1024.0 * 1024.0)
        tqdm.write(f"Kept boosting weights snapshot -> {weights_path} ({szz:.1f} MB)")


print("Done.")

