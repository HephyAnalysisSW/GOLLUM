#!/usr/bin/env python
from __future__ import annotations
import os, sys, argparse, importlib, yaml
import numpy as np

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer

from ML.BIT.MultiBoostedInformationTree import MultiBoostedInformationTree
from pdf.PDFParametrization import PDFParametrization

# ---------------- args ----------------
p = argparse.ArgumentParser(description="BIT training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model file?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
with open(cfg_path, "r") as f:
    CFG = yaml.safe_load(f) or {}
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
feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")
input_dim = len(feat_names)

# observers: must contain generator columns in this order
GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

# ---------------- PDF parametrization & combinations ----------------
pdf_n = int(J.get("pdf", {}).get("cheb_n", 5))
pdf = PDFParametrization(n=pdf_n)                     # defines variables: ['c0',..,'cN']
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
        gx1  = G[:, on2idx["Generator_x1"]]
        gx2  = G[:, on2idx["Generator_x2"]]
        gid1 = G[:, on2idx["Generator_id1"]]
        gid2 = G[:, on2idx["Generator_id2"]]
        yield (X.astype(np.float32, copy=False),
               gx1.astype(np.float32, copy=False),
               gx2.astype(np.float32, copy=False),
               gid1.astype(np.int32,  copy=False),
               gid2.astype(np.int32,  copy=False),
               w.astype(np.float32,   copy=False))

Xs = []
targets_acc = []  # list of (N_i, len(combos)) arrays
for X, x1, x2, id1, id2, w in iterate_all():
    # Unweighted derivatives aligned with pdf.combinations
    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2)            # (N_i, M)
    # Multiply each column by the event weight (treating derivatives as reweights)
    deriv_w = deriv * w.reshape(-1, 1).astype(np.float32, copy=False)   # (N_i, M)
    Xs.append(X)
    targets_acc.append(deriv_w)

X_all   = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
DER_all = np.concatenate(targets_acc, axis=0) if len(targets_acc) > 1 else targets_acc[0]

# Build the dict that BIT expects: {combination: vector}; () term is now the nominal weight
training_weights = {combos[i]: DER_all[:, i] for i in range(len(combos))}

# ---------------- build & train BIT ----------------
cfg_base = os.path.splitext(os.path.basename(cfg_path))[0]
model_dir = os.path.join(user.model_directory, "BIT", cfg_base, J["id"])
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))

bit = None
if (not args.overwrite) and os.path.exists(model_path):
    try:
        print(f"Trying to load BIT from {model_path}")
        bit = MultiBoostedInformationTree.load(model_path)
    except Exception:
        bit = None

if bit is None:
    # pull BIT hyperparameters from YAML
    mcfg = J.get("model", {}) or {}
    bit = MultiBoostedInformationTree(
        training_features = X_all,
        training_weights  = training_weights,   # targets = derivatives ((), ('c0',), ..., quadratic)
        base_points       = base_points,        # as in legacy script
        feature_names     = feat_names,
        **mcfg,
    )
    bit.boost()
    bit.save(model_path)
    print(f"Saved BIT -> {model_path}")
else:
    print("Loaded existing BIT.")

# ---------------- optional training plots ----------------
rt = J.get("runtime", {}) or {}
if bool(rt.get("training_plots", False)):
    # reuse the existing plotting block inside your previous script;
    # the BIT instance and data arrays are now available here:
    #   - bit (trained)
    #   - X_all (features)
    #   - training_weights (truth derivatives; training_weights[()] is the () term)
    import ROOT, math
    from data.plot_options import plot_options as PLOT_OPTS
    plot_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"])
    os.makedirs(plot_dir, exist_ok=True)

    tex = ROOT.TLatex(); ROOT.gStyle.SetOptStat(0); ROOT.gROOT.SetBatch(True)
    # simple scalar-color map for derivatives:
    colors = {}
    i_lin, i_diag, i_mix = 0, 0, 0
    for der in bit.derivatives:
        if len(der) == 1:         colors[der] = ROOT.kAzure + i_lin;  i_lin  += 1
        elif len(der) == 2 and len(set(der)) == 1:
                                  colors[der] = ROOT.kRed + i_diag;   i_diag += 1
        else:                     colors[der] = ROOT.kGreen + i_mix;  i_mix  += 1

    # choose a few iterations
    iters = list(range(1, min(10, bit.n_trees)+1)) + list(range(10, bit.n_trees+1, 10))
    w0 = training_weights[()]  # () term from derivatives

    def _safe_div(n, d):
        d2 = d.reshape(-1,1)
        out = np.zeros_like(n, dtype=float)
        np.divide(n, d2, out=out, where=(d2!=0))
        return out

    for t in iters:
        pred = bit.vectorized_predict(X_all, max_n_tree=t)  # shape (N, M-1) ? in BIT code it's aligned to derivatives[1:]
        # Build truth ratios per derivative (skip () which is index 0 in combos)
        # Map derivatives order to combos order held by BIT:
        ders = bit.derivatives
        truth_mat = np.stack([training_weights.get(der if der in training_weights else tuple(reversed(der))) for der in ders], axis=1)
        # feature-binned overview
        from data.plot_options import plot_options as PLOT_OPTS
        feat_bins = {f: np.linspace(PLOT_OPTS[f]['binning'][1], PLOT_OPTS[f]['binning'][2], PLOT_OPTS[f]['binning'][0]+1) for f in feat_names}

        for feat in feat_names:
            bins = feat_bins[feat]
            idx  = feat_names.index(feat)
            binned = np.digitize(X_all[:, idx], bins)
            mask = (binned.reshape(-1,1) == np.arange(1, len(bins))).T  # (B,N)

            h_w0 = np.array([w0[m].sum() for m in mask])  # (B,)
            # pred is missing the () column; align:
            # build pred_full with ()=zeros then following derivatives:
            pred_full = np.zeros((pred.shape[0], len(ders)), dtype=float)
            pred_full[:,1:] = pred
            h_pred = np.array([(w0.reshape(-1,1)*pred_full)[m].sum(axis=0) for m in mask])  # (B, M)
            h_truth = np.array([(truth_mat)[m].sum(axis=0) for m in mask])                   # (B, M)

            r_pred  = _safe_div(h_pred,  h_w0)   # (B, M)
            r_truth = _safe_div(h_truth, h_w0)

            # quick plot for a couple of first derivatives
            import ROOT
            c = ROOT.TCanvas(f"c_{feat}_{t}", "", 900, 700); ROOT.gStyle.SetOptStat(0)
            leg = ROOT.TLegend(0.2,0.1,0.9,0.85); leg.SetNColumns(2); leg.SetBorderSize(0); leg.SetFillStyle(0)
            first = True
            for i_der, der in enumerate(ders[: min(6, len(ders))]):  # limit clutter
                hT = ROOT.TH1F(f"hT_{i_der}", "", len(bins)-1, bins[0], bins[-1])
                hP = ROOT.TH1F(f"hP_{i_der}", "", len(bins)-1, bins[0], bins[-1])
                for b in range(1, len(bins)):
                    hT.SetBinContent(b, r_truth[b-1, i_der])
                    hP.SetBinContent(b, r_pred[b-1, i_der])
                col = colors[der]
                for h, sty in [(hT, 2), (hP, 1)]:
                    h.SetLineColor(col); h.SetLineStyle(sty); h.SetLineWidth(2); h.SetMarkerStyle(0)
                    h.GetXaxis().SetTitle(PLOT_OPTS[feat]['tex'])
                    if first: h.Draw("hist")
                    else:     h.Draw("histsame")
                    first = False
                leg.AddEntry(hT, f"R{der}", "l"); leg.AddEntry(hP, f"Rhat{der}", "l")
            leg.Draw()
            out_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"], "train", feat)
            os.makedirs(out_dir, exist_ok=True)
            c.Print(os.path.join(out_dir, f"iter_{t:04d}.png"))
            c.Close()

    syncer.sync()

print("Done.")

