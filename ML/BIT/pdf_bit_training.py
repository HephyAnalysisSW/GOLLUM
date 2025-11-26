#!/usr/bin/env python
from __future__ import annotations
import os, sys, argparse, importlib, yaml
import numpy as np

# project roots
sys.path.insert(0, '..'); sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader

from pdf.PDFParametrization import PDFParametrization

# ---------------- args ----------------
p = argparse.ArgumentParser(description="BIT training (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model file?")
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
    if args.overwrite: flags.append("--overwrite")
    if args.small:     flags.append("--small")
    if args.numba:     flags.append("--numba")
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
L.setFeatures( J["features"] )
feat_names = list(getattr(L, "feature_names", []) or [])
if not feat_names:
    raise RuntimeError("Loader has no feature_names.")
input_dim = len(feat_names)

# observers: must contain generator columns in this order
GEN_OBS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]
obs_names = list(getattr(L, "observer_names", []) or [])
missing_gen = [n for n in GEN_OBS if n not in obs_names]
if missing_gen:
    raise RuntimeError(f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'.")

# ---------------- PDF parametrization & combinations ----------------
pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", 'Chebyshev')
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

# Build the dict that BIT expects: {combination: vector}; () term is now the nominal weight
training_weights = {combos[i]: DER_all[:, i] for i in range(len(combos))}
if args.small:
    n_max = len(X_all)//30
    X_all   = X_all[:n_max]
    DER_all = DER_all[:n_max]
    training_weights = {key:val[:n_max] for key, val in training_weights.items()}

# ---------------- build & train BIT ----------------
cfg_base = os.path.join( CFG.get("version", "default"), J['region'] )
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, J.get("output", {}).get("filename", "BIT.pkl"))
if args.small:
    model_path = model_path[:-4]+"_small.pkl"

bit = None
if not args.overwrite:
    print(f"Attempt to load BIT from {model_path}")
    if os.path.exists(model_path):
        try:
            bit = MultiBoostedInformationTree.load(model_path)
            print(f"Loaded BIT from {model_path}")
        except Exception:
            pass
            print("Failed. Training new.")
    else:
        print("Not found. Training new.")

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

    # --- matrix plotting per iteration (one canvas, many pads) ---
    # expects: iters, bit, X_all (features), w0 (nominal weights),
    #          training_weights (dict keyed by derivative tuples),
    #          feat_names, user, cfg_base, J, PLOT_OPTS, colors (map der->color)
    import ROOT, os, math
    ROOT.gStyle.SetOptStat(0)

    stuff = []  # keep ROOT objects alive

    for t in iters:
        # predictions: shape (N, M-1) aligned to derivatives[1:]
        pred = bit.predict(X_all, max_n_tree=t)

        # Build truth ratios per derivative (all, including () at position 0)
        ders = bit.derivatives
        truth_mat = np.stack([
            training_weights.get(der, training_weights.get(tuple(reversed(der))))
            for der in ders
        ], axis=1)  # (N, M)

        # binning cache
        feat_bins = {
            f: np.linspace(PLOT_OPTS[f]['binning'][1],
                           PLOT_OPTS[f]['binning'][2],
                           PLOT_OPTS[f]['binning'][0] + 1)
            for f in feat_names
        }

        # --- build all histos first, then draw in a matrix ---
        # Per feature we’ll collect yield, truth and pred histos for all derivatives
        per_feat = {}  # f -> dict(yield, truth{der}, pred{der}, bins)
        for feat in feat_names:
            bins = feat_bins[feat]
            idx  = feat_names.index(feat)
            binned = np.digitize(X_all[:, idx], bins)
            mask = (binned.reshape(-1, 1) == np.arange(1, len(bins))).T  # (B, N)

            # yields per bin
            h_w0 = np.array([w0[m].sum() for m in mask])  # (B,)

            # sums per derivative per bin
            h_pred   = np.array([(w0.reshape(-1, 1) * pred)[m].sum(axis=0) for m in mask])   # (B, M)
            h_truth  = np.array([ truth_mat[m].sum(axis=0)                    for m in mask])      # (B, M)

            # ratios
            def _safe_div(numer, denom):
                denom2 = denom.reshape(-1, 1)
                out = np.zeros_like(numer, dtype=float)
                np.divide(numer, denom2, out=out, where=(denom2 != 0))
                return out

            r_pred  = _safe_div(h_pred,  h_w0)  # (B, M)
            r_truth = _safe_div(h_truth, h_w0)  # (B, M)

            # ROOT histos
            # yield (normalize into visible band later)
            hY = ROOT.TH1F(f"hY_{feat}", "", len(bins)-1, bins[0], bins[-1])
            for b in range(1, len(bins)):
                hY.SetBinContent(b, h_w0[b-1])
            hY.SetLineColor(ROOT.kGray + 2); hY.SetMarkerStyle(0); hY.SetLineWidth(2)
            hY.GetXaxis().SetTitle(PLOT_OPTS[feat]['tex']); hY.SetTitle("")
            stuff.append(hY)

            H_truth = {}
            H_pred  = {}
            for i_der, der in enumerate(ders):
                # skip drawing the () baseline ratio (always 1); keep objects small but stable
                if len(der) == 0:
                    continue
                hT = ROOT.TH1F(f"hT_{feat}_{i_der}", "", len(bins)-1, bins[0], bins[-1])
                hP = ROOT.TH1F(f"hP_{feat}_{i_der}", "", len(bins)-1, bins[0], bins[-1])
                for b in range(1, len(bins)):
                    hT.SetBinContent(b, r_truth[b-1, i_der])
                    hP.SetBinContent(b, r_pred[b-1,  i_der])
                col = colors.get(der, ROOT.kBlue)
                for h, sty in ((hT, 2), (hP, 1)):
                    h.SetLineColor(col); h.SetLineStyle(sty); h.SetLineWidth(2); h.SetMarkerStyle(0)
                    h.GetXaxis().SetTitle(PLOT_OPTS[feat]['tex'])
                    stuff.append(h)
                H_truth[der] = hT; H_pred[der] = hP

            per_feat[feat] = dict(yield_=hY, truth=H_truth, pred=H_pred, bins=bins)

        # --- draw on a single canvas with pads (features + legend) ---
        n_feat = len(feat_names)
        total_pads = n_feat + 1
        gx = int(math.ceil(math.sqrt(total_pads)))
        gy = int(math.ceil(total_pads / gx))
        c = ROOT.TCanvas(f"c_iter_{t}", f"BIT iter {t}", 500*gx, 500*gy)
        c.Divide(gx, gy)

        # Legend goes into last pad
        leg = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
        leg.SetBorderSize(0); leg.SetFillStyle(0)
        leg.SetNColumns( min(3, 1 + len(ders)//10) )
        stuff.append(leg)

        # Determine y-range based on truth curves across all features (data-driven)
        # We use per-feature truth ranges when drawing; this is just for reference.

        # Draw each feature into its pad
        for i, feat in enumerate(feat_names):
            pad = c.cd(i + 1)
            pad.SetTicks(1, 1); pad.SetBottomMargin(0.15); pad.SetLeftMargin(0.15)
            pad.SetLogy(False)  # ratio-like plots

            bins = per_feat[feat]['bins']
            n_bins = len(bins) - 1

            # find dynamic Y range from truth curves (data-driven)
            y_max = 0.0
            y_min = +1e9
            for der, hT in per_feat[feat]['truth'].items():
                if hT.GetMaximum() > y_max: y_max = hT.GetMaximum()
                # scan bins for min > 0 (allow negative if present)
                for b in range(1, n_bins+1):
                    y = hT.GetBinContent(b)
                    if y < y_min: y_min = y
            if not np.isfinite(y_min): y_min = 0.0
            if not np.isfinite(y_max): y_max = 1.0
            # pad margins
            y_pad = 0.2 * (y_max - y_min if y_max > y_min else 1.0)
            y_low = y_min - y_pad
            y_hi  = y_max + y_pad
            if y_hi <= y_low: y_hi = y_low + 1.0

            # frame
            hframe = ROOT.TH2F(f"hf_{feat}", f";{PLOT_OPTS[feat]['tex']};ratio",
                               n_bins, bins[0], bins[-1], 100, y_low, y_hi)
            hframe.GetYaxis().SetTitleOffset(1.3)
            hframe.Draw()
            stuff.append(hframe)

            # draw yield normalized into the same band
            hY = per_feat[feat]['yield_']
            # normalize yield into [y_low, y_hi] gently
            y_min0, y_max0 = 0.0, hY.GetMaximum()
            if y_max0 > 0:
                for b in range(1, n_bins+1):
                    v = hY.GetBinContent(b)
                    scaled = y_low + 0.92*(y_hi - y_low) * (v - y_min0) / max(1e-12, y_max0)
                    hY.SetBinContent(b, scaled)
            hY.SetLineColor(ROOT.kGray + 2)
            hY.Draw("hist same")

            first = True
            # draw all derivatives (truth dashed, pred solid)
            for der in ders:
                if len(der) == 0:  # skip ()
                    continue
                hT = per_feat[feat]['truth'][der]
                hP = per_feat[feat]['pred'][der]
                if first:
                    hT.Draw("hist same"); hP.Draw("hist same")
                    first = False
                else:
                    hT.Draw("hist same"); hP.Draw("hist same")

        # Legend in the last pad
        pad = c.cd(n_feat + 1)
        pad.SetTicks(1,1); pad.SetBottomMargin(0.15); pad.SetLeftMargin(0.15)
        # Fill legend entries once
        added = set()
        for der in ders:
            if len(der) == 0:  # skip ()
                continue
            # take any feature’s histos for legend prototypes
            sample_feat = feat_names[0]
            hT = per_feat[sample_feat]['truth'][der]
            hP = per_feat[sample_feat]['pred'][der]
            if der not in added:
                tex = "R" + str(der)
                texh= "#hat{R}" + str(der)
                leg.AddEntry(hT, tex, "l")
                leg.AddEntry(hP, texh, "l")
                added.add(der)

        # also add yield descriptor
        leg.AddEntry(per_feat[feat_names[0]]['yield_'], "yield (SM, scaled)", "l")
        leg.Draw()
        stuff.append(leg)

        # annotate iteration
        tl = ROOT.TLatex(); tl.SetNDC(); tl.SetTextSize(0.07); tl.SetTextAlign(11)
        tl.DrawLatex(0.30, 0.95, f"Trees = {t:04d}")
        stuff.append(tl)

        out_dir = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"], "train")
        os.makedirs(out_dir, exist_ok=True)
        c.Print(os.path.join(out_dir, f"iter_{t:04d}.png"))
        c.Close()

        syncer.sync()

print("Done.")

