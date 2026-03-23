#!/usr/bin/env python
from __future__ import annotations
import os
import sys
import math
import argparse
import importlib
import io
import contextlib
from tqdm import tqdm

import numpy as np

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader
from pdf.PDFParametrization import PDFParametrization


# ---------------- args ----------------
p = argparse.ArgumentParser(description="Truth-only PDF reweight plots (YAML-driven)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="BIT job id to run (omit to list)")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--postfix", default="")
p.add_argument(
    "--point",
    action="append",
    nargs="*",
    metavar="POI=VALUE",
    help=(
        "Parameter point to plot. Can be given multiple times. "
        "Example: --point c1=1.0 c2=-0.5 --point c1=-1.0"
    ),
)
p.add_argument(
    "--active-pdgids",
    nargs="*",
    default="all",
    help='List of active PDG IDs, e.g. --active_pdgids 11 13 22. Defaults to "all".',
)
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

    script = os.path.basename(__file__)
    extra = " --small" if args.small else ""
    for j in jobs:
        print(f"python {script} {args.config}{extra} --job {j['id']} --point c1=1.0")
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
L.set_n_split(20)
sel = J.get("selection", None)
sel_f = J.get("selection_features", [])
if sel:
    L.addSelection(sel, sel_f)
    print(f"Added selection to loader: {sel} and selection_features {sel_f}")

print(L)

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
    raise RuntimeError(
        f"Observer_names must include {GEN_OBS}, missing {missing_gen} in loader '{loader_name}'."
    )

# ---------------- PDF parametrization ----------------
#pdf_basis = "gluon_POD_nongluon_NNPDF31_hessian"
#pdf_basis = "gluon_POD_nongluon_NNPDF40"
#pdf_basis = "gluon_POD_nongluon_PDF4LHC21"
pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
pdf_rescale_pod_amplitudes = J.get("pdf", {}).get("rescale_pod_amplitudes", True)

pdf = PDFParametrization(n=pdf_n, typ=pdf_type, basis=pdf_basis, 
     active_pids=args.active_pdgids if args.active_pdgids=='all' else list(map(int, args.active_pdgids)),
     rescale_pod_amplitudes=pdf_rescale_pod_amplitudes,
    )

pdf.print()

combos = [tuple(sorted(c)) for c in pdf.combinations]
allowed_pois = set(pdf.variables)

# ---------------- parse parameter points ----------------
def parse_points(raw_points):
    if not raw_points:
        return [dict()]  # SM by default

    out = []
    for raw in raw_points:
        point = {}
        for item in raw:
            if "=" not in item:
                raise RuntimeError(f"Malformed POI assignment '{item}'. Use POI=VALUE.")
            name, val = item.split("=", 1)
            if name not in allowed_pois:
                raise RuntimeError(
                    f"Unknown POI '{name}'. Allowed POIs: {', '.join(sorted(allowed_pois))}"
                )
            point[name] = float(val)
        out.append(point)
    return out

points = parse_points(args.point)


def point_label(point):
    if not point:
        return "SM"
    return ", ".join(f"{k}={v:g}" for k, v in point.items())


def sanitize_token(s):
    return (
        str(s)
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "")
    )


def point_tag(point):
    if not point:
        return "SM"
    return "__".join(f"{k}_{sanitize_token(f'{v:g}')}" for k, v in point.items())


# ---------------- collect all data ----------------
def iterate_all():
    n_shards = len(L)
    if args.small:
        n_shards = 1

    on2idx = {n: i for i, n in enumerate(obs_names)}
    i_Q = on2idx["Generator_scalePDF"]
    i_x1 = on2idx["Generator_x1"]
    i_x2 = on2idx["Generator_x2"]
    i_id1 = on2idx["Generator_id1"]
    i_id2 = on2idx["Generator_id2"]

    i_run = on2idx['run']
    i_event = on2idx['event']
    i_lumi = on2idx['luminosityBlock']

    for shard in range(n_shards):
        X, G, w = L.materialize(shard=shard, what="fow")
        yield (
            X.astype(np.float32, copy=False),
            G[:, i_Q].astype(np.float32, copy=False),
            G[:, i_x1].astype(np.float32, copy=False),
            G[:, i_x2].astype(np.float32, copy=False),
            G[:, i_id1].astype(np.int32, copy=False),
            G[:, i_id2].astype(np.int32, copy=False),
            w.astype(np.float32, copy=False),
            G[:, i_run].astype(np.int32, copy=False),
            G[:, i_event].astype(np.int32, copy=False),
            G[:, i_lumi].astype(np.int32, copy=False),
        )


Xs, targets = [], []
x1s, x2s, id1s, id2s, Qs, Rs, Es, Ls = [], [], [], [], [], [], [], []
for X, Q, x1, x2, id1, id2, w, R, E, Lum in tqdm( iterate_all(), total=(1 if args.small else len(L)), desc="Shards", unit="shard"):
    deriv = pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q)
    deriv_w = deriv * w.reshape(-1, 1).astype(np.float32, copy=False)
    Xs.append(X)
    targets.append(deriv_w)
    x1s.append(x1)
    x2s.append(x2)
    id1s.append(id1)
    id2s.append(id2)
    Qs.append(Q)

    Rs.append(R)
    Es.append(E)
    Ls.append(Lum)

X = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
Q = np.concatenate(Qs, axis=0) if len(Qs) > 1 else Qs[0]
x1 = np.concatenate(x1s, axis=0) if len(x1s) > 1 else x1s[0]
x2 = np.concatenate(x2s, axis=0) if len(x2s) > 1 else x2s[0]
id1 = np.concatenate(id1s, axis=0) if len(id1s) > 1 else id1s[0]
id2 = np.concatenate(id2s, axis=0) if len(id2s) > 1 else id2s[0]
run   = np.concatenate(Rs, axis=0) if len(Rs) > 1 else Rs[0]
event = np.concatenate(Es, axis=0) if len(Es) > 1 else Es[0]
lumi  = np.concatenate(Ls, axis=0) if len(Ls) > 1 else Ls[0]


DER = np.concatenate(targets, axis=0) if len(targets) > 1 else targets[0]
training_weights = {combos[i]: DER[:, i] for i in range(len(combos))}

def weight_at_point_old(point):
    """
    Quadratic polynomial in the derivative-combination basis:
      w(theta) = sum_c [ prod_{v in c} theta_v ] * w_c
    with coefficient 1 for the empty tuple ().
    """
    w = training_weights[()].astype(np.float64, copy=True)
    for comb, wc in training_weights.items():
        if len(comb) == 0:
            continue
        coeff = 1.0
        for v in comb:
            coeff *= point.get(v, 0.0)
        if coeff != 0.0:
            w += coeff * wc
    return w

def weight_at_point(point):
    w = training_weights[()].astype(np.float64, copy=True)
    for comb, wc in training_weights.items():
        if len(comb) == 0:
            continue
        coeff = 1.0
        for v in comb:
            coeff *= point.get(v, 0.0)
        if len(comb) == 2 and comb[0] == comb[1]:
            coeff *= 0.5
        if coeff != 0.0:
            w += coeff * wc
    return w

import ROOT
from data.plot_options import plot_options as PLOT_OPTS

plot_feats = [f for f in feat_names if f in PLOT_OPTS]
if not plot_feats:
    raise RuntimeError("No plotable features found in PLOT_OPTS.")

cfg_base = os.path.join(CFG.get("version", "default"), J["region"])
out_dir = os.path.join(user.plot_directory, "PDF", cfg_base, J["id"], "truth")
os.makedirs(out_dir, exist_ok=True)

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

colors = [
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kCyan + 1,
    ROOT.kViolet + 1,
    ROOT.kSpring + 5,
    ROOT.kPink + 7,
]

curves = []
for i, point in enumerate(points):
    curves.append(
        {
            "point": point,
            "label": point_label(point),
            "tag": point_tag(point),
            "weight": weight_at_point(point),
            "color": colors[i % len(colors)],
        }
    )

w0 = training_weights[()]

total_pads = len(plot_feats) + 1
gx = int(math.ceil(math.sqrt(total_pads)))
gy = int(math.ceil(total_pads / gx))
c = ROOT.TCanvas("c_truth", "truth", 500 * gx, 500 * gy)
c.Divide(gx, gy)

keep = []

def safe_ratio(num, den):
    den2 = den.copy()
    den2[den2 == 0] = 1.0
    return num / den2

for i_feat, feat in enumerate(plot_feats):
    pad = c.cd(i_feat + 1)
    pad.SetTicks(1, 1)
    pad.SetBottomMargin(0.15)
    pad.SetLeftMargin(0.15)

    n, lo, hi = PLOT_OPTS[feat]["binning"]
    edges = np.linspace(lo, hi, n + 1)
    col = feat_names.index(feat)
    x = X[:, col]

    h_sm, _ = np.histogram(x, bins=edges, weights=w0)

    ratios = []
    for crv in curves:
        h, _ = np.histogram(x, bins=edges, weights=crv["weight"])
        crv["hist_" + feat] = h
        crv["ratio_" + feat] = safe_ratio(h, h_sm)
        ratios.append(crv["ratio_" + feat])

    vals = np.concatenate(ratios) if ratios else np.array([1.0])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        y_min, y_max = 0.0, 2.0
    else:
        y_min = float(np.min(vals))
        y_max = float(np.max(vals))
        if y_max <= y_min:
            y_max = y_min + 1.0

    pad_frac = 0.20
    y_low = y_min - pad_frac * (y_max - y_min)
    y_hi = y_max + pad_frac * (y_max - y_min)

    hframe = ROOT.TH2F(
        f"hf_{feat}",
        f";{PLOT_OPTS[feat]['tex']};ratio to SM",
        n, lo, hi, 100, y_low, y_hi
    )
    hframe.GetYaxis().SetTitleOffset(1.3)
    hframe.Draw()
    keep.append(hframe)

    # scaled nominal yield, same idea as in the training plot
    hY = ROOT.TH1F(f"hY_{feat}", "", n, lo, hi)
    for b in range(1, n + 1):
        hY.SetBinContent(b, float(h_sm[b - 1]))
    y_max0 = float(np.max(h_sm) if len(h_sm) else 0.0)
    if y_max0 > 0:
        for b in range(1, n + 1):
            v = hY.GetBinContent(b)
            scaled = y_low + 0.92 * (y_hi - y_low) * (v / max(1e-12, y_max0))
            hY.SetBinContent(b, scaled)
    hY.SetLineColor(ROOT.kGray + 2)
    hY.SetLineWidth(2)
    hY.SetMarkerStyle(0)
    hY.Draw("hist same")
    keep.append(hY)

    for crv in curves:
        hR = ROOT.TH1F(f"hR_{feat}_{crv['tag']}", "", n, lo, hi)
        for b in range(1, n + 1):
            hR.SetBinContent(b, float(crv["ratio_" + feat][b - 1]))
        hR.SetLineColor(crv["color"])
        hR.SetLineStyle(1)
        hR.SetLineWidth(2)
        hR.SetMarkerStyle(0)
        hR.Draw("hist same")
        keep.append(hR)

# legend panel
pad = c.cd(len(plot_feats) + 1)
pad.SetTicks(1, 1)
pad.SetBottomMargin(0.15)
pad.SetLeftMargin(0.15)

leg = ROOT.TLegend(0.08, 0.08, 0.92, 0.92)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetNColumns(1)

for crv in curves:
    htmp = ROOT.TH1F(f"leg_{crv['tag']}", "", 1, 0, 1)
    htmp.SetLineColor(crv["color"])
    htmp.SetLineStyle(1)
    htmp.SetLineWidth(2)
    leg.AddEntry(htmp, crv["label"], "l")
    keep.append(htmp)

hy = ROOT.TH1F("leg_yield", "", 1, 0, 1)
hy.SetLineColor(ROOT.kGray + 2)
hy.SetLineWidth(2)
leg.AddEntry(hy, "SM yield (scaled)", "l")
keep.append(hy)

leg.Draw()
keep.append(leg)

basename = "truth_" + "__".join(crv["tag"] for crv in curves)
if args.postfix:
    basename+="_"+args.postfix
out_png = os.path.join(out_dir, basename + ".png")
c.Print(out_png)
c.Close()

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    syncer.sync()
out = buf.getvalue().strip()
if out:
    print(out)

print(f"Wrote {out_png}")

