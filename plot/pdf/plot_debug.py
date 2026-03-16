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
pdf_n = J.get("pdf", {}).get("pdf_n", None)
pdf_type = J.get("pdf", {}).get("pdf_type", None)
pdf_basis = J.get("pdf", {}).get("pdf_basis", None)
#pdf_basis = "gluon_POD_nongluon_NNPDF31_hessian"
#pdf_basis = "gluon_POD_nongluon_NNPDF40"
#pdf_basis = "gluon_POD_nongluon_PDF4LHC21"

pdf = PDFParametrization(n=pdf_n, typ=pdf_type, basis=pdf_basis, active_pids=args.active_pdgids if args.active_pdgids=='all' else list(map(int, args.active_pdgids)))
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


mask=(X[:,11] > 770) & (X[:,11]<820)

## ---------------- compact debug table: only c8 ----------------
#
#debug_mask = mask.copy()
#max_rows = 80
#
#dbg = np.flatnonzero(debug_mask)
#if max_rows is not None:
#    dbg = dbg[:max_rows]
#
#k = 8
#scales = pdf.scale_c if pdf.scale_c is not None else np.ones(pdf.nvariations)
#
#xx1  = x1[dbg]
#xx2  = x2[dbg]
#QQ   = Q[dbg]
#iid1 = id1[dbg]
#iid2 = id2[dbg]
#llum = lumi[dbg]
#eevt = event[dbg]
#
#mask1 = np.isin(iid1, pdf.active_pids)
#mask2 = np.isin(iid2, pdf.active_pids)
#if pdf.x_max is not None:
#    mask1 &= (xx1 < pdf.x_max)
#    mask2 &= (xx2 < pdf.x_max)
#mask1f = mask1.astype(float)
#mask2f = mask2.astype(float)
#
#gen1 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(xx1), tuple(QQ)), iid1)])
#gen2 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(xx2), tuple(QQ)), iid2)])
#
#ref1 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(xx1), tuple(QQ)), iid1)])
#ref2 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(xx2), tuple(QQ)), iid2)])
#
#var1 = np.array([t.get(i) for t, i in zip(pdf.var_pdfs[k].xfxQ(tuple(xx1), tuple(QQ)), iid1)])
#var2 = np.array([t.get(i) for t, i in zip(pdf.var_pdfs[k].xfxQ(tuple(xx2), tuple(QQ)), iid2)])
#
#phi1_8 = scales[k] * (var1 - ref1) * mask1f
#phi2_8 = scales[k] * (var2 - ref2) * mask2f
#
#coeffs = np.zeros(pdf.nvariations)
#coeffs[k] = 1.0
#
#f1_eval = pdf.evaluate(xx1, iid1, QQ, coeffs)
#f2_eval = pdf.evaluate(xx2, iid2, QQ, coeffs)
#
#with np.errstate(divide='ignore', invalid='ignore'):
#    rw8_man = ((ref1 + phi1_8) / gen1) * ((ref2 + phi2_8) / gen2)
#
#rw8_pdf = pdf.product_parametrizations(xx1, xx2, iid1, iid2, coeffs, QQ)
#
#header = (
#    f"{'idx':>7s} {'lumi':>7s} {'event':>11s} | "
#    f"{'x1':>8s} {'x2':>8s} {'Q':>8s} {'id1':>4s} {'id2':>4s} | "
#    f"{'gen1':>9s} {'gen2':>9s} {'ref1':>9s} {'ref2':>9s} | "
#    f"{'phi1_8':>9s} {'phi2_8':>9s} | "
#    f"{'f1_eval':>9s} {'f2_eval':>9s} | "
#    f"{'rw8_man':>9s} {'rw8_pdf':>9s}"
#)
#print(header)
#print("-" * len(header))
#
#for j, i_evt in enumerate(dbg):
#    row = (
#        f"{i_evt:7d} {int(llum[j]):7d} {int(eevt[j]):11d} | "
#        f"{xx1[j]:8.2e} {xx2[j]:8.2e} {QQ[j]:8.2e} {int(iid1[j]):4d} {int(iid2[j]):4d} | "
#        f"{gen1[j]:9.2e} {gen2[j]:9.2e} {ref1[j]:9.2e} {ref2[j]:9.2e} | "
#        f"{phi1_8[j]:9.2e} {phi2_8[j]:9.2e} | "
#        f"{f1_eval[j]:9.2e} {f2_eval[j]:9.2e} | "
#        f"{rw8_man[j]:9.2e} {rw8_pdf[j]:9.2e}"
#    )
#    print(row)


#debug_mask = mask.copy()
#max_rows = 80
#
#dbg = np.flatnonzero(debug_mask)
#if max_rows is not None:
#    dbg = dbg[:max_rows]
#
#ks = [6, 7, 8]
#scales = pdf.scale_c if pdf.scale_c is not None else np.ones(pdf.nvariations)
#
#xx1  = x1[dbg]
#xx2  = x2[dbg]
#QQ   = Q[dbg]
#iid1 = id1[dbg]
#iid2 = id2[dbg]
#llum = lumi[dbg]
#eevt = event[dbg]
#
#mask1 = np.isin(iid1, pdf.active_pids)
#mask2 = np.isin(iid2, pdf.active_pids)
#if pdf.x_max is not None:
#    mask1 &= (xx1 < pdf.x_max)
#    mask2 &= (xx2 < pdf.x_max)
#mask1f = mask1.astype(float)
#mask2f = mask2.astype(float)
#
#f_gen1 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(xx1), tuple(QQ)), iid1)])
#f_gen2 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(xx2), tuple(QQ)), iid2)])
#
#f_ref1 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(xx1), tuple(QQ)), iid1)])
#f_ref2 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(xx2), tuple(QQ)), iid2)])
#
#var1 = np.array([
#    [t.get(i) for t, i in zip(pdf.var_pdfs[k].xfxQ(tuple(xx1), tuple(QQ)), iid1)]
#    for k in ks
#])
#var2 = np.array([
#    [t.get(i) for t, i in zip(pdf.var_pdfs[k].xfxQ(tuple(xx2), tuple(QQ)), iid2)]
#    for k in ks
#])
#
#phi1 = np.array([
#    scales[k] * (var1[j] - f_ref1) * mask1f
#    for j, k in enumerate(ks)
#])
#phi2 = np.array([
#    scales[k] * (var2[j] - f_ref2) * mask2f
#    for j, k in enumerate(ks)
#])
#
#with np.errstate(divide='ignore', invalid='ignore'):
#    rw_single = np.array([
#        ((f_ref1 + phi1[j]) / f_gen1) * ((f_ref2 + phi2[j]) / f_gen2)
#        for j in range(len(ks))
#    ])
#
#header = (
#    f"{'idx':>7s} {'lumi':>7s} {'event':>11s} | "
#    f"{'x1':>8s} {'x2':>8s} {'Q':>8s} {'id1':>4s} {'id2':>4s} | "
#    f"{'f_gen1':>9s} {'f_gen2':>9s} {'f_ref1':>9s} {'f_ref2':>9s} | "
#    f"{'phi1_6':>9s} {'phi1_7':>9s} {'phi1_8':>9s} | "
#    f"{'phi2_6':>9s} {'phi2_7':>9s} {'phi2_8':>9s} | "
#    f"{'rw6':>9s} {'rw7':>9s} {'rw8':>9s}"
#)
#print(header)
#print("-" * len(header))
#
#for j, i_evt in enumerate(dbg):
#    row = (
#        f"{i_evt:7d} {int(llum[j]):7d} {int(eevt[j]):11d} | "
#        f"{xx1[j]:8.2e} {xx2[j]:8.2e} {QQ[j]:8.2e} {int(iid1[j]):4d} {int(iid2[j]):4d} | "
#        f"{f_gen1[j]:9.2e} {f_gen2[j]:9.2e} {f_ref1[j]:9.2e} {f_ref2[j]:9.2e} | "
#        f"{phi1[0,j]:9.2e} {phi1[1,j]:9.2e} {phi1[2,j]:9.2e} | "
#        f"{phi2[0,j]:9.2e} {phi2[1,j]:9.2e} {phi2[2,j]:9.2e} | "
#        f"{rw_single[0,j]:9.2e} {rw_single[1,j]:9.2e} {rw_single[2,j]:9.2e}"
#    )
#    print(row)

DER = np.concatenate(targets, axis=0) if len(targets) > 1 else targets[0]
training_weights = {combos[i]: DER[:, i] for i in range(len(combos))}

 ---------------- reweighting ----------------
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

# ---------------- plotting ----------------
def plot_truth_root():
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


plot_truth_root()
print("Done.")

#c=np.zeros(pdf.nvariations); 
#c[8]=1.; 
#rdir=pdf.product_parametrizations(x1,x2,id1,id2,c,Q); 
#idx=np.argsort(np.abs(rdir))[-20:][::-1]; 
#print(np.c_[idx,run[idx],lumi[idx],event[idx],id1[idx],id2[idx],x1[idx],x2[idx],Q[idx],rdir[idx]])
#
#
#c = np.zeros(pdf.nvariations)
#c[8] = 1.
#
#f1 = pdf.evaluate(x1, id1, Q, c)
#f2 = pdf.evaluate(x2, id2, Q, c)
#
#ref1 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(x1), tuple(Q)), id1)])
#ref2 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(x2), tuple(Q)), id2)])
#
#gen1 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(x1), tuple(Q)), id1)])
#gen2 = np.array([t.get(i) for t, i in zip(pdf.gen_pdf.xfxQ(tuple(x2), tuple(Q)), id2)])
#
#fixed  = (ref1 / gen1) * (ref2 / gen2)
#deform = (f1 / ref1) * (f2 / ref2)
#full   = fixed * deform
#
#idx = np.argsort(np.abs(full))[-20:][::-1]
#print(np.c_[idx, run[idx], lumi[idx], event[idx],
#            id1[idx], id2[idx], x1[idx], x2[idx], Q[idx],
#            fixed[idx], deform[idx], full[idx]])
#
#
#c = np.zeros(pdf.nvariations)
#c[8] = 1.
#
#f1 = pdf.evaluate(x1, id1, Q, c)
#f2 = pdf.evaluate(x2, id2, Q, c)
#
#r = pdf.product_parametrizations(x1, x2, id1, id2, c, Q)
#idx = np.argsort(np.abs(r))[-20:][::-1]
#
#print(np.c_[idx, run[idx], lumi[idx], event[idx],
#            id1[idx], id2[idx], x1[idx], x2[idx], Q[idx],
#            f1[idx], f2[idx], r[idx]])
#
#
#c = np.zeros(pdf.nvariations)
#c[8] = 1.
#
#f1 = pdf.evaluate(x1, id1, Q, c)
#f2 = pdf.evaluate(x2, id2, Q, c)
#
#ref1 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(x1), tuple(Q)), id1)])
#ref2 = np.array([t.get(i) for t, i in zip(pdf.reference_pdf.xfxQ(tuple(x2), tuple(Q)), id2)])
#
#d1 = f1 / ref1
#d2 = f2 / ref2
#
#r = pdf.product_parametrizations(x1, x2, id1, id2, c, Q)
#idx = np.argsort(np.abs(r))[-20:][::-1]
#
#print(np.c_[idx, id1[idx], id2[idx],
#            x1[idx], x2[idx], Q[idx],
#            d1[idx], d2[idx], d1[idx] * d2[idx], r[idx]])
#
#
#
## for debugging
max_x    =  np.max(np.column_stack((x1,x2)),axis=1)
ratio_c8 = weight_at_point({'c8':1})/weight_at_point({'c8':0})
ratio_c7 = weight_at_point({'c7':1})/weight_at_point({'c7':0})
ratio_c6 = weight_at_point({'c6':1})/weight_at_point({'c6':0})
x_prod = x1*x2

assert False, ""

def plot_np_2d_root(
    x,
    y,
    xtitle="x",
    ytitle="y",
    bins_x=None,   # e.g. [100, 0.0, 1.0]
    bins_y=None,   # e.g. [ 80, -4.0, 4.0]
    name="np2d",
    title="",
    draw_option="COLZ",
    canvas=None,
    filename=None,
    save=True,
    subdir="debug",
    log_z=True,
):
    import os
    import ROOT
    import numpy as np
    import common.user as user

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    if len(x) == 0:
        raise ValueError("x and y are empty")

    def _auto_root_binning(a):
        edges = np.histogram_bin_edges(a, bins="auto")
        n = len(edges) - 1
        lo = float(edges[0])
        hi = float(edges[-1])

        if n <= 0:
            n = 1
        if lo == hi:
            lo -= 0.5
            hi += 0.5

        return [int(n), lo, hi]

    if bins_x is None:
        bins_x = _auto_root_binning(x)
    else:
        bins_x = [int(bins_x[0]), float(bins_x[1]), float(bins_x[2])]

    if bins_y is None:
        bins_y = _auto_root_binning(y)
    else:
        bins_y = [int(bins_y[0]), float(bins_y[1]), float(bins_y[2])]

    nx, xlo, xhi = bins_x
    ny, ylo, yhi = bins_y

    if nx <= 0 or ny <= 0:
        raise ValueError("Number of bins must be positive")
    if not xlo < xhi:
        raise ValueError(f"Need xlo < xhi, got {xlo} >= {xhi}")
    if not ylo < yhi:
        raise ValueError(f"Need ylo < yhi, got {ylo} >= {yhi}")

    # Fast event aggregation in numpy
    H, _, _ = np.histogram2d(
        x,
        y,
        bins=[nx, ny],
        range=[[xlo, xhi], [ylo, yhi]],
    )

    ROOT.gStyle.SetOptStat(0)

    if canvas is None:
        canvas = ROOT.TCanvas(f"c_{name}", title or name, 800, 700)

    if not hasattr(canvas, "_keep"):
        canvas._keep = []

    hist = ROOT.TH2F(
        f"h2_{name}",
        f"{title};{xtitle};{ytitle}",
        nx, xlo, xhi,
        ny, ylo, yhi,
    )

    # Loop only over bins, not events
    for ix in range(nx):
        for iy in range(ny):
            hist.SetBinContent(ix + 1, iy + 1, float(H[ix, iy]))

    if log_z:
        canvas.SetLogz(True)
        if hist.GetMaximum() > 0:
            hist.SetMinimum(0.5)

    hist.Draw(draw_option)
    canvas._keep.append(hist)
    canvas.Modified()
    canvas.Update()

    out_path = None
    if save:
        plot_dir = os.path.join(user.plot_directory, subdir)
        os.makedirs(plot_dir, exist_ok=True)

        if filename is None:
            filename = f"{name}.png"

        out_path = os.path.join(plot_dir, filename)
        canvas.SaveAs(out_path)
        print(f"Wrote {out_path}")

    syncer.sync()
    #return canvas, hist, out_path, bins_x, bins_y
