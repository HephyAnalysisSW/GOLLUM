#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your likelihood machinery
sys.path.insert(0, '..')

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL,
)
from fit.N2LLExtensions import N2LLExtensions 

import common.yaml_loader as yaml_loader
import common.syncer as syncer
from data.plot_options import plot_options

def set_free_values(hyp, names, values):
    """Assign values (array-like) to the corresponding parameter names in hyp."""
    for nm, v in zip(names, values):
        getattr(hyp, nm).val = float(v)


def clone_with_delta(hyp, names, delta):
    """Clone hypothesis and shift free params by delta vector."""
    h = hyp.clone()
    for nm, dv in zip(names, delta):
        getattr(h, nm).val = float(getattr(h, nm).val) + float(dv)
    return h


p = argparse.ArgumentParser(
    description="Compare true -2logL to Fisher quadratic approximation near (0,0) Asimov."
)
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--plot_directory",     action="store",      default="orthogonalize", help="plot sub-directory")
p.add_argument("--cache-root", default=None, help="Override cache root (optional)")
p.add_argument("--overwrite", action="store_true", help="Overwrite caches")
p.add_argument("--fi-step-scale", type=float, default=1e-4,
               help="Step scale for Fisher binned/penalty finite diffs")
args = p.parse_args()

# ---- Load YAML + surrogates ----
cfg = yaml_loader.load_yaml(args.config)
yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, args.config, overwrite=False)

# ---- Likelihood info & hypothesis scaffold ----
like_info = load_likelihood(cfg)
hyp = build_hypothesis_from_likelihood(like_info, name="SR")

# --- optionally disable all nuisances ---
if True:
    for p in hyp.nuisances:
        p.val = 0.0
        p.freeze()
    print("[opts] --no_syst: all nuisances set to 0 and frozen.")

hyp.print()

# Keep track of the free parameters (not frozen and not ignored)
free_params = [p for p in hyp.parameters
               if not p.isFrozen and not getattr(p, "isIgnored", False)]
free_names  = [p.name for p in free_params]
if not free_names:
    raise RuntimeError("No free parameters to probe.")
print("\n[free] Parameters to probe:")
for nm in free_names:
    print("   -", nm)

# ---- Build caches, prepare runtime ----
n2ll = N2LLExtensions(
    like_info,
    cfg['defaults']['module_samples'],
    os.path.join(
        "NN2LCache",
        os.path.splitext(os.path.basename(args.config))[0],
        str(cfg.get("version", "v0")),
    ),
    cache_root=args.cache_root,
    overwrite=args.overwrite,
)
n2ll.build_cache()
n2ll.prepare_runtime()

# ---- Set A-simov at (0,0) ----
# By convention setAsimov(None) activates A-simov with zero bias term.
n2ll.setAsimov()  # A-simov (c' = 0, ν' = 0)

# ---- Baseline -2logL at (0,0) ----
# All params were initialized to 0 in build_hypothesis_from_likelihood.
f0 = float(n2ll(hyp))
print(f"\n[baseline] -2 log L (Asimov, at 0) = {f0:.6e}")

# ---- Fisher information at (0,0) ----
# N2LL.fisher_information() returns the regular Fisher matrix I_ab
# (i.e. E[∂_a log L ∂_b log L]). For small displacements δ, the
# quadratic approximation of Δ(-2logL) is δ^T I δ.
print("\n[FI] Computing Fisher information at (0,0) in Asimov A-mode…")
I = n2ll.fisher_information(hyp, step_scale=args.fi_step_scale, verbose=True)

# Numerical symmetrization for safety
I = 0.5 * (I + I.T)

# ================================================================
#   Build rate + shape basis in POI space (Option A coordinates)
#   d are the basis coefficients:  c = V_new @ d  and  d = D @ c
#   => Fisher in d-space is identity.
# ================================================================

# ---- 1) Select only POIs among free parameters ----
free_params = [p for p in hyp.parameters
               if not p.isFrozen and not getattr(p, "isIgnored", False)]
poi_free_params = [p for p in free_params if p.isPOI]
poi_names       = [p.name for p in poi_free_params]
n_poi           = len(poi_names)
if n_poi == 0:
    raise RuntimeError("No free POIs found for rate/shape basis construction.")
print("\n[orthogonalization] Free POIs used for basis:")
for nm in poi_names:
    print("   -", nm)

# ---- 2) Build s_a and M_ab from unbinned Asimov events at (0,0) ----
#     s_a = sum_i w0_i R_a(x_i)
#     M_ab = sum_i w0_i R_a(x_i) R_b(x_i)
M = np.zeros((n_poi, n_poi), dtype=np.float64)
s = np.zeros(n_poi, dtype=np.float64)
chunk = n2ll.eval_chunk_size

print("\n[orthogonalization] Accumulating s_a and M_ab from unbinned regions only…")
for R in n2ll.regions:
    rid = R["id"]
    class_ids = n2ll._class_ids_by_region.get(rid, [])
    if not class_ids:
        continue
    N = n2ll._N_region.get(rid, 0)
    if N == 0:
        continue
    print(f"  - region '{rid}': N = {N}")

    poi_order_by_cid = {cid: n2ll._poi_order[(rid, cid)] for cid in class_ids}

    for start in tqdm(range(0, N, chunk),
                      desc=f"[orthogonalization] region {rid}",
                      unit="chunk", leave=False):
        stop = min(start + chunk, N)
        f0  = n2ll._h5[(rid, class_ids[0])]
        w0  = np.asarray(f0["w0"][start:stop], dtype=np.float64)  # (M_chunk,)

        R_event = np.zeros((stop-start, n_poi), dtype=np.float64)
        for cid in class_ids:
            f   = n2ll._h5[(rid, cid)]
            g   = np.asarray(f["g"][start:stop], dtype=np.float64)
            R_A = np.asarray(f["R"][start:stop, :], dtype=np.float64)
            local_pois = poi_order_by_cid[cid]
            n_local    = len(local_pois)
            for a_global, pname in enumerate(poi_names):
                if pname not in local_pois:
                    continue
                loc_idx = local_pois.index(pname)
                if loc_idx >= n_local:
                    raise RuntimeError(f"Inconsistent BIT dims for ({rid},{cid}) and POI '{pname}'.")
                R_event[:, a_global] += g * R_A[:, loc_idx]

        WR = w0[:, None] * R_event
        s  += WR.sum(axis=0)
        M  += R_event.T @ WR

# Symmetrize M
M = 0.5 * (M + M.T)
print("\n[orthogonalization] Per-L Fisher matrix in POI space (unbinned only) M_ab:")
print(M)
print("\n[orthogonalization] Rate derivative vector s_a:")
print(s)

# ---- 3) Rate direction v_rate = M^{-1} s / sqrt(s^T M^{-1} s) ----
Minv = np.linalg.pinv(M, rcond=1e-12)
num  = Minv @ s
den2 = float(s @ (Minv @ s))
if den2 <= 0:
    raise RuntimeError("Non-positive s^T M^{-1} s; check inputs.")
v_rate = num / np.sqrt(den2)

def M_inner(u, v): return float(u @ (M @ v))

print("\n=== Fisher-basis (unbinned per-L) summary ===")
print(f"||v_rate||_M^2 = {M_inner(v_rate, v_rate):.6f} (should be ~1)")
print(f"s · v_rate     = {float(s @ v_rate):.6f}")
print(f"sqrt(s^T M^-1 s) = {np.sqrt(den2):.6f}")

# ---- 4) Fisher-orthonormal basis via Gram–Schmidt (M-metric), then tidy ordering ----
V_cols = [v_rate.copy()]
for k in range(n_poi):
    e = np.zeros(n_poi); e[k] = 1.0
    u = e.copy()
    for v in V_cols:
        u -= M_inner(v, u) * v
    nu2 = M_inner(u, u)
    if nu2 > 1e-10 * max(1.0, np.max(np.abs(np.diag(M)))):
        V_cols.append(u / np.sqrt(nu2))

V = np.stack(V_cols, axis=1)
m_cols = V.shape[1]
print(f"\n[orthogonalization] #basis vectors built (incl. rate) = {m_cols}/{n_poi}")
print("max|offdiag(V^T M V)| =", np.max(np.abs(V.T @ (M @ V) - np.eye(m_cols))))

# Diagonalize M in shape subspace for nice ordering
if m_cols > 1:
    V_perp = V[:, 1:]
    M_perp = V_perp.T @ (M @ V_perp)
    lam, U = np.linalg.eigh(M_perp)
    V_new = np.column_stack([V[:, 0], V_perp @ U])
else:
    lam   = np.array([])
    V_new = V

labels_d = ["rate"] + [f"shape_{i+1}" for i in range(V_new.shape[1] - 1)]
print("\n=== Eigensystem summary (per-L, POI order = {}) ===".format(poi_names))
np.set_printoptions(precision=4, suppress=True)
for j, lab in enumerate(labels_d):
    print(f"{lab:>8s}: {V_new[:, j]}")

# ---- 5) Option A coordinates: d are basis coefficients, so c = V_new @ d  and  d = D @ c ----
# Prefer exact inverse if square/full-rank; fall back to pinv otherwise.
if V_new.shape[0] == V_new.shape[1]:
    D_matrix = np.linalg.inv(V_new)
else:
    D_matrix = np.linalg.pinv(V_new, rcond=1e-12)

print("\n[rotation] Define new parameters d_j by")
print("  (d_rate, d_shape_1, ..., d_shape_{})^T = D · c".format(V_new.shape[1]-1))
print("with c in POI order:", poi_names)
print("and D = V_new^{-1} (basis coefficients), i.e.")
print(D_matrix)


# ---- 6) Store rotation matrix and meta info ----
try:
    import common.user as user
    out_dir = getattr(user, "output_directory", "./outputs")
except Exception:
    out_dir = "./outputs"
os.makedirs(out_dir, exist_ok=True)

base_name = os.path.splitext(os.path.basename(args.config))[0]
rot_path_json = os.path.join(out_dir, f"orthogonal_basis_{base_name}_{cfg['version']}.json")
rot_path_npz  = os.path.join(out_dir, f"orthogonal_basis_{base_name}_{cfg['version']}.npz")

payload = {
    "config": os.path.basename(args.config),
    "poi_order": poi_names,
    "basis_labels": labels_d,
    "M": M.tolist(),
    "s": s.tolist(),
    "V_new": V_new.tolist(),   # columns = basis vectors in c-space
    "D": D_matrix.tolist(),    # d = D @ c, and in THIS basis, the Fisher information is the identity
    "shape_eigenvalues": lam.tolist() if lam.size else [],
}
with open(rot_path_json, "w") as fjs:
    json.dump(payload, fjs, indent=2)
np.savez(rot_path_npz,
         poi_order=np.array(poi_names, dtype=object),
         basis_labels=np.array(labels_d, dtype=object),
         M=M, s=s, V_new=V_new, D=D_matrix, shape_eigenvalues=lam)

print(f"\n[write] Saved orthogonal basis to:\n  {rot_path_json}\n  {rot_path_npz}")



# ================================================================
#  ROOT feature plots with rotated Fisher basis (Option A)
#  Steps are 1σ along each basis vector: c_step = V_new @ e_j
# ================================================================
import ROOT, os
from tqdm import tqdm
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)
from common import helpers
import common.user as user

plot_directory = os.path.join(
    user.plot_directory, args.plot_directory, os.path.splitext(os.path.basename(args.config))[0],
)
os.makedirs(plot_directory, exist_ok=True)

rate_color = ROOT.kMagenta + 1
_shape_palette = [
    ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kOrange + 1,
    ROOT.kCyan + 2, ROOT.kViolet + 1, ROOT.kTeal + 1, ROOT.kPink + 1,
    ROOT.kAzure + 2, ROOT.kSpring + 5, ROOT.kYellow + 2, ROOT.kGray + 1,
]
n_dirs = V_new.shape[1]
shape_colors = [_shape_palette[i % len(_shape_palette)] for i in range(max(1, n_dirs - 1))]
labels_d = ["rate"] + [f"shape_{i+1}" for i in range(n_dirs - 1)]

# 1σ steps in the d-coordinates (basis coefficients)
steps = [1.0 for _ in range(n_dirs)]

stuff = []
print("\n[plots] Making feature distributions with rotated basis directions (per region)…")

def make_hist_feature(weights, feature_idx, feature_name, region_features):
    bins = plot_options[feature_name]['binning']  # [nbins, xmin, xmax]
    arr  = region_features[:, feature_idx]
    hist, edges = np.histogram(arr, np.linspace(bins[1], bins[2], bins[0] + 1), weights=weights)
    return helpers.make_TH1F((hist, edges))

def clone_and_divide(num, den):
    h = num.Clone(f"{num.GetName()}__ratio"); h.Divide(den); return h

for R in n2ll.regions:
    rid = R["id"]
    asimov_list = R.get("_asimov_samples", [])
    if not asimov_list:
        continue

    print(f"\n[plots:{rid}] Collecting A-simov events and nominal weights…")
    region_feats, region_w0s = [], []
    feature_names_ref = None
    for sname in asimov_list:
        L = getattr(n2ll.samples_mod, sname)
        feat_names = list(getattr(L, "feature_names", []) or [])
        if feature_names_ref is None: feature_names_ref = feat_names
        elif feat_names != feature_names_ref:
            raise RuntimeError(f"[plots:{rid}] Feature mismatch across Asimov samples.")
        nsplits = int(getattr(L, "n_split", 1))
        for shard in range(nsplits):
            X, w0 = L.materialize(shard=shard, what="fw", n=None)
            if X is None or len(X) == 0: continue
            region_feats.append(np.asarray(X, dtype=np.float64))
            region_w0s.append(np.asarray(w0, dtype=np.float64))

    if not region_feats:
        print(f"[plots:{rid}] No events found; skipping.")
        continue

    region_features = np.concatenate(region_feats, axis=0)
    w0 = np.concatenate(region_w0s, axis=0)
    N = region_features.shape[0]
    print(f"[plots:{rid}] N = {N} events; {len(feature_names_ref)} features.")

    # warm up per-region surrogate cache
    _ = n2ll.evaluate_ratio(rid, region_features, feature_names_ref, hyp, cached=True, return_T=False)

    name2idx = {n: i for i, n in enumerate(feature_names_ref)}
    plot_feature_names = [f for f in plot_options.keys() if f in name2idx]
    print(f"[plots:{rid}] #features to plot = {len(plot_feature_names)}")

    def build_hyp_along_direction(j, alpha):
        """Option A: c_step = alpha * column j of V_new (1σ step when alpha=1)."""
        h_step = hyp.clone()
        for a, pname in enumerate(poi_names):
            getattr(h_step, pname).val = float(alpha * V_new[a, j])
        return h_step

    for feature_name in tqdm(plot_feature_names, desc=f"[plots:{rid}] features", unit="feat"):
        feature_idx = name2idx[feature_name]

        # Nominal
        h_nom = make_hist_feature(w0, feature_idx, feature_name, region_features)
        h_nom.SetLineColor(ROOT.kGray + 2); h_nom.SetMarkerStyle(0); h_nom.SetLineWidth(2)

        # Variations along each rotated direction (1σ)
        h_dirs = []
        for j in range(n_dirs):
            alpha = steps[j]  # 1σ
            if alpha == 0.0:
                w_step = w0.copy()
            else:
                h_step = build_hyp_along_direction(j, alpha)
                Rj = n2ll.evaluate_ratio(rid, region_features, feature_names_ref, h_step, cached=True, return_T=False)
                w_step = w0 * np.asarray(Rj, dtype=np.float64)

            h_j = make_hist_feature(w_step, feature_idx, feature_name, region_features)
            h_j.SetLineColor(rate_color if j == 0 else shape_colors[j - 1])
            h_j.SetLineWidth(3 if j == 0 else 2); h_j.SetMarkerStyle(0)
            h_dirs.append(h_j)

        # 1) Yields
        for logY in (False, True):
            c = ROOT.TCanvas(f"c_yield_{rid}_{feature_name}_{int(logY)}", "", 800, 650)
            leg = ROOT.TLegend(0.20, 0.76, 0.90, 0.88)
            leg.SetNColumns(3); leg.SetFillStyle(0); leg.SetShadowColor(ROOT.kWhite); leg.SetBorderSize(0)
            h_nom.GetXaxis().SetTitle(plot_options[feature_name]['tex']); h_nom.GetYaxis().SetTitle("events")
            h_nom.Draw("hist"); leg.AddEntry(h_nom, f"{rid}: nominal", "l")
            for j, hj in enumerate(h_dirs):
                hj.Draw("histsame"); leg.AddEntry(hj, f"{labels_d[j]} (1σ)", "l")
            stuff.append(leg); leg.Draw()
            ROOT.gPad.SetLogy(logY); c.Update()
            outdir = os.path.join(plot_directory, rid, "basis_yield", "log" if logY else "lin")
            os.makedirs(outdir, exist_ok=True); helpers.copyIndexPHP(outdir)
            c.Print(os.path.join(outdir, f"{feature_name}.png")); c.Print(os.path.join(outdir, f"{feature_name}.pdf")); c.Close()

        # 2) Ratios
        for logY in (False, ):
            c = ROOT.TCanvas(f"c_ratio_{feature_name}_{int(logY)}", "", 800, 650)
            leg = ROOT.TLegend(0.20, 0.76, 0.90, 0.88); leg.SetNColumns(3); leg.SetFillStyle(0); leg.SetShadowColor(ROOT.kWhite); leg.SetBorderSize(0)
            h_unity = clone_and_divide(h_nom, h_nom)
            h_unity.SetLineColor(ROOT.kGray + 1); h_unity.SetLineStyle(2); h_unity.SetLineWidth(2)
            h_unity.GetXaxis().SetTitle(plot_options[feature_name]['tex']); h_unity.GetYaxis().SetTitle("variation / nominal")
            h_unity.SetMinimum(0.95); h_unity.SetMaximum(1.05); h_unity.SetTitle(""); h_unity.Draw("hist"); leg.AddEntry(h_unity, "nominal", "l")
            for j, hj in enumerate(h_dirs):
                h_ratio = clone_and_divide(hj, h_nom)
                h_ratio.SetLineColor(rate_color if j == 0 else shape_colors[j - 1])
                h_ratio.SetLineWidth(3 if j == 0 else 2); h_ratio.SetMarkerStyle(0); h_ratio.SetTitle("")
                h_ratio.Draw("histsame"); stuff.append(h_ratio); leg.AddEntry(h_ratio, f"{labels_d[j]} (1#sigma)", "l")
            stuff.append(leg); leg.Draw()
            ROOT.gPad.SetLogy(logY); c.Modified(); c.Update()
            outdir = os.path.join(plot_directory, rid, "basis_ratio", "log" if logY else "lin")
            os.makedirs(outdir, exist_ok=True); helpers.copyIndexPHP(outdir)
            c.Print(os.path.join(outdir, f"{feature_name}.png")); c.Print(os.path.join(outdir, f"{feature_name}.pdf")); c.Close()

    print("\n[done] Yield and ratio plots written under:")
    print(f"  {os.path.join(plot_directory, '<region>', 'basis_yield')}")
    print(f"  {os.path.join(plot_directory, '<region>', 'basis_ratio')}")

