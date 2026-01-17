import os
import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
import numpy as np
import subprocess
import argparse
import json
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user   as user
import common.syncer as syncer
import common.helpers as helpers

import common.yaml_loader as yaml_loader
from fit.Likelihood import load_likelihood
from fit.Likelihood import build_hypothesis_from_likelihood
from fit.Modeling import Rotated

from pdf.PDFParametrization import PDFParametrization

# grouping of systematics (nuisances are "nu_*")
from sys_grouping import sys_grouping

# CMS palette (explicit, non-transparent)
import cmsstyle

# ---------------- args ----------------
p = argparse.ArgumentParser(description="Gluon PDF plotting")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--yes", "-y", action="store_true",
               help="Automatically run missing surrogate trainings without asking")
p.add_argument("--postfit", required=True,
               help="JSON file with post-fit parameters and covariance")
p.add_argument("--postfix", default='v1',
               help="A txt string")
p.add_argument("--freezePOI", action="store_true",
               help="Do not sample POIs, use only central values")
p.add_argument("--Q", type=float, default=1.65,
               help="Scale Q for PDF evaluation (same units as PDF grid, default: 1.65)")
p.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")
p.add_argument("--subdir", action="store", default='', help="Subdirectory for plotting")

# grouped uncertainty decomposition knobs
p.add_argument("--x_ref", type=float, default=0.06,
               help="Reference Bjorken-x used to rank grouped uncertainties (default: 0.06)")
p.add_argument("--n_toys", type=int, default=10000,
               help="Number of toys per band (default: 10000)")
p.add_argument("--seed", type=int, default=42,
               help="RNG seed for toys (default: 42)")

args = p.parse_args()

plot_label = "gluonPDF"

# ---------------- load YAML CFG ----------------
print(f"[info] Loading config from: {args.config}")
cfg = yaml_loader.load_yaml(args.config)

yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
missing_cmds = yaml_loader.load_surrogates(
    cfg,
    args.config,
    overwrite=False,
    prefer_numba=False,
)

# Are there missing commands? If so, let's do those. Ask the user (or require --yes)
if missing_cmds:
    print(f"[info] Found {len(missing_cmds)} missing surrogate trainings.")
    if not args.yes:
        ans = input(f"{len(missing_cmds)} surrogates missing. Run training now? [y/N] ")
        if ans.lower() not in ("y", "yes"):
            print("[info] Not running trainings, exiting.")
            sys.exit(1)

    for cmd in missing_cmds:
        print("[info] Running:", cmd)
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"[error] Command failed with exit code {ret.returncode}")
            sys.exit(ret.returncode)

    # try again
    print("[info] Re-checking for missing surrogates after training...")
    missing_cmds = yaml_loader.load_surrogates(
        cfg,
        args.config,
        overwrite=False,
        prefer_numba=False,
    )
    if missing_cmds:
        print("[error] Still missing surrogates after running trainings:")
        for cmd in missing_cmds:
            print("  ", cmd)
        sys.exit(1)

print("[info] All required surrogates available. Loading likelihood...")
like_info = load_likelihood(cfg)

# output directory for PDF plots
pdf_plot_directory = os.path.join(
    user.plot_directory,
    plot_label,
    args.subdir,
    os.path.splitext(os.path.basename(args.config))[0]
)
os.makedirs(pdf_plot_directory, exist_ok=True)
print(f"[info] {plot_label} plots will be written under: {pdf_plot_directory}")

# load postfit result (JSON with parameters + covariance)
with open(args.postfit, "r") as f:
    fit_results = json.load(f)

fit_par_names = [p["name"] for p in fit_results["parameters"]]
params_best   = np.array([p["value"] for p in fit_results["parameters"]], dtype=float)
cov           = np.array(fit_results["covariance"]["matrix"], dtype=float)

print(f"[info] Loaded fit result with {len(fit_par_names)} parameters.")

# build hypothesis from likelihood for consistency checks
hyp = build_hypothesis_from_likelihood(like_info)

rotated = bool(args.rotate)
hyp_rotated = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

# --- consistency check between likelihood parameters and fit results ---
like_par_names = [p.name for p in hyp_rotated.parameters]

set_like = set(like_par_names)
set_fit  = set(fit_par_names)

only_in_like = sorted(list(set_like - set_fit))
only_in_fit  = sorted(list(set_fit  - set_like))

if only_in_like or only_in_fit:
    print("[error] Inconsistent parameter definitions between likelihood and fit results.")
    if only_in_like:
        print("  Present in likelihood but missing in fit:")
        for n in only_in_like:
            print("   -", n)
    if only_in_fit:
        print("  Present in fit but missing in likelihood:")
        for n in only_in_fit:
            print("   -", n)
    sys.exit(1)

if like_par_names != fit_par_names:
    print("[warning] Parameter names match as a set, but order differs between likelihood and fit results.")
    print("  Likelihood order:")
    print("   ", ", ".join(like_par_names))
    print("  Fit result order:")
    print("   ", ", ".join(fit_par_names))
else:
    print(f"[info] Parameter list consistent between likelihood and fit ({len(like_par_names)} parameters).")

# ----------------------------------------------------------------------
# Find first POI-dependent BIT entry in likelihood and corresponding job
# ----------------------------------------------------------------------
like_params = None
poi_job_id = None

for region in like_info.get("regions", []):
    for cls in region.get("classes", []):
        poi = cls.get("POI", None)
        if poi and poi.get("type") == "bit" and poi.get("parameters"):
            like_params = poi["parameters"]
            poi_job_id = poi["job"]
            break
    if like_params is not None:
        break

if like_params is None or poi_job_id is None:
    print("[error] Could not find a POI-dependent BIT term in the likelihood.")
    sys.exit(1)

print(f"[info] Using POI-dependent BIT job '{poi_job_id}' with POIs: {', '.join(like_params)}")

# find corresponding job J in cfg['jobs']
J = None
for job in cfg.get("jobs", []):
    if job.get("id") == poi_job_id:
        J = job
        break

if J is None:
    print(f"[error] No job with id '{poi_job_id}' found in cfg['jobs'].")
    sys.exit(1)

pdf_cfg   = J.get("pdf", {})
pdf_n     = pdf_cfg.get("pdf_n", None)
pdf_type  = pdf_cfg.get("pdf_type", None)

if pdf_n is None or pdf_type is None:
    print(f"[error] Job '{poi_job_id}' has no 'pdf' configuration (pdf_n / pdf_type).")
    sys.exit(1)

print(f"[info] PDF configuration: pdf_n={pdf_n}, pdf_type={pdf_type}")

if len(pdf_n) != len(like_params):
    print("[warning] Length mismatch: len(pdf_n) != len(POI parameters).")
    print(f"  len(pdf_n)   = {len(pdf_n)}")
    print(f"  len(POIs)    = {len(like_params)}")

# instantiate PDF parametrization
pdf = PDFParametrization(n=pdf_n, typ=pdf_type)

# map parameter names -> indices in fit result
idx_map = {name: i for i, name in enumerate(fit_par_names)}

# POIs are taken from the (possibly rotated) hypothesis object
poi_names = [p.name for p in hyp_rotated.POIs]
poi_indices = [idx_map[name] for name in poi_names]

# central POI coefficients (MLE) in the *current* POI basis
coeffs_central = params_best[poi_indices]
print("[info] Central POI coefficients (MLE):")
for name, val in zip(poi_names, coeffs_central):
    print(f"  {name:>20s} = {val: .5e}")

# covariance submatrix for POIs:
# this is the *marginal* POI covariance (all nuisances included via marginalization)
cov_poi = cov[np.ix_(poi_indices, poi_indices)]

# sampling knobs
n_toys       = int(args.n_toys)
fit_rng_seed = int(args.seed)

# counts for plot annotations
n_poi = len(poi_names)
all_nu_names_fit = [n for n in fit_par_names if n.startswith("nu_")]
n_nu  = len(all_nu_names_fit)

postfit_tag = os.path.splitext(os.path.basename(args.postfit))[0]
rotate_tag  = os.path.splitext(os.path.basename(args.rotate))[0] if rotated else None

# ----------------------------------------------------------------------
# Grouped uncertainty decomposition helpers
# ----------------------------------------------------------------------

def _cond_cov_poi_given_fixed_nu(cov_full, poi_idx, fixed_nu_idx):
    """
    Conditional covariance of POIs c given a subset of nuisances nu_fixed is held fixed:

        Cov(c | nu_fixed) = A - B D^{-1} C

    where the joint covariance over (c, nu_fixed) is:
        [ A  B ]
        [ C  D ]

    This is the Schur complement of D.
    """
    if len(fixed_nu_idx) == 0:
        return cov_full[np.ix_(poi_idx, poi_idx)]
    A = cov_full[np.ix_(poi_idx, poi_idx)]
    B = cov_full[np.ix_(poi_idx, fixed_nu_idx)]
    C = cov_full[np.ix_(fixed_nu_idx, poi_idx)]
    D = cov_full[np.ix_(fixed_nu_idx, fixed_nu_idx)]
    return A - B @ np.linalg.inv(D) @ C


def _rot_poi_to_base_coeffs(poi_vec_rot):
    """
    Convert POI coefficients from the (possibly rotated) POI basis to the *base* POI basis
    used by pdf.evaluate(..., coeffs=...).

    - If not rotated: identity.
    - If rotated: instantiate a hypothesis with rotated POIs and read out base().POIs.
    """
    if not rotated:
        return np.array(poi_vec_rot, dtype=float)

    hyp_tmp = hyp_rotated.cloneModify(**dict(zip(poi_names, poi_vec_rot)))
    return np.array([p.val for p in hyp_tmp.base().POIs], dtype=float)


def _stage_cov_poi(fixed_group_names, group_to_nu_names):
    """
    Build Cov(POI | nu_fixed) where nu_fixed is the *union* of all nuisances in
    fixed_group_names.
    """
    fixed_nu_names = []
    for gname in fixed_group_names:
        fixed_nu_names += group_to_nu_names[gname]
    fixed_nu_idx = [idx_map[n] for n in fixed_nu_names]
    return _cond_cov_poi_given_fixed_nu(cov, poi_indices, fixed_nu_idx)


def _grad_f_wrt_poi_rot_at_xref(x_ref, eps0=1e-5):
    """
    Numerical gradient g_i = d f(x_ref) / d c_i in the *current POI coordinate system*.
    Used only for ranking at x_ref via: Var[f(x_ref)] ~ g^T Cov(c|nu_fixed) g.

    The PDF evaluation itself is always performed in the base coefficient basis.
    """
    c0_rot  = np.array(coeffs_central, dtype=float)
    c0_base = _rot_poi_to_base_coeffs(c0_rot)

    f0 = float(pdf.evaluate(
        x=np.array([x_ref], dtype=float),
        id=np.array([pid], dtype=int),
        Q=np.array([Q_val], dtype=float),
        coeffs=c0_base
    )[0])

    g = np.zeros(len(c0_rot), dtype=float)
    for i in range(len(c0_rot)):
        eps = eps0 * (abs(c0_rot[i]) + 1.0)

        cp = np.array(c0_rot, dtype=float); cp[i] += eps
        cm = np.array(c0_rot, dtype=float); cm[i] -= eps

        fp = float(pdf.evaluate(
            x=np.array([x_ref], dtype=float),
            id=np.array([pid], dtype=int),
            Q=np.array([Q_val], dtype=float),
            coeffs=_rot_poi_to_base_coeffs(cp)
        )[0])
        fm = float(pdf.evaluate(
            x=np.array([x_ref], dtype=float),
            id=np.array([pid], dtype=int),
            Q=np.array([Q_val], dtype=float),
            coeffs=_rot_poi_to_base_coeffs(cm)
        )[0])
        g[i] = (fp - fm) / (2.0 * eps)

    return f0, g


def _sample_pdf_band_from_cov_poi(cov_poi_stage):
    """
    Sample POIs from N(coeffs_central, cov_poi_stage), convert to base coeffs, evaluate
    PDF on the full x-grid, and return (q_low, q_high) quantiles.

    Quantiles: [0.32, 0.68].
    """
    if args.freezePOI:
        return central_pdf.copy(), central_pdf.copy()

    # identical RNG stream per stage -> easier comparisons between bands
    np.random.seed(fit_rng_seed)
    poi_samples_stage = np.random.multivariate_normal(
        mean=coeffs_central,
        cov=cov_poi_stage,
        size=n_toys
    )

    if rotated:
        poi_samples_base = np.zeros_like(poi_samples_stage)
        for itoy in range(n_toys):
            poi_samples_base[itoy, :] = _rot_poi_to_base_coeffs(poi_samples_stage[itoy, :])
    else:
        poi_samples_base = poi_samples_stage

    pdf_samples = np.zeros((n_toys, n_x), dtype=float)
    for itoy in range(n_toys):
        pdf_samples[itoy, :] = pdf.evaluate(
            x=x_vals, id=id_arr, Q=Q_arr, coeffs=poi_samples_base[itoy, :]
        )

    q_low  = np.quantile(pdf_samples, 0.32, axis=0)
    q_high = np.quantile(pdf_samples, 0.68, axis=0)
    return q_low, q_high


# ------------------- build x grid and evaluate PDFs --------------------

x_min = 0.015
x_max = 0.3
n_x   = 200
x_vals = np.logspace(np.log10(x_min), np.log10(x_max), n_x)

# gluon id=21, scale Q from CLI
pid   = 21
Q_val = float(args.Q)
id_arr = np.full(n_x, pid, dtype=int)
Q_arr  = np.full(n_x, Q_val, dtype=float)

# central coefficients in the *base* coefficient basis (needed for pdf.evaluate)
coeffs_central_base = _rot_poi_to_base_coeffs(coeffs_central)

# central gluon PDF
central_pdf = np.array(
    pdf.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs_central_base),
    dtype=float
)

# ----------------------------------------------------------------------
# Grouped uncertainty decomposition:
#   - rank nuisance groups greedily at x_ref
#   - build staged bands
# ----------------------------------------------------------------------

# Discover the systematics era, because we added little years everywhere
sys_grouping_era = None
for p in all_nu_names_fit:
    if "2016" in p:
        sys_grouping_era = sys_grouping[2016]
        break
    if "2017" in p:
        sys_grouping_era = sys_grouping[2017]
        break
    if "2018" in p:
        sys_grouping_era = sys_grouping[2018]
        break

group_to_nu_names = {gname: nu_list for gname, nu_list in sys_grouping_era}
group_names       = [gname for gname, _ in sys_grouping_era]

# enforce that grouping covers *all* nuisances present in the fit
all_nu_names_grp  = [n for _, nu_list in sys_grouping_era for n in nu_list]
assert set(all_nu_names_fit) == set(all_nu_names_grp)

x_ref = float(args.x_ref)
f_ref, g_ref = _grad_f_wrt_poi_rot_at_xref(x_ref)
print(f"[info] Ranking grouped nuisances at x_ref = {x_ref:g} (Bjorken-x)")
print(f"[info] Central f_g(x_ref, Q={Q_val:.2f}) = {f_ref:.6e}")

# ----------------------------------------------------------------------
# (OLD) Greedy "fix largest" ordering (commented out; keep for easy switch)
# ----------------------------------------------------------------------
# fixed_groups = []
# remaining    = list(group_names)
# ranked       = []
#
# cov_curr = _stage_cov_poi(fixed_groups, group_to_nu_names)
# var_curr = float(g_ref.T @ cov_curr @ g_ref)
#
# for istep in range(len(group_names)):
#     best_g   = None
#     best_red = None
#     best_var = None
#
#     for gname in remaining:
#         cov_try = _stage_cov_poi(fixed_groups + [gname], group_to_nu_names)
#         var_try = float(g_ref.T @ cov_try @ g_ref)
#         red = var_curr - var_try
#         if (best_red is None) or (red > best_red):
#             best_red = red
#             best_g   = gname
#             best_var = var_try
#
#     ranked.append(best_g)
#     fixed_groups.append(best_g)
#     remaining.remove(best_g)
#     var_curr = best_var
#
#     print(f"  step {istep+1:2d}: fix {best_g:>12s}  ->  Var_ref = {var_curr:.6e}")
#
# print("[info] Final ordering (largest incremental impact first):")
# print("  " + "  >  ".join(ranked))


# ----------------------------------------------------------------------
# (NEW) Greedy "add smallest" ordering:
#   Start at stats-only (all nu fixed), then add back groups with smallest
#   incremental increase in Var[f(x_ref)] at each step.
# ----------------------------------------------------------------------
fixed_groups = list(group_names)   # all fixed -> stats-only
remaining    = list(group_names)   # candidates to "add back" (unfix)
ranked_add   = []

cov_curr = _stage_cov_poi(fixed_groups, group_to_nu_names)  # stats-only cov
var_curr = float(g_ref.T @ cov_curr @ g_ref)

for istep in range(len(group_names)):
    print(f"At {istep}/{len(group_names)}")
    best_g   = None
    best_inc = None
    best_var = None

    for gname in remaining:
        fixed_try = [g for g in fixed_groups if g != gname]  # unfix gname
        cov_try = _stage_cov_poi(fixed_try, group_to_nu_names)
        var_try = float(g_ref.T @ cov_try @ g_ref)
        inc = var_try - var_curr  # incremental variance increase

        if (best_inc is None) or (inc < best_inc):
            best_inc = inc
            best_g   = gname
            best_var = var_try

    ranked_add.append(best_g)
    fixed_groups.remove(best_g)
    remaining.remove(best_g)
    var_curr = best_var

    print(f"  step {istep+1:2d}: add {best_g:>12s}  ->  Var_ref = {var_curr:.6e}")

print("[info] Final ordering (smallest incremental impact first):")
print("  " + "  <  ".join(ranked_add))

# Build staged bands:
#   stage 0: stats-only          (all nu fixed)
#   stage k: add ranked_add[:k]  (unfix these k groups)
#   stage G: all nuisances       (none fixed)
stage_labels = []
stage_q_low  = []
stage_q_high = []

for k in range(len(ranked_add) + 1):
    print(f"At {k}/{len(ranked_add)}")
    fixed_k = [g for g in group_names if g not in ranked_add[:k]]
    cov_k   = _stage_cov_poi(fixed_k, group_to_nu_names)
    ql, qh  = _sample_pdf_band_from_cov_poi(cov_k)

    stage_q_low.append(np.array(ql, dtype=float))
    stage_q_high.append(np.array(qh, dtype=float))

    if k == 0:
        stage_labels.append("Stats-only (all #nu fixed)")
    elif k == len(ranked_add):
        stage_labels.append(f"Add {ranked_add[k-1]} #rightarrow all")
    else:
        stage_labels.append(f"Add {ranked_add[k-1]}")

# --------------- build ratio (PDF / central) for bottom pad ------------

ratio_central = np.ones_like(central_pdf)
mask = central_pdf > 0

stage_ratio_low  = []
stage_ratio_high = []
for k in range(len(stage_q_low)):
    rl = np.ones_like(central_pdf)
    rh = np.ones_like(central_pdf)
    rl[mask] = stage_q_low[k][mask]  / central_pdf[mask]
    rh[mask] = stage_q_high[k][mask] / central_pdf[mask]
    stage_ratio_low.append(rl)
    stage_ratio_high.append(rh)

# y-range for plotting (PDF pad) based on all staged bands
all_vals = [central_pdf] + stage_q_low + stage_q_high
all_vals = np.concatenate([a[a > 0] for a in all_vals if np.any(a > 0)])

if all_vals.size > 0:
    y_min_top = 0.5 * float(np.min(all_vals))
    y_max_top = 1.5 * float(np.max(all_vals))
else:
    y_min_top = 1e-8
    y_max_top = 1.0

# y-range for ratio
r_min, r_max = 0.89, 1.11

# --------------------------- ROOT plotting -----------------------------

# Taller canvas to accommodate a dedicated legend/text pad
c = ROOT.TCanvas("c_gluonPDF", "gluonPDF", 800, 1400)

# Three pads:
#   - padLegend: top, contains legend + annotations
#   - padTop:    middle, PDF with bands
#   - padBottom: bottom, ratio
#
# Keep PDF and ratio pads same height; legend pad is additional.
padLegend = ROOT.TPad("padLegend", "padLegend", 0.0, 0.73, 1.0, 1.0)
padTop    = ROOT.TPad("padTop",    "padTop",    0.0, 0.41, 1.0, 0.73)
padBottom = ROOT.TPad("padBottom", "padBottom", 0.0, 0.00, 1.0, 0.41)

# Legend pad: no axes; just text + legend
padLegend.SetTopMargin(0.10)
padLegend.SetBottomMargin(0.0)
padLegend.SetLeftMargin(0.10)
padLegend.SetRightMargin(0.05)
padLegend.SetTicks(0, 0)

padLegend.SetFillColor(0)
padLegend.SetFrameFillColor(0)
padLegend.SetBorderMode(0)
padLegend.SetFrameBorderMode(0)

# PDF pad
padTop.SetBottomMargin(0.00)
padTop.SetTopMargin(0.0)
padTop.SetLeftMargin(0.10)
padTop.SetRightMargin(0.05)
padTop.SetTicks(1, 1)
padTop.SetLogx(True)

# Ratio pad
padBottom.SetTopMargin(0.00)
padBottom.SetBottomMargin(0.22)
padBottom.SetLeftMargin(0.10)
padBottom.SetRightMargin(0.05)
padBottom.SetTicks(1, 1)
padBottom.SetLogx(True)

padLegend.Draw()
padTop.Draw()
padBottom.Draw()

# ------------------- LEGEND PAD: annotations + legend ------------------
padLegend.cd()

latex = ROOT.TLatex()
latex.SetNDC(True)
latex.SetTextFont(42)
latex.SetTextSize(0.065)

# line 1: counts
latex.DrawLatex(0.12, 0.90, f"nPOI = {n_poi},  nNuis = {n_nu}   (total = {n_poi + n_nu})")

# line 2: postfit tag
latex.DrawLatex(0.12, 0.80, f"postfit: {postfit_tag}")

# line 3 (optional): rotate tag
if rotated:
    latex.DrawLatex(0.12, 0.70, f"rotate:  {rotate_tag}")

# The actual legend object will be created after the band graphs exist.
# We place it low in this pad to avoid overlap with the annotation lines.

# ------------------- TOP PAD: f_g(x, Q) with bands ----------------------
padTop.cd()

frame_top = ROOT.TH1F(
    "frame_gluonPDF_top",
    f";;f_{{g}}(x, Q = {Q_val:.2f})",
    100, x_min, x_max
)
frame_top.SetMinimum(y_min_top)
frame_top.SetMaximum(y_max_top)
frame_top.GetXaxis().SetMoreLogLabels(True)
frame_top.GetXaxis().SetNoExponent(True)
frame_top.GetXaxis().SetLabelSize(0)      # no x labels on PDF pad
frame_top.GetXaxis().SetTitleSize(0)      # no x title on PDF pad

frame_top.GetYaxis().SetTitleSize(0.060)
frame_top.GetYaxis().SetLabelSize(0.055)
frame_top.GetYaxis().SetTitleOffset(0.80)
frame_top.Draw()

# non-transparent, fixed palette cycle (cmsstyle p10)
colors = [
    cmsstyle.p10.kBlue,
    cmsstyle.p10.kYellow,
    cmsstyle.p10.kRed,
    cmsstyle.p10.kAsh,
    cmsstyle.p10.kViolet,
    cmsstyle.p10.kBrown,
    cmsstyle.p10.kOrange,
    cmsstyle.p10.kGreen,
    cmsstyle.p10.kGray,
    cmsstyle.p10.kCyan,
]

g_stage_bands_top = []
for k in range(len(stage_q_low)):
    col = colors[k % len(colors)]
    g_band = ROOT.TGraphAsymmErrors(n_x)
    g_band.SetFillColor(col)
    g_band.SetFillStyle(1001)
    g_band.SetLineWidth(0)
    g_band.SetLineColor(0)

    for i in range(n_x):
        y = central_pdf[i]
        dy_down = y - stage_q_low[k][i]
        dy_up   = stage_q_high[k][i] - y
        if dy_down < 0: dy_down = 0.0
        if dy_up   < 0: dy_up   = 0.0
        g_band.SetPoint(i, x_vals[i], y)
        g_band.SetPointError(i, 0.0, 0.0, dy_down, dy_up)

    g_stage_bands_top.append(g_band)

# In the "add smallest" staging, the band grows with k, so draw largest first
# (i.e. draw reversed) to keep stats-only visible on top.
for g_band in reversed(g_stage_bands_top):
    g_band.Draw("3 SAME")

# central line (PDF pad)
g_central_top = ROOT.TGraph(n_x)
for i in range(n_x):
    g_central_top.SetPoint(i, x_vals[i], central_pdf[i])
g_central_top.SetLineColor(ROOT.kBlack)
g_central_top.SetLineWidth(2)
g_central_top.Draw("L SAME")
padTop.RedrawAxis()          # or: ROOT.gPad.RedrawAxis()
padTop.Update()

# ------------------- BOTTOM PAD: ratio to central ----------------------
padBottom.cd()

frame_ratio = ROOT.TH1F(
    "frame_gluonPDF_ratio",
    ";x;variation / central",
    100, x_min, x_max
)
frame_ratio.SetMinimum(r_min)
frame_ratio.SetMaximum(r_max)
frame_ratio.GetXaxis().SetMoreLogLabels(True)
frame_ratio.GetXaxis().SetNoExponent(True)

frame_ratio.GetXaxis().SetTitleSize(0.065)
frame_ratio.GetXaxis().SetLabelSize(0.055)

frame_ratio.GetYaxis().SetTitleSize(0.050)
frame_ratio.GetYaxis().SetLabelSize(0.045)
frame_ratio.GetYaxis().CenterTitle(True)
frame_ratio.GetYaxis().SetTitleOffset(1.05)
frame_ratio.GetYaxis().SetNdivisions(505)
frame_ratio.Draw()

g_stage_bands_ratio = []
for k in range(len(stage_ratio_low)):
    col = colors[k % len(colors)]
    g_band = ROOT.TGraphAsymmErrors(n_x)
    g_band.SetFillColor(col)
    g_band.SetFillStyle(1001)
    g_band.SetLineWidth(0)
    g_band.SetLineColor(0)

    for i in range(n_x):
        y = ratio_central[i]
        dy_down = y - stage_ratio_low[k][i]
        dy_up   = stage_ratio_high[k][i] - y
        if dy_down < 0: dy_down = 0.0
        if dy_up   < 0: dy_up   = 0.0
        g_band.SetPoint(i, x_vals[i], y)
        g_band.SetPointError(i, 0.0, 0.0, dy_down, dy_up)

    g_stage_bands_ratio.append(g_band)

for g_band in reversed(g_stage_bands_ratio):
    g_band.Draw("3 SAME")

# central ratio line at 1
g_central_ratio = ROOT.TGraph(n_x)
for i in range(n_x):
    g_central_ratio.SetPoint(i, x_vals[i], 1.0)
g_central_ratio.SetLineColor(ROOT.kBlack)
g_central_ratio.SetLineWidth(2)
g_central_ratio.Draw("L SAME")
padBottom.RedrawAxis()       # or: ROOT.gPad.RedrawAxis()
padBottom.Update()

# ------------------- Finalize legend in legend pad ---------------------
padLegend.cd()

leg = ROOT.TLegend(0.05, 0.05, 0.95, 0.68)  # taller
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.055)
leg.SetNColumns(3)

leg.AddEntry(g_central_top, "Central (MLE)", "l")
for k in range(len(g_stage_bands_top)):
    leg.AddEntry(g_stage_bands_top[k], stage_labels[k], "f")
leg.Draw()

c.cd()
c.Update()

helpers.copyIndexPHP(pdf_plot_directory)

# Q tag with zero padding for sorted filenames
Q_int = int(round(Q_val * 1000.0))
Q_tag = f"Q{Q_int:06d}"

r_suf = "_ROT_"+os.path.splitext(os.path.basename(args.rotate))[0] if rotated else ""
f_suf = "_FIT_"+os.path.splitext(os.path.basename(args.postfit))[0]
out_png = os.path.join(pdf_plot_directory, f"{plot_label}_gluon_{args.postfix}{f_suf}{r_suf}_{Q_tag}.png")
out_pdf = os.path.join(pdf_plot_directory, f"{plot_label}_gluon_{args.postfix}{f_suf}{r_suf}_{Q_tag}.pdf")
c.SaveAs(out_png)
c.SaveAs(out_pdf)

print(f"[info] {plot_label} PDF plot written to:\n  {out_png}\n  {out_pdf}")

syncer.sync()

