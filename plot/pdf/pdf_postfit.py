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
pdf_plot_directory = os.path.join(user.plot_directory,  plot_label, args.subdir, os.path.splitext(os.path.basename(args.config))[0])
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

poi_names = [p.name for p in hyp_rotated.POIs]
try:
    #poi_indices = [idx_map[name] for name in like_params]
    poi_indices = [idx_map[name] for name in poi_names] 
except KeyError as e:
    print(f"[error] POI '{e.args[0]}' not found in fit result parameter list.")
    sys.exit(1)

# central POI coefficients (MLE)
coeffs_central = params_best[poi_indices]
print("[info] Central POI coefficients (MLE):")
for name, val in zip(poi_names, coeffs_central):
    print(f"  {name:>20s} = {val: .5e}")

# covariance submatrix for POIs
cov_poi = cov[np.ix_(poi_indices, poi_indices)]

# knobs for sampling
n_toys       = 10000
fit_rng_seed = 42

# sample POIs unless freezePOI is set
if args.freezePOI:
    print("[info] --freezePOI given: not sampling POIs, will only show central PDF.")
    poi_samples = None
else:
    print(f"[info] Sampling {len(like_params)} POIs with {n_toys} toys...")
    np.random.seed(fit_rng_seed)
    poi_samples = np.random.multivariate_normal(
        mean=coeffs_central,
        cov=cov_poi,
        size=n_toys
    )

    # rotate into original coefficients using hyp_rotated
    poi_samples_base = []
    if rotated:
        print("Un-rotating samples.") 
        hyp_central = hyp_rotated.cloneModify(**dict(zip(poi_names, coeffs_central)))
        coeffs_central_base =  np.array( [p.val for p in hyp_central.base().POIs])
        for sample in poi_samples:
            hyp_sample = hyp_rotated.cloneModify(**dict(zip(poi_names, sample)))
            poi_samples_base.append([p.val for p in hyp_sample.base().POIs])
        poi_samples_base = np.array(poi_samples_base)
        print("Done.")
    else:
        coeffs_central_base = coeffs_central
        poi_samples_base = poi_samples

# ------------------- build x grid and evaluate PDFs --------------------

x_min = 1e-3
x_max = 0.8
n_x   = 200
x_vals = np.logspace(np.log10(x_min), np.log10(x_max), n_x)

# gluon id=21, scale Q from CLI
pid   = 21
Q_val = float(args.Q)
id_arr = np.full(n_x, pid, dtype=int)
Q_arr  = np.full(n_x, Q_val, dtype=float)

# central gluon PDF
central_pdf = np.array(
    pdf.evaluate(x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs_central_base),
    dtype=float
)

# toy PDFs
if poi_samples_base is not None:
    pdf_samples = np.zeros((n_toys, n_x), dtype=float)
    for itoy in range(n_toys):
        coeffs_toy = poi_samples_base[itoy]
        pdf_samples[itoy, :] = pdf.evaluate(
            x=x_vals, id=id_arr, Q=Q_arr, coeffs=coeffs_toy
        )
    q_low  = np.quantile(pdf_samples, 0.32, axis=0)
    q_high = np.quantile(pdf_samples, 0.68, axis=0)
else:
    # no sampling: band collapses to central
    q_low  = central_pdf.copy()
    q_high = central_pdf.copy()

# --------------- build ratio (PDF / central) for bottom pad ------------

ratio_central = np.ones_like(central_pdf)
ratio_low     = np.ones_like(central_pdf)
ratio_high    = np.ones_like(central_pdf)

mask = central_pdf > 0
ratio_low[mask]  = q_low[mask]  / central_pdf[mask]
ratio_high[mask] = q_high[mask] / central_pdf[mask]

# y-range for plotting (top pad) based on central PDF only
positive_central = central_pdf[central_pdf > 0]
if positive_central.size > 0:
    y_min_top = 0.5 * float(np.min(positive_central))
    y_max_top = 1.5 * float(np.max(positive_central))
else:
    y_min_top = 1e-8
    y_max_top = 1.0

# y-range for ratio (bottom pad) fixed to [0, 2]
r_min, r_max = 0.75, 1.25

# --------------------------- ROOT plotting -----------------------------

c = ROOT.TCanvas("c_gluonPDF", "gluonPDF", 800, 800)

padTop    = ROOT.TPad("padTop", "padTop", 0.0, 0.30, 1.0, 1.0)
padBottom = ROOT.TPad("padBottom", "padBottom", 0.0, 0.00, 1.0, 0.30)

padTop.SetBottomMargin(0.0)
padTop.SetTopMargin(0.08)
padTop.SetLeftMargin(0.10)
padTop.SetRightMargin(0.05)
padTop.SetTicks(1, 1)
padTop.SetLogx(True)

padBottom.SetTopMargin(0.0)
padBottom.SetBottomMargin(0.30)
padBottom.SetLeftMargin(0.10)
padBottom.SetRightMargin(0.05)
padBottom.SetTicks(1, 1)
padBottom.SetLogx(True)

padTop.Draw()
padBottom.Draw()

# ------------------- TOP PAD: f_g(x, Q) with band ----------------------
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
frame_top.GetXaxis().SetLabelSize(0)      # no x labels on top pad
frame_top.GetXaxis().SetTitleSize(0)      # no x title on top pad
frame_top.GetYaxis().SetTitleSize(0.05)
frame_top.GetYaxis().SetLabelSize(0.045)
frame_top.GetYaxis().SetTitleOffset(0.85)
frame_top.Draw()

# band from quantiles (top)
g_band_top = ROOT.TGraphAsymmErrors(n_x)
g_band_top.SetFillColor(ROOT.kGray + 1)
g_band_top.SetFillStyle(3345)
g_band_top.SetLineWidth(0)
g_band_top.SetLineColor(0)  # no band outline

for i in range(n_x):
    y  = central_pdf[i]
    dy_down = y - q_low[i]
    dy_up   = q_high[i] - y
    if dy_down < 0: dy_down = 0.0
    if dy_up   < 0: dy_up   = 0.0
    g_band_top.SetPoint(i, x_vals[i], y)
    g_band_top.SetPointError(i, 0.0, 0.0, dy_down, dy_up)

g_band_top.Draw("3 SAME")

# quantile lines (top)
g_q_low_top = ROOT.TGraph(n_x)
g_q_high_top = ROOT.TGraph(n_x)
for i in range(n_x):
    g_q_low_top.SetPoint(i, x_vals[i], q_low[i])
    g_q_high_top.SetPoint(i, x_vals[i], q_high[i])
for g in (g_q_low_top, g_q_high_top):
    g.SetLineColor(ROOT.kBlack)
    g.SetLineWidth(1)
    g.Draw("L SAME")

# central line (top)
g_central_top = ROOT.TGraph(n_x)
for i in range(n_x):
    g_central_top.SetPoint(i, x_vals[i], central_pdf[i])
g_central_top.SetLineColor(ROOT.kBlack)
g_central_top.SetLineWidth(2)
g_central_top.Draw("L SAME")

# legend (top)
leg = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(g_central_top, "Central (MLE)", "l")
leg.AddEntry(g_band_top,    "68% interval (POI)", "f")
leg.Draw()

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
frame_ratio.GetXaxis().SetTitleSize(0.10)
frame_ratio.GetXaxis().SetLabelSize(0.08)
frame_ratio.GetYaxis().SetTitleSize(0.09)
frame_ratio.GetYaxis().SetLabelSize(0.08)
frame_ratio.GetYaxis().SetTitleOffset(0.5)
frame_ratio.GetYaxis().SetNdivisions(505)
frame_ratio.Draw()

# band from quantiles in ratio space
g_band_ratio = ROOT.TGraphAsymmErrors(n_x)
g_band_ratio.SetFillColor(ROOT.kGray + 1)
g_band_ratio.SetFillStyle(3345)
g_band_ratio.SetLineWidth(0)
g_band_ratio.SetLineColor(0)  # no band outline

for i in range(n_x):
    y  = ratio_central[i]
    dy_down = y - ratio_low[i]
    dy_up   = ratio_high[i] - y
    if dy_down < 0: dy_down = 0.0
    if dy_up   < 0: dy_up   = 0.0
    g_band_ratio.SetPoint(i, x_vals[i], y)
    g_band_ratio.SetPointError(i, 0.0, 0.0, dy_down, dy_up)

g_band_ratio.Draw("3 SAME")

# quantile lines (ratio)
g_q_low_ratio = ROOT.TGraph(n_x)
g_q_high_ratio = ROOT.TGraph(n_x)
for i in range(n_x):
    g_q_low_ratio.SetPoint(i, x_vals[i], ratio_low[i])
    g_q_high_ratio.SetPoint(i, x_vals[i], ratio_high[i])
for g in (g_q_low_ratio, g_q_high_ratio):
    g.SetLineColor(ROOT.kBlack)
    g.SetLineWidth(1)
    g.Draw("L SAME")

# central ratio line at 1
g_central_ratio = ROOT.TGraph(n_x)
for i in range(n_x):
    g_central_ratio.SetPoint(i, x_vals[i], ratio_central[i])
g_central_ratio.SetLineColor(ROOT.kBlack)
g_central_ratio.SetLineWidth(2)
g_central_ratio.Draw("L SAME")

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

