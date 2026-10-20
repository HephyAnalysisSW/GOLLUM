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
import glob
import json

from ctypes import c_double
import numpy as np
import uproot
import awkward as ak
import ROOT

# project roots
notebook_dir = '/users/ang.li/public/GluonPDF/HEPHY-uncertainty/user/ang'
os.chdir(notebook_dir)
sys.path.insert(0, os.path.abspath(os.path.join(notebook_dir, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(notebook_dir, '../../')))

import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader
from pdf.PDFParametrization import PDFParametrization

from fit.Likelihood import load_likelihood
from fit.Likelihood import build_hypothesis_from_likelihood
from fit.Modeling import Rotated

args = argparse.ArgumentParser()
args.add_argument("--ntoys", type=int, default=100, help="Number of toy samples to generate.")
args.add_argument("--seed", type=int, default=42, help="Random seed for toy generation.")
args.add_argument("--rotate", type=str, help="Json file for the rotation.")
args.add_argument("--postfit", type=str, help="Json file for the post-fit results.")
args.add_argument("--config", type=str, help="YAML config file defining the likelihood and PDF parametrization.")
args.add_argument("--postfix", type=str, help="Output ROOT file postfix for histograms.")
args.add_argument("--output", type=str, default="./output", help="Output directory for histograms.")

args = args.parse_args()

def getsamples(config, postfit, rotate, n_toys, fit_rng_seed):
    cfg = yaml_loader.load_yaml(config)
    with open(postfit, "r") as f:
        fit_results = json.load(f)
    fit_par_names = [p["name"] for p in fit_results["parameters"]]
    params_best   = np.array([p["value"] for p in fit_results["parameters"]], dtype=float)
    cov           = np.array(fit_results["covariance"]["matrix"], dtype=float)
    rotated = bool(rotate)
    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info)
    hyp_rotated = Rotated(hyp, rotate, name="Fisher-basis")
    like_par_names = [p.name for p in hyp_rotated.parameters]
    poi_name_set   = {p.name for p in hyp_rotated.POIs}
    set_like = set(like_par_names)
    set_fit  = set(fit_par_names)

    only_in_like = sorted(list(set_like - set_fit))
    only_in_fit  = sorted(list(set_fit  - set_like))

    handled_missing_pois = []
    for n in only_in_like:
        if n in poi_name_set:
            print(f"[warning] POI '{n}' is in the likelihood but missing in the fit result; setting it to zero and freezing it.")
            getattr(hyp_rotated, n).freeze(value=0.0)
            handled_missing_pois.append(n)

    only_in_like = [n for n in only_in_like if n not in handled_missing_pois]
    like_par_names_in_fit = [p.name for p in hyp_rotated.parameters if p.name in set_fit]

    like_params = None
    poi_job_id = None
    region_string = None

    # binned
    if like_info.get("regions", []):
        region_string = "regions"
        poi_type = "bit"
    elif like_info.get("binned", []) != []:
        region_string = "binned"
        poi_type = "ich"
    else:
        raise KeyError("Missing definition of regions.")

    for region in like_info.get(region_string, []):
        for cls in region.get("classes", []):
            poi = cls.get("POI", None)
            if poi and poi.get("type") == poi_type and poi.get("parameters"):
                like_params = poi["parameters"]
                poi_job_id = poi["job"]
                break
        if like_params is not None:
            break

    if like_params is None or poi_job_id is None:
        print("[error] Could not find a POI-dependent term in the likelihood.")
        sys.exit(1)

    print(f"[info] Using POI-dependent job '{poi_job_id}' with POIs: {', '.join(like_params)}")

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
    pdf_basis = pdf_cfg.get("pdf_basis", None)
    pdf_rescale_pod_amplitudes = pdf_cfg.get("rescale_pod_amplitudes", True)

    pdf = PDFParametrization(n=pdf_n, typ=pdf_type, basis=pdf_basis, rescale_pod_amplitudes=pdf_rescale_pod_amplitudes)

    # map parameter names -> indices in fit result
    idx_map = {name: i for i, name in enumerate(fit_par_names)}

    poi_names = [p.name for p in hyp_rotated.POIs]
    poi_names_in_fit = [name for name in poi_names if name in idx_map]
    poi_names_missing = [name for name in poi_names if name not in idx_map]

    if poi_names_missing:
        if rotated and all(getattr(hyp_rotated, name).isFrozen for name in poi_names_missing):
            print("[warning] Frozen rotated POIs missing in fit result; setting them to zero:")
            for name in poi_names_missing:
                print(f"  {name:>20s} =  0.00000e+00")
                getattr(hyp_rotated, name).val = 0.0
        else:
            print("[error] These POIs are missing in the fit result parameter list:")
            for name in poi_names_missing:
                print("  ", name)
            sys.exit(1)

    poi_indices = [idx_map[name] for name in poi_names_in_fit]

    coeffs_central = params_best[poi_indices]
    print("[info] Central POI coefficients (MLE):")
    for name in poi_names:
        if name in idx_map:
            val = params_best[idx_map[name]]
        else:
            val = 0.0
        print(f"  {name:>20s} = {val: .5e}")

        cov_poi = cov[np.ix_(poi_indices, poi_indices)] if poi_indices else np.zeros((0, 0), dtype=float)

    print(f"[info] Sampling {len(poi_names_in_fit)} POIs with {n_toys} toys...")
    np.random.seed(fit_rng_seed)
    poi_samples = np.random.multivariate_normal(
        mean=coeffs_central,
        cov=cov_poi,
        size=n_toys
    ) if len(poi_names_in_fit) > 0 else np.zeros((n_toys, 0), dtype=float)

    # rotate into original coefficients using hyp_rotated
    poi_samples_base = []
    if rotated:
        print("Un-rotating samples.")
        hyp_central = hyp_rotated.cloneModify(**dict(zip(poi_names_in_fit, coeffs_central)))
        coeffs_central_base = np.array([p.val for p in hyp_central.base().POIs])
        for sample in poi_samples:
            hyp_sample = hyp_rotated.cloneModify(**dict(zip(poi_names_in_fit, sample)))
            poi_samples_base.append([p.val for p in hyp_sample.base().POIs])
        poi_samples_base = np.array(poi_samples_base)
        print("Done.")
    else:
        coeffs_central_base = coeffs_central
        poi_samples_base = poi_samples
    return pdf, coeffs_central_base, poi_samples_base

def makehists(x,bins,weights,name):
    h, _ = np.histogram(x, bins=bins, weights=weights)
    hroot = ROOT.TH1D("h"+name, "h", len(bins)-1, np.array(bins,dtype=float))
    for b in range(1,len(bins)):
        hroot.SetBinContent(b, float(h[b-1]))
    return hroot

pdf, coeffs_central_base, poi_samples_base = getsamples(args.config, args.postfit, args.rotate, n_toys=args.ntoys, fit_rng_seed=args.seed)

input_paths=[os.path.join(
        "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/Hgg-gen-ntuples/",
        "RunIISummer20UL16NanoAODAPVv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_preVFP_v11-v2/",
    )]

for p in input_paths:
    files = glob.glob(f"{p}/*.root")

print(f"Found {len(files)} files")

branches = [
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "H_pt",
    "H_y",
    "Generator_weight",
    "LHEPdfWeight"
]

arrays = uproot.concatenate(
    {file: "Events" for file in files},
    expressions=branches,
    library="ak"   # awkward arrays
)

# Access data
x1 = arrays["Generator_x1"]
x2 = arrays["Generator_x2"]
id1 = arrays["Generator_id1"]
id2 = arrays["Generator_id2"]
Q = arrays["Generator_scalePDF"]
h_pt = arrays["H_pt"]
h_y = arrays["H_y"]
gen_weight = arrays["Generator_weight"]
pdf_weights = arrays["LHEPdfWeight"]

h_y_abs = np.abs(h_y)
#lumi = 19.5  # in fb^-1
lumi = 1
xsec = 36200  # in fb
# Calculate event weights
xsec_weights = (lumi * xsec) / np.sum(gen_weight)

y_bins = [0,1,1.5,2.0,2.5]
xmin = 0
xmax = 200
nxbins = 10
bins = np.linspace(xmin,xmax,nxbins+1)
bin_width = (xmax - xmin) / nxbins

notp = np.logical_not((h_y_abs<1.5) & (h_y_abs>1.0) & (h_pt>100) & (h_pt<120) & (id1==21) & (id2==-2) & (0.00368>x1) & (x1>0.00367) & (0.43416>x2) & (x2>0.43415), dtype=int)

xsec_weights = xsec_weights/bin_width

newpdf_weights = pdf.product_parametrizations(np.array(x1),np.array(x2),np.array(id1),np.array(id2),poi_samples_base,np.array(Q))
newpdf_weights_nominal = pdf.product_parametrizations(np.array(x1),np.array(x2),np.array(id1),np.array(id2),coeffs_central_base,np.array(Q))

hs = {}
n_slices = len(y_bins) - 1
for i in range(n_slices):
    #sel = (h_y_abs >= y_bins[i]) & (h_y_abs < y_bins[i+1]) & (id1==21) & (id2==21)
    sel = (h_y_abs >= y_bins[i]) & (h_y_abs < y_bins[i+1]) & (np)
    hs[f"slice_{i}"] = makehists(h_pt[sel], bins=bins, weights=newpdf_weights_nominal[sel]*xsec_weights*gen_weight[sel],name=f"nom_y{i}")

hs_vars = {}
for i in range(args.ntoys):
    hs_vars[i] = {}
    for j in range(n_slices):
        #sel = (h_y_abs >= y_bins[j]) & (h_y_abs < y_bins[j+1]) & (id1==21) & (id2==21)
        sel = (h_y_abs >= y_bins[j]) & (h_y_abs < y_bins[j+1]) & (np)
        hs_vars[i][f"slice_{j}"] = makehists(h_pt[sel], bins=bins, weights=newpdf_weights[i][sel]*xsec_weights*gen_weight[sel],name=f"toy_y{j}_var{i}")

if not os.path.exists(args.output):
    os.makedirs(args.output)
fnew = ROOT.TFile(os.path.join(args.output,f"hists_toys_{args.postfix}.root"), "RECREATE")
fnew.cd()
for key, hist in hs.items():
    hist.Write()
for toy_idx, toy_hists in hs_vars.items():
    for key, hist in toy_hists.items():
        hist.Write()
