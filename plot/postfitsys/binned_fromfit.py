#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import json
import argparse
from array import array

import numpy as np
import ROOT

import common.user as user
import common.syncer as syncer
import common.helpers as helpers
import common.yaml_loader as yaml_loader

from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL
from fit.Modeling import Rotated
from data.plot_options import plot_options
from data.colors import get_color

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

_KEEPALIVE = []


def _safe_name(text):
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(text))


def _flatten(arr):
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        return out.copy()
    if out.ndim == 2:
        return out.reshape(-1).copy()
    raise RuntimeError(f"Expected 1D or 2D array, got shape {out.shape}")


def evaluate_class(cls, h_base):
    poi = cls.get("POI", {}) or {}
    pred = poi.get("predictor", None)
    if pred is None:
        raise RuntimeError(f"Missing ICH predictor for class {cls.get('id', '?')}")

    poi_params = list(poi.get("parameters", []) or [])
    cvec = np.array([float(h_base[name].val) for name in poi_params], dtype=np.float64)
    vals = _flatten(pred.predict(cvec))

    for syst in cls.get("systematics", []) or []:
        stype = syst.get("type")

        if stype == "icph":
            spred = syst.get("predictor", None)
            if spred is None:
                raise RuntimeError(f"Missing ICPH predictor for {cls.get('id','?')}/{syst.get('id','?')}")
            sparams = list(syst.get("parameters", []) or [])
            x = np.array([float(h_base[name].val) for name in sparams], dtype=np.float64)
            rel = _flatten(spred.predict(x))
            if len(rel) != len(vals):
                raise RuntimeError(
                    f"Shape mismatch in {cls.get('id','?')}/{syst.get('id','?')}: "
                    f"{len(rel)} vs {len(vals)}"
                )
            vals *= rel

        elif stype == "lnN":
            sparams = list(syst.get("parameters", []) or [])
            if len(sparams) != 1:
                raise RuntimeError(f"Bad lnN systematic: {syst}")
            alpha = float(syst.get("value", 0.0))
            vals *= np.exp(np.log1p(alpha) * float(h_base[sparams[0]].val))

    return vals


p = argparse.ArgumentParser(description="Simple postfit binned template plots from an external fit result")
p.add_argument("fit_result", help="Fit JSON file")
p.add_argument("--config", default=None, help="Config YAML. If omitted, try to read it from the fit JSON.")
p.add_argument("--rotate", default=None, help="Rotation JSON, same logic as Likelihood.py")
p.add_argument("--outdir", default=None, help="Output directory")
p.add_argument("--n-toys", default=1000, type=int, help="Number of covariance toys")
p.add_argument("--seed", default=42, type=int, help="Random seed")
args = p.parse_args()

fit = json.load(open(args.fit_result))

config_path = args.config
if config_path is None:
    config_path = (
        fit.get("args", {}).get("config", None)
        or fit.get("config", None)
    )
if config_path is None:
    raise RuntimeError("Need --config, because no config path was found inside the fit JSON.")

print(f"[info] fit_result : {args.fit_result}")
print(f"[info] config     : {config_path}")
print(f"[info] rotate     : {args.rotate}")

cfg = yaml_loader.load_yaml(config_path)
yaml_loader.print_summary(cfg, config_path, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, config_path, overwrite=False)

like_info = load_likelihood(cfg)
hyp = build_hypothesis_from_likelihood(like_info, name="SR")
rotated = bool(args.rotate)
hyp_for_fit = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

fit_names = list(fit["free_parameter_order"])
fit_param_map = {p["name"]: float(p["value"]) for p in fit["parameters"]}
fit_cov = np.asarray(fit["covariance"]["matrix"], dtype=np.float64)

all_unfrozen = [par for par in hyp_for_fit.parameters if not par.isFrozen]
all_unfrozen_names = [par.name for par in all_unfrozen]
poi_name_set = {par.name for par in hyp_for_fit.POIs}

missing_in_fit = [n for n in all_unfrozen_names if n not in fit_param_map or n not in fit_names]
missing_pois_in_fit = [n for n in missing_in_fit if n in poi_name_set]
missing_other_in_fit = [n for n in missing_in_fit if n not in poi_name_set]

if missing_pois_in_fit:
    print(f"[warning] These POIs are missing in the fit JSON; setting them to zero and not sampling them: {missing_pois_in_fit}")
    for n in missing_pois_in_fit:
        if hasattr(hyp_for_fit, n):
            getattr(hyp_for_fit, n).val = 0.0
        if hasattr(hyp, n):
            getattr(hyp, n).val = 0.0

if missing_other_in_fit:
    raise RuntimeError(f"These active non-POI parameters are missing in the fit JSON: {missing_other_in_fit}")

extra_in_fit = [n for n in fit_names if n not in all_unfrozen_names]
if extra_in_fit:
    print(f"[warning] Ignoring parameters present in fit JSON but not active here: {extra_in_fit}")

active_params = [par for par in all_unfrozen if par.name in fit_param_map and par.name in fit_names]
active_names = [par.name for par in active_params]

fit_idx = {n: i for i, n in enumerate(fit_names)}
cov_idx = [fit_idx[n] for n in active_names]
cov_active = fit_cov[np.ix_(cov_idx, cov_idx)]
mean_active = np.array([fit_param_map[n] for n in active_names], dtype=np.float64)

best_fit_hyp = hyp_for_fit.cloneModify(**{n: fit_param_map[n] for n in active_names})
best_base_hyp = best_fit_hyp._base

np.random.seed(args.seed)
if len(active_names) > 0 and args.n_toys > 0:
    theta_samples = np.random.multivariate_normal(mean_active, cov_active, size=args.n_toys)
else:
    theta_samples = np.zeros((0, len(active_names)), dtype=np.float64)

default_binning = cfg.get("defaults", {}).get("default_binning", None)
if default_binning:
    var_name, var_edges_default = default_binning[0]
    x_title_default = plot_options.get(var_name, {}).get("tex", var_name)
    logY_default = plot_options.get(var_name, {}).get("logY", False)
else:
    var_name = "bin"
    var_edges_default = None
    x_title_default = "bin"
    logY_default = False

base = os.path.splitext(os.path.basename(config_path))[0]
version = str(cfg.get("version", "v0"))
suffix = "_rotate" if rotated else ""
if args.outdir is None:
    outdir = os.path.join(user.plot_directory, "binned_postfit_fromfit", base, f"{version}{suffix}")
else:
    outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
helpers.copyIndexPHP(outdir)

print(f"[info] output dir : {outdir}")
print(f"[info] n_toys     : {args.n_toys}")

region_canvases = {}

for region in like_info.get("binned", []) or []:
    region_id = region["id"]
    print(f"[info] plotting region {region_id}")

    first_pred = ((region.get("classes", []) or [])[0].get("POI") or {}).get("predictor", None)
    if first_pred is None:
        raise RuntimeError(f"Region {region_id} has no first ICH predictor")

    unroll = N2LL._unroll_bins_from_ich(first_pred)
    shape = tuple(unroll.get("shape", ()))
    n_bins_region = len(unroll["flat_bins"])

    use_default_1d_binning = (
        len(shape) == 1
        and var_edges_default is not None
        and (len(var_edges_default) - 1 == n_bins_region)
    )

    if use_default_1d_binning:
        plot_edges = array('d', var_edges_default)
        x_title = x_title_default
        logY = logY_default
        separators = []
    else:
        plot_edges = array('d', np.arange(n_bins_region + 1, dtype=np.float64))
        x_title = "unrolled bin"
        logY = False
        separators = []
        if len(shape) == 2:
            nb1, nb2 = shape
            separators = [i * nb2 for i in range(1, nb1)]

    class_infos = []
    total_central = np.zeros(n_bins_region, dtype=np.float64)

    for cls in region.get("classes", []) or []:
        class_id = cls["id"]
        sample_name = cls.get("sample", class_id)

        vals = evaluate_class(cls, best_base_hyp)
        if len(vals) != n_bins_region:
            raise RuntimeError(
                f"Region {region_id}, class {class_id}: got {len(vals)} bins, expected {n_bins_region}"
            )

        class_infos.append({
            "id": class_id,
            "sample": sample_name,
            "cls": cls,
            "central": vals,
        })
        total_central += vals

    if not class_infos:
        print(f"[warning] Region {region_id} has no classes, skipping.")
        continue

    if len(theta_samples) > 0:
        total_samples = np.zeros((len(theta_samples), n_bins_region), dtype=np.float64)

        for itoy, theta in enumerate(theta_samples):
            toy_pars = {name: float(theta[i]) for i, name in enumerate(active_names)}
            toy_hyp = hyp_for_fit.cloneModify(**toy_pars)
            toy_base = toy_hyp._base

            total_this = np.zeros(n_bins_region, dtype=np.float64)
            for ci in class_infos:
                total_this += evaluate_class(ci["cls"], toy_base)
            total_samples[itoy, :] = total_this

        q_low = np.quantile(total_samples, 0.16, axis=0)
        q_high = np.quantile(total_samples, 0.84, axis=0)
    else:
        q_low = total_central.copy()
        q_high = total_central.copy()

    class_hists = []
    class_labels = []

    for ci in class_infos:
        color = get_color(ci["sample"]) if callable(get_color) else ROOT.kGray + 1
        h_cls = ROOT.TH1F(
            f"h_postfit_{_safe_name(region_id)}_{_safe_name(ci['id'])}",
            "",
            len(plot_edges) - 1,
            plot_edges
        )
        h_cls.SetDirectory(0)
        for ib, y in enumerate(ci["central"], start=1):
            h_cls.SetBinContent(ib, float(y))
        h_cls.SetLineColor(ROOT.kBlack)
        h_cls.SetFillColor(color)
        h_cls.SetLineWidth(1)
        class_hists.append(h_cls)
        class_labels.append(ci["sample"])

    h_total = ROOT.TH1F(
        f"h_postfit_total_{_safe_name(region_id)}",
        "",
        len(plot_edges) - 1,
        plot_edges
    )
    h_total.SetDirectory(0)
    for ib, y in enumerate(total_central, start=1):
        h_total.SetBinContent(ib, float(y))

    h_unc = h_total.Clone(f"h_postfit_unc_{_safe_name(region_id)}")
    h_unc.SetDirectory(0)
    for ib, (nom, lo, hi) in enumerate(zip(total_central, q_low, q_high), start=1):
        err = max(abs(nom - lo), abs(hi - nom))
        h_unc.SetBinError(ib, float(err))
    h_unc.SetFillColor(ROOT.kGray + 1)
    h_unc.SetFillStyle(3345)
    h_unc.SetLineWidth(0)
    h_unc.SetMarkerSize(0)

    h_unc_up = h_total.Clone(f"h_postfit_unc_up_{_safe_name(region_id)}")
    h_unc_down = h_total.Clone(f"h_postfit_unc_down_{_safe_name(region_id)}")
    h_unc_up.SetDirectory(0)
    h_unc_down.SetDirectory(0)
    for ib in range(1, n_bins_region + 1):
        nom = h_total.GetBinContent(ib)
        err = h_unc.GetBinError(ib)
        h_unc_up.SetBinContent(ib, nom + err)
        h_unc_down.SetBinContent(ib, max(0.0, nom - err))
        h_unc_up.SetBinError(ib, 0.0)
        h_unc_down.SetBinError(ib, 0.0)
    h_unc_up.SetLineColor(ROOT.kGray + 2)
    h_unc_down.SetLineColor(ROOT.kGray + 2)
    h_unc_up.SetLineWidth(1)
    h_unc_down.SetLineWidth(1)
    h_unc_up.SetFillStyle(0)
    h_unc_down.SetFillStyle(0)

    h_data = h_total.Clone(f"h_postfit_data_{_safe_name(region_id)}")
    h_data.SetDirectory(0)
    for ib in range(1, n_bins_region + 1):
        y = h_data.GetBinContent(ib)
        h_data.SetBinError(ib, np.sqrt(max(0.0, y)))
    h_data.SetMarkerStyle(ROOT.kFullCircle)
    h_data.SetMarkerSize(1.0)
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetFillStyle(0)

    h_ratio_data = h_data.Clone(f"h_postfit_ratio_data_{_safe_name(region_id)}")
    h_ratio_data.SetDirectory(0)
    for ib in range(1, n_bins_region + 1):
        nom = h_total.GetBinContent(ib)
        y   = h_data.GetBinContent(ib)
        ey  = h_data.GetBinError(ib)   # stat-only = sqrt(y)

        if nom > 0.0:
            h_ratio_data.SetBinContent(ib, y / nom)
            h_ratio_data.SetBinError(ib, ey / nom)
        else:
            h_ratio_data.SetBinContent(ib, 0.0)
            h_ratio_data.SetBinError(ib, 0.0)

    h_ratio_data.SetMarkerStyle(ROOT.kFullCircle)
    h_ratio_data.SetMarkerSize(1.0)
    h_ratio_data.SetLineColor(ROOT.kBlack)
    h_ratio_data.SetMarkerColor(ROOT.kBlack)
    h_ratio_data.SetLineWidth(1)
    h_ratio_data.SetFillStyle(0)

    integrals = [h.Integral(1, n_bins_region) for h in class_hists]
    order = sorted(range(len(class_hists)), key=lambda i: integrals[i])
    class_hists_sorted = [class_hists[i] for i in order]
    class_labels_sorted = [class_labels[i] for i in order]

    hs = ROOT.THStack(f"stack_postfit_{_safe_name(region_id)}", "")
    for h in class_hists_sorted:
        hs.Add(h, "hist")

    c = ROOT.TCanvas(f"c_postfit_{_safe_name(region_id)}", f"c_postfit_{_safe_name(region_id)}", 900, 900)
    padTop = ROOT.TPad(c.GetName() + "_top", c.GetName() + "_top", 0.0, 0.30, 1.0, 1.0)
    padBottom = ROOT.TPad(c.GetName() + "_bottom", c.GetName() + "_bottom", 0.0, 0.00, 1.0, 0.30)

    padTop.SetBottomMargin(0.0)
    padTop.SetTopMargin(0.08)
    padTop.SetLeftMargin(0.10)
    padTop.SetRightMargin(0.05)
    padTop.SetTicks(1, 1)

    padBottom.SetTopMargin(0.0)
    padBottom.SetBottomMargin(0.30)
    padBottom.SetLeftMargin(0.10)
    padBottom.SetRightMargin(0.05)
    padBottom.SetTicks(1, 1)

    padTop.Draw()
    padBottom.Draw()

    padTop.cd()
    if logY:
        padTop.SetLogy(True)

    hs.Draw("HIST")
    hs.GetYaxis().SetTitle("Events")
    hs.GetYaxis().SetTitleSize(0.05)
    hs.GetYaxis().SetTitleOffset(1.1)
    hs.GetYaxis().SetLabelSize(0.045)
    hs.GetXaxis().SetLabelSize(0)
    hs.GetXaxis().SetTitleSize(0)

    max_y = max(hs.GetMaximum(), h_data.GetMaximum())
    if logY:
        hs.SetMinimum(0.5)
        hs.SetMaximum(10.0 * max_y if max_y > 0 else 1.0)
    else:
        hs.SetMinimum(0.0)
        hs.SetMaximum(1.5 * max_y if max_y > 0 else 1.0)

    h_unc.Draw("E2 SAME")
    h_unc_up.Draw("HIST SAME")
    h_unc_down.Draw("HIST SAME")
    h_data.Draw("E SAME")

    top_sep_lines = []
    for x in separators:
        line = ROOT.TLine(x, hs.GetMinimum(), x, hs.GetMaximum())
        line.SetLineStyle(ROOT.kDashed)
        line.SetLineColor(ROOT.kGray + 1)
        line.Draw("SAME")
        top_sep_lines.append(line)

    leg = ROOT.TLegend(0.50, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.AddEntry(h_data, "Data (Asimov)", "lep")
    for h, lbl in zip(class_hists_sorted, class_labels_sorted):
        leg.AddEntry(h, lbl, "f")
    leg.AddEntry(h_unc, "Uncertainty", "f")
    leg.Draw()

    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextSize(0.035)
    label.DrawLatex(0.12, 0.93, region_id)

    padBottom.cd()

    h_ratio = h_total.Clone(f"h_postfit_ratio_{_safe_name(region_id)}")
    h_ratio.SetDirectory(0)
    h_ratio.Divide(h_total)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(2)
    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle("var / nominal")
    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.09)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.GetYaxis().SetLabelSize(0.08)
    h_ratio.GetXaxis().SetTitle(x_title)
    h_ratio.GetXaxis().SetTitleSize(0.10)
    h_ratio.GetXaxis().SetLabelSize(0.08)

    ratio_boxes = []
    h_ratio_up = h_ratio.Clone(f"h_postfit_ratio_up_{_safe_name(region_id)}")
    h_ratio_down = h_ratio.Clone(f"h_postfit_ratio_down_{_safe_name(region_id)}")
    h_ratio_up.SetDirectory(0)
    h_ratio_down.SetDirectory(0)
    h_ratio_up.SetFillStyle(0)
    h_ratio_down.SetFillStyle(0)
    h_ratio_up.SetLineColor(ROOT.kGray + 2)
    h_ratio_down.SetLineColor(ROOT.kGray + 2)
    h_ratio_up.SetLineWidth(1)
    h_ratio_down.SetLineWidth(1)

    max_dev = 0.0
    for ib in range(1, n_bins_region + 1):
        x1 = plot_edges[ib - 1]
        x2 = plot_edges[ib]
        nom = h_total.GetBinContent(ib)
        err = h_unc.GetBinError(ib)

        if nom > 0.0:
            rel = err / nom
            y_low = 1.0 - rel
            y_high = 1.0 + rel

            box = ROOT.TBox(x1, y_low, x2, y_high)
            box.SetFillColor(ROOT.kGray + 1)
            box.SetFillStyle(3345)
            box.SetLineWidth(0)
            ratio_boxes.append(box)

            h_ratio_up.SetBinContent(ib, y_high)
            h_ratio_down.SetBinContent(ib, y_low)
            max_dev = max(max_dev, rel)
        else:
            h_ratio_up.SetBinContent(ib, 1.0)
            h_ratio_down.SetBinContent(ib, 1.0)

    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    h_ratio.SetMinimum(r_min)
    h_ratio.SetMaximum(r_max)
    h_ratio.Draw("HIST")
    for box in ratio_boxes:
        box.Draw("SAME")
    h_ratio_up.Draw("HIST SAME")
    h_ratio_down.Draw("HIST SAME")
    h_ratio_data.Draw("E1 X0 SAME")
    ratio_sep_lines = []
    for x in separators:
        line = ROOT.TLine(x, r_min, x, r_max)
        line.SetLineStyle(ROOT.kDashed)
        line.SetLineColor(ROOT.kGray + 1)
        line.Draw("SAME")
        ratio_sep_lines.append(line)

    line1 = ROOT.TLine(plot_edges[0], 1.0, plot_edges[-1], 1.0)
    line1.SetLineStyle(ROOT.kDashed)
    line1.SetLineColor(ROOT.kBlack)
    line1.Draw("SAME")

    c.cd()
    c.Update()

    out_png = os.path.join(outdir, f"{_safe_name(region_id)}__postfit.png")
    out_pdf = os.path.join(outdir, f"{_safe_name(region_id)}__postfit.pdf")
    c.SaveAs(out_png)
    c.SaveAs(out_pdf)
    print(f"[info] wrote\n  {out_png}\n  {out_pdf}")

    region_canvases[region_id] = c
    _KEEPALIVE.extend(
        [c, padTop, padBottom, hs, h_total, h_unc, h_unc_up, h_unc_down, h_data, h_ratio, h_ratio_data, h_ratio_up, h_ratio_down, leg, label, line1]
        + class_hists_sorted + ratio_boxes + top_sep_lines + ratio_sep_lines
    )

print(f"[info] done, produced {len(region_canvases)} region plots")
syncer.sync()
