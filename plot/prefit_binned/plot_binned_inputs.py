#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import argparse
from typing import Dict, List, Tuple, Any

import common.syncer as syncer

import numpy as np
import ROOT

from fit.Likelihood import load_likelihood, build_hypothesis_from_likelihood, N2LL  # noqa: E402
import common.yaml_loader as yaml_loader  # noqa: E402
import common.helpers as helpers  # noqa: E402

try:
    from common.user import plot_directory as DEFAULT_PLOT_DIRECTORY  # noqa: E402
except Exception:
    DEFAULT_PLOT_DIRECTORY = os.path.join(os.getcwd(), "plots")

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

_KEEPALIVE = []

SYS_GROUPS: List[Tuple[str, set[str]]] = [
    ("JER", {
        'nu_CMS_res_j_0',
        'nu_CMS_res_j_1',
        'nu_CMS_res_j_2',
        'nu_CMS_res_j_3',
        'nu_CMS_res_j_4',
        'nu_CMS_res_j_5',
    }),
    ("JES", {
        'nu_CMS_scale_j_FlavorPureBottom',
        'nu_CMS_scale_j_FlavorPureCharm',
        'nu_CMS_scale_j_FlavorPureGluon',
        'nu_CMS_scale_j_FlavorPureQuark',
        'nu_CMS_scale_j_Regrouped_Absolute',
        'nu_CMS_scale_j_Regrouped_Absolute_2016',
        'nu_CMS_scale_j_Regrouped_Absolute_2017',
        'nu_CMS_scale_j_Regrouped_Absolute_2018',
        'nu_CMS_scale_j_Regrouped_BBEC1',
        'nu_CMS_scale_j_Regrouped_BBEC1_2016',
        'nu_CMS_scale_j_Regrouped_BBEC1_2017',
        'nu_CMS_scale_j_Regrouped_BBEC1_2018',
        'nu_CMS_scale_j_Regrouped_EC2',
        'nu_CMS_scale_j_Regrouped_EC2_2016',
        'nu_CMS_scale_j_Regrouped_EC2_2017',
        'nu_CMS_scale_j_Regrouped_EC2_2018',
        'nu_CMS_scale_j_Regrouped_HF',
        'nu_CMS_scale_j_Regrouped_HF_2016',
        'nu_CMS_scale_j_Regrouped_HF_2017',
        'nu_CMS_scale_j_Regrouped_HF_2018',
        'nu_CMS_scale_j_Regrouped_RelativeBal',
        'nu_CMS_scale_j_Regrouped_RelativeSample_2016',
        'nu_CMS_scale_j_Regrouped_RelativeSample_2017',
        'nu_CMS_scale_j_Regrouped_RelativeSample_2018',
        'nu_Uncl',
        'nu_CMS_scale_j_Total_EtaBin0',
        'nu_CMS_scale_j_Total_EtaBin1',
        'nu_CMS_scale_j_Total_EtaBin2',
        'nu_CMS_scale_j_Total_EtaBin3',
        'nu_CMS_scale_j_Total_EtaBin4',
        'nu_CMS_scale_j_Total_EtaBin5',
    }),
    ("leptons", {
        'nu_EleSF',
        'nu_MuSF',
    }),
    ('#alpha_{S}', {
        'nu_alphaS',
    }),
    ('b-tagging', {
        'nu_btag_b',
        'nu_btag_l',
    }),
    ("scales", {
        'nu_mu_fac',
        'nu_mu_ren',
    }),
    ("ISR/FSR", {
        'nu_showerFSR',
        'nu_showerISR',
    }),
    ("PU/L1Pre", {
        'nu_pu',
        'nu_l1prefire',
    }),
    ("t#bar{t} norm.", {
        'nu_norm_TTLep',
    }),
]


def _safe_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(text))


def _palette() -> List[int]:
    return [
        ROOT.kBlack,
        ROOT.kRed + 1,
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 7,
        ROOT.kCyan + 1,
        ROOT.kViolet + 1,
        ROOT.kAzure + 2,
        ROOT.kSpring + 5,
        ROOT.kPink + 7,
        ROOT.kTeal + 3,
    ]


def _region_pois(region: Dict[str, Any]) -> List[str]:
    names = set()
    for cls in region.get("classes", []) or []:
        for nm in ((cls.get("POI") or {}).get("parameters", []) or []):
            names.add(nm)
    return sorted(names)


def _pretty_combo(combo: Any) -> str:
    if combo is None:
        return ""
    if isinstance(combo, str):
        return combo
    seq = list(combo)
    if not seq:
        return ""
    return "*".join(str(x) for x in seq)


def _clean_nuis_label(text: str) -> str:
    out = str(text)
    out = out.replace("nu_", "")
    out = out.replace("CMS_", "")
    return out


def _make_unroll_info(region: Dict[str, Any]) -> Dict[str, Any]:
    classes = region.get("classes", []) or []
    if not classes:
        raise RuntimeError(f"Region {region.get('id', '?')} has no classes.")
    first_pred = (classes[0].get("POI") or {}).get("predictor", None)
    if first_pred is None:
        raise RuntimeError(f"Region {region.get('id', '?')} is missing its first ICH predictor.")
    return N2LL._unroll_bins_from_ich(first_pred)


def _flatten_prediction(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        return out.copy()
    if out.ndim == 2:
        return out.reshape(-1).copy()
    raise RuntimeError(f"Expected 1D or 2D prediction, got shape {out.shape}")


def _region_total_ich_template(region: Dict[str, Any], point: Dict[str, float]) -> np.ndarray:
    total = None
    for cls in region.get("classes", []) or []:
        poi = cls.get("POI") or {}
        pred = poi.get("predictor", None)
        if pred is None:
            raise RuntimeError(f"Missing ICH predictor for {region.get('id','?')}/{cls.get('id','?')}")
        par_names = list(poi.get("parameters", []) or [])
        cvec = np.array([float(point.get(name, 0.0)) for name in par_names], dtype=np.float64)
        vals = _flatten_prediction(pred.predict(cvec))
        if total is None:
            total = np.zeros_like(vals, dtype=np.float64)
        if vals.shape != total.shape:
            raise RuntimeError(
                f"Inconsistent ICH shape in region {region.get('id','?')}: "
                f"class {cls.get('id','?')} has shape {vals.shape}, expected {total.shape}"
            )
        total += vals
    if total is None:
        raise RuntimeError(f"Region {region.get('id','?')} has no plottable classes.")
    return total


def _build_bin_hist(name: str, values: np.ndarray, ytitle: str) -> ROOT.TH1D:
    n = int(len(values))
    h = ROOT.TH1D(name, "", n, 0.5, n + 0.5)
    h.SetDirectory(0)
    for i, val in enumerate(values, start=1):
        h.SetBinContent(i, float(val))
        h.SetBinError(i, 0.0)
    h.GetXaxis().SetTitle("unrolled bin")
    h.GetYaxis().SetTitle(ytitle)
    return h


def _ratio_to_central(num: np.ndarray, den: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.ones_like(num, dtype=np.float64)
    mask = np.abs(den) > eps
    out[mask] = num[mask] / den[mask]
    bad = (~mask) & (np.abs(num) > eps)
    if np.any(bad):
        print(f"[warning] Found {int(np.count_nonzero(bad))} bins with central=0 and varied!=0; ratio set to 1 there.")
    return out


def _vertical_separators(unroll: Dict[str, Any], ymin: float, ymax: float, color: int = ROOT.kGray + 1) -> List[ROOT.TLine]:
    lines = []
    shape = tuple(unroll.get("shape", ()))
    if len(shape) == 2:
        nb1, nb2 = shape
        for i in range(1, nb1):
            x = i * nb2 + 0.5
            line = ROOT.TLine(x, ymin, x, ymax)
            line.SetLineColor(color)
            line.SetLineStyle(ROOT.kDashed)
            line.SetLineWidth(1)
            lines.append(line)
    return lines


def _draw_unroll_label(pad_or_canvas, region_id: str, unroll: Dict[str, Any], extra: str = "") -> ROOT.TLatex:
    axes = list(unroll.get("axes", []) or [])
    shape = tuple(unroll.get("shape", ()))
    if len(shape) == 1:
        desc = f"{region_id}: {axes[0] if axes else 'axis'}"
    elif len(shape) == 2:
        ax0 = axes[0] if len(axes) > 0 else "x"
        ax1 = axes[1] if len(axes) > 1 else "y"
        desc = f"{region_id}: unrolled {ax0} x {ax1} ({shape[0]} blocks x {shape[1]} bins)"
    else:
        desc = f"{region_id}: unrolled"
    if extra:
        desc = f"{desc}   |   {extra}"
    pad_or_canvas.cd()
    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextSize(0.032)
    text.DrawLatex(0.11, 0.93, desc)
    return text


def _save_canvas(canvas: ROOT.TCanvas, outdir: str, stem: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    try:
        helpers.copyIndexPHP(outdir)
    except Exception:
        pass
    png = os.path.join(outdir, f"{stem}.png")
    pdf = os.path.join(outdir, f"{stem}.pdf")
    canvas.SaveAs(png)
    canvas.SaveAs(pdf)
    print(f"[info] Wrote\n  {png}\n  {pdf}")


def plot_region_ich(region: Dict[str, Any], outdir: str, logy: bool = False) -> ROOT.TCanvas:
    region_id = region["id"]
    unroll = _make_unroll_info(region)
    region_pois = _region_pois(region)

    central = _region_total_ich_template(region, point={})
    variations: List[Tuple[str, np.ndarray]] = [("central", central)]
    for poi in region_pois:
        variations.append((f"{poi}=1", _region_total_ich_template(region, point={poi: 1.0})))

    cname = f"c_ich_{_safe_name(region_id)}"
    canvas = ROOT.TCanvas(cname, cname, 900, 850)
    pad_top = ROOT.TPad(cname + "_top", cname + "_top", 0.0, 0.30, 1.0, 1.0)
    pad_bot = ROOT.TPad(cname + "_bot", cname + "_bot", 0.0, 0.00, 1.0, 0.30)

    pad_top.SetBottomMargin(0.02)
    pad_top.SetTopMargin(0.08)
    pad_top.SetLeftMargin(0.12)
    pad_top.SetRightMargin(0.04)
    pad_top.SetTicks(1, 1)

    pad_bot.SetTopMargin(0.03)
    pad_bot.SetBottomMargin(0.30)
    pad_bot.SetLeftMargin(0.12)
    pad_bot.SetRightMargin(0.04)
    pad_bot.SetTicks(1, 1)

    pad_top.Draw()
    pad_bot.Draw()

    colors = _palette()
    top_hists: List[ROOT.TH1D] = []
    ratio_hists: List[ROOT.TH1D] = []

    ymax = 0.0
    positive_min = None
    all_ratio_vals = []

    for idx, (label, vals) in enumerate(variations):
        h = _build_bin_hist(f"h_top_{_safe_name(region_id)}_{idx}", vals, "ICH prediction")
        h.SetLineColor(colors[idx % len(colors)])
        h.SetLineWidth(3 if idx == 0 else 2)
        h.SetLineStyle(ROOT.kSolid)
        h.SetTitle("")
        ymax = max(ymax, float(np.max(vals)) if len(vals) else 0.0)
        pos = vals[vals > 0.0]
        if len(pos):
            candidate = float(np.min(pos))
            positive_min = candidate if positive_min is None else min(positive_min, candidate)
        top_hists.append(h)

        r = _ratio_to_central(vals, central)
        all_ratio_vals.append(r)
        hr = _build_bin_hist(f"h_ratio_{_safe_name(region_id)}_{idx}", r, "var / central")
        hr.SetLineColor(colors[idx % len(colors)])
        hr.SetLineWidth(3 if idx == 0 else 2)
        hr.SetLineStyle(ROOT.kSolid)
        hr.SetTitle("")
        ratio_hists.append(hr)

    pad_top.cd()
    if logy:
        pad_top.SetLogy(True)

    first = top_hists[0]
    if logy:
        ymin = 0.5 * positive_min if (positive_min is not None and positive_min > 0.0) else 1e-4
        ymax_draw = 20.0 * ymax if ymax > 0.0 else 1.0
    else:
        ymin = 0.0
        ymax_draw = 1.25 * ymax if ymax > 0.0 else 1.0

    first.SetMinimum(ymin)
    first.SetMaximum(ymax_draw)
    first.GetYaxis().SetTitleSize(0.055)
    first.GetYaxis().SetTitleOffset(1.05)
    first.GetYaxis().SetLabelSize(0.045)
    first.GetXaxis().SetLabelSize(0.0)
    first.GetXaxis().SetTitleSize(0.0)
    first.Draw("HIST")
    for h in top_hists[1:]:
        h.Draw("HIST SAME")

    sep_top = _vertical_separators(unroll, ymin if not logy else max(ymin, 1e-6), ymax_draw)
    for line in sep_top:
        line.Draw("SAME")

    leg = ROOT.TLegend(0.58, 0.60, 0.94, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(1 if len(variations) <= 7 else 2)
    for h, (label, _) in zip(top_hists, variations):
        leg.AddEntry(h, label, "l")
    leg.Draw()

    label_top = _draw_unroll_label(pad_top, region_id, unroll, extra="total region template")

    pad_bot.cd()
    ratio0 = ratio_hists[0]
    ratio_vals = np.concatenate([np.asarray(x, dtype=np.float64) for x in all_ratio_vals])
    finite = ratio_vals[np.isfinite(ratio_vals)]

    if len(finite):
        rmin = float(np.min(finite))
        rmax = float(np.max(finite))
    else:
        rmin, rmax = 0.9, 1.1

    if not np.isfinite(rmin) or not np.isfinite(rmax):
        rmin, rmax = 0.9, 1.1

    if abs(rmax - rmin) < 1e-6:
        rmin, rmax = 0.9, 1.1
    else:
        center = 1.0
        half = max(abs(rmax - center), abs(center - rmin))
        half *= 1.15
        rmin = center - half
        rmax = center + half

    ratio0.SetMinimum(rmin)
    ratio0.SetMaximum(rmax)
    ratio0.GetYaxis().SetTitleSize(0.10)
    ratio0.GetYaxis().SetTitleOffset(0.55)
    ratio0.GetYaxis().SetLabelSize(0.085)
    ratio0.GetYaxis().SetNdivisions(505)
    ratio0.GetXaxis().SetTitle("unrolled bin")
    ratio0.GetXaxis().SetTitleSize(0.11)
    ratio0.GetXaxis().SetTitleOffset(1.0)
    ratio0.GetXaxis().SetLabelSize(0.085)
    ratio0.Draw("HIST")
    for h in ratio_hists[1:]:
        h.Draw("HIST SAME")

    sep_bot = _vertical_separators(unroll, rmin, rmax)
    for line in sep_bot:
        line.Draw("SAME")

    unit = ROOT.TLine(0.5, 1.0, len(central) + 0.5, 1.0)
    unit.SetLineColor(ROOT.kBlack)
    unit.SetLineStyle(ROOT.kDashed)
    unit.SetLineWidth(1)
    unit.Draw("SAME")

    canvas.cd()
    canvas.Update()

    stem = f"{_safe_name(region_id)}__ich_templates"
    _save_canvas(canvas, outdir, stem)

    _KEEPALIVE.extend([canvas, pad_top, pad_bot, leg, label_top, unit] + top_hists + ratio_hists + sep_top + sep_bot)
    return canvas


def plot_region_icph(region: Dict[str, Any], outdir: str) -> List[ROOT.TCanvas]:
    region_id = region["id"]
    unroll = _make_unroll_info(region)
    colors = _palette()
    canvases = []

    group_lookup = {name: nuis for name, nuis in SYS_GROUPS}

    for cls in region.get("classes", []) or []:
        class_id = cls.get("id", "class")
        grouped_entries = {name: [] for name, _ in SYS_GROUPS}

        for syst in cls.get("systematics", []) or []:
            if syst.get("type") != "icph":
                continue

            pred = syst.get("predictor", None)
            if pred is None:
                raise RuntimeError(f"Missing ICPH predictor for {region_id}/{class_id}/{syst.get('id', '?')}")

            raw_deltas = getattr(pred, "deltas", None)
            if raw_deltas is None:
                raise RuntimeError(f"ICPH predictor has no deltas for {region_id}/{class_id}/{syst.get('id', '?')}")

            deltas = np.asarray(raw_deltas, dtype=np.float64)
            if deltas.ndim not in (2, 3):
                raise RuntimeError(
                    f"Expected ICPH deltas with ndim 2 or 3 for {region_id}/{class_id}/{syst.get('id', '?')}, got {deltas.shape}"
                )

            if deltas.ndim == 2:
                curves = [deltas[i, :].reshape(-1) for i in range(deltas.shape[0])]
            else:
                curves = [deltas[i, :, :].reshape(-1) for i in range(deltas.shape[0])]

            combos = list(syst.get("combinations", []) or getattr(pred, "combinations", []) or [])
            params = list(syst.get("parameters", []) or getattr(pred, "parameters", []) or [])

            if combos:
                labels = [_clean_nuis_label(_pretty_combo(cmb)) for cmb in combos]
            elif params:
                labels = [_clean_nuis_label(p) for p in params]
            else:
                labels = [f"term{i}" for i in range(len(curves))]

            matched_groups = []
            pset = set(params)
            for group_name, group_nuis in SYS_GROUPS:
                if pset.intersection(group_nuis):
                    matched_groups.append(group_name)

            if not matched_groups:
                continue

            base_entries = list(zip(labels, curves))
            for group_name in matched_groups:
                grouped_entries[group_name].extend(base_entries)

        for group_name, _ in SYS_GROUPS:
            entries = grouped_entries[group_name]
            if not entries:
                continue

            cname = f"c_icph_{_safe_name(region_id)}_{_safe_name(class_id)}_{_safe_name(group_name)}"
            canvas = ROOT.TCanvas(cname, cname, 1100, 850)

            pad_leg = ROOT.TPad(cname + "_leg", cname + "_leg", 0.0, 0.72, 1.0, 1.0)
            pad_plot = ROOT.TPad(cname + "_plot", cname + "_plot", 0.0, 0.00, 1.0, 0.72)

            pad_leg.SetBottomMargin(0.02)
            pad_leg.SetTopMargin(0.08)
            pad_leg.SetLeftMargin(0.08)
            pad_leg.SetRightMargin(0.04)

            pad_plot.SetTopMargin(0.02)
            pad_plot.SetBottomMargin(0.12)
            pad_plot.SetLeftMargin(0.12)
            pad_plot.SetRightMargin(0.04)
            pad_plot.SetTicks(1, 1)

            pad_leg.Draw()
            pad_plot.Draw()

            hists = []
            ymin = +np.inf
            ymax = -np.inf

            for i, (label, vals) in enumerate(entries):
                h = _build_bin_hist(
                    f"h_icph_{_safe_name(region_id)}_{_safe_name(class_id)}_{_safe_name(group_name)}_{i}",
                    vals,
                    "log variation",
                )
                h.SetLineColor(colors[i % len(colors)])
                h.SetLineWidth(2)
                h.SetLineStyle(ROOT.kSolid)
                h.SetTitle("")
                ymin = min(ymin, float(np.min(vals)) if len(vals) else ymin)
                ymax = max(ymax, float(np.max(vals)) if len(vals) else ymax)
                hists.append((h, label))

            if not np.isfinite(ymin) or not np.isfinite(ymax):
                ymin, ymax = -1.0, 1.0
            elif abs(ymax - ymin) < 1e-12:
                ymin -= 1.0
                ymax += 1.0
            else:
                span = ymax - ymin
                ymin -= 0.15 * span
                ymax += 0.15 * span

            pad_plot.cd()
            h0 = hists[0][0]
            h0.SetMinimum(ymin)
            h0.SetMaximum(ymax)
            h0.GetYaxis().SetTitleSize(0.050)
            h0.GetYaxis().SetTitleOffset(1.08)
            h0.GetYaxis().SetLabelSize(0.040)
            h0.GetXaxis().SetTitle("unrolled bin")
            h0.GetXaxis().SetTitleSize(0.045)
            h0.GetXaxis().SetLabelSize(0.040)
            h0.Draw("HIST")
            for h, _ in hists[1:]:
                h.Draw("HIST SAME")

            zero = ROOT.TLine(0.5, 0.0, h0.GetNbinsX() + 0.5, 0.0)
            zero.SetLineColor(ROOT.kBlack)
            zero.SetLineStyle(ROOT.kDashed)
            zero.SetLineWidth(1)
            zero.Draw("SAME")

            sep = _vertical_separators(unroll, ymin, ymax)
            for line in sep:
                line.Draw("SAME")

            pad_leg.cd()
            label_top = _draw_unroll_label(
                pad_leg,
                region_id,
                unroll,
                extra=f"ICPH group: {group_name}   |   process: {class_id}",
            )

            leg = ROOT.TLegend(0.03, 0.05, 0.97, 0.78)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetNColumns(2)
            for h, label in hists:
                leg.AddEntry(h, label, "l")
            leg.Draw()

            canvas.cd()
            canvas.Update()

            stem = f"{_safe_name(region_id)}__icph_group__{_safe_name(class_id)}__{_safe_name(group_name)}"
            _save_canvas(canvas, outdir, stem)

            _KEEPALIVE.extend([canvas, pad_leg, pad_plot, leg, label_top, zero] + [h for h, _ in hists] + sep)
            canvases.append(canvas)

    return canvases


def make_output_dir(cfg: Dict[str, Any], config_path: str, outdir: str | None) -> str:
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        return outdir
    base = os.path.splitext(os.path.basename(config_path))[0]
    version = str(cfg.get("version", "v0"))
    out = os.path.join(DEFAULT_PLOT_DIRECTORY, "binned_inputs_2", base, version)
    os.makedirs(out, exist_ok=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot binned ICH/ICPH inputs from a likelihood config")
    parser.add_argument("config", help="Path to global YAML config")
    parser.add_argument("--outdir", default=None, help="Output directory for plots")
    parser.add_argument("--logy", action="store_true", help="Use log y-scale for the ICH template top pad")
    args = parser.parse_args()

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False)

    like_info = load_likelihood(cfg)
    hypothesis_template = build_hypothesis_from_likelihood(like_info, name="plot_binned_inputs")

    binned_regions = list(like_info.get("binned", []) or [])
    if not binned_regions:
        raise RuntimeError("No binned regions found in config.")

    outdir = make_output_dir(cfg, args.config, args.outdir)
    print(f"[info] Output directory: {outdir}")

    region_ich_canvases = {}
    region_icph_canvases = {}
    all_canvases = []

    for region in binned_regions:
        region_id = region["id"]
        print(f"[info] Plotting region {region_id}")

        c_ich = plot_region_ich(region, outdir, logy=args.logy)
        c_icph = plot_region_icph(region, outdir)

        region_ich_canvases[region_id] = c_ich
        region_icph_canvases[region_id] = c_icph

        all_canvases.append(c_ich)
        all_canvases.extend(c_icph)

    print(f"[info] Done. Produced {len(all_canvases)} canvas objects.")

    syncer.sync()
