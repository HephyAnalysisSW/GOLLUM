import os
import re
import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)
import numpy as np
from array import array

import subprocess
import argparse
import copy
import yaml
import json

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user   as user
import common.syncer as syncer
import common.helpers as helpers

import common.yaml_loader as yaml_loader
from fit.Likelihood import load_likelihood
from data.plot_options import plot_options

# ---------------- args ----------------
p = argparse.ArgumentParser(description="Plotting")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--yes", "-y", action="store_true", help="Automatically run missing surrogate trainings without asking")
p.add_argument("--feature", required=True, help="Feature to plot (overrides config default_binning variable)")
p.add_argument("--binning", nargs="+", type=float,
               help="List of bin edges for the feature (thresholds). If not given, use plot_options binning.")
p.add_argument("--rotate", action="store_true", help="Rotate?")
p.add_argument("--no_syst", action="store_true", help="Disable all nuisances (freeze to 0).")
p.add_argument("--postfit", default = None, help="Post fit file")
p.add_argument("--freezePOI", action="store_true", help="Do not modify the POI when sampling.")

args = p.parse_args()

# ---------------- load & patch YAML CFG ----------------

print(f"[info] Loading config from: {args.config}")
orig_cfg = yaml_loader.load_yaml(args.config)

# Decide which feature to use
if args.feature:
    feature_name = args.feature
else:
    # fall back to the feature in the original config
    try:
        feature_name = orig_cfg['defaults']['default_binning'][0][0]
    except (KeyError, IndexError, TypeError):
        feature_name = None

# Decide on bin edges
edges = None
if args.binning:
    # explicit list of thresholds
    edges = [float(x) for x in args.binning]
    print(f"[info] Using explicit bin edges from --binning ({len(edges)-1} bins).")
elif feature_name is not None and feature_name in plot_options and 'binning' in plot_options[feature_name]:
    # build thresholds from plot_options: [nBins, low, high]
    nBins, low, high = plot_options[feature_name]['binning']
    edges = [low + i*(high - low) / nBins for i in range(nBins + 1)]
    print(f"[info] Using binning from plot_options for '{feature_name}': "
          f"nBins={nBins}, low={low}, high={high} -> {len(edges)-1} bins.")
else:
    # fall back to what is already in the config, if available
    try:
        _, edges = orig_cfg['defaults']['default_binning'][0]
        print(f"[info] Falling back to binning from config defaults for feature '{feature_name}'.")
    except (KeyError, IndexError, TypeError):
        edges = None
        print("[warning] No binning information found; leaving defaults unchanged.")

# Make a deep copy to patch
cfg = copy.deepcopy(orig_cfg)

# Patch defaults if we have feature and edges
if feature_name is not None and edges is not None:
    print(f"[info] Setting default feature to '{feature_name}' with {len(edges)-1} bins.")
    cfg.setdefault('defaults', {})
    cfg['defaults']['default_binning'] = [[feature_name, edges]]
else:
    print("[info] Keeping original defaults.default_binning.")

# Update version and construct model/output directory
base_version = cfg.get('version', 'plot')
if args.feature:
    suffix = f"_{args.feature}"
    if not base_version.endswith(suffix):
        version = base_version + suffix
    else:
        version = base_version
else:
    version = base_version

cfg['version'] = version
print(f"[info] Using version: {version}")

output_directory = os.path.join(user.model_directory, version)
os.makedirs(output_directory, exist_ok=True)
print(f"[info] Model/output directory: {output_directory}")

# Write patched YAML into output_directory
yaml_basename   = os.path.basename(args.config)
patched_yaml_path = os.path.join(output_directory, yaml_basename)

with open(patched_yaml_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[info] Patched config written to: {patched_yaml_path}")

# From here on, use cfg + patched_yaml_path
yaml_loader.print_summary(cfg, patched_yaml_path, yaml_loader._INCLUDE_TRACE)
missing_cmds = yaml_loader.load_surrogates(
    cfg,
    patched_yaml_path,
    overwrite=False,
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
        patched_yaml_path,
        overwrite=False,
    )
    if missing_cmds:
        print("[error] Still missing surrogates after running trainings:")
        for cmd in missing_cmds:
            print("  ", cmd)
        sys.exit(1)

print("[info] All required surrogates available. Loading likelihood...")
like_info = load_likelihood(cfg)

plot_directory = os.path.join(user.plot_directory, 'binned_templates', args.feature)
print(f"[info] Plots will be written under: {plot_directory}")

if args.postfit:
    plot_label = "postfit"
else:
    plot_label = "prefit"

# output directory for pre/postfit plots
fit_plot_directory = os.path.join(user.plot_directory, plot_label+'_stacks')
os.makedirs(fit_plot_directory, exist_ok=True)
print(f"[info] {plot_label} plots will be written under: {fit_plot_directory}")

# get default binning & x-axis label / logY info
var_name, var_edges = cfg['defaults']['default_binning'][0]
bin_edges = array('d', var_edges)
n_bins_default = len(var_edges) - 1

logY = plot_options.get(var_name, {}).get('logY', False)

# ---------------- prefit template plots ----------------
if args.postfit:
    print("Git postfit argument. Don't make templates.")
else:
    x_title = plot_options.get(var_name, {}).get('tex', var_name)
    # simple color palette
    colors = [
        ROOT.kRed + 1,
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 1,
        ROOT.kCyan + 1,
    ]
    
    # legend columns (configurable)
    legend_columns = 3
    
    # group definitions (regex-style, '*' -> '.*')
    syst_groups = {
        'MODELING': [
            'Scales',
            'ShowerISR', 
            'ShowerFSR',
            'AlphaS'
        ],
        'EXPERIMENTAL': [
            'L1Prefire',
            'PU',
            'MuSF',
            'EleSF',
            'BTag_b',
            'BTag_l',
        ],
        'JER': [
            'CMS_res_j_0',
            'CMS_res_j_1',
            'CMS_res_j_2',
            'CMS_res_j_3',
            'CMS_res_j_4',
            'CMS_res_j_5',
        ],
        'JES1': [
            'CMS_scale_j_FlavorPureBottom',
            'CMS_scale_j_FlavorPureCharm',
            'CMS_scale_j_FlavorPureGluon',
            'CMS_scale_j_FlavorPureQuark',
        ],
        'JES2': [
            'CMS_scale_j_Regrouped_Absolute*',
            'CMS_scale_j_Regrouped_BBEC1*',
            'CMS_scale_j_Regrouped_EC2*',
        ],
        'JES3': [
            'CMS_scale_j_Regrouped_HF',
            'Uncl',
        ],
    }
  
    for region in like_info['binned']:
        region_id = region['id']
        print("Region:", region_id)

        for cls in region['classes']:
            class_id = cls['id']
            print("  Class:", class_id)

            poi = cls['POI']
            poi_predictor = poi['predictor']
            poi_params = poi.get('parameters', [])
            #poi_point = [0.0] * len(poi_params)
            #Here we should pass the MLE if we want postfit values
            if args.postfit:
                param_map = {p["name"]: p["value"] for p in fit_results["parameters"]}
                poi_point = [param_map[name] for name in poi_params]
            else:
                poi_point  = [0.0] * len(poi_params)


            # central prediction
            central = np.asarray(poi_predictor.predict(poi_point), dtype='float64')
            n_bins = len(central)

            if n_bins != n_bins_default:
                print(f"WARNING: n_bins ({n_bins}) != default binning ({n_bins_default}) for {region_id}, {class_id}")
            # use min to avoid crashes if mismatch
            n_bins_use = min(n_bins, n_bins_default)

            for group_name, patterns in syst_groups.items():

                # collect systematics in this group for this class
                systs_in_group = []
                for syst in cls['systematics']:
                    # skip flat lnN normalizations
                    if syst.get('type') != 'icph':
                        continue

                    sys_id = syst['id']
                    matched = False
                    for pat in patterns:
                        regex = '^' + pat.replace('*', '.*') + '$'
                        if re.match(regex, sys_id):
                            matched = True
                            break
                    if matched:
                        systs_in_group.append(syst)

                if not systs_in_group:
                    continue

                print("    Group:", group_name, "(", ", ".join(s['id'] for s in systs_in_group), ")")

                out_dir = os.path.join(plot_directory, region_id, class_id)
                helpers.copyIndexPHP(out_dir)
                os.makedirs(out_dir, exist_ok=True)

                canvas_name = f"{region_id}_{class_id}_{group_name}"
                # stretch in y
                c = ROOT.TCanvas(canvas_name, canvas_name, 800, 900)

                # three pads: legend (top), yields (middle), ratios (bottom)
                padLegend = ROOT.TPad(canvas_name + "_legend", canvas_name + "_legend", 0.0, 0.80, 1.0, 1.0)
                padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 0.80)
                padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

                padLegend.SetBottomMargin(0.05)
                padLegend.SetTopMargin(0.10)
                padLegend.SetLeftMargin(0.10)
                padLegend.SetRightMargin(0.10)
                padLegend.SetFillStyle(0)

                padTop.SetBottomMargin(0.)
                padTop.SetTopMargin(0.08)
                padTop.SetLeftMargin(0.10)
                padTop.SetRightMargin(0.05)

                padBottom.SetTopMargin(0.0)
                padBottom.SetBottomMargin(0.30)
                padBottom.SetLeftMargin(0.10)
                padBottom.SetRightMargin(0.05)

                padLegend.Draw()
                padTop.Draw()
                padBottom.Draw()

                # legend (created here, drawn later in padLegend)
                legend = ROOT.TLegend(0.02, 0.10, 0.98, 0.90)
                legend.SetBorderSize(0)
                legend.SetFillStyle(0)
                legend.SetNColumns(legend_columns)

                # ------------- TOP PAD: absolute yields -------------
                padTop.cd()
                padTop.SetTicks(1, 1)
                if logY:
                    padTop.SetLogy(True)

                # central histogram with variable binning
                h_central_name = f"h_central_{region_id}_{class_id}_{group_name}"
                h_central = ROOT.TH1F(h_central_name, "", n_bins_default, bin_edges)
                for i in range(n_bins_use):
                    h_central.SetBinContent(i + 1, central[i])

                h_central.SetLineColor(ROOT.kBlack)
                h_central.SetLineWidth(2)
                # no title on the top pad
                h_central.SetTitle("")
                h_central.GetXaxis().SetTitle(x_title)

                h_central.GetYaxis().SetTitle("Events")
                h_central.GetYaxis().SetTitleSize(0.06)
                h_central.GetYaxis().SetLabelSize(0.045)
                # x label on bottom pad only
                h_central.GetXaxis().SetLabelSize(0)
                h_central.GetXaxis().SetTitleSize(0)


                legend.AddEntry(h_central, "nominal", "l")

                # keep references alive
                h_variations = [h_central]

                color_index = 0

                for syst in systs_in_group:
                    sys_id = syst['id']
                    predictor = syst['predictor']
                    syst_params = syst['parameters']
                    n_syst_params = len(syst_params)

                    for ip, p_name in enumerate(syst_params):
                        color = colors[color_index % len(colors)]
                        color_index += 1

                        p_tex_name = p_name.lstrip('nu_').lstrip('CMS_')

                        # +1 sigma
                        vec_up = [0.0] * n_syst_params
                        vec_up[ip] = 1.0
                        rel_up = np.asarray(predictor.predict(vec_up), dtype='float64')
                        vals_up = central * rel_up

                        h_up_name = f"h_{group_name}_{sys_id}_{p_name}_Up"
                        h_up = ROOT.TH1F(h_up_name, "", n_bins_default, bin_edges)
                        for i in range(n_bins_use):
                            h_up.SetBinContent(i + 1, vals_up[i])
                        h_up.SetLineColor(color)
                        h_up.SetLineStyle(ROOT.kSolid)
                        h_up.SetLineWidth(1)
                        h_up.Draw("HIST SAME")
                        legend.AddEntry(h_up, f"{p_tex_name} +1#sigma", "l")
                        h_variations.append(h_up)

                        # -1 sigma
                        vec_down = [0.0] * n_syst_params
                        vec_down[ip] = -1.0
                        rel_down = np.asarray(predictor.predict(vec_down), dtype='float64')
                        vals_down = central * rel_down

                        h_down_name = f"h_{group_name}_{sys_id}_{p_name}_Down"
                        h_down = ROOT.TH1F(h_down_name, "", n_bins_default, bin_edges)
                        for i in range(n_bins_use):
                            h_down.SetBinContent(i + 1, vals_down[i])
                        h_down.SetLineColor(color)
                        h_down.SetLineStyle(ROOT.kDashed)
                        h_down.SetLineWidth(1)
                        h_down.Draw("HIST SAME")
                        legend.AddEntry(h_down, f"{p_tex_name} -1#sigma", "l")
                        h_variations.append(h_down)

                # y range (absolute yields)
                max_y = max(h.GetMaximum() for h in h_variations)
                if logY:
                    h_central.SetMinimum(0.8)
                    h_central.SetMaximum(1.2 * max_y if max_y > 0 else 1.0)
                else:
                    h_central.SetMinimum(0.0)
                    h_central.SetMaximum(1.2 * max_y if max_y > 0 else 1.0)

                h_central.Draw("HIST")
                for h in h_variations[1:]:
                    h.Draw("HIST SAME")

                # ------------- BOTTOM PAD: ratios -------------
                padBottom.cd()
                padBottom.SetTicks(1, 1)

                # ratio central
                ratio_central_name = h_central_name + "_ratio"
                h_ratio_central = h_central.Clone(ratio_central_name)
                h_ratio_central.SetDirectory(0)
                h_ratio_central.Divide(h_central)
                h_ratio_central.SetLineColor(ROOT.kBlack)
                h_ratio_central.SetLineWidth(2)
                h_ratio_central.SetTitle("")

                h_ratio_central.GetYaxis().SetTitle("var / nominal")
                h_ratio_central.GetYaxis().SetNdivisions(505)
                h_ratio_central.GetYaxis().SetTitleSize(0.09)
                h_ratio_central.GetYaxis().SetTitleOffset(0.5)
                h_ratio_central.GetYaxis().SetLabelSize(0.08)

                h_ratio_central.GetXaxis().SetTitle(x_title)
                h_ratio_central.GetXaxis().SetTitleSize(0.1)
                h_ratio_central.GetXaxis().SetLabelSize(0.08)

                # build ratio histos for variations
                h_ratio_vars = [h_ratio_central]
                for h in h_variations[1:]:
                    r_name = h.GetName() + "_ratio"
                    h_r = h.Clone(r_name)
                    h_r.SetDirectory(0)
                    h_r.Divide(h_central)
                    h_ratio_vars.append(h_r)

                # ratio y-range based on max relative deviation from 1
                max_dev = 0.0
                for h in h_ratio_vars:
                    for i in range(1, n_bins_use + 1):
                        val = h.GetBinContent(i)
                        if val != 0:
                            dev = abs(val - 1.0)
                            if dev > max_dev:
                                max_dev = dev

                if max_dev <= 0.0:
                    r_min, r_max = 0.9, 1.1
                else:
                    # 30% larger than max deviation, symmetric around 1
                    half_range = 1.3 * max_dev
                    r_min = 1.0 - half_range
                    r_max = 1.0 + half_range

                h_ratio_central.SetMinimum(r_min)
                h_ratio_central.SetMaximum(r_max)

                h_ratio_central.Draw("HIST")
                for h_r in h_ratio_vars[1:]:
                    h_r.Draw("HIST SAME")

                # line at 1
                line = ROOT.TLine(var_edges[0], 1.0, var_edges[-1], 1.0)
                line.SetLineStyle(ROOT.kDashed)
                line.SetLineColor(ROOT.kBlack)
                line.Draw("SAME")

                # ------------- LEGEND PAD -------------
                padLegend.cd()
                # no frame, no axes, just the legend
                legend.Draw()

                c.cd()
                c.Update()

                out_png = os.path.join(out_dir, canvas_name + ".png")
                out_pdf = os.path.join(out_dir, canvas_name + ".pdf")
                c.SaveAs(out_png)
                c.SaveAs(out_pdf)

# ---------------- pre/post fit stack plots ----------------
from fit.Likelihood import build_hypothesis_from_likelihood
from data.colors import get_color

# knobs
n_toys          = 1000  # number of random nuisance samples
fit_legend_columns  = 2
fit_rng_seed        = 42

# load provided covariance
if args.postfit:
    with open(args.postfit, "r") as f:
        fit_results = json.load(f)

    param_order = fit_results["free_parameter_order"]
    params_best = [p["value"] for p in fit_results["parameters"]]
    params_best = np.array(params_best)
    cov = np.array(fit_results["covariance"]["matrix"])

print(f"[info] Building fit hypothesis and sampling with {n_toys} toys...")

# binning & axis info from (possibly patched) cfg
var_name_fit, var_edges_fit = cfg['defaults']['default_binning'][0]
bin_edges_fit = array('d', var_edges_fit)
n_bins_fit    = len(var_edges_fit) - 1

x_title_fit = plot_options.get(var_name_fit, {}).get('tex', var_name_fit)
logY_fit    = plot_options.get(var_name_fit, {}).get('logY', False)

# hypothesis (to get parameter structure)
hyp = build_hypothesis_from_likelihood(like_info)

# --- consistency check between likelihood parameters and fit results ---

like_par_names = [p.name for p in hyp.parameters]
fit_par_names  = [p["name"] for p in fit_results["parameters"]]

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

# optional: also check ordering, since you rely on aligned vectors
if like_par_names != fit_par_names:
    print("[warning] Parameter names match as a set, but order differs between likelihood and fit results.")
    print("  Likelihood order:")
    print("   ", ", ".join(like_par_names))
    print("  Fit result order:")
    print("   ", ", ".join(fit_par_names))
else:
    print(f"[info] Parameter list consistent between likelihood and fit ({len(like_par_names)} parameters).")

# choose which parameters to sample
if getattr(args, "freezePOI", False):
    # old behaviour: sample nuisances only, POIs fixed
    active_params = [par for par in hyp.nuisances if not par.isFrozen]
    print(f"[info] Sampling {len(active_params)} active nuisances only (POIs frozen) for {plot_label}.")
else:
    # new default: sample POIs + nuisances (all non-frozen parameters)
    active_params = [par for par in hyp.parameters if not par.isFrozen]
    print(f"[info] Sampling {len(active_params)} active POIs + nuisances for {plot_label}.")

active_param_names = [par.name for par in active_params]
name_to_idx        = {name: i for i, name in enumerate(active_param_names)}
n_active_params    = len(active_params)

# global toy samples for all active parameters
if n_active_params > 0:
    np.random.seed(fit_rng_seed)
    if args.postfit:
        # params_best and cov must be ordered consistently with active_param_names
        theta_samples = np.random.multivariate_normal(
            mean=params_best,
            cov=cov,
            size=n_toys
        )
    else:
        # prefit: standard normal for all active params
        theta_samples = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(n_toys, n_active_params)
        )
else:
    theta_samples = None

print(f"[info] Generated toys for {plot_label} sampling.")

for region in like_info['binned']:
    region_id = region['id']
    print(f"[info] {plot_label} stack for region: {region_id}")

    # ---- prepare class info: central yields, POI info and systematics ----
    class_infos = []
    for cls in region['classes']:
        class_id    = cls['id']
        sample_name = cls.get('sample', class_id)

        poi        = cls['POI']
        poi_pred   = poi['predictor']
        poi_params = poi.get('parameters', [])

        # central POI point: MLE for postfit, 0 for prefit (as before)
        if args.postfit:
            param_map = {p["name"]: p["value"] for p in fit_results["parameters"]}
            poi_point = [param_map[name] for name in poi_params]
        else:
            poi_point = [0.0] * len(poi_params)

        central = np.asarray(poi_pred.predict(poi_point), dtype='float64')
        n_bins_region = len(central)
        n_bins_use    = min(n_bins_region, n_bins_fit)
        central       = central[:n_bins_use]

        # indices of POI parameters in the active parameter vector (may be None if frozen)
        poi_cols = [name_to_idx.get(pname, None) for pname in poi_params]

        # pick up all icph systematics for this class and map their parameters
        syst_list = []
        for syst in cls['systematics']:
            if syst.get('type') != 'icph':
                continue
            syst_params = syst['parameters']
            cols = [name_to_idx.get(pname, None) for pname in syst_params]
            syst_list.append({
                'id'        : syst['id'],
                'predictor' : syst['predictor'],
                'params'    : syst_params,
                'cols'      : cols,
            })

        class_infos.append({
            'id'        : class_id,
            'sample'    : sample_name,
            'central'   : central,
            'poi_pred'  : poi_pred,
            'poi_params': poi_params,
            'poi_cols'  : poi_cols,
            'syst_list' : syst_list,
        })

    if not class_infos:
        print(f"[warning] Region {region_id} has no classes, skipping.")
        continue

    # enforce same n_bins_use across all classes (take minimum to be safe)
    n_bins_use = min(n_bins_fit, min(len(ci['central']) for ci in class_infos))

    # ---- central total and per-class histograms ----
    total_central = np.zeros(n_bins_use, dtype='float64')
    class_hists   = []
    class_labels  = []
    class_colors  = []

    for ci in class_infos:
        sample_name = ci['sample']
        color = get_color(sample_name) if callable(get_color) else ROOT.kGray + 1

        h_name = f"h_{plot_label}_{region_id}_{ci['id']}"
        h_cls  = ROOT.TH1F(h_name, "", n_bins_fit, bin_edges_fit)
        for ib in range(n_bins_use):
            h_cls.SetBinContent(ib + 1, ci['central'][ib])

        h_cls.SetLineColor(ROOT.kBlack)
        h_cls.SetFillColor(color)
        h_cls.SetLineWidth(1)

        class_hists.append(h_cls)
        class_labels.append(sample_name)
        class_colors.append(color)

        total_central += ci['central'][:n_bins_use]

    # ---- sampling total prediction over parameters ----
    if n_active_params > 0:
        total_samples = np.zeros((n_toys, n_bins_use), dtype='float64')

        for itoy in range(n_toys):
            theta = theta_samples[itoy]
            total_this = np.zeros(n_bins_use, dtype='float64')

            for ci in class_infos:
                # start from POI prediction if POIs are sampled, otherwise from central
                if getattr(args, "freezePOI", False) or len(ci['poi_params']) == 0:
                    vals = ci['central'].copy()
                else:
                    # build POI point for this toy from theta
                    poi_point_toy = []
                    for name, col in zip(ci['poi_params'], ci['poi_cols']):
                        if col is None:
                            # parameter not active in sampling (frozen): use central value
                            if args.postfit:
                                poi_point_toy.append(param_map[name])
                            else:
                                poi_point_toy.append(0.0)
                        else:
                            poi_point_toy.append(theta[col])
                    vals = np.asarray(ci['poi_pred'].predict(poi_point_toy),
                                      dtype='float64')[:n_bins_use]

                # apply multiplicative icph systematics
                for syst in ci['syst_list']:
                    x = []
                    for col in syst['cols']:
                        if col is None:
                            x.append(0.0)
                        else:
                            x.append(theta[col])
                    rel = np.asarray(syst['predictor'].predict(x), dtype='float64')[:n_bins_use]
                    vals *= rel

                total_this += vals[:n_bins_use]

            total_samples[itoy, :] = total_this

        q_low  = np.quantile(total_samples, 0.16, axis=0)
        q_high = np.quantile(total_samples, 0.84, axis=0)
    else:
        # no parameters to sample: no extra uncertainty
        q_low  = total_central.copy()
        q_high = total_central.copy()

    # ---- total prediction histogram and uncertainty band ----
    h_total = ROOT.TH1F(f"h_{plot_label}_total_{region_id}", "", n_bins_fit, bin_edges_fit)
    for ib in range(n_bins_use):
        h_total.SetBinContent(ib + 1, total_central[ib])

    # uncertainty band
    h_unc = h_total.Clone(f"h_{plot_label}_unc_{region_id}")
    h_unc.SetDirectory(0)
    for ib in range(n_bins_use):
        nominal = total_central[ib]
        lo = q_low[ib]
        hi = q_high[ib]
        err = max(abs(nominal - lo), abs(hi - nominal))
        h_unc.SetBinError(ib + 1, err)

    h_unc.SetFillColor(ROOT.kGray + 1)
    h_unc.SetFillStyle(3345)
    h_unc.SetLineWidth(0)
    h_unc.SetMarkerSize(0)

    # lines at nominal ± uncertainty (absolute)
    h_unc_up   = h_total.Clone(f"h_{plot_label}_unc_up_{region_id}")
    h_unc_down = h_total.Clone(f"h_{plot_label}_unc_down_{region_id}")
    h_unc_up.SetDirectory(0)
    h_unc_down.SetDirectory(0)
    for ib in range(1, n_bins_use + 1):
        nom = h_total.GetBinContent(ib)
        err = h_unc.GetBinError(ib)
        h_unc_up.SetBinContent(ib, nom + err)
        h_unc_down.SetBinContent(ib, max(0.0, nom - err))
        h_unc_up.SetBinError(ib, 0.0)
        h_unc_down.SetBinError(ib, 0.0)
    h_unc_up.SetLineColor(ROOT.kGray + 2)
    h_unc_up.SetLineStyle(ROOT.kSolid)
    h_unc_up.SetLineWidth(1)
    h_unc_down.SetLineColor(ROOT.kGray + 2)
    h_unc_down.SetLineStyle(ROOT.kSolid)
    h_unc_down.SetLineWidth(1)
    h_unc_up.SetFillStyle(0)
    h_unc_up.SetFillColor(0)
    h_unc_down.SetFillStyle(0)
    h_unc_down.SetFillColor(0)

    # "data" histogram: copy of total, errors = sqrt(yield)
    h_data = h_total.Clone(f"h_{plot_label}_data_{region_id}")
    h_data.SetDirectory(0)
    for ib in range(1, n_bins_fit + 1):
        y = h_data.GetBinContent(ib)
        h_data.SetBinError(ib, np.sqrt(y))
    h_data.SetMarkerStyle(ROOT.kFullCircle)
    h_data.SetMarkerSize(1.0)
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetFillStyle(0)

    # ---- sort classes by yield (highest on top in stack) ----
    integrals = [h.Integral(1, n_bins_use) for h in class_hists]
    order = sorted(range(len(class_hists)), key=lambda i: integrals[i])  # small first, big last

    class_hists_sorted  = [class_hists[i]  for i in order]
    class_labels_sorted = [class_labels[i] for i in order]

    # ---- build stack ----
    stack_name = f"stack_{plot_label}_{region_id}"
    hs = ROOT.THStack(stack_name, "")
    for h in class_hists_sorted:
        hs.Add(h, "hist")

    # ---- canvas and pads ----
    canvas_name = f"c_{plot_label}_{region_id}"
    c_fit = ROOT.TCanvas(canvas_name, canvas_name, 800, 800)

    padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 1.0)
    padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

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

    # ---- TOP PAD: absolute yields ----
    padTop.cd()
    if logY_fit:
        padTop.SetLogy(True)

    hs.Draw("HIST")
    hs.GetXaxis().SetTitle(x_title_fit)
    hs.GetYaxis().SetTitle("Events")

    # font sizes / alignment (top pad)
    hs.GetYaxis().SetTitleSize(0.05)     # a bit smaller
    hs.GetYaxis().SetTitleOffset(1.1)    # helps align with bottom pad title
    hs.GetYaxis().SetLabelSize(0.045)
    hs.GetXaxis().SetLabelSize(0)
    hs.GetXaxis().SetTitleSize(0)

    # y-range
    max_y = max(hs.GetMaximum(), h_data.GetMaximum())
    if logY_fit:
        hs.SetMinimum(0.5)
        hs.SetMaximum(10.0 * max_y if max_y > 0 else 1.0)
    else:
        hs.SetMinimum(0.0)
        hs.SetMaximum(1.5 * max_y if max_y > 0 else 1.0)

    # draw uncertainty band, lines, and data
    h_unc.Draw("E2 SAME")
    h_unc_up.Draw("HIST SAME")
    h_unc_down.Draw("HIST SAME")
    h_data.Draw("E SAME")

    # legend
    leg = ROOT.TLegend(0.50, 0.60, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(fit_legend_columns)

    leg.AddEntry(h_data, "Data (Asimov)", "lep")
    for h, lbl in zip(class_hists_sorted, class_labels_sorted):
        leg.AddEntry(h, lbl, "f")
    leg.AddEntry(h_unc, "Uncertainty", "f")
    leg.Draw()

    # ---- BOTTOM PAD: ratios ----
    padBottom.cd()

    # ratio central
    h_ratio_central = h_total.Clone(f"h_{plot_label}_ratio_{region_id}")
    h_ratio_central.SetDirectory(0)
    h_ratio_central.Divide(h_total)  # becomes 1 where non-zero
    h_ratio_central.SetLineColor(ROOT.kBlack)
    h_ratio_central.SetLineWidth(2)
    h_ratio_central.SetTitle("")

    h_ratio_central.GetYaxis().SetTitle("var / nominal")
    h_ratio_central.GetYaxis().SetNdivisions(505)
    h_ratio_central.GetYaxis().SetTitleSize(0.09)
    h_ratio_central.GetYaxis().SetTitleOffset(0.5)
    h_ratio_central.GetYaxis().SetLabelSize(0.08)

    h_ratio_central.GetXaxis().SetTitle(x_title_fit)
    h_ratio_central.GetXaxis().SetTitleSize(0.10)
    h_ratio_central.GetXaxis().SetLabelSize(0.08)

    # ratio uncertainty band via TBoxes + line-only histos
    ratio_boxes = []

    h_ratio_line_up   = h_ratio_central.Clone(f"h_{plot_label}_ratio_up_{region_id}")
    h_ratio_line_down = h_ratio_central.Clone(f"h_{plot_label}_ratio_down_{region_id}")
    h_ratio_line_up.SetDirectory(0)
    h_ratio_line_down.SetDirectory(0)

    # lines only, no fill
    h_ratio_line_up.SetFillStyle(0)
    h_ratio_line_up.SetFillColor(0)
    h_ratio_line_down.SetFillStyle(0)
    h_ratio_line_down.SetFillColor(0)
    h_ratio_line_up.SetLineColor(ROOT.kGray + 2)
    h_ratio_line_down.SetLineColor(ROOT.kGray + 2)
    h_ratio_line_up.SetLineWidth(1)
    h_ratio_line_down.SetLineWidth(1)

    for ib in range(1, n_bins_use + 1):
        x1 = var_edges_fit[ib-1]
        x2 = var_edges_fit[ib]
        nom = h_total.GetBinContent(ib)
        err = h_unc.GetBinError(ib)

        if nom > 0.0:
            rel = err / nom
            y_low  = 1.0 - rel
            y_high = 1.0 + rel

            box = ROOT.TBox(x1, y_low, x2, y_high)
            box.SetFillColor(ROOT.kGray + 1)
            box.SetFillStyle(3345)
            box.SetLineWidth(0)
            ratio_boxes.append(box)

            h_ratio_line_up.SetBinContent(ib, y_high)
            h_ratio_line_down.SetBinContent(ib, y_low)
        else:
            # no prediction in this bin -> keep lines at 1
            h_ratio_line_up.SetBinContent(ib, 1.0)
            h_ratio_line_down.SetBinContent(ib, 1.0)

    # ratio y-range from max relative deviation
    max_dev = 0.0
    for ib in range(1, n_bins_use + 1):
        nom = h_total.GetBinContent(ib)
        err = h_unc.GetBinError(ib)
        if nom > 0:
            dev = err / nom
            if dev > max_dev:
                max_dev = dev

    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    h_ratio_central.SetMinimum(r_min)
    h_ratio_central.SetMaximum(r_max)

    # draw ratio
    h_ratio_central.Draw("HIST")
    for box in ratio_boxes:
        box.Draw("SAME")
    h_ratio_line_up.Draw("HIST SAME")
    h_ratio_line_down.Draw("HIST SAME")

    # line at 1
    line = ROOT.TLine(var_edges_fit[0], 1.0, var_edges_fit[-1], 1.0)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("SAME")

    c_fit.cd()
    c_fit.Update()

    helpers.copyIndexPHP(fit_plot_directory)
    out_png = os.path.join(fit_plot_directory, f"{region_id}_{var_name_fit}_{plot_label}.png")
    out_pdf = os.path.join(fit_plot_directory, f"{region_id}_{var_name_fit}_{plot_label}.pdf")
    c_fit.SaveAs(out_png)
    c_fit.SaveAs(out_pdf)

    print(f"[info] {plot_label} plot written to:\n  {out_png}\n  {out_pdf}")

syncer.sync()
# binning & axis info from (possibly patched) cfg
