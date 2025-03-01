#!/usr/bin/env python3
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import copy
import numpy as np
import pickle
import argparse
import time
import yaml
import os

import common.syncer
import common.helpers as helpers

# ROOT imports
import ROOT
ROOT.gROOT.SetBatch(True)  # Run in batch mode so we don't pop up windows.

from common.logger import get_logger
from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user

def update_dict(d, keys, value):
    """Recursively update a nested dictionary."""
    key = keys[0]
    if len(keys) == 1:
        # Convert value to appropriate type
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        d[key] = value
    else:
        d = d.setdefault(key, {})
        update_dict(d, keys[1:], value)

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="ML inference and plotting of 1D scan.")
    parser.add_argument('--logLevel',
                        action='store',
                        nargs='?',
                        choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG','TRACE','NOTSET'],
                        default='INFO',
                        help="Log level for logging")
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("--small", action="store_true", help="Run a subset.")

    # The user-specific parameter scanning
    parser.add_argument("--var",
                        type=str,
                        default="mu",
                        choices=["mu","nu_bkg","nu_tt","nu_diboson","nu_tes","nu_jes","nu_met"],
                        help="Which parameter to vary (except asimov_mu).")
    parser.add_argument("--range",
                        nargs=3,
                        type=float,
                        default=[2.7, 2.9, 0.01],
                        help="start end step for scanning the chosen parameter.")
    
    # Asimov parameters (fixed or optional if needed)
    parser.add_argument("--asimov_mu", type=float, default=None, help="Fix asimov mu for the scan.")
    parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Fix asimov nu_bkg.")
    parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Fix asimov nu_tt.")
    parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Fix asimov nu_diboson.")

    parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
    parser.add_argument("--postfix", default=None, type=str, help="Append this to the fit result.")

    args = parser.parse_args()
    logger = get_logger(args.logLevel, logFile=None)

    # Construct postfix for filenames based on asimov parameters
    postfix = []
    if args.asimov_mu is not None:
        postfix.append(f"mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_bkg is not None:
        postfix.append(f"nu_bkg_{args.asimov_nu_bkg:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_tt is not None:
        postfix.append(f"nu_tt_{args.asimov_nu_tt:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_diboson is not None:
        postfix.append(f"nu_diboson_{args.asimov_nu_diboson:.3f}".replace("-", "m").replace(".", "p"))
    if args.postfix is not None:
        postfix.append(args.postfix)
    postfix = "_".join(postfix)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from {}".format(args.config))

    # Process modifications
    if args.modify:
        for mod in args.modify:
            if "=" not in mod:
                raise ValueError(f"Invalid modify argument: {mod}. Must be in 'key=value' format.")
            key, value = mod.split("=", 1)
            logger.warning("Updating cfg with: %s=%r" % (key, value))
            key_parts = key.split(".")
            update_dict(cfg, key_parts, value)

    # Define output directory
    config_name = os.path.basename(args.config).replace(".yaml", "")
    output_directory = os.path.join(user.output_directory, config_name)
    plot_directory   = os.path.join(user.plot_directory,   "LLplots", config_name)
    os.makedirs(plot_directory, exist_ok=True)

    cfg['tmp_path'] = os.path.join(output_directory, f"tmp_data{'_small' if args.small else ''}")
    os.makedirs(cfg['tmp_path'], exist_ok=True)

    # Two copies of config: no CSI and CSI
    cfg_noCSI = copy.deepcopy(cfg)
    cfg_noCSI["CSI"]["use"] = False
    cfg_CSI  = copy.deepcopy(cfg)
    cfg_CSI["CSI"]["use"]   = True

    # Initialize inference
    infer_noCSI = Inference(cfg_noCSI, small=args.small)
    infer_CSI   = Inference(cfg_CSI,   small=args.small)

    # Define the likelihood functions
    # They each expect: mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met, + asimov parameters
    ll_noCSI = lambda mu,nu_bkg,nu_tt,nu_diboson,nu_tes,nu_jes,nu_met: \
        infer_noCSI.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson,
                            nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met,
                            asimov_mu=args.asimov_mu,
                            asimov_nu_bkg=args.asimov_nu_bkg,
                            asimov_nu_tt=args.asimov_nu_tt,
                            asimov_nu_diboson=args.asimov_nu_diboson)

    ll_CSI = lambda mu,nu_bkg,nu_tt,nu_diboson,nu_tes,nu_jes,nu_met: \
        infer_CSI.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson,
                          nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met,
                          asimov_mu=args.asimov_mu,
                          asimov_nu_bkg=args.asimov_nu_bkg,
                          asimov_nu_tt=args.asimov_nu_tt,
                          asimov_nu_diboson=args.asimov_nu_diboson)

    best_fit = {
        'mu': 2.5,
        'nu_bkg': 0.,
        'nu_tt': 0.,
        'nu_diboson': 0.,
        'nu_tes': 0.,
        'nu_jes': 0.,
        'nu_met': 0.,
    }

    #-------------------------------------------------------
    # 1) Read the range from command line
    #    e.g. --range 2.7 2.9 0.01 means start=2.7, stop=2.9, step=0.01
    #-------------------------------------------------------
    start_val, end_val, step_val = args.range
    var_to_scan = args.var  # e.g. "mu", "nu_bkg", ...

    #-------------------------------------------------------
    # 2) Evaluate noCSI and CSI likelihoods over the range
    #-------------------------------------------------------
    x_vals     = []
    y_vals_noCSI = []
    y_vals_CSI   = []

    # Copy best_fit so we can mutate
    p = copy.deepcopy(best_fit)

    # Make sure to include the end_val in the scan if step divides exactly
    n_steps = int(round((end_val - start_val) / step_val)) + 1
    # Alternatively, we can just do np.arange but be sure about the last bin
    for i in range(n_steps):
        this_val = start_val + i*step_val
        if this_val > end_val + 1e-9:
            break
        p[var_to_scan] = this_val
        # Evaluate
        lnL_noCSI = ll_noCSI(**p)
        lnL_CSI   = ll_CSI(**p)
        x_vals.append(this_val)
        y_vals_noCSI.append(lnL_noCSI)
        y_vals_CSI.append(lnL_CSI)

    # --- Add best-fit point to the arrays ---
    bf_param = best_fit[var_to_scan]
    p[var_to_scan] = bf_param
    bf_y_noCSI = ll_noCSI(**p)
    bf_y_CSI   = ll_CSI(**p)

    x_vals.append(bf_param)
    y_vals_noCSI.append(bf_y_noCSI)
    y_vals_CSI.append(bf_y_CSI)

    # --- Sort the arrays by x for a clean line in the TGraph ---
    combined = list(zip(x_vals, y_vals_noCSI, y_vals_CSI))
    combined.sort(key=lambda row: row[0])  # sort by x

    x_vals_arr, y_vals_noCSI_arr, y_vals_CSI_arr = zip(*combined)

    #-------------------------------------------------------
    # 3) Create two TGraph objects
    #-------------------------------------------------------
    from array import array
    gr_noCSI = ROOT.TGraph(len(x_vals_arr),
                           array('d', x_vals_arr),
                           array('d', y_vals_noCSI_arr))
    gr_CSI   = ROOT.TGraph(len(x_vals_arr),
                           array('d', x_vals_arr),
                           array('d', y_vals_CSI_arr))

    # Give them some styles
    gr_noCSI.SetLineColor(ROOT.kRed)
    gr_noCSI.SetLineWidth(2)
    gr_noCSI.SetMarkerColor(ROOT.kRed)
    gr_noCSI.SetMarkerStyle(20)
    gr_noCSI.SetTitle("no CSI")

    gr_CSI.SetLineColor(ROOT.kBlue)
    gr_CSI.SetLineWidth(2)
    gr_CSI.SetMarkerColor(ROOT.kBlue)
    gr_CSI.SetMarkerStyle(21)
    gr_CSI.SetTitle("CSI")

    #-------------------------------------------------------
    # 4) Make a TMultiGraph to put them both on the same plot
    #-------------------------------------------------------
    mg = ROOT.TMultiGraph()
    mg.Add(gr_noCSI, "LP")
    mg.Add(gr_CSI,   "LP")
    mg.SetTitle(f"LL scan vs {var_to_scan};{var_to_scan};Log-Likelihood")

    # You can manually adjust the y-axis. For instance, to zoom in on a small
    # range where differences are small, find the overall min & max:
    y_all = np.concatenate([y_vals_noCSI_arr, y_vals_CSI_arr])
    y_min, y_max = np.min(y_all), np.max(y_all)
    # Optionally expand the range a little:
    y_margin = 0.01 * abs(y_max - y_min)
    y_min_plot = y_min - y_margin
    y_max_plot = y_max + y_margin

    #-------------------------------------------------------
    # 5) Plot to a canvas and add a legend
    #-------------------------------------------------------
    c = ROOT.TCanvas("c","c",800,600)
    mg.Draw("AL")  # "A" means draw axis, "L" means connect points with lines

    mg.GetXaxis().SetLimits(start_val, end_val)  # x-axis range
    mg.SetMinimum(y_min_plot)
    mg.SetMaximum(y_max_plot)

    legend = ROOT.TLegend(0.15,0.75,0.4,0.85)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(gr_noCSI, "no CSI", "l")
    legend.AddEntry(gr_CSI,   "CSI",    "l")
    legend.Draw()

    # Optionally draw a grid
    c.SetGrid()

    # Save the plot
    png_name = os.path.join(plot_directory,
                f"likelihoodScan_{var_to_scan}_{postfix}.png" if postfix else
                f"likelihoodScan_{var_to_scan}.png")
    c.SaveAs(png_name)
    logger.info(f"Saved plot as {png_name}")

helpers.copyIndexPHP( plot_directory )
common.syncer.sync()
