#!/usr/bin/env python3
import sys
sys.path.insert(0, "..")
import copy
import numpy as np
import pickle
import argparse
import time
import yaml
import os

import common.syncer

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

    # Best-fit dictionary /scratch/robert.schoefbeck/batch_output/batch.13392608.out

    best_fit = {
        'mu': 2.790832418674728,
        'nu_bkg': -0.035687869355687206,
        'nu_tt': -0.01529719518782724,
        'nu_diboson': -0.027784415682560314,
        'nu_tes': -0.00971769862493762,
        'nu_jes': -0.002120059893820499,
        'nu_met': 0.028296744776968728,
    }

    import common.approximate_hessian as approximate_hessian

    # Create the parameter vector in the same order that your objective function expects
    x0 = [
        best_fit["mu"],
        best_fit["nu_bkg"],
        best_fit["nu_tt"],
        best_fit["nu_diboson"],
        best_fit["nu_tes"],
        best_fit["nu_jes"],
        best_fit["nu_met"],
    ]

    # Match each parameter to a bound (low, high). 
    # For instance, from your Inference code or the minuit config:
    param_bounds = [
        (0.0, None),   # mu
        (-10., 10.),   # nu_bkg
        (-10., 10.),   # nu_tt
        (-4., 4.),     # nu_diboson
        (-10., 10.),   # nu_tes
        (-10., 10.),   # nu_jes
        (0., 5.),      # nu_met
    ]

    # Example objective in SciPy style:
    def objective(x):
        return ll_CSI(
            mu=x[0], nu_bkg=x[1], nu_tt=x[2], nu_diboson=x[3],
            nu_tes=x[4], nu_jes=x[5], nu_met=x[6]
        )

    # Call the approximate_hessian with some step size (e.g. 0.2).
    step_size = 0.2

    # Compute the Hessian:
    hess = approximate_hessian.approximate_hessian(
        func=objective,
        x=np.array(x0, dtype=float),
        step=step_size,
        bounds=param_bounds
    )

    # Print it
    print("Hessian:")
    print(hess)

