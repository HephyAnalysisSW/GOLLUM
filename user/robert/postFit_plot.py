import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os
import numpy as np
import pickle
import argparse
import time
import yaml
#from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user

import cProfile, pstats


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
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
    parser.add_argument("--config", help="Path to the config file.")
    parser.add_argument("--impacts", action="store_true", help="Run post-fit uncertainties.")
    parser.add_argument("--scan", action="store_true", help="Run likelihood scan.")
    parser.add_argument("--small", action="store_true", help="Run a subset.")
    parser.add_argument("--doHesse", action="store_true", help="Run Hesse after Minuit?")
    parser.add_argument("--minimizer", type=str, default="minuit", choices=["minuit", "bfgs", "robust"], help="Which minimizer?")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--asimov_mu", type=float, default=None, help="Modify asimov weights according to mu.")
    parser.add_argument("--start_mu", type=float, default=1.0, help="Modify asimov weights according to mu.")
    parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify asimov weights according to nu_bkg.")
    parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Modify asimov weights according to nu_ttbar.")
    parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify asimov weights according to nu_diboson.")
    parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
    parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
    parser.add_argument("--CSI", nargs="+", default = [], help="Make only those CSIs")
    parser.add_argument("--toy", default = None, type=str,  help="Specify toy with path to h5 file.")

    args = parser.parse_args()

    from common.logger import get_logger
    logger = get_logger(args.logLevel, logFile = None)

    # Construct postfix for filenames based on asimov parameters
    postfix = []
    if args.asimov_mu is not None:
        postfix.append(f"mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_bkg is not None:
        postfix.append(f"nu_bkg_{args.asimov_nu_bkg:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_tt is not None:
        postfix.append(f"nu_ttbar_{args.asimov_nu_tt:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_diboson is not None:
        postfix.append(f"nu_diboson_{args.asimov_nu_diboson:.3f}".replace("-", "m").replace(".", "p"))

    if args.postfix is not None:
        postfix.append( args.postfix )

    postfix = "_".join( postfix )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from {}".format(args.config))

    # Process modifications
    if args.modify:
        for mod in args.modify:
            if "=" not in mod:
                raise ValueError(f"Invalid modify argument: {mod}. Must be in 'key=value' format.")
            key, value = mod.split("=", 1)
            logger.warning( "Updating cfg with: %s=%r"%( key, value) )
            key_parts = key.split(".")
            update_dict(cfg, key_parts, value)

    ## Define output directory
    #config_name = os.path.basename(args.config).replace(".yaml", "")
    #output_directory = os.path.join ( user.output_directory, config_name)

    #fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
    #os.makedirs(fit_directory, exist_ok=True)
    #cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data{'_small' if args.small else ''}" )
    #os.makedirs(cfg['tmp_path'], exist_ok=True)

    # Define plot directory
    config_name = os.path.basename(args.config).replace(".yaml", "")
    plot_directory = os.path.join ( user.plot_directory, config_name)

    from common.likelihoodFit import likelihoodFit

    # Initialize inference object
    toy_origin = "config"
    toy_path = None
    toy_from_memory = None
    if args.toy is not None:
        toy_origin = "path"
        toy_path = args.toy

    infer = Inference(cfg, small=args.small, overwrite=args.overwrite, toy_origin=toy_origin, toy_path=toy_path, toy_from_memory=toy_from_memory)

    # Define the likelihood function
    likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
        infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                      nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                      asimov_mu=args.asimov_mu, \
                      asimov_nu_bkg=args.asimov_nu_bkg, \
                      asimov_nu_tt=args.asimov_nu_tt, \
                      asimov_nu_diboson=args.asimov_nu_diboson)

    # Perform global fit
    logger.info("Start global fit.")
    fit = likelihoodFit(likelihood_function, doHesse=args.doHesse)

    #profiler = cProfile.Profile()
    #profiler.enable()

    q_mle, parameters_mle, cov, limits = fit.fit(start_mu=args.start_mu)
    logger.info("Fit done.")

    #profiler.disable()

    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.print_stats()
