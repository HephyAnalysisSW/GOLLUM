import sys
sys.path.insert(0, "..")

from common.logger import get_logger
import os
import numpy as np
import pickle
import argparse
import time
import yaml
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
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the ML predictions for the simulation.")
    parser.add_argument("-p", "--predict", action="store_true", help="Run predictions.")
    parser.add_argument("-i", "--impacts", action="store_true", help="Run post-fit uncertainties.")
    parser.add_argument("-g", "--scan", action="store_true", help="Run likelihood scan.")
    parser.add_argument("--small", action="store_true", help="Run a subset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--asimov_mu", type=float, default=None, help="Modify toy weights according to mu.")
    parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify toy weights according to nu_bkg.")
    parser.add_argument("--asimov_nu_ttbar", type=float, default=None, help="Modify toy weights according to nu_ttbar.")
    parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify toy weights according to nu_diboson.")
    parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
    parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
    parser.add_argument("--CSI", nargs="+", default = [], help="Make only those CSIs")

    args = parser.parse_args()

    logger = get_logger(args.logLevel, logFile = None)

    # Construct postfix for filenames based on asimov parameters
    postfix = []
    if args.asimov_mu is not None:
        postfix.append(f"mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_bkg is not None:
        postfix.append(f"nu_bkg_{args.asimov_nu_bkg:.3f}".replace("-", "m").replace(".", "p"))
    if args.asimov_nu_ttbar is not None:
        postfix.append(f"nu_ttbar_{args.asimov_nu_ttbar:.3f}".replace("-", "m").replace(".", "p"))
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

    # Define output directory
    config_name = os.path.basename(args.config).replace(".yaml", "") 
    output_directory = os.path.join ( user.output_directory, config_name)

    fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
    os.makedirs(fit_directory, exist_ok=True)
    cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data{'_small' if args.small else ''}" )
    os.makedirs(cfg['tmp_path'], exist_ok=True)

    # Initialize inference object
    infer = Inference(cfg, small=args.small, overwrite=args.overwrite)

    # Save the dataset if requested
    if args.save:
        infer.save(restrict_csis = args.CSI)

    if args.predict:
        # Define the likelihood function
        likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
            infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                          nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                          asimov_mu=args.asimov_mu, \
                          asimov_nu_bkg=args.asimov_nu_bkg, \
                          asimov_nu_ttbar=args.asimov_nu_ttbar, \
                          asimov_nu_diboson=args.asimov_nu_diboson)

        # Perform global fit
        logger.info("Start global fit.")
        fit = likelihoodFit(likelihood_function)
        q_mle, parameters_mle, cov = fit.fit()
        logger.info("Fit done.")

        logger.info(f"q_mle = {q_mle}")
        logger.info(f"parameters = {parameters_mle}")

        # Save fit results
        data_to_save = {"q_mle": q_mle, "mu_mle": parameters_mle["mu"]}
        for param1 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
            data_to_save[param1] = parameters_mle[param1]
            for param2 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
                data_to_save[f"cov__{param1}__{param2}"] = cov[param1, param2]

        result_file = os.path.join(output_directory, f"fitResult.{config_name}{'_' + postfix if postfix else ''}.pkl")
        with open(result_file, 'wb') as file:
            pickle.dump(data_to_save, file)
        logger.info("Saved fit results: "+result_file)

        # Perform likelihood scan if requested
        if args.scan:
            logger.info("Start scan.")
            deltaQ, muPoints = fit.scan(Npoints=20, mumin=0, mumax=2)
            logger.info("Scan done.")
            np.savez(f"likelihoodScan.{config_name}.npz", deltaQ=np.array(deltaQ), mu=np.array(muPoints))

        # Compute impacts if requested
        if args.impacts:
            logger.info("Start impacts.")
            postFitUncerts = fit.impacts()
            logger.info("Impacts done.")

            logger.info(f"postFit parameter boundaries: {postFitUncerts}")
            impacts_file = os.path.join(output_directory, f"postFitUncerts.{config_name}{'_' + postfix if postfix else ''}.pkl")
            with open(impacts_file, 'wb') as file:
                pickle.dump(postFitUncerts, file)

        infer.clossMLresults()

    # Deprecated feature (commented out for now)
    # r = infer.testStat(1, 0)
    # print(r)
