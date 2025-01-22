import sys
sys.path.insert(0, "..")
import os
import numpy as np
import pickle
import argparse
import time
from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user

# Logger function for timestamped messages
def logger(message):
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{formatted_time}: {message}")

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="ML inference.")
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

    args = parser.parse_args()

    # Construct postfix for filenames based on asimov parameters
    postfix = ""
    if args.asimov_mu is not None:
        postfix += f"mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p")
    if args.asimov_nu_bkg is not None:
        postfix += f"nu_bkg_{args.asimov_nu_bkg:.3f}".replace("-", "m").replace(".", "p")
    if args.asimov_nu_ttbar is not None:
        postfix += f"nu_ttbar_{args.asimov_nu_ttbar:.3f}".replace("-", "m").replace(".", "p")
    if args.asimov_nu_diboson is not None:
        postfix += f"nu_diboson_{args.asimov_nu_diboson:.3f}".replace("-", "m").replace(".", "p")

    # Initialize inference object
    infer = Inference(args.config, small=args.small, overwrite=args.overwrite)
    config_name = args.config.replace(".yaml", "")

    # Define output directory
    output_directory = os.path.join(
        user.output_directory,
        os.path.basename(args.config).replace(".yaml", ""),
        f"fit_data{'_small' if args.small else ''}"
    )
    os.makedirs(output_directory, exist_ok=True)

    # Save the dataset if requested
    if args.save:
        infer.save()

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
        logger("Start global fit.")
        fit = likelihoodFit(likelihood_function)
        q_mle, parameters_mle, cov = fit.fit()
        logger("Fit done.")

        print(f"q_mle = {q_mle}")
        print(f"parameters = {parameters_mle}")

        # Save fit results
        data_to_save = {"q_mle": q_mle, "mu_mle": parameters_mle["mu"]}
        for param1 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
            data_to_save[param1] = parameters_mle[param1]
            for param2 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
                data_to_save[f"cov__{param1}__{param2}"] = cov[param1, param2]

        result_file = os.path.join(output_directory, f"fitResult.{config_name}{'_' + postfix if postfix else ''}.pkl")
        with open(result_file, 'wb') as file:
            pickle.dump(data_to_save, file)

        # Perform likelihood scan if requested
        if args.scan:
            logger("Start scan.")
            deltaQ, muPoints = fit.scan(Npoints=20, mumin=0, mumax=2)
            logger("Scan done.")
            np.savez(f"likelihoodScan.{config_name}.npz", deltaQ=np.array(deltaQ), mu=np.array(muPoints))

        # Compute impacts if requested
        if args.impacts:
            logger("Start impacts.")
            postFitUncerts = fit.impacts()
            logger("Impacts done.")

            print(f"postFit parameter boundaries: {postFitUncerts}")
            impacts_file = os.path.join(output_directory, f"postFitUncerts.{config_name}{'_' + postfix if postfix else ''}.pkl")
            with open(impacts_file, 'wb') as file:
                pickle.dump(postFitUncerts, file)

        infer.clossMLresults()

    # Deprecated feature (commented out for now)
    # r = infer.testStat(1, 0)
    # print(r)

