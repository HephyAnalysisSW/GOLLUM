#!/usr/bin/env python

import sys
sys.path.insert(0, "..")
import numpy as np
import pickle

import argparse
import common.user
from common.likelihoodFit import likelihoodFit
from common.LikelihoodScanPlotter import LikelihoodScanPlotter
from Workflow.Inference import Inference
import time

def logger(message):
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{formatted_time}: {message}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument("-c","--config", help="Path to the config file.")
    parser.add_argument("-s","--save", action="store_true", help="Whether to save the ML predictions for the simulation.")
    parser.add_argument("-p","--predict", action="store_true", help="Whether to predict.")
    parser.add_argument("-i","--impacts", action="store_true", help="Whether to run postFit uncertainties.")
    parser.add_argument("-g","--scan", action="store_true", help="Whether to run likelihood scan.")
    parser.add_argument("--small", action="store_true", help="Run a subset?")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite?")
    parser.add_argument("--asimov_mu",          action="store", default=None, type=float, help="Modify the toy weights according to mu?")
    parser.add_argument("--asimov_nu_bkg",      action="store", default=None, type=float, help="Modify the toy weights according to nu_bkg?")
    parser.add_argument("--asimov_nu_ttbar",    action="store", default=None, type=float, help="Modify the toy weights according to nu_?")
    parser.add_argument("--asimov_nu_diboson",  action="store", default=None, type=float, help="Modify the toy weights according to nu_bkg?")
    
    args = parser.parse_args()

    postfix=""
    if args.asimov_mu is not None:
        postfix += "mu_"+("%4.3f"%args.asimov_mu).replace("-", "m").replace(".", "p")
    if args.asimov_nu_bkg is not None:
        postfix += "nu_bkg_"+("%4.3f"%args.asimov_nu_bkg).replace("-", "m").replace(".", "p")
    if args.asimov_nu_ttbar is not None:
        postfix += "nu_ttbar_"+("%4.3f"%args.asimov_nu_ttbar).replace("-", "m").replace(".", "p")
    if args.asimov_nu_diboson is not None:
        postfix += "nu_diboson_"+("%4.3f"%args.asimov_nu_diboson).replace("-", "m").replace(".", "p")

    infer = Inference(args.config, small=args.small, overwrite=args.overwrite)
    configName = args.config.replace(".yaml", "")

    # Save  the dataset
    if args.save:
        infer.save()

    if args.predict:
        likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met:\
            infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, 
                asimov_mu=args.asimov_mu, asimov_nu_bkg=args.asimov_nu_bkg, asimov_nu_ttbar=args.asimov_nu_ttbar, asimov_nu_diboson=args.asimov_nu_diboson)

        # Fit Asimov
        fit = likelihoodFit(likelihood_function)
        logger("Start global fit.")
        q_mle, parameters_mle, cov = fit.fit()
        logger("Fit done.")
        print(f"q_mle = {q_mle}")
        print(f"parameters = {parameters_mle}")
        data_to_save = {
            "q_mle": q_mle,
            "mu_mle": parameters_mle["mu"],
        }

        for param1 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
            data_to_save[param1] = parameters_mle[param1]
            for param2 in ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]:
                data_to_save["cov__"+param1+"__"+param2] = cov[param1, param2]
        with open('fitResult.'+configName+'.pkl', 'wb') as file:
            pickle.dump(data_to_save, file)
        # Perform a likelihood scan over mu and store result in arrays
        if args.scan:
            logger("Start scan.")
            deltaQ,muPoints = fit.scan(Npoints=20, mumin=0, mumax=2)
            logger("Scan done.")
            np.savez('likelihoodScan.'+configName+'.npz', deltaQ=np.array(deltaQ), mu=np.array(muPoints))
        # Get constraints for all nuisances and store in a dictionary
        # For each parameter, there is a tuple with values (nu_mle, nu_lower, nu_upper)
        if args.impacts:
            logger("Start impacts.")
            postFitUncerts = fit.impacts()
            logger("Impacts done.")
            print(f"postFit parameter boundaries: {postFitUncerts}")
            with open('postFitUncerts.'+configName+'.pkl', 'wb') as file:
                pickle.dump(postFitUncerts, file)
        infer.clossMLresults()
    # Below is the deprecated feature that calculates the ML prediction on the fly
    #r = infer.testStat(1,0)
    #print(r)
