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

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="ML inference.")
  parser.add_argument("-c","--config", help="Path to the config file.")
  parser.add_argument("-s","--save", action="store_true", help="Whether to save the ML predictions for the simulation.")
  parser.add_argument("-p","--predict", action="store_true", help="Whether to predict.")
  parser.add_argument("-i","--impacts", action="store_true", help="Whether to run postFit uncertainties.")
  parser.add_argument("-g","--scan", action="store_true", help="Whether to run likelihood scan.")
  args = parser.parse_args()

  infer = Inference(args.config)
  configName = args.config.replace(".yaml", "")
  if args.save:
    infer.save(filename="test",isData=False)
  if args.predict:
    # Fit Asimov
    function_asimov = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met: infer.predict(mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met,False)
    fit = likelihoodFit(function_asimov)
    q_mle, parameters_mle = fit.fit()
    print(f"q_mle = {q_mle}")
    print(f"parameters = {parameters_mle}")
    # Perform a likelihood scan over mu and store result in arrays
    if args.scan:
        deltaQ,muPoints = fit.scan(Npoints=20, mumin=0, mumax=2)
        np.savez('likelihoodScan.'+configName+'.npz', deltaQ=np.array(deltaQ), mu=np.array(muPoints))
    # Get constraints for all nuisances and store in a dictionary
    # For each parameter, there is a tuple with values (nu_mle, nu_lower, nu_upper)
    if args.impacts:
        postFitUncerts = fit.impacts()
        print(f"postFit parameter boundaries: {postFitUncerts}")
        with open('postFitUncerts.'+configName+'.pkl', 'wb') as file:
            pickle.dump(postFitUncerts, file)
    infer.clossMLresults()
  # Below is the deprecated feature that calculates the ML prediction on the fly
  #r = infer.testStat(1,0)
  #print(r)
