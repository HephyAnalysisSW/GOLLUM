import sys
sys.path.insert(0, "..")

import os
import numpy as np
import pickle
import argparse
import time
import yaml
from Workflow.Inference import Inference
import common.user as user

class Model:
    def __init__(self, get_train_set=None, systematics=None):
        self.cfg = self.loadConfig( os.path.join( os.getcwd(), "../Workflow/config_reference.yaml" ) )

    def predict(self, test_set):

        # Initialize inference object
        infer = Inference(self.cfg, small=False, overwrite=False, toy_origin="memory", toy_path=None, toy_from_memory=test_set)

        # Define likelihood function
        likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
            infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                          nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                          asimov_mu=None, \
                          asimov_nu_bkg=None, \
                          asimov_nu_tt=None, \
                          asimov_nu_diboson=None)

        # Perform global fit
        fit = likelihoodFit(likelihood_function)
        q_mle, parameters_mle, cov = fit.fit(start_mu=args.start_mu)

        mu = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu - delta_mu
        p84 = mu + delta_mu

        # Calibrate mu?
        # from common.muCalibrator import muCalibrator
        # calibration_file = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/calibration.pkl"
        # calibrator = muCalibrator(calibration_file)
        # mu = calibrator.getMu( \
        #     mu=mu, \
        #     nu_jes=parameters_mle["nu_jes"], \
        #     nu_tes=parameters_mle["nu_tes"], \
        #     nu_met=parameters_mle["nu_met"])
        # mu = calibrator.getMu( \
        #     mu=p16, \
        #     nu_jes=parameters_mle["nu_jes"], \
        #     nu_tes=parameters_mle["nu_tes"], \
        #     nu_met=parameters_mle["nu_met"])
        # mu = calibrator.getMu( \
        #     mu=p84, \
        #     nu_jes=parameters_mle["nu_jes"], \
        #     nu_tes=parameters_mle["nu_tes"], \
        #     nu_met=parameters_mle["nu_met"])

        # Check mu boundaries
        if p16 < 0.0:
            p16 = 0.0
        if p84 > 3.0:
            p84 = 3.0

        return {
            "mu_hat": mu,
            "delta_mu_hat": delta_mu,
            "p16": p16,
            "p84": p84,
        }



    def loadConfig(self, config_path):
        assert os.path.exists(config_path), "Config does not exist: {}".format(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
