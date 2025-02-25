import sys
sys.path.insert(0, "..")

import os
import numpy as np
import yaml
from Workflow.Inference import Inference
from common.likelihoodFit import likelihoodFit
import common.user as user
from common.logger import get_logger

class Model:
    def __init__(self, get_train_set=None, systematics=None):
        self.cfg = self.loadConfig( os.path.join( os.getcwd(), "../Workflow/configs/config_reference_v2_calib.yaml" ) )
        self.calibrate = True
        # TODO: Set tmp_path for ML ntuples an CSI stuff
        output_directory = os.path.join( user.output_directory, "config_reference_v2_calib")
        self.cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data" )
        logger = get_logger("INFO", logFile = None)

    def predict(self, test_set):
        # Initialize inference object
        infer = Inference(cfg=self.cfg, small=False, overwrite=False, toy_origin="memory", toy_path=None, toy_from_memory=test_set)
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
        fit.parameterBoundaries["mu"] = (0.1, 3.0)
        q_mle, parameters_mle, cov, limits = fit.fit(start_mu=1.0)

        mu = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu - delta_mu
        p84 = mu + delta_mu

        # Check mu boundaries
        if p16 < 0.1:
            p16 = 0.09
        if p84 > 3.0:
            p84 = 3.01

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


    def predict_allParam(self, test_set, limits=None):
        # Initialize inference object
        infer = Inference(cfg=self.cfg, small=False, overwrite=False, toy_origin="memory", toy_path=None, toy_from_memory=test_set)
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
        if limits is not None:
            for p in limits.keys():
                fit.parameterBoundaries[p] = limits[p]
        fit.parameterBoundaries["mu"] = (0.1, 3.0)
        q_mle, parameters_mle, cov, limits = fit.fit(start_mu=1.0)

        mu = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu - delta_mu
        p84 = mu + delta_mu

        if p16 < 0.1:
            p16 = 0.09
        if p84 > 3.0:
            p84 = 3.01

        return {
            "mu_hat": mu,
            "delta_mu_hat": delta_mu,
            "p16": p16,
            "p84": p84,
            "nu_jes": parameters_mle["nu_jes"],
            "nu_tes": parameters_mle["nu_tes"],
            "nu_met": parameters_mle["nu_met"],
            "nu_bkg": parameters_mle["nu_bkg"],
            "nu_tt": parameters_mle["nu_tt"],
            "nu_diboson": parameters_mle["nu_diboson"],
        }
