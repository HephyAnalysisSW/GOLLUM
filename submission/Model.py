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
        self.cfg = self.loadConfig( os.path.join( os.getcwd(), "../Workflow/configs/config_reference.yaml" ) )
        self.calibrate = True
        # TODO: Set tmp_path for ML ntuples an CSI stuff
        output_directory = os.path.join( user.output_directory, "config_reference")
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
        q_mle, parameters_mle, cov, limits = fit.fit(start_mu=0.0)

        mu = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu - delta_mu
        p84 = mu + delta_mu

        # Calibrate mu?
        if self.calibrate:
            from common.muCalibrator import muCalibrator
            calibration_file = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/calibration.pkl"
            calibrator = muCalibrator(calibration_file)
            correction = calibrator.getCorrection( \
                mu=mu, \
                nu_jes=parameters_mle["nu_jes"], \
                nu_tes=parameters_mle["nu_tes"], \
                nu_met=parameters_mle["nu_met"])

        # TODO SET HARD CODED BOUNDARIES?
        # Check mu boundaries
        # if p16 < 0.0:
        #     p16 = -0.01
        # if p84 > 3.0:
        #     p84 = 3.01
        # if we do boundaries, we can also adjust deltaMu

        return {
            "mu_hat": mu,
            "delta_mu_hat": delta_mu,
            "p16": p16,
            "p84": p84,
            "correction": correction, # TO BE REMOVED
        }



    def loadConfig(self, config_path):
        assert os.path.exists(config_path), "Config does not exist: {}".format(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
