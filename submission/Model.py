import sys
sys.path.insert(0, "..")

import os
import numpy as np
import yaml
from Workflow.Inference import Inference
from common.likelihoodFit import likelihoodFit
import common.user as user
from common.logger import get_logger
from common.intervalFinder import intervalFinder


class Model:
    def __init__(self, get_train_set=None, systematics=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))#.replace('/submission', '')
        self.cfg = self.loadConfig( os.path.join( self.script_dir, "config_submission.yaml" ) )
        self.calibrate = True
        output_directory = os.path.join( self.script_dir, "data")
        self.cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data" )
        logger = get_logger("INFO", logFile = None)

    def fit(self):
        pass

    def predict(self, test_set):
        # Initialize inference object
        infer = Inference(cfg=self.cfg, small=False, overwrite=False, toy_origin="memory", toy_path=None, toy_from_memory=test_set)
        infer.ignore_loading_check()
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

        mu_mle = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu_mle - delta_mu
        p84 = mu_mle + delta_mu

        # Now do NON-PROFILED scan
        Npoints = 21
        mumin = mu_mle - 3*delta_mu
        mumax = mu_mle + 3*delta_mu

        # Now go to MLE point and only evaluate mu
        deltaQ = []
        muPoints = [mumin+i*(mumax-mumin)/Npoints for i in range(Npoints)]

        for i, mu in enumerate(muPoints):
            q = likelihood_function(mu=mu, nu_bkg=parameters_mle["nu_bkg"], nu_tt=parameters_mle["nu_tt"], nu_diboson=parameters_mle["nu_diboson"], nu_tes=parameters_mle["nu_tes"], nu_jes=parameters_mle["nu_jes"], nu_met=parameters_mle["nu_met"])
            deltaQ.append(q-q_mle)

        # Interval finder interpolates and returns crossing points
        # if a boundary is below best fit mu, it is the lower boundary, if above it is the upper

        intFinder = intervalFinder(muPoints, deltaQ, 1.0)
        boundaries = intFinder.getInterval()
        for b in boundaries:
            if b < mu_mle:
                p16 = b
            if b > mu_mle:
                p84 = b

        # inflate and offset
        offset = 0.0
        inflate = 1.0
        ####REPLACEOFFSET####
        ####REPLACEINFLATE####
        p16 = mu_mle - inflate*(mu_mle-p16) + offset
        p84 = mu_mle + inflate*(p84-mu_mle) + offset
        mu_mle = mu_mle + offset

        # Check mu boundaries
        if p16 < 0.1:
            p16 = 0.09
        if p84 > 3.0:
            p84 = 3.01

        delta_mu = (p84-p16)/2

        return {
            "mu_hat": mu_mle,
            "delta_mu_hat": delta_mu,
            "p16": p16,
            "p84": p84,
            "nu_bkg": parameters_mle["nu_bkg"],
            "nu_tt":  parameters_mle["nu_tt"],
            "nu_diboson": parameters_mle["nu_diboson"],
            "nu_tes": parameters_mle["nu_tes"],
            "nu_jes": parameters_mle["nu_jes"],
            "nu_met": parameters_mle["nu_met"],
        }


    def loadConfig(self, config_path):
        assert os.path.exists(config_path), "Config does not exist: {}".format(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        for task in cfg["Tasks"]:
            for selection in cfg["Selections"]:
                for item in ["calibration", "icp_file", "model_path"]:
                    if item in cfg[task][selection]:
                        cfg[task][selection][item] = os.path.join(self.script_dir, cfg[task][selection][item])
        return cfg
