import os
import sys
sys.path.insert(0, "..")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pythonpackages'))

import numpy as np
import yaml
from Workflow.Inference import Inference
from common.likelihoodFit import likelihoodFit
import common.user as user
from common.intervalFinder import intervalFinder

import logging
logger = logging.getLogger('UNC')

class Model:
    def __init__(self, get_train_set=None, systematics=None, config_path=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            self.cfg = self.loadConfig( os.path.join( self.script_dir, "config_submission.yaml" ) )
        else:
            self.cfg = self.loadConfig( config_path )
        self.calibrate = True
        output_directory = os.path.join( self.script_dir, "data")
        self.cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data" )

        self.infer = None

        ## Removing non-picklable modules
        #for _, mc in self.infer.models['MultiClassifier'].items():
        #    mc.config = None
        #    mc.optimizer = None
        #    mc.lr_schedule = None
        #    mc.loss_fn = None
        #    mc.metrics = None

        #for _, r in self.infer.icps.items():
        #    for _, icp in r.items():
        #        icp.config = None


    def fit(self):
        pass

    def predict(self, test_set):

        if self.infer is None:
            self.infer = Inference(cfg=self.cfg, small=False, overwrite=False, toy_origin="memory", toy_path=None, toy_from_memory=None)
            self.infer.ignore_loading_check()

        # Initialize inference object
        self.infer.setToyFromMemory(test_set)
        self.infer._dcr_cache = {}
        # Define likelihood function
        likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
            self.infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
            nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
            asimov_mu=None, \
            asimov_nu_bkg=None, \
            asimov_nu_tt=None, \
            asimov_nu_diboson=None)
        # Perform global fit
        fit = likelihoodFit(likelihood_function)
        fit.parameterBoundaries["mu"] = (0, None)
        q_mle, parameters_mle, cov, limits = fit.fit(start_mu=1.0)

        mu_mle = parameters_mle["mu"]
        delta_mu = np.sqrt(cov["mu", "mu"])
        p16 = mu_mle - delta_mu
        p84 = mu_mle + delta_mu

        # Now do NON-PROFILED scan
        Npoints = 21
        mumin = min(mu_mle - 3*delta_mu, mu_mle-0.5) # if delta_mu is too small, scan from mu-0.5 to mu+0.5
        mumax = max(mu_mle + 3*delta_mu, mu_mle+0.5) # if delta_mu is too small, scan from mu-0.5 to mu+0.5

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
        if config_path.endswith(".pkl"):
            use_yaml = False
        else:
            try:
                import yaml
                use_yaml = True
            except:
                import pickle
                use_yaml = False
                config_path = config_path.replace(".yaml", ".pkl")

        assert os.path.exists(config_path), "Config does not exist: {}".format(config_path)

        if use_yaml:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                cfg = pickle.load(f, 'rb')

        for task in cfg["Tasks"]:
            for selection in cfg["Selections"]:
                for item in ["calibration", "icp_file", "model_path"]:
                    if item in cfg[task][selection]:
                        cfg[task][selection][item] = os.path.join(self.script_dir, cfg[task][selection][item])

        if "Poisson" in cfg:
            for sel in cfg["Poisson"].keys():
                if 'model_path' in cfg["Poisson"][sel]:
                    cfg["Poisson"][sel]["model_path"] = os.path.join(self.script_dir, cfg["Poisson"][sel]["model_path"])
                cfg["Poisson"][sel]["IC"] = os.path.join(self.script_dir, cfg["Poisson"][sel]["IC"])
                for process in cfg["Poisson"][sel]["ICP"].keys():
                    cfg["Poisson"][sel]["ICP"][process] = os.path.join(self.script_dir, cfg["Poisson"][sel]["ICP"][process])

        return cfg
