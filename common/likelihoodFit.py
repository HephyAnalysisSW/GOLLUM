'''
This is a class to perform likelihood fits.
There is a function implemented that acts as a likelihood but it can be
exchanged with any function that takes mu and nus as arguments.
'''
from iminuit import Minuit

import logging
logger = logging.getLogger("UNC")
print_level=0

def likelihood_test_function( mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    # this defines the function that is minimized
    logger.warning( "Using likelihood_test_function! Result will not depend on data!" )
    penalty = 0.5*(nu_bkg**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2)
    return (mu - 1.1**nu_bkg)**2 + (mu + 2.1**nu_tt)**2 + mu * mu + 2*penalty


class likelihoodFit:
    def __init__(self, function):

        self.function = function

        self.parameterBoundaries = {
            "mu": (0.0, None),
            "nu_bkg": (-10., 10.),
            "nu_tt": (-10., 10.),
            "nu_diboson": (-4., 4.),
            "nu_jes": (-10., 10.),
            "nu_tes": (-10., 10.),
            "nu_met": (0., 5.),
        }
        self.tolerance = 0.1 # default 0.1
        self.eps = 0.1 # default
        self.q_mle = None
        self.parameters_mle = None

    def fit(self, start_mu=1.0, start_nu_bkg=0.0, start_nu_tt=0.0, start_nu_diboson=0.0, start_nu_jes=0.0, start_nu_tes=0.0, start_nu_met=0.0):

        # function to find the global minimum, minimizing mu and nus
        logger.info("Fit global minimum")

        m = Minuit(self.function, mu=start_mu, nu_bkg=start_nu_bkg, nu_tt=start_nu_tt, nu_diboson=start_nu_diboson, nu_jes=start_nu_jes, nu_tes=start_nu_tes, nu_met=start_nu_met)
        m.errordef = Minuit.LIKELIHOOD
        m.print_level=print_level

        m.limits["mu"] = self.parameterBoundaries["mu"]

        for nuname in ["nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]:
            m.limits[nuname] = self.parameterBoundaries[nuname]

        m.tol = self.tolerance

        for param in m.parameters:
            m.errors[param] = self.eps  # Set the step size for all parameters

        m.migrad()
        logger.info("Before 'm.hesse()")
        print(m)
        m.hesse()
        logger.info("After 'm.hesse()")
        print(m)

        self.q_mle = m.fval
        self.parameters_mle = m.values
        limits = None
        return m.fval, m.values, m.covariance, limits

    def scan(self, Npoints=100, mumin=-5, mumax=5):
        # Scan over points of mu
        logger.info("Scan signal strength")
        # First find global Min and store MLE values
        if self.q_mle is None or self.parameters_mle is None:
            q_mle, parameters_mle, cov = self.fit()
        else:
            logger.info("No need to re-run global fit, take existing results")
            q_mle = self.q_mle
            parameters_mle = self.parameters_mle
        # Now make scan over nu
        qDeltas = []
        muList = [mumin+i*(mumax-mumin)/Npoints for i in range(Npoints)]
        fmu = 1
        for i_mu, mu in enumerate(muList):
            # Create a function that fixes mu and only uses the nu as arguments
            fixed_mu = mu
            likelihood_fixedMu = lambda nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met: self.function(fixed_mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
            m = Minuit(likelihood_fixedMu, nu_bkg=0.0, nu_tt=0.0, nu_diboson=0.0, nu_jes=0.0, nu_tes=0.0, nu_met=0.0)
            m.errordef = Minuit.LIKELIHOOD
            for nuname in ["nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]:
                m.limits[nuname] = self.parameterBoundaries[nuname]
            m.tol = self.tolerance
            m.migrad()
            print(m)
            qDeltas.append(m.fval-q_mle)
            if (i_mu+1)/len(muList)*100 >= 10*fmu:
                logger.info(f"Scanned {i_mu+1}/{len(muList)}")
                fmu += 1
        return qDeltas, muList

    def impacts(self):
        logger.info("Calculate impacts")
        # First find global Min and store MLE values
        if self.q_mle is None or self.parameters_mle is None:
            q_mle, parameters_mle, cov = self.fit()
        else:
            logger.info("No need to re-run global fit, take existing results")
            q_mle = self.q_mle
            parameters_mle = self.parameters_mle
        mu_mle = parameters_mle["mu"]
        nu_bkg_mle = parameters_mle["nu_bkg"]
        nu_tt_mle = parameters_mle["nu_tt"]
        nu_diboson_mle = parameters_mle["nu_diboson"]
        nu_jes_mle = parameters_mle["nu_jes"]
        nu_tes_mle = parameters_mle["nu_tes"]
        nu_met_mle = parameters_mle["nu_met"]

        # Define functions that fix all parameters but a single nu
        nu_bkg_function = lambda nu_bkg: abs(self.function(mu_mle, nu_bkg, nu_tt_mle, nu_diboson_mle, nu_jes_mle, nu_tes_mle, nu_met_mle)-q_mle-1.0)
        nu_tt_function =      lambda nu_tt: abs(self.function(mu_mle, nu_bkg_mle, nu_tt, nu_diboson_mle, nu_jes_mle, nu_tes_mle, nu_met_mle)-q_mle-1.0)
        nu_diboson_function = lambda nu_diboson: abs(self.function(mu_mle, nu_bkg_mle, nu_tt_mle, nu_diboson, nu_jes_mle, nu_tes_mle, nu_met_mle)-q_mle-1.0)
        nu_jes_function = lambda nu_jes: abs(self.function(mu_mle, nu_bkg_mle, nu_tt_mle, nu_diboson_mle, nu_jes, nu_tes_mle, nu_met_mle)-q_mle-1.0)
        nu_tes_function = lambda nu_tes: abs(self.function(mu_mle, nu_bkg_mle, nu_tt_mle, nu_diboson_mle, nu_jes_mle, nu_tes, nu_met_mle)-q_mle-1.0)
        nu_met_function = lambda nu_met: abs(self.function(mu_mle, nu_bkg_mle, nu_tt_mle, nu_diboson_mle, nu_jes_mle, nu_tes_mle, nu_met)-q_mle-1.0)
        nu_functions = {
            "nu_bkg": nu_bkg_function,
            "nu_tt": nu_tt_function,
            "nu_diboson": nu_diboson_function,
            "nu_jes": nu_jes_function,
            "nu_tes": nu_tes_function,
            "nu_met": nu_met_function,
        }

        # Now go through each nu and find point where q - q_mle == 1
        # Do this from nu_mle to both sides to find lower and upper boundaries
        limits = {}
        for nuname in ["nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]:
            upper, lower = None, None

            # Use **kwargs in order to dynamically change the parameters
            param_up = {nuname: parameters_mle[nuname] + 0.01}
            m_up = Minuit(nu_functions[nuname], **param_up)
            m_up.errordef = Minuit.LEAST_SQUARES
            m_up.print_level=print_level
            for nuname2 in ["nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]:
                if nuname == nuname2:
                    m_up.limits[nuname] = (parameters_mle[nuname], self.parameterBoundaries[nuname][1])
                else:
                    m_up.limits[nuname] = self.parameterBoundaries[nuname]
            m_up.tol = self.tolerance
            m_up.migrad()
            print(m_up)
            upper = m_up.values[nuname]

            param_down = {nuname: parameters_mle[nuname] - 0.01}
            m_down = Minuit(nu_functions[nuname], **param_down)
            m_down.errordef = Minuit.LEAST_SQUARES
            m_down.print_level=print_level
            for nuname2 in ["nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]:
                if nuname == nuname2:
                    m_down.limits[nuname] = (self.parameterBoundaries[nuname][0], parameters_mle[nuname])
                else:
                    m_down.limits[nuname] = self.parameterBoundaries[nuname]
            m_down.tol = self.tolerance
            m_down.migrad()
            print(m_down)
            lower = m_down.values[nuname]
            limits[nuname] = (parameters_mle[nuname], lower, upper)
        return limits
