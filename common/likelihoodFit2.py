'''
This is a class to perform likelihood fits.
There is a function implemented that acts as a likelihood but it can be
exchanged with any function that takes mu and nus as arguments.
'''
from iminuit import Minuit

import logging
logger = logging.getLogger("UNC")

# Miniut debug
if logger.getEffectiveLevel() <= logging.DEBUG:
    print_level=2
else:
    print_level=0

def likelihood_test_function( mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    # this defines the function that is minimized
    logger.warning( "Using likelihood_test_function! Result will not depend on data!" )
    penalty = 0.5*(nu_bkg**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2)
    return (mu - 1.1**nu_bkg)**2 + (mu + 2.1**nu_tt)**2 + mu * mu + 2*penalty

def my_nelder_mead_fixed_bounds(fun, x0, args=(), jac=None, hess=None, hessp=None,
                                constraints=None, callback=None, **kwargs):
    """
    A custom Nelder-Mead minimizer that ignores any incoming bounds and always
    uses the fixed bounds stored in my_nelder_mead_fixed_bounds.fixed_bounds.
    If any parameter is out of bounds, the function returns a huge penalty.
    """
    import scipy.optimize as so

    # Retrieve the fixed bounds from a stored attribute.
    fixed_bounds = my_nelder_mead_fixed_bounds.fixed_bounds

    # Define a wrapped objective that penalizes out-of-bound parameters.
    def fun_bounded(x, *args):
        for xi, (lb, ub) in zip(x, fixed_bounds):
            if xi < lb or xi > ub:
                return 1e10  # A huge penalty
        return fun(x, *args)

    local_options = {
        "maxiter": 2000,  # maximum iterations
        "xatol": 1e-2,    # parameter convergence tolerance
        "fatol": 1e-2,    # function-value convergence tolerance
    }

    result = so.minimize(
        fun_bounded,
        x0,
        args=args,
        method="Nelder-Mead",
        # Do not pass any bounds to so.minimize (since Nelder-Mead ignores them anyway).
        callback=callback,
        options=local_options,
    )
    return result


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

        # step sizes
        self.eps = 0.1

        # Will store fit results here
        self.q_mle = None
        self.parameters_mle = None
        self.covariance_mle = None


    def fit(self, start_mu=1.0, start_nu_bkg=0.0, start_nu_tt=0.0,
            start_nu_diboson=0.0, start_nu_jes=0.0, start_nu_tes=0.0, start_nu_met=0.0):

        logger.info("Fit global minimum")

        m = Minuit(
            self.function,
            mu=start_mu,
            nu_bkg=start_nu_bkg,
            nu_tt=start_nu_tt,
            nu_diboson=start_nu_diboson,
            nu_jes=start_nu_jes,
            nu_tes=start_nu_tes,
            nu_met=start_nu_met,
        )
        m.print_level = print_level
        for parname, lim in self.parameterBoundaries.items():
            m.limits[parname] = lim

        # Build SciPy-style bounds in the desired order:
        # Explicit parameter order: mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met
        from math import inf
        param_order = ["mu", "nu_bkg", "nu_tt", "nu_diboson", "nu_tes", "nu_jes", "nu_met"]
        sc_bounds = []
        for pn in param_order:
            low, high = self.parameterBoundaries[pn]
            if low is None:
                low = -inf
            if high is None:
                high = inf
            sc_bounds.append((low, high))
        # For debugging, you mentioned sc_bounds is:
        # [(0.0, inf), (-10.0, 10.0), (-10.0, 10.0), (-4.0, 4.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, 5.0)]

        # Set step sizes as before.
        for par in m.parameters:
            m.errors[par] = self.eps

        m.errordef = 1.0

        # Store our custom bounds in the attribute used by our custom Nelder-Mead.
        my_nelder_mead_fixed_bounds.fixed_bounds = sc_bounds

        # Call the derivative-free optimization with our custom method that enforces bounds.
        m.scipy(method=my_nelder_mead_fixed_bounds, ncall=5000)

        # Refine with a quick gradient-based step.
        m.migrad()

        # Request MINOS errors on mu.
        m.minos("mu")

        self.q_mle = m.fval
        self.parameters_mle = m.values
        self.covariance_mle = m.covariance

        mu_hat = m.values["mu"]
        mu_err = m.merrors["mu"]
        mu_lower = mu_hat + mu_err.lower
        mu_upper = mu_hat + mu_err.upper

        logger.info("Minimum of function = %f at mu = %f", m.fval, mu_hat)
        logger.info("mu_lower = %f    mu_upper = %f", mu_lower, mu_upper)

        print(m)

        return m.fval, m.values, m.covariance, (mu_hat, mu_lower, mu_upper)

#    def fit(self, start_mu=1.0, start_nu_bkg=0.0, start_nu_tt=0.0, start_nu_diboson=0.0, start_nu_jes=0.0, start_nu_tes=0.0, start_nu_met=0.0):
#
#        # function to find the global minimum, minimizing mu and nus
#        logger.info("Fit global minimum")
#        errordef = Minuit.LEAST_SQUARES
#
#        m = Minuit(self.function, mu=start_mu, nu_bkg=start_nu_bkg, nu_tt=start_nu_tt, nu_diboson=start_nu_diboson, nu_jes=start_nu_jes, nu_tes=start_nu_tes, nu_met=start_nu_met)
#        # Set parameter limits
#        for parname, lim in self.parameterBoundaries.items():
#            m.limits[parname] = lim
#
#        # Use step sizes (important for simplex or any derivative-free approach)
#        for par in m.parameters:
#            m.errors[par] = self.eps
#
#        # If we literally want a +1 rise in this function to define the error,
#        # set errordef = 1.0.  (If you had a negative log-likelihood, you'd usually do 0.5)
#        m.errordef = 1.0
#
#        # Use derivative-free optimization from SciPy. 
#        # You can try method="Nelder-Mead" or "Powell" or "SLSQP" depending on your problem.
#        m.scipy(method="Nelder-Mead")
#
#        # Now that we have a good candidate minimum,
#        # ask for MINOS errors on mu at Delta(function) = +1.
#        m.minos("mu")
#
#        # Store results
#        self.q_mle = m.fval
#        self.parameters_mle = m.values
#        self.covariance_mle = m.covariance  # may be None if Nelder-Mead doesn't provide one
#
#        # Extract the profile points for mu
#        mu_hat = m.values["mu"]
#        mu_err = m.merrors["mu"]  # MINOS returns an object with 'lower' and 'upper'
#        mu_lower = mu_hat + mu_err.lower
#        mu_upper = mu_hat + mu_err.upper
#
#        logger.info("Minimum of function = %f at mu = %f", m.fval, mu_hat)
#        logger.info("mu_lower = %f    mu_upper = %f", mu_lower, mu_upper)
#
#        print(m)
#
#        return m.fval,  m.values, m.covariance, (mu_hat, mu_lower, mu_upper)
