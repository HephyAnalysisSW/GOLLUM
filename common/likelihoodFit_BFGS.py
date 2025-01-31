import logging
logger = logging.getLogger("UNC")
import sys
sys.path.insert(0, '..')
import numpy as np
from scipy.optimize import minimize

def likelihood_test_function(mu, nu_bkg, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    """
    A simple test function.
    """
    logger.warning("Using likelihood_test_function! Result does not depend on real data.")
    penalty = 0.5 * (nu_bkg**2 + nu_tt**2 + nu_diboson**2 + nu_jes**2 + nu_tes**2 + nu_met**2)
    return (mu - 1.1**nu_bkg)**2 + (mu + 2.1**nu_tt)**2 + mu * mu + 2 * penalty


class likelihoodFit:
    """
    A class to perform likelihood fits using SciPy's L-BFGS-B minimizer.
    We implement a simplified Hessian approximation:
      - If a parameter is NOT pinned at boundary, do central difference.
      - If pinned at EXACTLY one side, do one-sided difference.
      - If pinned on both sides or if for cross partials we see two pinned
        parameters, set that Hessian entry to 0.
    """

    def __init__(self, function):
        # Store the user-provided likelihood function
        self.function = function

        # Parameter boundaries in the same style as your original code
        self.parameterBoundaries = {
            "mu": (0.0, None),
            "nu_bkg": (-10., 10.),
            "nu_tt": (-10., 10.),
            "nu_diboson": (-4., 4.),
            "nu_jes": (-10., 10.),
            "nu_tes": (-10., 10.),
            "nu_met": (0., 5.),
        }

        # Tolerance for the minimizer -- sets 'ftol' in SciPy
        self.tolerance = 0.1  
        # Step size for finite differences in L-BFGS-B
        self.eps = 0.1        
        # Separate step for Hessian approximation
        self.hessian_step = 0.2

        # Results
        self.q_mle = None
        self.parameters_mle = None
        self.covariance = None

        # If we consider "pinned" if x[i] is within this tolerance of the boundary
        self.boundary_tol = 1e-7

    def is_at_lower_bound(self, x_val, bound):
        """Check if x_val is at or very close to the lower boundary."""
        low, _ = bound
        if low is None:
            return False
        return (x_val - low) < self.boundary_tol

    def is_at_upper_bound(self, x_val, bound):
        """Check if x_val is at or very close to the upper boundary."""
        _, high = bound
        if high is None:
            return False
        return (high - x_val) < self.boundary_tol

    def approximate_hessian(self, func, x, step, bounds):
        """
        Numerically approximate the Hessian with simplified boundary handling:
          - If parameter i is strictly inside the domain, do central difference.
          - If pinned at exactly one boundary, do one-sided difference.
          - If pinned at both boundaries, set that diagonal to 0.
          - For cross partials, if either i or j is pinned at both boundaries
            (or if i and j are pinned in any combination), set to 0.

        'func'  : objective function
        'x'     : point at which we approximate the Hessian
        'step'  : nominal finite-difference step size
        'bounds': list of (low, high) for each param
        """
        n = len(x)
        hess = np.zeros((n, n), dtype=float)
        f0 = func(x)

        # Precompute pinned info
        pinned_info = []
        for i in range(n):
            at_low = self.is_at_lower_bound(x[i], bounds[i])
            at_high = self.is_at_upper_bound(x[i], bounds[i])
            pinned_info.append((at_low, at_high))

        # 1) Diagonal terms
        for i in range(n):
            at_low, at_high = pinned_info[i]

            if at_low and at_high:
                # pinned on both sides => no movement => second deriv = 0
                hess[i, i] = 0.0
                continue

            if (not at_low) and (not at_high):
                # fully inside => central difference
                x_fwd = x.copy()
                x_bwd = x.copy()
                x_fwd[i] += step
                x_bwd[i] -= step
                f_fwd = func(x_fwd)
                f_bwd = func(x_bwd)
                hess[i, i] = (f_fwd - 2.0*f0 + f_bwd) / (step**2)
            elif at_low and (not at_high):
                # pinned at lower bound => forward difference only
                x_f1 = x.copy()
                x_f2 = x.copy()
                x_f1[i] += step
                x_f2[i] += 2.0*step
                f_f1 = func(x_f1)
                f_f2 = func(x_f2)
                hess[i, i] = (f_f2 - 2.0*f_f1 + f0) / (step**2)
            elif at_high and (not at_low):
                # pinned at upper bound => backward difference only
                x_b1 = x.copy()
                x_b2 = x.copy()
                x_b1[i] -= step
                x_b2[i] -= 2.0*step
                f_b1 = func(x_b1)
                f_b2 = func(x_b2)
                hess[i, i] = (f_b2 - 2.0*f_b1 + f0) / (step**2)

        # 2) Off-diagonal terms
        for i in range(n):
            for j in range(i+1, n):
                # If either param is pinned at both sides, skip
                i_pinned_low, i_pinned_high = pinned_info[i]
                j_pinned_low, j_pinned_high = pinned_info[j]
                if (i_pinned_low and i_pinned_high) or (j_pinned_low and j_pinned_high):
                    # Both boundaries pinned => set cross partial to 0
                    hess[i, j] = 0.0
                    hess[j, i] = 0.0
                    continue

                # If *both* i and j are strictly inside, do normal 4-point difference
                # If either is pinned at exactly one boundary, let's skip or set to 0
                #   (since we said "Ignore the case where more than one param is pinned")
                i_strictly_in = (not i_pinned_low and not i_pinned_high)
                j_strictly_in = (not j_pinned_low and not j_pinned_high)
                if i_strictly_in and j_strictly_in:
                    x_pp = x.copy(); x_pm = x.copy()
                    x_mp = x.copy(); x_mm = x.copy()
                    x_pp[i] += step; x_pp[j] += step
                    x_pm[i] += step; x_pm[j] -= step
                    x_mp[i] -= step; x_mp[j] += step
                    x_mm[i] -= step; x_mm[j] -= step
                    f_pp = func(x_pp)
                    f_pm = func(x_pm)
                    f_mp = func(x_mp)
                    f_mm = func(x_mm)
                    hess_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
                    hess[i, j] = hess_ij
                    hess[j, i] = hess_ij
                else:
                    # If either i or j is pinned (exactly one boundary),
                    # to keep code simpler we do no cross partial => set 0
                    hess[i, j] = 0.0
                    hess[j, i] = 0.0

        return hess

    def fit(
        self,
        start_mu=1.0,
        start_nu_bkg=0.0,
        start_nu_tt=0.0,
        start_nu_diboson=0.0,
        start_nu_jes=0.0,
        start_nu_tes=0.0,
        start_nu_met=0.0
    ):
        """
        Fit the global minimum using SciPy's L-BFGS-B.
        Then approximate the Hessian with simplified boundary handling.
        """

        logger.info("Fit global minimum with L-BFGS-B")

        # 1) Set initial guess
        x0 = np.array([
            start_mu,
            start_nu_bkg,
            start_nu_tt,
            start_nu_diboson,
            start_nu_jes,
            start_nu_tes,
            start_nu_met
        ], dtype=float)

        # 2) Build the bounds in the same order
        bounds_ordered = [
            self.parameterBoundaries["mu"],
            self.parameterBoundaries["nu_bkg"],
            self.parameterBoundaries["nu_tt"],
            self.parameterBoundaries["nu_diboson"],
            self.parameterBoundaries["nu_jes"],
            self.parameterBoundaries["nu_tes"],
            self.parameterBoundaries["nu_met"],
        ]

        # 3) Objective in SciPy style
        def objective(x):
            return self.function(
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                x[6]
            )

        logger.info("Starting minimization...")
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds_ordered,
            options={
                "finite_diff_rel_step": self.eps,
                "ftol": self.tolerance,
                "disp": True
            }
        )
        logger.info("Minimization done.")

        self.q_mle = res.fun
        self.parameters_mle = res.x

        logger.info(f"Minimization success: {res.success}, status={res.status}")
        logger.info(f"Message: {res.message}")
        logger.info(f"Function value at minimum: {res.fun}")
        logger.info(f"Solution: {res.x}")

        # Approximate Hessian with simplified boundary logic
        logger.info("Computing approximate Hessian with step=%f", self.hessian_step)
        hess = self.approximate_hessian(
            objective,
            res.x,
            step=self.hessian_step,
            bounds=bounds_ordered
        )

        try:
            self.covariance = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            self.covariance = None
            logger.warning("Hessian is singular or near-singular. Covariance not available.")

        logger.info("Hessian approximation completed.")
        logger.debug(f"Hessian:\n{hess}")
        if self.covariance is not None:
            logger.debug(f"Covariance (inverse Hessian):\n{self.covariance}")
    
        param_names = ["mu", "nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]
        values_dict = {name: val for name, val in zip(param_names, self.parameters_mle)}

        cov_dict = {}
        if self.covariance is not None:
            for i, name_i in enumerate(param_names):
                for j, name_j in enumerate(param_names):
                    cov_dict[(name_i, name_j)] = self.covariance[i, j]

        return self.q_mle, values_dict, cov_dict

        #return self.q_mle, self.parameters_mle, self.covariance

if __name__ == "__main__":
    from common.logger import get_logger
    logger = get_logger("DEBUG", logFile = None)
    # Demonstration code
    fitter = likelihoodFit(likelihood_test_function)

    # Run the fit method with default start values
    logger.info("Running demo fit with default start values:")
    q_mle, params, cov = fitter.fit()
    logger.info(f"Final function value (q_mle): {q_mle}")
    logger.info(f"Best-fit parameters: {params}")
    logger.info(f"Approximate covariance:\n{cov}")
