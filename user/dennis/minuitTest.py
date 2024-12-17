from iminuit import Minuit


# Likelihood class
class likelihoodFit:
    def __init__(self, features):
        self.features = features

    def likelihood_function(self, mu, nu1, nu2):
        # this defines the function that is minimized
        penalty = 0.5*(nu1**2+nu2**2)
        return (mu - 1.1**nu1)**2 + (mu + 2.1**nu2)**2 + mu * mu + penalty

    def fit(self):
        # function to find the global minimum, minimizing mu and nus
        print("Fit global minimum")
        errordef = Minuit.LEAST_SQUARES
        m = Minuit(self.likelihood_function, mu=0.0, nu1=0.0, nu2=0.0)
        m.migrad()
        return m.fval, m.values

    def scan(self, Npoints=100, mumin=-5, mumax=5):
        # Scan over points of mu
        print("Scan signal strength")
        # First find global Min and store MLE values
        q_mle, parameters_mle = self.fit()
        # Now make scan over nu
        qDeltas = []
        muList = [mumin+i*(mumax-mumin)/Npoints for i in range(Npoints)]
        errordef = Minuit.LEAST_SQUARES
        for mu in muList:
            # Create a function that fixes mu and only uses the nu as arguments
            fixed_mu = mu
            likelihood_fixedMu = lambda nu1, nu2: self.likelihood_function(fixed_mu, nu1, nu2)
            m = Minuit(likelihood_fixedMu, nu1=0.0, nu2=0.0)
            m.migrad()
            qDeltas.append(m.fval-q_mle)
        return qDeltas, muList

    def impacts(self):
        print("Calculate impacts")
        # First find global Min and store MLE values
        q_mle, parameters_mle = self.fit()
        mu_mle = parameters_mle["mu"]
        nu1_mle = parameters_mle["nu1"]
        nu2_mle = parameters_mle["nu2"]

        # Define functions that fix all parameters but a single nu
        nu1_function = lambda nu1: abs(self.likelihood_function(mu_mle, nu1, nu2_mle)-q_mle-1.0)
        nu2_function = lambda nu2: abs(self.likelihood_function(mu_mle, nu1_mle, nu2)-q_mle-1.0)
        nu_functions = {
            "nu1": nu1_function,
            "nu2": nu2_function,
        }

        # Now go through each nu and find point where q - q_mle == 1
        # Do this from nu_mle to both sides to find lower and upper boundaries
        limits = {}
        for nuname in ["nu1", "nu2"]:
            upper, lower = None, None

            # Use **kwargs in order to dynamically change the parameters (nu1 -> nu2, ....)
            param_up = {nuname: parameters_mle[nuname] + 0.01}
            m_up = Minuit(nu_functions[nuname], **param_up)
            m_up.limits[nuname] = (parameters_mle[nuname], None)
            m_up.migrad()
            upper = m_up.values[nuname]

            param_down = {nuname: parameters_mle[nuname] - 0.01}
            m_down = Minuit(nu_functions[nuname], **param_down)
            m_down.migrad()
            m_down.limits[nuname] = (None, parameters_mle[nuname])
            lower = m_down.values[nuname]
            limits[nuname] = (lower, upper)
        return limits

if __name__ == "__main__":
    print("1. Simple fit")
    fit = likelihoodFit(None)
    q_mle, parameters_mle = fit.fit()
    nu1_mle = parameters_mle["nu1"]
    nu2_mle = parameters_mle["nu2"]
    print(f" - q at MLE: {q_mle}")
    print(f" - Parameters at MLE: {parameters_mle}")
    print("-------------------------------------")
    print("2. Scan over mu")
    deltaQ,muPoints = fit.scan()
    print(f" - mus = {muPoints}")
    print(f" - deltaQs = {deltaQ}")
    print("-------------------------------------")
    print("3. Impacts")
    limits = fit.impacts()
    print(f" - constrained parameter boundaries: {limits}")
