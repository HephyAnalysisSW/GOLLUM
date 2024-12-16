from iminuit import Minuit


# Likelihood class
class likelihoodFit:
    def __init__(self, features):
        self.features = features

    def likelihood_function(self, mu, nu1, nu2):
        penalty = 0.5*(nu1**2+nu2**2)
        return (mu - 1.1**nu1)**2 + (mu + 2.1**nu2)**2 + mu * mu + penalty

    def fit(self):
        print("Fit global minimum")
        errordef = Minuit.LEAST_SQUARES
        m = Minuit(self.likelihood_function, mu=0.0, nu1=0.0, nu2=0.0)
        m.migrad()
        # print("Minimum value:", m.fval)
        # print("Best parameters:", m.values)
        return m.fval, m.values

    def scan(self):
        print("Scan signal strength")
        qPoints = []
        Nsteps = 100
        mumin = -5
        mumax = 5
        muList = [mumin+i*(mumax-mumin)/Nsteps for i in range(Nsteps)]
        errordef = Minuit.LEAST_SQUARES
        for mu in muList:
            fixed_mu = mu
            likelihood_fixedMu = lambda nu1, nu2: self.likelihood_function(fixed_mu, nu1, nu2)
            m = Minuit(likelihood_fixedMu, nu1=0.0, nu2=0.0)
            m.migrad()
            qPoints.append(m.fval)
        return qPoints, muList

    def impacts(self):
        print("Calculate impacts")

        # First find global Min and store MLE values
        errordef = Minuit.LEAST_SQUARES
        m = Minuit(self.likelihood_function, mu=0.0, nu1=0.0, nu2=0.0)
        m.migrad()
        q_mle = m.fval
        mu_mle = m.values["mu"]
        nu1_mle = m.values["nu1"]
        nu2_mle = m.values["nu2"]

        # Scan over nu1
        upper_nu1, lower_nu1 = None, None
        nu1_function = lambda nu1: abs(self.likelihood_function(mu_mle, nu1, nu2_mle)-q_mle-1.0)
        m_up = Minuit(nu1_function, nu1=nu1_mle+0.01)
        m_up.limits["nu1"] = (nu1_mle, None)
        m_up.migrad()
        upper_nu1 = m_up.values["nu1"]

        m_down = Minuit(nu1_function, nu1=nu1_mle-0.01)
        m_down.migrad()
        m_down.limits["nu1"] = (None, nu1_mle)
        lower_nu1 = m_down.values["nu1"]

        # Scan over nu2
        upper_nu2, lower_nu2 = None, None
        nu2_function = lambda nu2: abs(self.likelihood_function(mu_mle, nu1_mle, nu2)-q_mle-1.0)
        m_up = Minuit(nu2_function, nu2=nu2_mle+0.01)
        m_up.limits["nu2"] = (nu2_mle, None)
        m_up.migrad()
        upper_nu2 = m_up.values["nu2"]

        m_down = Minuit(nu2_function, nu2=nu2_mle-0.01)
        m_down.migrad()
        m_down.limits["nu2"] = (None, nu2_mle)
        lower_nu2 = m_down.values["nu2"]

        return (upper_nu1, lower_nu1), (upper_nu2, lower_nu2)


if __name__ == "__main__":
    fit = likelihoodFit(None)
    q_mle, parameters_mle = fit.fit()
    nu1_mle = parameters_mle["nu1"]
    nu2_mle = parameters_mle["nu2"]

    print(q_mle, parameters_mle)
    print("----------------")
    qPoints,muPoints = fit.scan()
    print(qPoints)
    print(muPoints)
    print("----------------")
    (upper_nu1, lower_nu1), (upper_nu2, lower_nu2) = fit.impacts()
    print(f"nu1 = {nu1_mle} + {upper_nu1-nu1_mle} - {nu1_mle-lower_nu1}")
    print(f"nu2 = {nu2_mle} + {upper_nu2-nu2_mle} - {nu2_mle-lower_nu2}")
