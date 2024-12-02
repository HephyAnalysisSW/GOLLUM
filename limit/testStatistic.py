


# q_{\mu} = \max_\ba u(\mathcal{D}|\mu,\ba) - \max_{\ba_0}u(\mathcal{D}|\mu_0,\ba_0),

# -\frac{1}{2}u(\mathcal{D}|\mu,\ba) = - \mathcal{L} \sigma(\mu,\ba) + \mathcal{L} \sigma(\text{SM}) + \sum_i^\text{N events} \log \frac{d\Sigma (x_i|\ba)}{d\Sigma(\text{SM})} - \frac{1}{2} \sum_k^K \alpha_k^2.

class testStatistic:
    def __init__():
        #
        #
        # lumi = const
        # getInclXS(mu, nu)
        # smXS = getInclXS(0, nu0)
        # getPenalty(nu)
        # getXSratio(features, mu, nu) <- This function should be configurable in order to stich multiple ML models together
        #       Comment from Robert: The x-sec value depends on the selection, not the features.
        #
        # Function to maximize with respect to nu
        # Function to scan over mu
        # return a funktion (of mu)
