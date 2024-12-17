###
### MODES:
### - Best Fit (find best fit mu, nu)
### - scan over mu
### - impacts


# We need to construct max[u(mu, nu)] - max[u(mu', nu')]
# where - 0.5 u(mu,nu) = -L sigma(mu, nu) + L sigma(SM) + dSigmaOverDSigmaSMsum + 0.5*nu^2

# L sigma(mu,nu) = sum_simEvents weights (1+alpha_S)^(nu_s) * exp(nu_A Delta_A(x)) * mu [ where mu is 0 for backgrounds]
# L sigma(SM) = sum_simEvents weights
# --> -L sigma(mu, nu) + L sigma(SM) = sum_simEvents weight * (1 - [(1+alpha_S)^(nu_s) * exp(nu_A Delta_A(x)) * mu]  )


def dSigmaOverDSigmaSM( self, features, mu=1, nu_jes=0 ):
  p_mc = self.models['MultiClassifier'].predict(features)
  p_pnn_jes = self.models['JES'].predict(features, nu=(nu_jes,))
  f_ztautau_rate = (1+alpha_ztautau)**(nu_ztautau)
  f_tt_rate = (1+alpha_tt)^(nu_tt)
  f_diboson_rate = (1+alpha_diboson)^(nu_diboson)
  return (mu*p_mc[:,0]/(p_mc[:,1]*f_ztautau_rate + p_mc[:,2]*f_tt_rate + p_mc[:,3]*f_diboson_rate) + 1)*p_pnn_jes


def getU(sim_events, observed_events, mu, nu):
    for e in sim_events:
        inclusivePartMuNu = (1+alpha_S)^(nu_S) * exp(nu_A * e.Delta_A)
        if e.isSignal:
            inclusivePartMuNu *= mu

        inclusivePart += e.weight * (1 - inclusivePartMuNu)
    for e in observed_events:
        dSigmaPart = dSigmaOverDSigmaSM(e.features, mu, nu)
        ts_batch = np.log(dSigmaPart).sum()
    return -2* (inclusivePart+dSigmaPart)


for mu in muList:
    # First scan mu over the first term
    # maximize over
    maxU1 = maximize( getU(sim_events, observed_events, mu, nu) ) # max wrt nu
    # in maximization also remember mu_mle and nu_mle
    maxU2 = getU(sim_events, observed_events, mu_mle, nu_mle)
    # now build q
    q = maxU1 - maxU2


################################################################################
# IMPACTS

# For impacts we need mu_mle, nu_mle

# 1. Get q_mle with nu_mle
# 2. For each nu, find points where q - q_mle == 1
# 3. These points define the new 1 sigma interval

for nu in nu_list:
    nu_min = -5
    mu_max = 5
    # find point wehere
    q - q_mle == 1
