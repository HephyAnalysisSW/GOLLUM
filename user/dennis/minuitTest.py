import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user as user
from common.likelihoodFit import likelihoodFit

print("1. Simple fit")
fit = likelihoodFit()
q_mle, parameters_mle = fit.fit()
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
print("-------------------------------------")
print("4. plug in custom function to minimize")

def f_custom(mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    penalty = 0.5*(nu_ztautau**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2)
    return (mu+0.2 - 1.1**nu_ztautau)**2 + (mu+0.2 + 2.1**nu_tt)**2 + (mu+0.2) * (mu+0.2) + penalty

fit_custom = likelihoodFit(f_custom)
q_mle, parameters_mle = fit_custom.fit()
print(f" - q at MLE: {q_mle}")
print(f" - Parameters at MLE: {parameters_mle}")
