import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.user as user
from common.likelihoodFit import likelihoodFit


def penalty(nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    return nu_ztautau**2+nu_tt**2+nu_diboson**2+nu_jes**2+nu_tes**2+nu_met**2

def dSigmaOverDSigmaSM(h5f, mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    # Classifier
    p_mc = h5f["MultiClassifier_predict"]
    # JES
    DA_pnn_jes = h5f["JES_DeltaA"]
    nu_A = self.models['JES'].nu_A((nu_jes,))
    p_pnn_jes = np.exp( np.dot(DA_pnn_jes, nu_A))
    # TES
    # to be implemented
    # MET
    # to be implemented
    # RATES
    f_ztautau_rate = (1+alpha_ztautau)**nu_ztautau
    f_tt_rate = (1+alpha_tt)**nu_tt
    f_diboson_rate = (1+alpha_diboson)**nu_diboson
    ##
    return ((mu*p_mc[:,0] + p_mc[:,1]*f_ztautau_rate + p_mc[:,2]*f_tt_rate + p_mc[:,3]*f_diboson_rate) / p_mc.sum(axis=1))*p_pnn_jes

def uTerm(mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    # define h5f file here
    weights = h5f["Weight"]
    dSoDS_toy = dSigmaOverDSigmaSM(h5f_toy, mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
    dSoDS_sim = dSigmaOverDSigmaSM(h5f_toy, mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
    incS = (weights[:]*(1-dSoDS_sim)).sum()
    penalty = penalty(nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
    return -2 *(incS+np.log(dSoDS_toy).sum())+penalty

def uTerm_asimov(mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met):
    # define h5f file here
    weights = h5f["Weight"]
    dSoDS_sim = dSigmaOverDSigmaSM(h5f_toy, mu, nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
    incS = (weights[:]*(1-dSoDS_sim)).sum()
    penalty = penalty(nu_ztautau, nu_tt, nu_diboson, nu_jes, nu_tes, nu_met)
    return -2 *(incS+(weights[:]*np.log(dSoDS_sim)).sum())+penalty


fit = likelihoodFit(uTerm_asimov)
q_mle, parameters_mle = fit.fit()
