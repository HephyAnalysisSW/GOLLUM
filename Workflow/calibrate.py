import pickle
import sys
import os
import argparse
import numpy as np
sys.path.insert(0, "..")
import common.syncer
import common.user as user

def getTuple(filename):
    if not os.path.exists(filename):
        return None

    with open(filename, 'rb') as file:
        fitResult = pickle.load(file)

    mu = fitResult["mu"]
    nu_jes = fitResult["nu_jes"]
    nu_tes = fitResult["nu_tes"]
    nu_met = fitResult["nu_met"]
    return (mu, nu_jes, nu_tes, nu_met)

def list_files_in_directory(directory_path):
    toyList = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            passedAll = True
            for veto in ["nominal", "ttbar", "htautau", "diboson", "ztautau"]:
                if veto in file:
                    passedAll = False
            if passedAll:
                toyList.append(file)
    return toyList

################################################################################
sysList = ["nominal.h5"]
toyPath = "/scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/lowMT_VBFJet/"
sysList += list_files_in_directory(toyPath)
fitResultDir = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference/"

muAll = [
    0.1,
    0.3,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
]

calibrationValues = np.array([])
muMeasured = np.array([])
nuJES = np.array([])
nuTES = np.array([])
nuMET = np.array([])
for sys in sysList:
    postfix = "" if sys == "nominal.h5" else "_"+sys.replace(".h5", "")
    for mu in muAll:
        integer_part = int(mu)
        fractional_part = abs(mu - integer_part)
        fractional_str = f"{fractional_part:.3f}"[2:]
        mustring = f"{integer_part}p{fractional_str}"
        filename = fitResultDir+"fitResult.config_reference_mu_"+mustring+postfix+".pkl"
        tuple = getTuple(filename)
        if tuple is None:
            continue

        (muMeas, nu_jes, nu_tes, nu_met) = tuple
        calibrationValues = np.append(calibrationValues, mu-muMeas)
        muMeasured = np.append(muMeasured, muMeas)
        nuJES = np.append(nuJES, nu_jes)
        nuTES = np.append(nuTES, nu_tes)
        nuMET = np.append(nuMET, nu_met)

data_to_save = {
    "calibration": calibrationValues,
    "mu": muMeasured,
    "nu_jes": nuJES,
    "nu_tes": nuTES,
    "nu_met": nuMET,
}
calibrationFile = os.path.join(user.output_directory, "calibration.pkl")
with open(calibrationFile, "wb") as f:
    pickle.dump(data_to_save, f)

print("Saved calibration:", calibrationFile)
