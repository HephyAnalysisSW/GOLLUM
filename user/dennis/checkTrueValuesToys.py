import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import numpy as np
import os
import pickle
from tqdm import tqdm
import ROOT
import common.user as user
import common.syncer

import argparse
parser = argparse.ArgumentParser(description="ML inference.")
args = parser.parse_args()


def list_files_in_directory(directory_path):
    toyList = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            passedAll = True
            for veto in ["ttbar", "htautau", "diboson", "ztautau"]:
                if veto in file:
                    passedAll = False
            if passedAll and ".h5" in file:
                toyList.append(os.path.join(directory_path, file))
    return toyList


directories = [
    "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_1/",
    "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_2/",
    "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_3/",
    "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_4/",
    "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_5/",
]

toys = []

for d in directories:
    toys += list_files_in_directory(d)

mu = []
nu_jes = []
nu_tes = []
nu_met = []
nu_ttbar = []
nu_diboson = []
nu_bkg = []

h_jes = ROOT.TH1F("jes", "", 21, 0.9, 1.11)
h_tes = ROOT.TH1F("tes", "", 21, 0.9, 1.11)
h_met = ROOT.TH1F("met", "", 21, -1, 5.1)
h_ttbar = ROOT.TH1F("ttbar", "", 21, 0.8, 1.21)
h_diboson = ROOT.TH1F("diboson", "", 21, 0.0, 2.01)
h_bkg = ROOT.TH1F("bkg", "", 21, 0.99, 1.011)

for i in tqdm(range(len(toys))):
    # Load sample and settings
    with open(toys[i].replace(".h5", ".pkl"), 'rb') as file:
        trueValues = pickle.load(file)

    mu.append(trueValues["mu"])
    nu_jes.append(trueValues["jes"])
    nu_tes.append(trueValues["tes"])
    sign_met = 1.0 if i%2==0 else -1.0
    nu_met.append(sign_met*trueValues["soft_met"])
    nu_ttbar.append(trueValues["ttbar_scale"])
    nu_diboson.append(trueValues["diboson_scale"])
    nu_bkg.append(trueValues["bkg_scale"])

    h_jes.Fill(trueValues["jes"])
    h_tes.Fill(trueValues["tes"])
    h_met.Fill(trueValues["soft_met"])
    h_ttbar.Fill(trueValues["ttbar_scale"])
    h_diboson.Fill(trueValues["diboson_scale"])
    h_bkg.Fill(trueValues["bkg_scale"])

print("<MU> =", np.mean(mu), "+-", np.std(mu, ddof=1))
print("<jes> =", np.mean(nu_jes), "+-", np.std(nu_jes, ddof=1))
print("<tes> =", np.mean(nu_tes), "+-", np.std(nu_tes, ddof=1))
print("<met> =", np.mean(nu_met), "+-", np.std(nu_met, ddof=1))
print("<ttbar> =", np.mean(nu_ttbar), "+-", np.std(nu_ttbar, ddof=1))
print("<diboson> =", np.mean(nu_diboson), "+-", np.std(nu_diboson, ddof=1))
print("<bkg> =", np.mean(nu_bkg), "+-", np.std(nu_bkg, ddof=1))

c = ROOT.TCanvas()
c.Divide(2,3)
c.cd(1)
h_jes.Draw("HIST")
c.cd(2)
h_tes.Draw("HIST")
c.cd(3)
h_met.Draw("HIST")
c.cd(4)
h_ttbar.Draw("HIST")
c.cd(5)
h_diboson.Draw("HIST")
c.cd(6)
h_bkg.Draw("HIST")
c.Print(user.plot_directory+"/gauss/test.pdf")
