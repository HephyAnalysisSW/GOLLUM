import ROOT
import pickle
import sys
import os
import argparse
import numpy as np
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import common.syncer
import common.user as user
from common.muCalibrator import muCalibrator
from common.calibrationPlotter import calibrationPlotter
from helpers import calculateScore,alphaToNu
ROOT.gROOT.SetBatch(ROOT.kTRUE)


def getKey(uncert):
    key = uncert
    if uncert == "met":
        key = "soft_met"
    if uncert in ["ttbar", "diboson", "bkg"]:
        key = uncert + "_scale"
    return key

parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument("--inflate", type=float, default=None)
args = parser.parse_args()

scoreFiles = [
    "output_mu_1p0_20250217_184059_cf51c989dc0640a1b0b8501aedc8ee8a.npz",
    "output_mu_2p0_20250217_184606_66b7dba84928488f98a1d8c4c01575b3.npz",
    "output_mu_3p0_20250217_183437_b08a7960adce404eb6e7b19926981cf7.npz",
    "output_mu_4p0_20250217_183522_e2f58a6535dd4fcaa8e59f5cbc62225b.npz",
    "output_mu_5p0_20250217_183522_5d6ac929b48a433c9cd1f7554c4dc8b5.npz",
]

for i,file in enumerate(scoreFiles):
    filename = os.path.join("/users/dennis.schwarz/HEPHY-uncertainty/submission/", file)
    data = np.load(filename, allow_pickle=True)
    if i==0:
        mu_true=data["mu_true"]
        mu_measured=data["mu_measured"]
        mu_measured_up=data["mu_measured_up"]
        mu_measured_down=data["mu_measured_down"]
        toypaths=data["toypaths"]
    else:
        mu_true = np.append(mu_true, data["mu_true"] )
        mu_measured = np.append(mu_measured, data["mu_measured"] )
        mu_measured_up = np.append(mu_measured_up, data["mu_measured_up"] )
        mu_measured_down = np.append(mu_measured_down, data["mu_measured_down"] )
        toypaths = np.append(toypaths, data["toypaths"])

if args.inflate is not None:
    uncert_factor = args.inflate
    for i in range(len(mu_measured)):
        mu_measured_up[i] = mu_measured[i] + uncert_factor * (mu_measured_up[i]-mu_measured[i])
        mu_measured_down[i] = mu_measured[i] - uncert_factor * (mu_measured[i]-mu_measured_down[i])

output_name = os.path.join( user.plot_directory, "ClosureTests", "Toys.pdf" )
p = calibrationPlotter(output_name)
p.setMus(mu_true, mu_measured, mu_measured_down, mu_measured_up)
p.draw()

score, average_width, coverage = calculateScore(mu_true, mu_measured_down, mu_measured_up)
print("SCORE =", score)
print("AVG. WIDTH =", average_width)
print("COVERAGE =", coverage)
print("-------------------------")
print("now colour points")

# sigma_binning = [
#     (-10, -3, ROOT.kRed),
#     ( -3, -2, ROOT.kRed)-1, 0, 1, 2, 3, 10]

for uncert in ["jes", "tes", "met", "bkg", "ttbar", "diboson"]:
    graph1D = ROOT.TGraph(len(mu_measured))
    graph2D = ROOT.TGraph2D(len(mu_measured))
    for i in range(len(mu_measured)):
        with open(toypaths[i].replace(".h5", ".pkl"), 'rb') as file:
            trueValues = pickle.load(file)
        alpha = trueValues[getKey(uncert)]
        sigma = alphaToNu(alpha, uncert)
        graph2D.SetPoint(i, mu_true[i], mu_measured[i], sigma)
        graph1D.SetPoint(i, sigma, mu_measured[i]-mu_true[i])

    c = ROOT.TCanvas("c","",0,0,600,400)
    # c.SetTheta(90.);
    # c.SetPhi(0.);
    ROOT.gStyle.SetPalette(1)
    graph2D.SetMarkerStyle(20)
    graph2D.Draw("pcol")
    plotname2d = os.path.join( user.plot_directory, "ClosureTests", f"Toys_2D_{uncert}.pdf" )
    c.Print(plotname2d)

    c = ROOT.TCanvas("c","",0,0,600,400)
    ROOT.gStyle.SetPalette(1)
    graph1D.SetMarkerStyle(20)
    graph1D.Draw("AP")
    graph1D.GetXaxis().SetTitle("#sigma_{"+uncert+"}")
    graph1D.GetYaxis().SetTitle("#mu_{measured} - #mu_{true}")
    plotname1d = os.path.join( user.plot_directory, "ClosureTests", f"Toys_1D_{uncert}.pdf" )
    c.Print(plotname1d)
