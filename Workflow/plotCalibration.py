import ROOT
import pickle
import sys
import os
import argparse
import numpy as np
sys.path.insert(0, "..")
import common.syncer
import common.user as user
from common.muCalibrator import muCalibrator

ROOT.gROOT.SetBatch(ROOT.kTRUE)
################################################################################

calibrator = muCalibrator("/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/calibration.pkl")

Npoints = 1000

ranges = {
    "mu": (0.0, 3.0),
    "nu_jes": (-3.0, 3.0),
    "nu_tes": (-3.0, 3.0),
    "nu_met": (0.0, 3.0),
}

pairs = [
    ("mu","nu_jes"),
    ("mu","nu_tes"),
    ("mu","nu_met"),
    ("nu_jes","nu_tes"),
    ("nu_jes","nu_met"),
    ("nu_tes","nu_met"),
]


for (p1, p2) in pairs:
    stepsize1 = (ranges[p1][1]-ranges[p1][0])/Npoints
    stepsize2 = (ranges[p2][1]-ranges[p2][0])/Npoints
    list1 = [ ranges[p1][0] + i*stepsize1 for i in range(Npoints) ]
    list2 = [ ranges[p2][0] + i*stepsize2 for i in range(Npoints) ]

    hist = ROOT.TH2D(p1+"_"+p2, "", len(list1), list1[0], list1[-1], len(list2), list2[0], list2[-1])

    for i, low1 in enumerate(list1):
        for j, low2 in enumerate(list2):
            val1 = low1+0.5*stepsize1
            val2 = low2+0.5*stepsize2

            mu = 1.0
            nu_jet = 0.0
            nu_tes = 0.0
            nu_met = 0.0

            if p1 == "mu":
                mu = val1
            elif p1 == "nu_jes":
                nu_jes = val1
            elif p1 == "nu_tes":
                nu_tes = val1
            elif p1 == "nu_met":
                nu_met = val1

            if p2 == "mu":
                mu = val2
            elif p2 == "nu_jes":
                nu_jes = val2
            elif p2 == "nu_tes":
                nu_tes = val2
            elif p2 == "nu_met":
                nu_met = val2

            mu_corrected = calibrator.getMu(mu=mu, nu_jes=nu_jes, nu_tes=nu_tes, nu_met=nu_met)
            calibration = mu_corrected - mu
            if abs(calibration) > 1.5:
                print(mu, nu_jes, nu_tes, nu_met)
            hist.SetBinContent(i+1, j+1, calibration)

    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas(p1+"_"+p2, "", 600, 600)
    ROOT.gPad.SetRightMargin(.15)
    hist.GetXaxis().SetTitle(p1)
    hist.GetYaxis().SetTitle(p2)
    hist.GetZaxis().SetRangeUser(-2, 2)
    hist.Draw("COLZ")
    plotname = os.path.join( user.plot_directory, "Calibration", "Scan_"+p1+"_"+p2+".pdf")
    c.Print(plotname)
