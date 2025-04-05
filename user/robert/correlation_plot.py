import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

import common.syncer
import common.helpers as helpers
import os
import numpy as np
import ROOT
import common.user as user

ROOT.gROOT.SetBatch(True)  # Run in batch mode so we don't pop up windows.
dir_path = os.path.dirname(os.path.realpath(__file__))

ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "correlation")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Create a hard-coded 7x7 covariance matrix
cov = np.array([
 [ 0.029,-0.001,-0.002,-0.001,-0.002,-0.001,-0.000],
 [-0.001, 0.388,-0.010,-0.003,-0.021,-0.001, 0.001],
 [-0.002,-0.010, 0.039,-0.005,-0.001,-0.000,-0.000],
 [-0.001,-0.003,-0.005, 0.021,-0.000, 0.000, 0.000],
 [-0.002,-0.021,-0.001,-0.000, 0.004, 0.000, 0.000],
 [-0.001,-0.001,-0.000, 0.000, 0.000, 0.001,-0.000],
 [-0.000, 0.001,-0.000, 0.000, 0.000,-0.000, 0.085],
 ]
)

# Compute the correlation matrix
std_devs = np.sqrt(np.diag(cov))
corr = cov / np.outer(std_devs, std_devs)

# Initialize ROOT
ROOT.gStyle.SetOptStat(0)

# Create a TH2F to hold the correlation matrix
n = 7
hist = ROOT.TH2F("", "", n, 0, n, n, 0, n)

# Fill the histogram
for i in range(n):
    for j in range(n):
        hist.SetBinContent(i + 1, j + 1, corr[i, j])

# Draw the histogram with color palette
canvas = ROOT.TCanvas("c", "c", 600, 600)
canvas.SetRightMargin(0.18)
canvas.SetTopMargin(0.18)
hist.SetMarkerSize(1.2)
ROOT.gStyle.SetPaintTextFormat(".2f")
ROOT.gStyle.SetHistMinimumZero(False) 
hist.Draw("COLZ TEXT")
canvas.Print(os.path.join( plot_directory, "correlation.png"))
canvas.Print(os.path.join( plot_directory, "correlation.pdf"))
common.syncer.sync()
