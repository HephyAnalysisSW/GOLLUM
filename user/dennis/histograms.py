import ROOT
import sys
import numpy as np
from array import array
from math import sqrt
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.common as common

npHistograms = {}
processes = []
for process in processes+["test", "nominal"]:
    npHistograms[process] = {}
    for feature in common.feature_names:
        nBins, lower, upper = common.plot_options[feature]['binning']
        # setup numpy hist
        binning = np.linspace(lower, upper, nBins+1)
        hist, binedges = np.histogram( np.array([]), binning, weights=np.array([]) )
        sumw2, binedges = np.histogram( np.array([]), binning, weights=np.array([]) )
        npHistograms[process][feature] = {
            "hist": hist,
            "sumw2": sumw2,
            "binning": binedges,
        }

def convertNPtoROOT( npHist, binedges, sumw2, name, axistitle):
    hist = ROOT.TH1F(name, axistitle, len(binedges)-1, array('d', binedges))
    for i, content in enumerate(npHist):
        hist.SetBinContent(i+1, content)
        if sumw2 is not None:
            hist.SetBinError(i+1, sqrt(sumw2[i]) )
    return hist
