import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')

import ROOT
import numpy as np
import common.syncer  
import common.helpers

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

def create_calibration_graph(values, labels, weights,
                             class_of_interest=0, nbins=10, dcr=False):
    """
    Create a TGraph for a weighted "calibration-like" plot of a single class among multiple classes.

    In each bin, we compute:
      - x = ( sum_{i in bin}[w_i * values_i] ) / ( sum_{i in bin}[w_i] )
            (the weighted mean of 'values' in that bin)
      - y = ( sum_{i in bin}[w_i] where label_i == class_of_interest ) / ( sum_{i in bin}[w_i] )
            (the weighted fraction of the chosen class in that bin)

    The bin range [v_min, v_max] is determined from the data's min and max.
    If a bin is empty (no weights), we place x at the bin center and set y=0.

    Parameters
    ----------
    values : np.ndarray of shape (N,)
        The quantity you want to place on the x-axis (not necessarily in [0,1]).
    labels : np.ndarray of shape (N,)
        Integer class labels for each event (e.g. 0,1,2,3,...).
    weights : np.ndarray of shape (N,)
        Event weights (nonnegative).
    class_of_interest : int
        Which label/class you want to measure the fraction of in each bin.
    nbins : int
        Number of bins in [min(values), max(values)].

    Returns
    -------
    graph : ROOT.TGraph
        A TGraph with nbins points: (weighted mean of `values` in bin, fraction of class_of_interest in bin).
    """
    # Ensure arrays are NumPy arrays
    values = np.asarray(values, dtype=float).flatten()
    labels = np.asarray(labels, dtype=int).flatten()
    weights = np.asarray(weights, dtype=float).flatten()
    assert len(values) == len(labels) == len(weights), \
        "values, labels, and weights must all have the same length."

    if len(values) == 0:
        # Edge case: no data
        return ROOT.TGraph(0)

    # Determine the bin edges from the data range
    v_min, v_max = values.min(), values.max()
    if v_min == v_max:
        # All values are identical -> trivial graph with 1 point
        fraction = float(np.sum(weights[labels == class_of_interest])) / np.sum(weights) \
                   if np.sum(weights) > 0.0 else 0.0
        g = ROOT.TGraph(1)
        g.SetPoint(0, v_min, fraction)
        return g

    # Sum of all weights in each bin
    bin_sum_w, bin_edges = np.histogram(
        values,
        bins=nbins,
        range=(v_min, v_max),
        weights=weights
    )

    # Sum of weights for 'class_of_interest' in each bin
    mask_class = (labels == class_of_interest)
    bin_sum_wclass, _ = np.histogram(
        values[mask_class],
        bins=nbins,
        range=(v_min, v_max),
        weights=weights[mask_class] if not dcr else weights[mask_class]
    )

    # Sum of (weight * value) for computing weighted mean of x
    bin_sum_wvalue, _ = np.histogram(
        values,
        bins=nbins,
        range=(v_min, v_max),
        weights=weights * values if not dcr else weights * values
    )

    # Create a TGraph with nbins points
    graph = ROOT.TGraph(nbins)

    # For each bin, compute the x-value (weighted mean) and y-value (fraction in class_of_interest)
    for i in range(nbins):
        # Bin edges
        bin_left = bin_edges[i]
        bin_right = bin_edges[i+1]
        bin_center = 0.5*(bin_left + bin_right)

        w_tot = bin_sum_w[i]
        if w_tot > 0.0:
            x_mean = bin_sum_wvalue[i] / w_tot
            frac_class = bin_sum_wclass[i] / w_tot
        else:
            # Empty bin -> place x at the center, y=0
            x_mean = bin_center
            frac_class = 0.0

        graph.SetPoint(i, x_mean, frac_class)

    return graph

import common.user as user
import common.data_structure as data_structure
import common.selections as selections
import common.datasets_hephy as datasets_hephy

small = True
# Calibrate DCR or Prob?
dcr = True
nbins = 50

soft_colors = [
    ROOT.TColor.GetColor("#779ECB"),  # Soft blue
    ROOT.TColor.GetColor("#03C03C"),  # Teal green
    ROOT.TColor.GetColor("#B39EB5"),  # Light purple
    ROOT.TColor.GetColor("#FFB347"),  # Soft orange
    ROOT.TColor.GetColor("#FFD1DC"),  # Pastel pink
    ROOT.TColor.GetColor("#AEC6CF"),  # Muted cyan
    ROOT.TColor.GetColor("#CFCFC4"),  # Light gray
    ROOT.TColor.GetColor("#77DD77")   # Pastel green
]


if small:
    n_split=100
else:
    n_split = 1

from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6")

#from ML.XGBMC.XGBMC import XGBMC
#xgbmc = XGBMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/XGBMC/lowMT_VBFJet/xgb_v1/v1")

# Iterate through the dataset
loader = datasets_hephy.get_data_loader(selection="lowMT_VBFJet", n_split=n_split)
for batch in loader:
    data, weights, labels = loader.split(batch)
    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )

    prob_tf = tfmc.predict(data, ic_scaling = dcr)
    #prob_xgb = xgbmc.predict(data, ic_scaling = dcr)

    break

region = "lowMT_VBFJet"

from ML.Calibration.MulticlassCalibration import MultiClassCalibration
mc_calib =  MultiClassCalibration.load(f"/groups/hephy/cms/robert.schoefbeck/Challenge/models/Calibration/{region}/config_reference_v3/{region}/calibrator_multi.pkl")
from ML.Calibration.Calibration import Calibration
calib    = Calibration.load(f"/groups/hephy/cms/robert.schoefbeck/Challenge/models/Calibration/{region}/config_reference_v2_calib/{region}/calibrator.pkl")

# Create a canvas with 2x2 pads
c = ROOT.TCanvas("c", "Calibration Plots", 1200, 900)
c.Divide(2, 2)

# We'll store TGraph pointers for easy reference (optional)
graphs_tf  = []
graphs_xgb = []

if not dcr:
    # probability calibration
    weights[labels==0]/=np.sum(weights[labels==0])
    weights[labels==1]/=np.sum(weights[labels==1])
    weights[labels==2]/=np.sum(weights[labels==2])
    weights[labels==3]/=np.sum(weights[labels==3])

stuff = []
for j in range(4):
    # Go to pad j+1
    c.cd(j+1)

    # Create graphs
    g_tf  = create_calibration_graph(prob_tf[:, j], labels, weights, class_of_interest=j, nbins=nbins, dcr=dcr)

    color = soft_colors[0]

    # Style the graphs
    g_tf.SetLineColor(color)
    g_tf.SetMarkerColor(color)
    g_tf.SetMarkerStyle(20)
    g_tf.SetLineWidth(2)

    # Give them a title and axis labels
    #title_str = f"Calibration Plot - Column {data_structure.label_encoding[j]};Mean Predicted Probability;Weighted True Probability"
    g_tf.SetTitle(";Mean Predicted Probability;Weighted True Probability")

    # Draw the first graph
    g_tf.Draw("ALP")
    # Ensure the axis range is [0,1] in both x and y
    g_tf.GetXaxis().SetRangeUser(0.0, 1.0)
    g_tf.GetYaxis().SetRangeUser(0.0, 1.0)

    # Optionally draw diagonal line y=x
    line = ROOT.TLine(0, 0, 1, 1)
    line.SetLineColor(ROOT.kGray+2)
    line.SetLineStyle(9)
    line.Draw("SAME")
    stuff.append(line)

    # Create a small legend (optional)
    legend = ROOT.TLegend(0.2, 0.75, 0.55, 0.85)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)
    legend.AddEntry(g_tf,  "Model TF",  "lp")
    legend.Draw("SAME")
    stuff.append(legend)

    # Keep references
    graphs_tf.append(g_tf)

c.RedrawAxis()
c.Update()
# You can save the canvas as an image/PDF:
c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "")+("calib_prob.pdf" if not dcr else "calib_dcr.pdf")))
c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "")+("calib_prob.png" if not dcr else "calib_dcr.png")))
common.helpers.copyIndexPHP( os.path.join(user.plot_directory, "calib" ) )
common.syncer.sync()
