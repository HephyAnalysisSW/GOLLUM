import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')

import ROOT
import numpy as np
import common.syncer  

def create_calibration_graph(values, labels, weights,
                             class_of_interest=0, nbins=10):
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
        weights=weights[mask_class]
    )

    # Sum of (weight * value) for computing weighted mean of x
    bin_sum_wvalue, _ = np.histogram(
        values,
        bins=nbins,
        range=(v_min, v_max),
        weights=weights * values
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
import common.datasets_hephy as datasets

# Calibrate DCR or Prob?
dcr = True
nbins = 100

from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6")

from ML.XGBMC.XGBMC import XGBMC
xgbmc = XGBMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/XGBMC/lowMT_VBFJet/xgb_v1/v1")

# Iterate through the dataset
loader = datasets.get_data_loader(selection="lowMT_VBFJet", n_split=1)
for batch in loader:
    data, weights, labels = loader.split(batch)
    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )

    prob_tf = tfmc.predict(data, ic_scaling = dcr)
    prob_xgb = xgbmc.predict(data, ic_scaling = dcr)

    break

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
    g_tf  = create_calibration_graph(prob_tf[:, j], labels, weights, class_of_interest=j, nbins=nbins)
    g_xgb = create_calibration_graph(prob_xgb[:, j], labels, weights, class_of_interest=j, nbins=nbins)

    # Style the graphs
    g_tf.SetLineColor(ROOT.kRed)
    g_tf.SetMarkerColor(ROOT.kRed)
    g_tf.SetMarkerStyle(20)
    g_tf.SetLineWidth(2)

    g_xgb.SetLineColor(ROOT.kBlue)
    g_xgb.SetMarkerColor(ROOT.kBlue)
    g_xgb.SetMarkerStyle(21)
    g_xgb.SetLineWidth(2)

    # Give them a title and axis labels
    title_str = f"Calibration Plot - Column {data_structure.label_encoding[j]};Mean Predicted Probability;Weighted True Probability"
    g_tf.SetTitle(title_str)

    # Draw the first graph
    g_tf.Draw("ALP")
    # Ensure the axis range is [0,1] in both x and y
    g_tf.GetXaxis().SetRangeUser(0.0, 1.0)
    g_tf.GetYaxis().SetRangeUser(0.0, 1.0)

    # Draw the second graph on top
    g_xgb.Draw("LP SAME")

    # Optionally draw diagonal line y=x
    line = ROOT.TLine(0, 0, 1, 1)
    line.SetLineColor(ROOT.kGray+2)
    line.SetLineStyle(2)
    line.Draw("SAME")
    stuff.append(line)

    # Create a small legend (optional)
    legend = ROOT.TLegend(0.15, 0.75, 0.45, 0.85)
    legend.AddEntry(g_tf,  "Model TF",  "lp")
    legend.AddEntry(g_xgb, "Model XGB", "lp")
    legend.Draw("SAME")
    stuff.append(legend)

    # Keep references
    graphs_tf.append(g_tf)
    graphs_xgb.append(g_xgb)

c.Update()
# You can save the canvas as an image/PDF:
c.Print(os.path.join(user.plot_directory, "calib", "calib_prob.pdf" if not dcr else "calib_dcr.pdf"))
c.Print(os.path.join(user.plot_directory, "calib", "calib_prob.png" if not dcr else "calib_dcr.png"))

common.syncer.sync()
