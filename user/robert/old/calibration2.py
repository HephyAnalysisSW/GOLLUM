import sys
import os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import ROOT
import numpy as np
import common.syncer  
import common.helpers

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

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
    n_split = 100
else:
    n_split = 1

from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6")

# Iterate through the dataset and get one batch for demonstration
loader = datasets_hephy.get_data_loader(selection="lowMT_VBFJet", n_split=n_split)
for batch in loader:
    data, weights, labels = loader.split(batch)
    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True))
    prob_tf = tfmc.predict(data, ic_scaling=dcr)
    break

region = "lowMT_VBFJet"

from ML.Calibration.MulticlassCalibration import MultiClassCalibration
mc_calib = MultiClassCalibration.load(f"/groups/hephy/cms/robert.schoefbeck/Challenge/models/Calibration/{region}/config_reference_v3/{region}/calibrator_multi.pkl")
from ML.Calibration.Calibration import Calibration
calib = Calibration.load(f"/groups/hephy/cms/robert.schoefbeck/Challenge/models/Calibration/{region}/config_reference_v2_calib/{region}/calibrator.pkl")

# Create a canvas with 2x2 pads
c = ROOT.TCanvas("c", "Calibration Plots", 1200, 900)
c.Divide(2, 2)

graphs_tf = []
graphs_xgb = []

if not dcr:
    # probability calibration: normalize weights for each class
    weights[labels == 0] /= np.sum(weights[labels == 0])
    weights[labels == 1] /= np.sum(weights[labels == 1])
    weights[labels == 2] /= np.sum(weights[labels == 2])
    weights[labels == 3] /= np.sum(weights[labels == 3])


dcr_range = {
    0: (0, 0.01),
    3: (0, 0.1),
}

stuff = []
for j in range(4):
    # Change to pad j+1
    c.cd(j+1)
    
    # Inline implementation of calibration graph for class j
    # Convert arrays to NumPy arrays
    #prob_tf_calib = mc_calib.predict(prob_tf)
    prob_tf_calib = calib.predict(prob_tf)
    values = np.asarray(prob_tf_calib[:, j], dtype=float).flatten()
    labs = np.asarray(labels, dtype=int).flatten()
    w = np.asarray(weights, dtype=float).flatten()
    assert len(values) == len(labs) == len(w), "values, labels, and weights must have the same length"

    v_min, v_max = values.min(), values.max()

    # Compute histograms for binning
    bin_sum_w, bin_edges = np.histogram(
        values,
        bins=nbins,
        range=(v_min, v_max),
        weights=w
    )
    mask_class = (labs == j)
    bin_sum_wclass, _ = np.histogram(
        values[mask_class],
        bins=nbins,
        range=(v_min, v_max),
        weights=w[mask_class]
    )
    bin_sum_wvalue, _ = np.histogram(
        values,
        bins=nbins,
        range=(v_min, v_max),
        weights=w * values
    )
    g_tf = ROOT.TGraph(nbins)
    for i in range(nbins):
        bin_left = bin_edges[i]
        bin_right = bin_edges[i+1]
        bin_center = 0.5 * (bin_left + bin_right)
        if bin_sum_w[i] > 0.0:
            x_mean = bin_sum_wvalue[i] / bin_sum_w[i] 
            frac_class = bin_sum_wclass[i] / bin_sum_w[i]
        else:
            x_mean = bin_center
            frac_class = 0.0
        g_tf.SetPoint(i, x_mean, frac_class)
    
    # Style the graph
    color = soft_colors[0]
    g_tf.SetLineColor(color)
    g_tf.SetMarkerColor(color)
    g_tf.SetMarkerStyle(20)
    g_tf.SetLineWidth(2)
    g_tf.SetTitle(";Mean Predicted Probability;Weighted True Probability")

    if dcr and j in dcr_range:
        frame = c.DrawFrame(dcr_range[j][0], dcr_range[j][0], dcr_range[j][1], dcr_range[j][1])
    else:
        frame = c.DrawFrame(0, 0, 1, 1)
    frame.GetXaxis().SetTitle("Mean Predicted Probability")
    frame.GetYaxis().SetTitle("Weighted True Probability")
    
    # Draw the graph
    g_tf.Draw("LPSAME")
    #g_tf.GetXaxis().SetRangeUser(0.0, 1.0)
    #g_tf.GetYaxis().SetRangeUser(0.0, 1.0)
    
    # Optionally draw the diagonal line y=x
    line = ROOT.TLine(0, 0, 1, 1)
    line.SetLineColor(ROOT.kGray+2)
    line.SetLineStyle(9)
    line.Draw("SAME")
    stuff.append(line)
    
    # Create and draw a legend
    legend = ROOT.TLegend(0.2, 0.75, 0.55, 0.85)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)
    legend.AddEntry(g_tf, "Model TF", "lp")
    legend.Draw("SAME")
    stuff.append(legend)
    
    graphs_tf.append(g_tf)

c.RedrawAxis()
c.Update()
# Save the canvas as PDF and PNG files (filenames depend on whether dcr is True)
if dcr:
    c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "") + "calib_dcr.pdf"))
    c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "") + "calib_dcr.png"))
else:
    c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "") + "calib_prob.pdf"))
    c.Print(os.path.join(user.plot_directory, "calib", ("small_" if small else "") + "calib_prob.png"))

common.helpers.copyIndexPHP(os.path.join(user.plot_directory, "calib"))
common.syncer.sync()

