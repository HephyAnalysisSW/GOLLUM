#!/usr/bin/env python

import sys, os
import numpy as np
import ROOT
from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer
import common.helpers as helpers
import common.data_structure as data_structure

# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
argParser.add_argument("--modelDir", action="store", default="/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6",  help="Directory containing the trained TFMC model.")
args = argParser.parse_args()

# let us load a classifier! 
from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load(args.modelDir)

import common.datasets_hephy as datasets_hephy
# Load the data
data_loader = datasets_hephy.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split)

# systmatics data_loader: values = (tes, jes, met) where for jes/tes we have 1.01 is +1 sigma, 0.99 is -1 sigma, etc. and for MET, the value directly is the sigma (between 0 and 3)
#datasets_hephy.get_data_loader( selection='lowMT_VBFJet', values=(1.01,1.01,0))
# all available combinations are here: ls /scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/lowMT_VBFJet

# This is how you change the normalization uncertainty to +1 sigma: Example for ttbar
# nu_tt = 1
# alpha_bkg = 0.001
# alpha_tt = 0.02
# alpha_diboson = 0.25

#weights[labels==data_structure.label_encoding['ttbar']] = weights[labels==data_structure.label_encoding['ttbar']]*(1+self.alpha_tt)**nu_tt

max_batch = 1 if args.small else -1

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "postfit_plots", args.selection)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Initialize histogram with zero (one per label and variable)
histograms = {}
for var, options in data_structure.plot_options.items():
    histograms[var] = {}
    for label in data_structure.labels:
        binning = options['binning']
        options["bin_edges"] = np.linspace(binning[1], binning[2], binning[0] + 1)
        histograms[var][label] = np.zeros(binning[0])

# Loop over data batches and calculate predictions
total_batches = len(data_loader)
for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
    features, weights, labels = data_loader.split(batch)
    #predictions = tfmc.predict(features, ic_scaling=False)

    # Accumulate predicted probabilities for each class and label
    for label_idx, label in enumerate(data_structure.labels):
        true_indices = (labels == label_idx)
        for i_var, var in enumerate( data_structure.feature_names ):
            if np.any(true_indices):
                histogram_batch, _ = np.histogram(
                    features[true_indices, i_var],
                    bins=data_structure.plot_options[var]['bin_edges'],
                    weights=weights[true_indices]
                )
                histograms[var][label] += histogram_batch 

    if max_batch > 0 and i_batch + 1 >= max_batch:
        break

# Plot the predicted probabilities for each predicted class using ROOT
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Loop over variables 
for i_var, var in enumerate( data_structure.feature_names ):
    canvas = ROOT.TCanvas("","", 800, 600)
    canvas.SetLogy()
    # Create a histogram stack for visualization
    max_y = 0  # To set uniform Y-axis limits
    histograms_for_class = []

    for label_idx, label in enumerate(data_structure.labels):
        # Create the histogram for the current true label
        h = ROOT.TH1F( var, var, *data_structure.plot_options[var]['binning']) 
        for i, value in enumerate(histograms[var][label]):
            h.SetBinContent(i + 1, value)

        # Style the histogram
        h.SetLineColor(data_structure.plot_styles[label]['fill_color'])
        h.SetLineWidth(2)

        # Add to the stack and find max Y value
        max_y = max(max_y, h.GetMaximum())
        histograms_for_class.append(h)

    first=True
    for h in histograms_for_class:
        if first:
            h.Draw("HIST")
            first=False
        else:
            h.Draw("HIST SAME")

    # Add legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)
    for label_idx, label in enumerate(data_structure.labels):
        legend.AddEntry(
            histograms_for_class[label_idx],
            f"{label}",
            "l"
        )
    legend.Draw()

    # Save the canvas
    output_file = os.path.join(plot_directory, f"{var}.png")
    canvas.SaveAs(output_file)
    print(f"Saved plot for predicted class {output_file}")

common.syncer.sync()
