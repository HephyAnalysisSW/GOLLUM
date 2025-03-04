#!/usr/bin/env python

import sys, os
import importlib
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
argParser.add_argument("--n_bins", action="store", default=50, type=int, help="Number of bins.")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--config", action="store", default="tfmc", help="Which config?")
argParser.add_argument("--configDir", action="store", default="configs", help="Where is the config?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
argParser.add_argument("--modelDir", action="store", required=True, help="Directory containing the trained TFMC model.")
args = argParser.parse_args()

# Import the data and config
import common.datasets_hephy as datasets_hephy
config = importlib.import_module(f"{args.configDir}.{args.config}")

# Load the trained TFMC model
from TFMC import TFMC
tfmc = TFMC.load(args.modelDir)

# Load the data
data_loader = datasets_hephy.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split)

max_batch = 1 if args.small else -1

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "TFMC", args.selection, args.config, "predicted_probabilities")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Accumulate predicted probabilities histograms
pred_histograms = {}
bin_edges = {}


# Initialize histograms based on num_classes and labels
n_bins, x_min, x_max = args.n_bins, 0, 1
histograms = {c: {label: np.zeros(n_bins) for label in data_structure.labels} for c in range(tfmc.num_classes)}
bin_edges = np.linspace(x_min, x_max, n_bins + 1)

# Loop over data batches and calculate predictions
total_batches = len(data_loader)
for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
    features, weights, labels = data_loader.split(batch)
    predictions = tfmc.predict(features, ic_scaling=False)

    # Accumulate predicted probabilities for each class and label
    for c in range(tfmc.num_classes):
        for label_idx, label in enumerate(data_structure.labels):
            true_indices = (labels == label_idx)
            if np.any(true_indices):
                pred_histogram, _ = np.histogram(
                    predictions[true_indices, c],
                    bins=bin_edges,
                    weights=weights[true_indices]
                )
                histograms[c][label] += pred_histogram

    if max_batch > 0 and i_batch + 1 >= max_batch:
        break

## Normalize histograms so that they sum to 1 for each predicted class and label
#for c in range(tfmc.num_classes):
#    for label in data_structure.labels:
#        total_sum = histograms[c][label].sum()
#        histograms[c][label] /= total_sum if total_sum > 0 else 1

# Plot the predicted probabilities for each predicted class using ROOT
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Define line styles for each true label
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta]  # Define colors

# Loop over predicted classes
for c, predicted_class in enumerate(tfmc.classes):
    canvas = ROOT.TCanvas(f"c_predicted_{predicted_class}", f"Predicted Class: {predicted_class}", 800, 600)
    canvas.SetLogy()
    # Create a histogram stack for visualization
    max_y = 0  # To set uniform Y-axis limits
    histograms_for_class = []

    for label_idx, true_label in enumerate(data_structure.labels):
        # Create the histogram for the current true label
        h_true = ROOT.TH1F(
            f"h_pred_{predicted_class}_{true_label}",
            f"Predicted {predicted_class} ({true_label})",
            n_bins, x_min, x_max
        )
        for i, value in enumerate(histograms[c][true_label]):
            h_true.SetBinContent(i + 1, value)

        # Style the histogram
        h_true.SetLineColor(colors[label_idx % len(colors)])
        h_true.SetLineWidth(2)

        # Add to the stack and find max Y value
        max_y = max(max_y, h_true.GetMaximum())
        histograms_for_class.append(h_true)

    # Draw the histograms
    frame = ROOT.TH2F(
        f"frame_{predicted_class}",
        f"Predicted Class: {predicted_class};Predicted Probability;Normalized Count",
        n_bins, x_min, x_max,
        100, 0.03, max_y * 1.2  # Y-axis starts at 0.03
    )
    frame.Draw()

    for h_true in histograms_for_class:
        h_true.Draw("HIST SAME")

    # Add legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)
    for label_idx, true_label in enumerate(data_structure.labels):
        legend.AddEntry(
            histograms_for_class[label_idx],
            f"{true_label}",
            "l"
        )
    legend.Draw()

    # Save the canvas
    output_file = os.path.join(plot_directory, f"predicted_class_{predicted_class}.png")
    canvas.SaveAs(output_file)
    print(f"Saved plot for predicted class {predicted_class} to {output_file}")

common.syncer.sync()
