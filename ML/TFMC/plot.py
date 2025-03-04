# Initialize histograms for predictions
pred_histograms = {}
bin_edges = {}

# Use binning for predictions (e.g., [30 bins, 0 to 1])
n_bins, x_min, x_max = 30, 0, 1
for c, class_name in enumerate(tfmc.classes):
    pred_histograms[class_name] = np.zeros((n_bins, len(data_structure.labels)))
    bin_edges[class_name] = np.linspace(x_min, x_max, n_bins + 1)

# Loop over data batches and calculate predictions
total_batches = len(data_loader)
for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
    features, weights, raw_labels = data_loader.split(batch)
    predictions = tfmc.predict(features, ic_scaling=False)

    # Accumulate predicted probabilities split by true class
    for c, class_name in enumerate(tfmc.classes):
        for true_c, true_label in enumerate(data_structure.labels):
            true_indices = (raw_labels == true_c)
            if np.any(true_indices):
                pred_histogram, _ = np.histogram(
                    predictions[true_indices, c],
                    bins=bin_edges[class_name],
                    weights=weights[true_indices]
                )
                pred_histograms[class_name][:, true_c] += pred_histogram

    if max_batch > 0 and i_batch + 1 >= max_batch:
        break

# Normalize histograms by total sum for each true class
for class_name in pred_histograms.keys():
    for true_c in range(len(data_structure.labels)):
        total_sum = pred_histograms[class_name][:, true_c].sum()
        pred_histograms[class_name][:, true_c] /= total_sum if total_sum > 0 else 1

# Plot each predicted probability using ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.setTDRStyle()

canvas = ROOT.TCanvas("c_predicted_probabilities", "Predicted Probabilities", 800, 600)

linestyles = [2, 1, 3, 4, 5]  # Define line styles for each true class

# Plot each predicted probability
for c, class_name in enumerate(tfmc.classes):
    canvas.Clear()
    h_frame = ROOT.TH2F(
        f"h_frame_{class_name}",
        f";Predicted Probability for {class_name};Normalized Probability",
        n_bins, x_min, x_max, 100, 0, 1.2 * np.max(pred_histograms[class_name])
    )
    h_frame.GetYaxis().SetTitleOffset(1.3)
    h_frame.Draw()

    # Add lines for each true class
    for true_c, true_label in enumerate(data_structure.labels):
        h_pred = ROOT.TH1F(
            f"h_pred_{class_name}_{true_label}",
            f"{class_name} (true {true_label})",
            n_bins, x_min, x_max
        )
        for i, value in enumerate(pred_histograms[class_name][:, true_c]):
            h_pred.SetBinContent(i + 1, value)

        h_pred.SetLineColor(ROOT.kBlue + true_c)
        h_pred.SetLineStyle(linestyles[true_c % len(linestyles)])
        h_pred.SetLineWidth(2)
        h_pred.Draw("HIST SAME")

    # Add legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    for true_c, true_label in enumerate(data_structure.labels):
        legend.AddEntry(f"h_pred_{class_name}_{true_label}", f"True {true_label}", "l")
    legend.Draw()

    # Save the plot
    output_file = os.path.join(plot_directory, f"predicted_{class_name}.png")
    canvas.SaveAs(output_file)
    print(f"Saved plot for predicted probabilities of {class_name} to {output_file}")
#!/usr/bin/env python

#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, '../..')
#import os
#import numpy as np
#from tqdm import tqdm
#import importlib
#import ROOT
#import common.data_structure as data_structure
#import common.helpers as helpers
#import argparse
#import common.user as user
#import common.syncer
#
## Parser
#argParser = argparse.ArgumentParser(description="Argument parser")
#argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
#argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
#argParser.add_argument("--small", action="store_true", help="Only one batch, for debugging")
#args = argParser.parse_args()
#
## Import datasets_hephy
#import common.datasets_hephy as datasets_hephy
#
## Set up directories
#plot_directory = os.path.join(user.plot_directory, "TFMC", args.selection, "features"+("_small" if args.small else ""))
#os.makedirs(plot_directory, exist_ok=True)
#helpers.copyIndexPHP(plot_directory)
#
## Load data
#data_loader = datasets_hephy.get_data_loader(selection=args.selection, selection_function=None, n_split=args.n_split if not args.small else 100)
#max_batch = 1 if args.small else -1
#
## Initialize histograms
#true_histograms = {}
#bin_edges = {}
#for feature_name in data_structure.plot_options.keys():
#    n_bins, x_min, x_max = data_structure.plot_options[feature_name]["binning"]
#    true_histograms[feature_name] = np.zeros((n_bins, len(data_structure.labels)))
#    bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)
#
## Accumulate histograms
#print("Accumulating histograms for true class probabilities...")
#total_batches = len(data_loader)
#for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
#    features, weights, raw_labels = data_loader.split(batch)
#
#    # Accumulate histograms per feature
#    for feature_idx, feature_name in enumerate(data_structure.feature_names):
#        feature_values = features[:, feature_idx]
#        for c, label in enumerate(data_structure.labels):
#            true_histogram, _ = np.histogram(
#                feature_values,
#                bins=bin_edges[feature_name],
#                weights=weights * (raw_labels == c),
#            )
#            true_histograms[feature_name][:, c] += true_histogram
#
#    if max_batch > 0 and i_batch + 1 >= max_batch:
#        break
#
## Plot histograms with ROOT
#print("Plotting histograms...")
#ROOT.gStyle.SetOptStat(0)
#for feature_name in data_structure.plot_options.keys():
#    n_bins, x_min, x_max = data_structure.plot_options[feature_name]["binning"]
#    canvas = ROOT.TCanvas(f"c_{feature_name}", f"Feature: {feature_name}", 800, 600)
#    
#    # Set logarithmic y-axis if specified
#    #if data_structure.plot_options[feature_name]['logY']:
#    canvas.SetLogy()
#
#    hist_stack = ROOT.THStack(f"stack_{feature_name}", f"{feature_name};{data_structure.plot_options[feature_name]['tex']};Count")
#    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta]
#
#    for c, label in enumerate(data_structure.labels):
#        hist = ROOT.TH1F(f"hist_{feature_name}_{label}", label, n_bins, x_min, x_max)
#        for b in range(n_bins):
#            hist.SetBinContent(b + 1, true_histograms[feature_name][b, c])
#        
#        # Set line styles and colors
#        hist.SetLineColor(colors[c % len(colors)])
#        hist.SetLineStyle(1)  # Solid line
#        hist.SetLineWidth(2)
#        hist_stack.Add(hist)
#
#    hist_stack.Draw("nostack HIST")
#    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
#    legend.SetBorderSize(0)
#    legend.SetShadowColor(0)
#
#    for c, label in enumerate(data_structure.labels):
#        legend.AddEntry(f"hist_{feature_name}_{label}", label, "l")
#    legend.Draw()
#
#    canvas.SaveAs(os.path.join(plot_directory, f"{feature_name}.png"))
#
#print(f"Histograms saved in {plot_directory}.")
#common.syncer.sync()
#
