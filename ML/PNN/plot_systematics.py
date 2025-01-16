#!/usr/bin/env python

import sys, os
import numpy as np
import ROOT
from tqdm import tqdm
import importlib

# Setup
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer
import common.helpers as helpers
import common.data_structure as data_structure

# Argument Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--config", action="store", default="pnn_quad_jes", help="Which config?")
argParser.add_argument("--configDir", action="store", default="configs", help="Where is the config?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
args = argParser.parse_args()

# Import the config and datasets
import common.datasets as datasets
config = importlib.import_module(f"{args.configDir}.{args.config}")

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "systematics", args.selection, args.config)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Load the data
data_loaders = {}
for base_point in config.base_points:
    base_point_tuple = tuple(base_point)
    values = config.get_alpha(base_point_tuple)
    data_loader = datasets.get_data_loader(
        selection=args.selection,
        values=values,
        selection_function=None,
        n_split=args.n_split if not args.small else 100
    )
    print(f"Loaded data for base point {base_point_tuple}, alpha = {values}, file = {data_loader.file_path}")
    data_loaders[base_point_tuple] = data_loader

max_batch = 1 if args.small else -1

# ROOT Setup
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)  # Disable interactive plots
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Initialize histograms
histograms = {
    feature: {base_point: None for base_point in data_loaders.keys()}
    for feature in data_structure.plot_options.keys()
}
bin_edges = {}

for feature_name, options in data_structure.plot_options.items():
    n_bins, x_min, x_max = options['binning']
    bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)
    for base_point in data_loaders.keys():
        histograms[feature_name][base_point] = np.zeros(n_bins)

# Accumulate histograms
print("Accumulating feature histograms...")
for base_point, data_loader in data_loaders.items():
    for i_batch, batch in enumerate(tqdm(data_loader, desc=f"Processing base point {base_point}")):
        features, weights, _ = data_loader.split(batch)

        for feature_idx, feature_name in enumerate(data_structure.feature_names):
            feature_values = features[:, feature_idx]
            histogram, _ = np.histogram(feature_values, bins=bin_edges[feature_name], weights=weights)
            histograms[feature_name][base_point] += histogram

        if max_batch > 0 and i_batch + 1 >= max_batch:
            break


# Plot histograms with a ratio pad
print("Plotting histograms...")
for feature_name, options in data_structure.plot_options.items():
    n_bins, x_min, x_max = options['binning']
    x_axis_title = options['tex']
    logY = options['logY']

    canvas = ROOT.TCanvas(f"c_{feature_name}", feature_name, 800, 800)

    # Create pads for main plot and ratio
    top_pad = ROOT.TPad("top_pad", "Top Pad", 0, 0.3, 1, 1.0)
    bottom_pad = ROOT.TPad("bottom_pad", "Bottom Pad", 0, 0, 1, 0.3)

    top_pad.SetBottomMargin(0)  # No margin for shared x-axis
    bottom_pad.SetTopMargin(0)
    bottom_pad.SetBottomMargin(0.35)  # Room for x-axis labels
    top_pad.Draw()
    bottom_pad.Draw()

    # Apply ticks to top and right axes
    top_pad.SetTicks(1, 1)
    bottom_pad.SetTicks(1, 1)

    # Top pad: main histogram
    top_pad.cd()
    if logY:
        top_pad.SetLogy()

    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)

    max_y = 0
    hists = []
    colors = [ROOT.kBlue, ROOT.kCyan, ROOT.kTeal - 1, ROOT.kViolet, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kRed]

    for i, (base_point, histogram) in enumerate(histograms[feature_name].items()):
        hist = ROOT.TH1F(f"h_{feature_name}_{i}", feature_name, n_bins, x_min, x_max)
        hist.GetXaxis().SetTitleFont(43)
        hist.GetYaxis().SetTitleFont(43)
        hist.GetXaxis().SetLabelFont(43)
        hist.GetYaxis().SetLabelFont(43)
        hist.GetXaxis().SetTitleSize(24)
        hist.GetYaxis().SetTitleSize(24)
        hist.GetXaxis().SetLabelSize(20)
        hist.GetYaxis().SetLabelSize(20)
        hist.GetYaxis().SetTitleOffset(1.6)

        for bin_idx, value in enumerate(histogram):
            hist.SetBinContent(bin_idx + 1, value)

        color = ROOT.kBlack if base_point == tuple(config.nominal_base_point) else colors[i % len(colors)]
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.Draw("HIST SAME" if i > 0 else "HIST")
        legend_label = "#nu = " + ", ".join(map(str, base_point)) if base_point != tuple(config.nominal_base_point) else "Nominal"
        legend.AddEntry(hist, legend_label, "l")

        hists.append(hist)
        max_y = max(max_y, hist.GetMaximum())

    for hist in hists:
        hist.GetYaxis().SetRangeUser(0.03 if logY else 0, 1.2 * max_y)

    legend.Draw()

    # Bottom pad: ratio plot
    bottom_pad.cd()
    nominal_hist = histograms[feature_name][tuple(config.nominal_base_point)]

    ratio_hists = []
    for i, (base_point, histogram) in enumerate(histograms[feature_name].items()):
        #if base_point == tuple(config.nominal_base_point):
        #    nominal_hist = histogram
        #    continue

        ratio_hist = ROOT.TH1F(f"r_{feature_name}_{i}", "Ratio", n_bins, x_min, x_max)
        for bin_idx in range(n_bins):
            ratio_value = histogram[bin_idx] / nominal_hist[bin_idx] if nominal_hist[bin_idx] > 0 else 0
            ratio_hist.SetBinContent(bin_idx + 1, ratio_value)

        ratio_hist.SetLineColor(colors[i % len(colors)])
        ratio_hist.SetLineWidth(2)
        ratio_hist.Draw("HIST SAME" if len(ratio_hists) > 0 else "HIST")
        ratio_hists.append(ratio_hist)

    # Format the ratio pad
    ratio_hists[0].GetXaxis().SetTitle(x_axis_title)
    ratio_hists[0].GetXaxis().SetTitleFont(43)
    ratio_hists[0].GetYaxis().SetTitleFont(43)
    ratio_hists[0].GetXaxis().SetLabelFont(43)
    ratio_hists[0].GetYaxis().SetLabelFont(43)
    ratio_hists[0].GetXaxis().SetTitleSize(24)
    ratio_hists[0].GetYaxis().SetTitleSize(24)
    ratio_hists[0].GetXaxis().SetLabelSize(20)
    ratio_hists[0].GetYaxis().SetLabelSize(20)
    ratio_hists[0].GetYaxis().SetTitle("Ratio")
    ratio_hists[0].GetYaxis().SetTitleOffset(1.6)
    ratio_hists[0].GetYaxis().SetRangeUser(0.8, 1.2)
    ratio_hists[0].GetYaxis().SetNdivisions(505)

    # Save the canvas
    canvas.SaveAs(os.path.join(plot_directory, f"{feature_name}.png"))
    print(f"Saved plot for {feature_name}.")

print(f"All plots saved in {plot_directory}.")
common.syncer.sync()

