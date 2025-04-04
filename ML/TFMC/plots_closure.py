#!/usr/bin/env python
import sys, os, importlib, argparse
import numpy as np
import ROOT
import tensorflow as tf
from tqdm import tqdm

# Add relative paths for common modules
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer
import common.helpers as helpers
import common.data_structure as data_structure
import common.datasets_hephy as datasets_hephy

# Argument parser (no --var argument anymore)
argParser = argparse.ArgumentParser(description="Plot histograms for true and predicted class probabilities for all features")
argParser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--config", action="store", default="tfmc", help="Which config?")
argParser.add_argument("--configDir", action="store", default="configs", help="Where is the config?")
argParser.add_argument("--modelDir", action="store", required=True, help="Directory containing the trained TFMC model.")
argParser.add_argument("--calib", action="store", default=None, help="Directory containing the trained calibration.")
argParser.add_argument("--small", action="store_true", help="Only one batch, for debugging")
args = argParser.parse_args()

# Logger
from common.logger import get_logger
logger = get_logger(args.logLevel, logFile=None)

# Import configuration
config = importlib.import_module(f"{args.configDir}.{args.config}")

# Load the trained TFMC model
from TFMC import TFMC
tfmc = TFMC.load(args.modelDir)

if args.calib is not None:
    from ML.Calibration.MulticlassCalibration import MultiClassCalibration
    calib = MultiClassCalibration.load(args.calib)

# Load the data
data_loader = datasets_hephy.get_data_loader(selection=args.selection, selection_function=None, n_split=args.n_split)
max_batch = 1 if args.small else -1

# Set up output directory for plots
plot_directory = os.path.join(user.plot_directory, "TFMC", args.selection, args.config, "closure")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Loop over all features that have plot_options defined
features_to_plot = [f for f in data_structure.feature_names if f in data_structure.plot_options]

# Determine the number of classes and the selected indices for filtering the events
classes = config.classes  # expecting config to define classes
num_classes = len(classes)
selected_indices = [data_structure.label_encoding[label] for label in classes]

# Create dictionaries to hold histograms and bin_edges for each feature
true_hist = {}
pred_hist = {}
pred_calib_hist = {}
bin_edges = {}

for feature in features_to_plot:
    n_bins, x_min, x_max = data_structure.plot_options[feature]['binning']
    true_hist[feature]      = np.zeros((n_bins, num_classes))
    pred_hist[feature]      = np.zeros((n_bins, num_classes))
    pred_calib_hist[feature] = np.zeros((n_bins, num_classes))
    bin_edges[feature]      = np.linspace(x_min, x_max, n_bins + 1)

# --- Compute weight sums ---
batch_counter = 0
weight_sums = np.zeros(num_classes)
for batch in tqdm(data_loader, total=(len(data_loader) if max_batch < 0 else max_batch), desc="Batches"):
    _ , weights, labels = data_loader.split(batch)
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    # Accumulate the sum of weights per class:
    weight_sums += np.sum(weights[:, None] * labels_one_hot, axis=0)
    batch_counter += 1
    if max_batch > 0 and batch_counter >= max_batch:
        break

# Class weights are sigma(total)/sigma(class)
class_weights = [weight_sums[data_structure.label_encoding[label]] for label in config.classes]
total = sum(class_weights)
class_weights = np.array([total / class_weights[i] for i in range(len(class_weights))])

# --- Accumulate histograms for each feature ---
print("Accumulating histograms ...")
batch_counter = 0
for batch in tqdm(data_loader, total=(len(data_loader) if max_batch < 0 else max_batch), desc="Batches"):
    features, weights, labels = data_loader.split(batch)
    # Filter events to only include selected classes
    mask = np.isin(labels, selected_indices)
    if np.sum(mask) == 0:
        continue
    features = features[mask]
    weights = weights[mask]
    labels = labels[mask]
    # Reweighting
    weights = weights * class_weights[labels.astype('int')]
    # Get predictions from the model
    predictions = tfmc.predict(features, ic_scaling=False)
    if args.calib is not None:
        predictions_calib = calib.predict(predictions)
    # Remap labels to sequential indices (for one-hot conversion)
    remapped_labels = np.array([selected_indices.index(l) for l in labels])
    labels_one_hot = tf.keras.utils.to_categorical(remapped_labels, num_classes=num_classes)

    # Loop over each feature and accumulate histograms
    for feature in features_to_plot:
        feature_idx = data_structure.feature_names.index(feature)
        feature_values = features[:, feature_idx]
        for c in range(num_classes):
            h_true, _ = np.histogram(feature_values, bins=bin_edges[feature], weights=weights * labels_one_hot[:, c])
            true_hist[feature][:, c] += h_true
            h_pred, _ = np.histogram(feature_values, bins=bin_edges[feature], weights=weights * predictions[:, c])
            pred_hist[feature][:, c] += h_pred
            if args.calib is not None:
                h_pred_calib, _ = np.histogram(feature_values, bins=bin_edges[feature], weights=weights * predictions_calib[:, c])
                pred_calib_hist[feature][:, c] += h_pred_calib

    batch_counter += 1
    if max_batch > 0 and batch_counter >= max_batch:
        break

# --- ROOT Plotting ---
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
macro_path = os.path.join(dir_path, "../../common/scripts/tdrstyle.C")
ROOT.gROOT.LoadMacro(macro_path)
ROOT.setTDRStyle()

# Loop over normalized and non-normalized versions
for normalized in [False, True]:
    # Loop over each feature
    for feature in features_to_plot:
        n_bins, x_min, x_max = data_structure.plot_options[feature]['binning']
        x_axis_title = data_structure.plot_options[feature]['tex']
        # logY = data_structure.plot_options[feature].get('logY', False)
        # Create a canvas (single plot per feature)
        canvas = ROOT.TCanvas("c", "Convergence Plot", 800, 600)
        if not normalized:
            logY = True
            if logY:
                canvas.SetLogy(True)
            else:
                canvas.SetLogy(False)
        else:
            canvas.SetLogy(False)

        # Work on local copies of the histograms to avoid changing the originals
        thist = true_hist[feature].copy()
        phist = pred_hist[feature].copy()
        if args.calib is not None:
            pchist = pred_calib_hist[feature].copy()
        else:
            pchist = phist

        if normalized:
            total_truth_sum = thist.sum(axis=1, keepdims=True)
            total_truth_sum = np.where(total_truth_sum == 0, 1, total_truth_sum)
            thist_norm  = thist /total_truth_sum
            phist_norm  = phist /total_truth_sum
            pchist_norm = pchist/total_truth_sum
        else:
            thist_norm  = thist /class_weights
            phist_norm  = phist /class_weights
            pchist_norm = pchist/class_weights
 
        # Determine y-axis range
        min_y = 0 if normalized else (0.9 if logY else 0)
        max_y = 0
        for c in range(num_classes):
            max_y = max(max_y, thist_norm[:, c].max(), phist_norm[:, c].max())
        if logY:
            max_y *= 9
        else:
            max_y *= 1.2
        if normalized:
            min_y, max_y = 0, 1

        # Create frame histogram for axis labels and draw it
        h_frame = ROOT.TH2F("h_frame", f";{x_axis_title};Probability" if normalized else f";{x_axis_title};Events", n_bins, x_min, x_max, 100, min_y, max_y)
        h_frame.GetYaxis().SetTitleOffset(1.3)
        h_frame.Draw()

        # Colors for classes (as in original code)
        colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta]
        stuff = [h_frame]

        # Draw histograms for each class: true (dashed) and predicted (solid)
        for c, class_name in enumerate(classes):
            h_true = ROOT.TH1F(f"h_true_{feature}_{c}", f"{feature} (true {class_name})", n_bins, x_min, x_max)
            for i in range(n_bins):
                h_true.SetBinContent(i + 1, thist_norm[i, c])
            h_true.SetLineColor(colors[c % len(colors)])
            h_true.SetLineStyle(2)  # dashed
            h_true.SetLineWidth(2)
            h_true.Draw("HIST SAME")
            stuff.append(h_true)

            h_pred = ROOT.TH1F(f"h_pred_{feature}_{c}", f"{feature} (pred {class_name})", n_bins, x_min, x_max)
            for i in range(n_bins):
                h_pred.SetBinContent(i + 1, pchist_norm[i, c])
            h_pred.SetLineColor(colors[c % len(colors)])
            h_pred.SetLineStyle(1)  # solid
            h_pred.SetLineWidth(2)
            h_pred.Draw("HIST SAME")
            stuff.append(h_pred)

        # Create a legend for the class entries
        legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
        legend.SetBorderSize(0)
        legend.SetShadowColor(0)
        for c, class_name in enumerate(classes):
            legend.AddEntry(stuff[1 + 2 * c], f"{class_name} (true)", "l")
            legend.AddEntry(stuff[2 + 2 * c], f"{class_name} (pred)", "l")
        legend.Draw()

        # Save the canvas to file (both PNG and PDF)
        file_suffix = "norm_" if normalized else ""
        output_file = os.path.join(plot_directory, file_suffix + f"{feature}.png")
        canvas.SaveAs(output_file)
        output_file = os.path.join(plot_directory, file_suffix + f"{feature}.pdf")
        canvas.SaveAs(output_file)
        print(f"Saved plot for feature '{feature}' to {output_file}.")

common.syncer.sync()

