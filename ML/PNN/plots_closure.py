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
from data_loader.data_loader_2 import H5DataLoader

# Argument parser (no --var argument anymore)
argParser = argparse.ArgumentParser(description="Plot histograms for true and predicted class probabilities for all features")
argParser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--config",        action="store",      default="pnn_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument("--process",       action="store",      default=None,                     help="Which process?")
argParser.add_argument("--training",      action="store",      default="v1",                     help="Training version")
argParser.add_argument("--rebin",         action="store",      default=1, type=int,            help="rebin plot?")
argParser.add_argument("--x_max",         action="store",      default=None, type=float,            help="x_max")
argParser.add_argument("--feature_names",  action="store",  nargs="*", default=["DER_pt_h"],                     help="Which feature?")
argParser.add_argument("--small", action="store_true", help="Only one batch, for debugging")
args = argParser.parse_args()

# Logger
from common.logger import get_logger
logger = get_logger(args.logLevel, logFile=None)

# Import configuration
config = importlib.import_module(f"{args.configDir}.{args.config}")

subdirs= [arg for arg in [args.process, args.selection] if arg is not None]

# Do we use ICP?
if config.icp is not None:
    from ML.ICP.ICP import InclusiveCrosssectionParametrization
    icp_name = "ICP_"+"_".join(subdirs)+"_"+config.icp+".pkl"
    icp = InclusiveCrosssectionParametrization.load(os.path.join(user.model_directory, "ICP", icp_name))
    config.icp_predictor = icp.get_predictor()
    print("We use this ICP:",icp_name)
    print(icp)

# Do we use a Scaler?
if config.use_scaler:
    from ML.Scaler.Scaler import Scaler
    scaler_name = "Scaler_"+"_".join(subdirs)+'.pkl'
    scaler = Scaler.load(os.path.join(user.model_directory, "Scaler", scaler_name))
    config.feature_means     = scaler.feature_means
    config.feature_variances = scaler.feature_variances

    print("We use this scaler:", scaler_name)
    print(scaler)

# Where to store the training
model_directory = os.path.join(user.model_directory, "PNN", *subdirs,  args.config, args.training)
os.makedirs(model_directory, exist_ok=True)

# Set up output directory for plots
plot_directory = os.path.join(user.plot_directory, "PNN", *subdirs, args.config, args.training, "paper")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

from ML.PNN.PNN import PNN

# Initialize model
print(f"Trying to load PNN from {model_directory}")
pnn = PNN.load(model_directory)

# Initialize for training
pnn.load_training_data(datasets_hephy=datasets_hephy, process=args.process, selection=args.selection, n_split=(args.n_split if not args.small else 100))

# -----------------------------------------------------------------
# Top-level code to accumulate histograms without training
# -----------------------------------------------------------------
# Set rebin factor (adjust as needed)
rebin = 1 if args.rebin is None else args.rebin

# Initialize histograms based on plot_options from data_structure
true_histograms = {}
pred_histograms = {}
bin_edges = {}

for feature_name in args.feature_names:
    n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
    if args.x_max is not None:
        x_max = args.x_max
    n_bins = n_bins // rebin  # make plots coarser
    true_histograms[feature_name] = np.zeros((n_bins, len(pnn.base_points)))
    pred_histograms[feature_name] = np.zeros((n_bins, len(pnn.base_points)))
    bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)

# Prepare the loaders from the training data for each base point
loaders = [pnn.training_data[tuple(base_point)] for base_point in pnn.base_points]

# Decide how many batches to process (use n_split from args)
max_batches = args.n_split if not args.small else 1
i_batch = 0

# -----------------------------------------------------------------
# Define a subset of base point variations to plot (no combinations)
# For TES and JES, we use +/-1, +/-2, and +/-3 (only one non-zero at a time);
# for MET, we use 1, 2, and 3 (only upward variations).
# -----------------------------------------------------------------
variations_to_plot = [
    (0, 0, 0),
    # TES variations (only nu_tes non-zero)
    (-1, 0, 0),
    (-2, 0, 0),
    (-3, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    # JES variations (only nu_jes non-zero)
    (0, -1, 0),
    (0, -2, 0),
    (0, -3, 0),
    (0, 1, 0),
    (0, 2, 0),
    (0, 3, 0),
    # MET variations (only nu_met non-zero)
    (0, 0, 1),
    (0, 0, 2),
    (0, 0, 3)
]

# -----------------------------------------------------------------
# Assign colors and build legend labels for each variation.
# For visual consistency, use similar colors for up and down variations:
#   - TES variations: shades of blue (symmetric in absolute value)
#   - JES variations: shades of red (symmetric in absolute value)
#   - MET variations: shades of green (only upward)
# -----------------------------------------------------------------
import ROOT

# Define color palettes using ROOT color constants.

# JES variations: light blue for positive, dark blue for negative.
jes_positive_colors = {1: ROOT.kAzure+2, 2: ROOT.kAzure+3, 3: ROOT.kAzure+4}
jes_negative_colors = {1: ROOT.kBlue-2, 2: ROOT.kBlue-3, 3: ROOT.kBlue-4}

# TES variations: light red for positive, dark red for negative.
tes_positive_colors = {1: ROOT.kPink+2, 2: ROOT.kPink+3, 3: ROOT.kPink+4}
tes_negative_colors = {1: ROOT.kRed-2, 2: ROOT.kRed-3, 3: ROOT.kRed-4}

# MET variations remain unchanged.
met_colors = {1: ROOT.kGreen+2, 2: ROOT.kGreen+3, 3: ROOT.kGreen+4}

variation_colors = {}
variation_legends = {}

for variation in variations_to_plot:
    nu_tes, nu_jes, nu_met = variation
    # Determine the color based on which variation is non-zero.
    if nu_tes != 0 and nu_jes == 0 and nu_met == 0:
        # TES variation: check the sign.
        if nu_tes > 0:
            color = tes_positive_colors[abs(nu_tes)]
        else:
            color = tes_negative_colors[abs(nu_tes)]
    elif nu_jes != 0 and nu_tes == 0 and nu_met == 0:
        # JES variation: check the sign.
        if nu_jes > 0:
            color = jes_positive_colors[abs(nu_jes)]
        else:
            color = jes_negative_colors[abs(nu_jes)]
    elif nu_met != 0 and nu_tes == 0 and nu_jes == 0:
        color = met_colors[abs(nu_met)]
    else:
        # In case of an unexpected combination, default to black.
        color = ROOT.kBlack

    variation_colors[variation] = color

    # Build the legend label using ROOT TeX syntax.
    legend_label = f"nu_{{tes}}={int(nu_tes)}  nu_{{jes}}={int(nu_jes)}  nu_{{met}}={int(nu_met)}"
    variation_legends[variation] = legend_label

variation_colors[(0,0,0)] = ROOT.kBlack
variation_legends[(0,0,0)] = "nominal"

# (Optional) Print out the variation info for verification.
print("Variations to plot with assigned colors and legend labels:")
for variation in variations_to_plot:
    print(f"{variation}: Color={variation_colors[variation]}, Legend='{variation_legends[variation]}'")


# Loop over batches (no gradient computation and no optimizer updates)
for batches in tqdm(zip(*loaders), desc="Accumulating Histograms"):
    # Process nominal batch
    nominal_batch = batches[pnn.nominal_base_point_index]
    features_nominal, weights_nominal, _ = H5DataLoader.split(nominal_batch)
    features_nominal_norm = (features_nominal - pnn.feature_means) / np.sqrt(pnn.feature_variances)
    features_nominal_tensor = tf.convert_to_tensor(features_nominal_norm, dtype=tf.float32)
    # In inference mode (no training)
    DeltaA_nominal = pnn.model(features_nominal_tensor, training=False)

    # Process each base point
    for i_base_point, (base_point, batch) in enumerate(zip(pnn.base_points, batches)):

        if tuple(base_point) not in variations_to_plot: continue
        
        features_nu, weights_nu, _ = H5DataLoader.split(batch)
        features_nu_norm = (features_nu - pnn.feature_means) / np.sqrt(pnn.feature_variances)
        features_nu_tensor = tf.convert_to_tensor(features_nu_norm, dtype=tf.float32)
        DeltaA_nu = pnn.model(features_nu_tensor, training=False)

        # Compute bias factor if ICP predictor is available
        if hasattr(pnn.config, "icp_predictor"):
            bias_factor = pnn.config.icp_predictor(**{k: v for k, v in zip(pnn.parameters, pnn.base_points[i_base_point])})
        else:
            bias_factor = 1

        # Accumulate histograms for each feature
        for _, feature_name in enumerate(args.feature_names):
            feature_idx = common.data_structure.feature_names.index( feature_name )
            feature_values_nominal = features_nominal[:, feature_idx]
            feature_values_nu = features_nu[:, feature_idx]
            # Get original binning info and adjust with rebin factor
            n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
            if args.x_max is not None:
                x_max = args.x_max
            n_bins = n_bins // rebin

            # Compute histogram for the nu (true) values using weights_nu
            true_hist, _ = np.histogram(feature_values_nu, bins=bin_edges[feature_name], weights=weights_nu)
            true_histograms[feature_name][:, i_base_point] += true_hist

            # Compute histogram for the predicted values.
            # The weight here includes bias_factor and the exponential of the model prediction for the nominal batch.
            pred_weights = weights_nominal * bias_factor * np.exp(
                tf.linalg.matvec(DeltaA_nominal, pnn.VkA[i_base_point]).numpy()
            )
            pred_hist, _ = np.histogram(feature_values_nominal, bins=bin_edges[feature_name], weights=pred_weights)
            pred_histograms[feature_name][:, i_base_point] += pred_hist

    i_batch += 1
    if i_batch >= max_batches:
        break

print("Histogram accumulation complete.")
for feature in true_histograms:
    print(f"Feature: {feature}")
    print(f"  True histogram shape: {true_histograms[feature].shape}")
    print(f"  Predicted histogram shape: {pred_histograms[feature].shape}")

# -----------------------------------------------------------------
# Additional code: Create individual ROOT plots (absolute and normalized) for each feature
# -----------------------------------------------------------------
import ROOT
# --- ROOT Plotting ---
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
macro_path = os.path.join(dir_path, "../../common/scripts/tdrstyle.C")
ROOT.gROOT.LoadMacro(macro_path)
ROOT.setTDRStyle()

stuff = []
for feature_name in true_histograms:
    # Retrieve binning information and adjust with rebin factor
    n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
    if args.x_max is not None:
        x_max = args.x_max
    n_bins = n_bins // rebin

    # -----------------------------
    # Absolute Histograms Plot
    # -----------------------------
    canvas_abs = ROOT.TCanvas(f"c_abs_{feature_name}", f"{feature_name} Absolute", 800, 600)
    
    # Determine maximum y value for scaling
    max_y_abs = 0
    for i in range(len(pnn.base_points)):
        max_y_abs = max(max_y_abs,
                        true_histograms[feature_name][:, i].max(),
                        pred_histograms[feature_name][:, i].max())
    max_y_abs *= 1.2

    # Create a dummy frame histogram to set axis ranges
    h_frame_abs = ROOT.TH1F(f"h_frame_abs_{feature_name}", f";{data_structure.plot_options[feature_name]['tex']};Counts", n_bins, x_min, x_max)
    h_frame_abs.SetMaximum(max_y_abs)

    logY = data_structure.plot_options[feature_name].get('logY', False)
    h_frame_abs.SetMinimum(0.9 if logY else 0)
    canvas_abs.SetLogy( logY )
    h_frame_abs.Draw("AXIS")

    # Create a legend for the class entries
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)

    # Draw true and predicted histograms for each base point on the same canvas
    for i_base_point, base_point in enumerate(pnn.base_points):

        if tuple(base_point) not in variations_to_plot: continue

        # True histogram (dashed line)
        h_true = ROOT.TH1F(f"h_true_{feature_name}_{i_base_point}", f"{feature_name} True", n_bins, x_min, x_max)
        for bin_idx in range(n_bins):
            h_true.SetBinContent(bin_idx+1, true_histograms[feature_name][bin_idx, i_base_point])
        h_true.SetLineColor(variation_colors[tuple(base_point)])
        h_true.SetLineStyle(2)
        h_true.SetLineWidth(2)
        h_true.Draw("SAME HIST")

        # Predicted histogram (solid line)
        h_pred = ROOT.TH1F(f"h_pred_{feature_name}_{i_base_point}", f"{feature_name} Pred", n_bins, x_min, x_max)
        for bin_idx in range(n_bins):
            h_pred.SetBinContent(bin_idx+1, pred_histograms[feature_name][bin_idx, i_base_point])
        h_pred.SetLineColor(variation_colors[tuple(base_point)])
        h_pred.SetLineStyle(1)
        h_pred.SetLineWidth(2)
        h_pred.SetMarkerStyle(0)
        h_pred.SetMarkerSize(0)
        h_pred.SetMarkerColor(variation_colors[tuple(base_point)])
        h_pred.Draw("SAME HIST")

        stuff.append( [h_true, h_pred] )
        legend.AddEntry(h_pred, variation_legends[tuple(base_point)])

    legend.Draw()

    abs_filename = os.path.join(plot_directory, f"{feature_name}_absolute.png")
    canvas_abs.SaveAs(abs_filename)
    abs_filename = os.path.join(plot_directory, f"{feature_name}_absolute.pdf")
    canvas_abs.SaveAs(abs_filename)
    print(f"Saved absolute plot for {feature_name} to {abs_filename}")

    # -----------------------------
    # Normalized Histograms Plot
    # -----------------------------
    canvas_norm = ROOT.TCanvas(f"c_norm_{feature_name}", f"{feature_name} Normalized", 800, 600)
    
    # For normalization, use the nominal true histogram as denominator
    nominal_true = true_histograms[feature_name][:, pnn.nominal_base_point_index].copy()
    nominal_true[nominal_true == 0] = 1  # avoid division by zero

#    # Determine maximum y value for normalized histograms
#    max_y_norm = 0
#    for i in range(len(pnn.base_points)):
#        norm_true = true_histograms[feature_name][:, i] / nominal_true
#        norm_pred = pred_histograms[feature_name][:, i] / nominal_true
#        max_y_norm = max(max_y_norm, norm_true.max(), norm_pred.max())
#    max_y_norm *= 1.2

    h_frame_norm = ROOT.TH1F(f"h_frame_norm_{feature_name}", f";{data_structure.plot_options[feature_name]['tex']};Normalized Counts", n_bins, x_min, x_max)
    h_frame_norm.SetMaximum(1.2)
    h_frame_norm.SetMinimum(0.8)
    h_frame_norm.Draw("AXIS")

    # Create a legend for the class entries
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)

    for i_base_point, base_point in enumerate(pnn.base_points):

        if tuple(base_point) not in variations_to_plot: continue

        norm_true = true_histograms[feature_name][:, i_base_point] / nominal_true
        norm_pred = pred_histograms[feature_name][:, i_base_point] / nominal_true

        h_true_norm = ROOT.TH1F(f"h_true_norm_{feature_name}_{i_base_point}", f"{feature_name} True Norm", n_bins, x_min, x_max)
        for bin_idx in range(n_bins):
            h_true_norm.SetBinContent(bin_idx+1, norm_true[bin_idx])
        h_true_norm.SetLineColor(variation_colors[tuple(base_point)])
        h_true_norm.SetLineStyle(2)
        h_true_norm.SetLineWidth(2)
        h_true_norm.Draw("SAME HIST")

        h_pred_norm = ROOT.TH1F(f"h_pred_norm_{feature_name}_{i_base_point}", f"{feature_name} Pred Norm", n_bins, x_min, x_max)
        for bin_idx in range(n_bins):
            h_pred_norm.SetBinContent(bin_idx+1, norm_pred[bin_idx])
        h_pred_norm.SetLineColor(variation_colors[tuple(base_point)])
        h_pred_norm.SetLineStyle(1)
        h_pred_norm.SetLineWidth(2)
        h_pred_norm.SetMarkerStyle(0)
        h_pred_norm.SetMarkerSize(0)
        h_pred_norm.SetMarkerColor(variation_colors[tuple(base_point)])
        h_pred_norm.Draw("SAME HIST")

        stuff.append( [h_true_norm, h_pred_norm] )

        legend.AddEntry(h_pred_norm, variation_legends[tuple(base_point)])

    legend.Draw()

    norm_filename = os.path.join(plot_directory, f"{feature_name}_normalized.png")
    canvas_norm.SaveAs(norm_filename)
    norm_filename = os.path.join(plot_directory, f"{feature_name}_normalized.pdf")
    canvas_norm.SaveAs(norm_filename)
    print(f"Saved normalized plot for {feature_name} to {norm_filename}")

common.syncer.sync()
