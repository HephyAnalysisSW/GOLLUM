import sys, os
import math
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from common.helpers import copyIndexPHP
import matplotlib
matplotlib.use("Agg")  # Set the backend
import common.syncer  # Re-import syncer after backend configuration
import common.data_structure as data_structure
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar

def accumulate_truth_histograms(data_loader, max_batch=-1):
    """
    Accumulate histograms for true class weights.

    Parameters:
    - data_loader: H5DataLoader, for loading batches of data.
    - max_batch: int, maximum number of batches to process (default: -1, process all).

    Returns:
    - truth_histograms: dict of histograms, one per feature.
    - bin_edges: dict of arrays, bin edges for each feature.
    """
    truth_histograms = {}
    bin_edges = {}

    total_batches = len(data_loader) if max_batch == -1 else min(len(data_loader), max_batch)
    i_batch = 0

    with tqdm(total=total_batches, desc="Accumulating Histograms", unit="batch") as pbar:
        for batch in data_loader:
            data, weights, labels = data_loader.split( batch )

            # Convert raw labels to one-hot encoded format
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(data_structure.labels))

            for k, feature in enumerate(data_structure.feature_names):
                feature_values = data[:, k]

                # Use plot_options for binning
                n_bins, lower, upper = data_structure.plot_options[feature]['binning']
                if feature not in bin_edges:
                    bin_edges[feature] = np.linspace(lower, upper, n_bins + 1)
                    truth_histograms[feature] = np.zeros((n_bins, len(data_structure.labels)))

                # Accumulate weights for each bin and class
                for b in range(n_bins):
                    in_bin = (feature_values >= bin_edges[feature][b]) & (feature_values < bin_edges[feature][b + 1])
                    bin_weights = weights[in_bin]

                    if bin_weights.sum() > 0:
                        truth_histograms[feature][b, :] += np.sum(bin_weights[:, None] * labels_one_hot[in_bin], axis=0)

            i_batch += 1
            pbar.update(1)

            if max_batch > 0 and i_batch >= max_batch:
                break

    return truth_histograms, bin_edges

def plot_truth_histograms(truth_histograms, bin_edges, output_dir="./plots/"):
    """
    Plot and save truth histograms as separate plots per feature with linear and logarithmic top panels.

    Parameters:
    - truth_histograms: dict, true class weights accumulated over bins.
    - bin_edges: dict, bin edges for each feature.
    - output_dir: str, directory to save the PNG files.
    """
    lin_dir = os.path.join(output_dir, "lin")
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(lin_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    copyIndexPHP(lin_dir)
    copyIndexPHP(log_dir)

    n_classes = len(data_structure.labels)
    colors = plt.cm.tab10(np.arange(n_classes))  # Use tab10 colormap for distinct colors

    for feature in data_structure.feature_names:
        # Get total yields for sorting
        total_yields = truth_histograms[feature].sum(axis=0)
        sorted_indices = np.argsort(total_yields)

        # Reorder histograms and colors by total yield
        stacked_histograms = np.cumsum(truth_histograms[feature][:, sorted_indices], axis=1)
        sorted_colors = colors[sorted_indices]
        sorted_class_labels = [data_structure.labels[i] for i in sorted_indices]

        for scale, subdir in zip(["linear", "log"], [lin_dir, log_dir]):
            # Create a figure with two panels
            fig, (ax_top, ax_bottom) = plt.subplots(
                2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
            )

            # Top panel: stacked yields
            for c, class_name in enumerate(sorted_class_labels):
                ax_top.step(
                    bin_edges[feature],  # Bin edges
                    np.append(stacked_histograms[:, c], stacked_histograms[-1, c]),  # Heights, cumulative
                    where="post",
                    color=sorted_colors[c],
                    linestyle="-",
                    label=class_name,
                )
            ax_top.set_title(feature)
            ax_top.set_ylabel("Number of Events")
            ax_top.legend(loc="upper right")
            ax_top.grid(False)

            if scale == "log":
                ax_top.set_yscale("log")

            # Bottom panel: fractional contributions
            total_weights = truth_histograms[feature].sum(axis=1, keepdims=True)
            fractional_weights = truth_histograms[feature] / np.where(total_weights == 0, 1, total_weights)

            for c, class_name in enumerate(data_structure.labels):
                ax_bottom.step(
                    bin_edges[feature],  # Bin edges
                    np.append(fractional_weights[:, c], 0),  # Heights
                    where="post",
                    color=colors[c],
                    linestyle="-",
                )
            ax_bottom.set_xlabel(feature)  # Feature name for x-axis
            ax_bottom.set_ylabel("Ratio")
            ax_bottom.grid(False)

            # Save the figure
            output_file = os.path.join(subdir, f"{feature.replace(' ', '_')}.png")
            fig.tight_layout()
            fig.savefig(output_file)
            plt.close(fig)

            # Register the file with the syncer
            common.syncer.file_sync_storage.append(output_file)

            print(f"Saved {scale} plot for {feature} to {output_file}.")

def plot_truth_histograms_root(truth_histograms, bin_edges, output_dir):
    """
    Plot and save truth histograms as separate ROOT canvases per feature with linear and logarithmic top panels.

    Parameters:
    - truth_histograms: dict, true class weights accumulated over bins.
    - bin_edges: dict, bin edges for each feature.
    - output_dir: str, directory to save the ROOT files.
    """
    import ROOT

    lin_dir = os.path.join(output_dir, "lin")
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(lin_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    copyIndexPHP(lin_dir)
    copyIndexPHP(log_dir)

    n_classes = len(data_structure.labels)

    for feature in data_structure.feature_names:
        # Get total yields for sorting
        total_yields = truth_histograms[feature].sum(axis=0)
        sorted_indices = np.argsort(total_yields)

        # Reorder histograms and styles by total yield
        stacked_histograms = truth_histograms[feature][:, sorted_indices]
        sorted_class_labels = [data_structure.labels[i] for i in sorted_indices]

        for scale, subdir in zip(["linear", "log"], [lin_dir, log_dir]):
            canvas = ROOT.TCanvas(f"c_{feature}_{scale}", feature, 800, 800)

            # Create pads
            top_pad = ROOT.TPad("top_pad", "Top Pad", 0, 0.3, 1, 1.0)
            bottom_pad = ROOT.TPad("bottom_pad", "Bottom Pad", 0, 0, 1, 0.3)
            top_pad.SetBottomMargin(0)
            bottom_pad.SetTopMargin(0)
            bottom_pad.SetBottomMargin(0.35)
            top_pad.Draw()
            bottom_pad.Draw()

            top_pad.SetTicks(1, 1)  # Enable ticks on top and right for the top pad
            bottom_pad.SetTicks(1, 1)  # Enable ticks on top and right for the bottom pad

            # Top pad: stacked yields
            top_pad.cd()
            if scale == "log":
                top_pad.SetLogy()

            h_stack = ROOT.THStack("h_stack", f";{data_structure.plot_options[feature]['tex']};Number of Events")

            hist_list = []
            for c, class_name in enumerate(sorted_class_labels):
                hist = ROOT.TH1F(f"{class_name}_{scale}", class_name, len(bin_edges[feature]) - 1, bin_edges[feature])
                for b in range(len(bin_edges[feature]) - 1):
                    hist.SetBinContent(b + 1, stacked_histograms[b, c])

                hist.SetFillColor(data_structure.plot_styles[class_name]["fill_color"])
                hist.SetLineColor(data_structure.plot_styles[class_name]["line_color"])
                hist.SetLineWidth(data_structure.plot_styles[class_name]["line_width"])

                h_stack.Add(hist)
                hist_list.append(hist)

            h_stack.Draw("HIST")
            h_stack.GetYaxis().SetTitle("Number of Events")
            h_stack.GetXaxis().SetTitleFont(43)
            h_stack.GetYaxis().SetTitleFont(43)
            h_stack.GetXaxis().SetLabelFont(43)
            h_stack.GetYaxis().SetLabelFont(43)
            h_stack.GetXaxis().SetTitleSize(24)
            h_stack.GetYaxis().SetTitleSize(24)
            h_stack.GetXaxis().SetLabelSize(20)
            h_stack.GetYaxis().SetLabelSize(20)
            h_stack.GetYaxis().SetTitleOffset( 1.6 )

            # Fix the y-axis label size and increase it by 15%
            #h_stack.GetYaxis().SetTitleSize(1.15 * h_stack.GetYaxis().GetTitleSize())  # Increase by 15%
            #h_stack.GetYaxis().SetLabelSize(20)  # Ensure axis numbers are displayed

            h_stack.SetMinimum(0.0005)

            # Hide x-axis labels on the top pad
            h_stack.GetXaxis().SetLabelSize(0)
            top_pad.Update()

            legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
            for c, class_name in enumerate(sorted_class_labels):
                legend.AddEntry(hist_list[c], class_name, "f" if class_name != "htautau" else "l")
            legend.Draw()

            # Bottom pad: fractional contributions
            bottom_pad.cd()

            h_ratio = ROOT.THStack("h_ratio", f";{data_structure.plot_options[feature]['tex']};Fraction")

            total_weights = truth_histograms[feature].sum(axis=1, keepdims=True)
            fractional_weights = truth_histograms[feature] / np.where(total_weights == 0, 1, total_weights)

            fraction_hists = []
            for c, class_name in enumerate(data_structure.labels):
                hist_frac = ROOT.TH1F(f"{class_name}_fraction", class_name, len(bin_edges[feature]) - 1, bin_edges[feature])
                for b in range(len(bin_edges[feature]) - 1):
                    hist_frac.SetBinContent(b + 1, fractional_weights[b, c])

                # Fractional contributions always use line styles
                hist_frac.SetLineColor(data_structure.plot_styles[class_name]["fill_color"])
                hist_frac.SetLineWidth(2)
                h_ratio.Add(hist_frac)
                fraction_hists.append(hist_frac)

            h_ratio.Draw("NOSTACK HIST")
            h_ratio.GetYaxis().SetTitle("Fraction")
            h_ratio.GetYaxis().SetTitleFont(43)
            h_ratio.GetXaxis().SetTitleFont(43)
            h_ratio.GetYaxis().SetLabelFont(43)
            h_ratio.GetXaxis().SetLabelFont(43)
            h_ratio.GetYaxis().SetTitleSize(24)
            h_ratio.GetXaxis().SetTitleSize(24)
            h_ratio.GetYaxis().SetLabelSize(20)
            h_ratio.GetXaxis().SetLabelSize(20)
            h_ratio.GetXaxis().SetTitle(data_structure.plot_options[feature]["tex"])
            #h_ratio.GetXaxis().SetTitleOffset(3.2)
            h_ratio.GetYaxis().SetTitleOffset(1.6)
            h_ratio.GetXaxis().SetTickLength(0.03 * 3)
            h_ratio.GetYaxis().SetTickLength(0.03)

            h_ratio.SetMinimum(0)
            h_ratio.SetMaximum(1)
            h_ratio.GetYaxis().SetNdivisions(505)

            bottom_pad.Update()

            canvas.Modified()
            canvas.Update()

            # Save the canvas
            output_file = os.path.join(subdir, f"{feature.replace(' ', '_')}.png")
            canvas.SaveAs(output_file)

            # Register with syncer
            common.syncer.file_sync_storage.append(output_file)

            print(f"Saved {scale} plot for {feature} to {output_file}.")


from data import get_data_loader

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--selection", type=str, default="inclusive", help="Event selection")
    parser.add_argument("--plot_directory", type=str, default="v2", help="Plot directory")
    parser.add_argument("--small", action="store_true" )

    return parser.parse_args()

args = parse_arguments()

import data

data_loader = data.get_data_loader( 
    name = "nominal", 
    n_split=10, 
    selection=args.selection, 
    selection_function=None, 
    )

output_path = os.path.join(user.plot_directory, "plots", args.plot_directory, args.selection+("_small" if args.small else ""))
os.makedirs(output_path, exist_ok=True)

# Accumulate histograms
truth_histograms, bin_edges = accumulate_truth_histograms(data_loader, max_batch=1 if args.small else -1)

# Plot and save
plot_truth_histograms_root(truth_histograms, bin_edges, output_path)

common.syncer.sync()
