import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from common.helpers import copyIndexPHP
import matplotlib
matplotlib.use("Agg")  # Set the backend
import common.syncer  # Re-import syncer after backend configuration

import numpy as np
from math import ceil, sqrt
from data import get_data_loader, feature_names
import tensorflow as tf
from matplotlib import pyplot as plt

def accumulate_truth_histograms(data_loader, class_labels, n_bins=30, max_batch=-1):
    """
    Accumulate histograms for true class weights.

    Parameters:
    - data_loader: H5DataLoader, for loading batches of data.
    - class_labels: list of str, class names in the dataset.
    - n_bins: int, number of bins for histograms (default: 30).
    - max_batch: int, maximum number of batches to process (default: -1, process all).

    Returns:
    - truth_histograms: dict of histograms, one per feature.
    - bin_edges: list of arrays, bin edges for each feature.
    """
    n_features = len(feature_names)
    n_classes = len(class_labels)
    bin_edges = [None] * n_features  # Initialize bin edges for each feature
    truth_histograms = {k: np.zeros((n_bins, n_classes)) for k in range(n_features)}

    i_batch = 0
    for batch in data_loader:
        data = batch['data']
        weights = batch['weights']
        raw_labels = batch['detailed_labels']

        # Convert raw labels to one-hot encoded format
        labels = np.array([class_labels.index(label.decode('utf-8')) for label in raw_labels])
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=n_classes)

        for k in range(n_features):
            feature_values = data[:, k]

            # Define bin edges only once
            if bin_edges[k] is None:
                bin_edges[k] = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)

            # Accumulate weights for each bin and class
            for b in range(n_bins):
                in_bin = (feature_values >= bin_edges[k][b]) & (feature_values < bin_edges[k][b + 1])
                bin_weights = weights[in_bin]

                if bin_weights.sum() > 0:
                    truth_histograms[k][b, :] += np.sum(bin_weights[:, None] * labels_one_hot[in_bin], axis=0)

        i_batch += 1
        if max_batch > 0 and i_batch >= max_batch:
            break

    return truth_histograms, bin_edges

def plot_truth_histograms(truth_histograms, bin_edges, class_labels, feature_names, output_dir):
    """
    Plot and save truth histograms as separate plots per feature, with top (yields) and bottom (ratios) panels.

    Parameters:
    - truth_histograms: dict, true class weights accumulated over bins.
    - bin_edges: list, bin edges for each feature.
    - class_labels: list of str, class names.
    - feature_names: list of str, feature names for the x-axis.
    - output_dir: str, directory to save the PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_features = len(truth_histograms)
    n_classes = len(class_labels)

    colors = plt.cm.tab10(np.arange(n_classes))  # Use tab10 colormap for distinct colors

    for k in range(n_features):
        # Create a figure with two panels
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Top panel: weighted yields
        for c, class_name in enumerate(class_labels):
            ax_top.step(
                bin_edges[k],  # Bin edges
                np.append(truth_histograms[k][:, c], 0),  # Heights, append 0 for the last step
                where="post",
                color=colors[c],
                linestyle="-",
                label=class_name,
            )
        ax_top.set_title(feature_names[k])
        ax_top.set_ylabel("Sum of Weights")
        ax_top.legend(loc="upper right")
        ax_top.grid(False)

        # Bottom panel: fractional contributions
        total_weights = truth_histograms[k].sum(axis=1, keepdims=True)
        fractional_weights = truth_histograms[k] / np.where(total_weights == 0, 1, total_weights)

        for c, class_name in enumerate(class_labels):
            ax_bottom.step(
                bin_edges[k],  # Bin edges
                np.append(fractional_weights[:, c], 0),  # Heights, append 0 for the last step
                where="post",
                color=colors[c],
                linestyle="-",
            )
        ax_bottom.set_xlabel(feature_names[k])  # Feature name for x-axis
        ax_bottom.set_ylabel("Fraction")
        ax_bottom.grid(False)

        # Save the figure
        output_file = os.path.join(output_dir, f"{feature_names[k].replace(' ', '_')}.png")
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)

        # Register the file with the syncer
        common.syncer.file_sync_storage.append(output_file)

        print(f"Saved plot for {feature_names[k]} to {output_file}.")

from data import get_data_loader, feature_names
data_loader = get_data_loader( n_split=1000 )
output_path = os.path.join(user.plot_directory, "plots")
os.makedirs(output_path, exist_ok=True)
copyIndexPHP( output_path)

# Initialize data loader and class labels
data_loader = get_data_loader(n_split=1000)
class_labels = [b'diboson', b'htautau', b'ttbar', b'ztautau']
class_labels = [label.decode('utf-8') for label in class_labels]  # Convert bytes to strings

# Accumulate histograms
truth_histograms, bin_edges = accumulate_truth_histograms(data_loader, class_labels, max_batch=1)

# Plot and save
plot_truth_histograms(truth_histograms, bin_edges, class_labels, feature_names, output_path)

common.syncer.sync()
