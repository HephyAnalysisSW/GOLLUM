#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os
import glob
import ROOT
ROOT.gStyle.SetOptStat(0)
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import random
import common.syncer
import common.user as user
import common.helpers

def load_data_from_dir(npz_dir):
    """
    Loads and concatenates data from all NPZ files in the given directory.
    Expects each NPZ file to contain keys 'mu_measured_down', 'mu_measured_up', and 'mu_true'.
    
    Parameters:
        npz_dir (str): Path to the directory containing NPZ files.
        
    Returns:
        tuple: (mu_measured_down_all, mu_measured_up_all, mu_true_all) as numpy arrays.
    """
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    if not npz_files:
        sys.exit(f"No NPZ files found in directory: {npz_dir}")
    
    down_list = []
    up_list = []
    center_list = []
    true_list = []
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
        except Exception as e:
            sys.exit(f"Error loading file {npz_file}: {e}")
        
        try:
            mu_measured_down = data['mu_measured_down']
            mu_measured_up   = data['mu_measured_up']
            mu_center        = data['mu_measured']
            mu_true          = data['mu_true']
        except KeyError as e:
            sys.exit(f"Missing expected key in NPZ file {npz_file}: {e}")
        
        down_list.append(mu_measured_down)
        up_list.append(mu_measured_up)
        center_list.append(mu_center)
        true_list.append(mu_true)
    
    # Concatenate the arrays along the first axis.
    mu_measured_down_all = np.concatenate(down_list, axis=0)
    mu_measured_up_all   = np.concatenate(up_list, axis=0)
    mu_measured_all      = np.concatenate(center_list, axis=0)
    mu_true_all          = np.concatenate(true_list, axis=0)
    
    return mu_measured_all, mu_measured_down_all, mu_measured_up_all, mu_true_all

def calc_penalty(coverage, Ntest=100):
    """
    Computes the piecewise function:
    
      f(c) = 1                                      if c ∈ [0.6827 − 2σ68, 0.6827 + 2σ68]
             1 + |(c − (0.6827 − 2σ68))/σ68|^4        if c < 0.6827 − 2σ68
             1 + |(c − (0.6827 + 2σ68))/σ68|^3        if c > 0.6827 + 2σ68
    
    """

    sigma68 = np.sqrt( (1-0.6827)*0.6827/Ntest )

    lower_bound = 0.6827 - 2 * sigma68
    upper_bound = 0.6827 + 2 * sigma68

    # If c is scalar, use simple if/elif/else.
    if np.isscalar(coverage):
        if lower_bound <= coverage <= upper_bound:
            return 1
        elif coverage < lower_bound:
            return 1 + abs((coverage - lower_bound) / sigma68)**4
        else:  # coverage > upper_bound
            return 1 + abs((coverage - upper_bound) / sigma68)**3
    else:
        # Assume c is an array-like; ensure it's a NumPy array.
        coverage = np.asarray(coverage)
        # Initialize result array with ones.
        result = np.ones_like(coverage, dtype=float)
        # Identify regions
        mask_lower = coverage < lower_bound
        mask_upper = coverage > upper_bound
        # Apply the piecewise formulas.
        result[mask_lower] = 1 + np.abs((coverage[mask_lower] - lower_bound) / sigma68)**4
        result[mask_upper] = 1 + np.abs((coverage[mask_upper] - upper_bound) / sigma68)**3
        return result
    
parser = argparse.ArgumentParser(
    description="Perform coverage tests on uncertainty predictions from NPZ files in a directory."
)
parser.add_argument(
    "--npzdir",
    default = "/scratch-cbe/users/robert.schoefbeck/Challenge/output/toyFits/v5_train/",
    type=str,
    help="Input directory containing NPZ files with 'mu_measured_down', 'mu_measured_up', and 'mu_true'."
)
#parser.add_argument( "--subdir", default = "", type=str,  )
#parser.add_argument( "--Ntest", default = 1000, type=int,  )
args = parser.parse_args()

#args.subdir += f"_Ntest{args.Ntest}"

quantiles = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]

# Create dictionaries to hold one TGraph per quantile for each observable.
score_graphs   = {q: ROOT.TGraph() for q in quantiles}
width_graphs   = {q: ROOT.TGraph() for q in quantiles}
cov_graphs     = {q: ROOT.TGraph() for q in quantiles}
penalty_graphs = {q: ROOT.TGraph() for q in quantiles}

# Set colors: ensure that 0.001 and 0.999 have the same color,
# and 0.01 and 0.99 have the same color.
color_map = {
    0.001: ROOT.kRed,
    0.01:  ROOT.kBlue,
    0.05:  ROOT.kMagenta,
    0.5:   ROOT.kBlack,
    0.95:  ROOT.kMagenta,
    0.99:  ROOT.kBlue,
    0.999: ROOT.kRed
}
# Apply color and marker style to each graph.
for q in quantiles:
    for graph in (score_graphs[q], width_graphs[q], cov_graphs[q], penalty_graphs[q]):
        graph.SetLineColor(color_map[q])
        graph.SetLineWidth(2)
        graph.SetMarkerStyle(20)
        graph.SetMarkerColor(color_map[q])

# Load and concatenate data from all NPZ files in the directory.
mu_measured, mu_measured_down, mu_measured_up, mu_true = load_data_from_dir(args.npzdir)

unique_mu_true = list(set(mu_true))

# make sure we relly have all 100 systematic variations for each mu_true value
good_mu_true = []
for mu_true_val in unique_mu_true:
    if list(mu_true).count(mu_true_val) == 100:
        good_mu_true.append( mu_true_val )

#lookup = {value: idx for idx, value in enumerate(mu_true)}

# apply nominal inflate for v5
inflate = 1.045
mu_measured_up_inf   = mu_measured + inflate*(mu_measured_up-mu_measured)
mu_measured_down_inf = mu_measured - inflate*(mu_measured-mu_measured_down) 

inside = (mu_true >= mu_measured_down_inf) & (mu_true <= mu_measured_up_inf)
width  = mu_measured_up_inf - mu_measured_down_inf

for i_plot in range(10):

    print ("i_plot", i_plot)
    N_meta = 10

    # Make these many coverage distributions
    coverages_meta = {} 
    for i_meta in range(N_meta):
        print("i_meta", i_meta) 
        random.shuffle(good_mu_true) 
        mu_true_selected = good_mu_true[:100]

        bootstraps_mu = np.random.choice(mu_true_selected, size=(1000, len(mu_true_selected)), replace=True)

        indices = np.array([np.array([np.where(mu_true == x) for x in bootstrap_mu]).flatten() for bootstrap_mu in bootstraps_mu ])

        coverages_meta[i_meta] = np.array( [inside[inds].mean() for inds in indices ])

    # Define a set of colors (you can add more if needed)
    color_list = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange, ROOT.kBlack, ROOT.kYellow+1]
    # Create a canvas to draw on
    c1 = ROOT.TCanvas("c1", "Multiple Histograms", 800, 600)
    # Container for the histograms
    hist_list = []
    for i_meta in range(N_meta):
        hist_list.append( common.helpers.make_TH1F( np.histogram(coverages_meta[i_meta], bins=np.linspace(0.63,0.7,50)) ) )
        hist_list[-1].SetLineColor(color_list[i_meta % len(color_list)])

    # Draw all histograms on the same canvas
    # The first is drawn normally; the rest use the "SAME" option
    for idx, hist in enumerate(hist_list):
        if idx == 0:
            hist.Draw()
        else:
            hist.Draw("SAME")

    # Update the canvas to display the histograms
    c1.Update()
    # Optionally, save the canvas as an image file
    c1.SetTitle("")

    c1.Print(os.path.join(user.plot_directory, "meta", f"coverages_{i_plot}.png")) 

    common.syncer.sync()
