#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import ROOT
import iminuit  # Import iMinuit since the covs come from iMinuit
from array import array
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

import common.user as user
import common.helpers
import common.syncer
# Parse the input directory argument.
parser = argparse.ArgumentParser(
    description="Concatenate npz file arrays into one dataset."
)
parser.add_argument("--input_dir", required=True,
                    help="Directory containing npz files")
args = parser.parse_args()

# Find all npz files in the input directory.
npz_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
if not npz_files:
    print("No npz files found in directory: {}".format(args.input_dir))
    exit(1)

# Define the keys expected in each npz file.
keys = [
    "mu_true", "mu_measured", "mu_measured_up", "mu_measured_down",
    "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson",
    "nu_tes_true", "nu_jes_true", "nu_met_true", "nu_bkg_true", "nu_tt_true", "nu_diboson_true"
]

# Initialize a dictionary to accumulate arrays for each key.
data_dict = {key: [] for key in keys}
covs_list = []  # To hold covariance matrices

# Loop over each npz file and load its data.
for file in npz_files:
    npz_data = np.load(file, allow_pickle=True)
    for key in keys:
        if key in npz_data:
            data_dict[key].append(npz_data[key])
    # Process the covariance matrices.
    covs_list.append(np.array(npz_data["covs"]))

# Concatenate arrays for each key along the first axis.
concatenated_data = {}
for key, arrays in data_dict.items():
    try:
        concatenated_data[key] = np.concatenate(arrays, axis=0)
    except Exception as e:
        print(f"Could not concatenate key {key}: {e}")
        concatenated_data[key] = arrays  # Fall back to a list if concatenation fails.

# Convert the list of covariance matrices to a 3D array: (N, Nvar, Nvar)
covs_array = np.concatenate(np.array(covs_list)) 

concatenated_data['nu_jes_true'] = (concatenated_data['nu_jes_true']-1)/0.01
concatenated_data['nu_tes_true'] = (concatenated_data['nu_tes_true']-1)/0.01
concatenated_data['nu_bkg_true'] = np.log(concatenated_data['nu_bkg_true'])/np.log(1+0.001)
concatenated_data['nu_tt_true'] = np.log(concatenated_data['nu_tt_true'])/np.log(1+0.02)
concatenated_data['nu_diboson_true'] = np.log(concatenated_data['nu_diboson_true'])/np.log(1+0.25)

# Print the shapes of the concatenated data.
print("Concatenated dataset:")
for key, value in concatenated_data.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape {value.shape}")
    else:
        print(f"  {key}: list of length {len(value)}")
print("Covariance matrices array shape:", covs_array.shape)

# Define the list of variables (base names) in the correct order.
variables = ["mu", "nu_tes", "nu_jes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kWaterMelon);

subdir = os.path.basename(os.path.normpath(args.input_dir))
# Loop over each variable to create the plots.
for idx, var in enumerate(variables):
    # Determine axis limits: for nu_met use [0, 5.1], for nmu use [0, 3.1], else [-5.1, 5.1].
    if var == "nu_met":
        axis_min = 0
        axis_max = 5.1
        nbins = 51
    elif var == "mu":
        axis_min = .1
        axis_max = 3
        nbins = 31
    else:
        axis_min = -5.1
        axis_max = 5.1
        nbins = 51

    # 1) Difference histogram: (measured - true) normalized by uncertainty.
    # For mu the keys are "mu_measured" and "mu_true", for others use "<var>" and "<var>_true".
    key_measured = f"{var}_measured" if var == "mu" else var
    key_true     = f"{var}_true"
    
    # Compute the difference.
    diff = concatenated_data[key_measured] - concatenated_data[key_true]
    # Get the uncertainty from the covariance matrix diagonal.
    uncertainty = np.sqrt(covs_array[:, idx, idx])
    # Normalize the difference.
    diff_norm = diff / uncertainty
    
    # Create a square canvas.
    c_diff = ROOT.TCanvas(f"c_diff_{var}", "", 600, 600)

    # Create a TH1F with 50 bins spanning the fixed axis limits.
    h_diff = ROOT.TH1F(f"h_diff_{var}", f"; (measured-true)/#sigma; Counts", nbins, axis_min, axis_max)
    # Fill the histogram with the normalized differences.
    for d in diff_norm:
        h_diff.Fill(d)
    # Force the x-axis limits.
    h_diff.GetXaxis().SetRangeUser(axis_min, axis_max)
    h_diff.Draw()
    c_diff.Update()

    # Save the difference histogram.
    dir_ = os.path.join(user.plot_directory, "toy_study", subdir)
    common.helpers.copyIndexPHP(dir_)
    os.makedirs(dir_, exist_ok=True)
    c_diff.Print(os.path.join(dir_, f"diff_{var}.png"))
    c_diff.Print(os.path.join(dir_, f"diff_{var}.pdf"))
    
    # 2) Correlation plot using TGraphErrors:
    # x-axis: true value, y-axis: measured value; error is from the covariance diagonal.
    x_vals = concatenated_data[key_true]
    y_vals = concatenated_data[key_measured]
    # For the error, use the square-root of the variance from the covariance matrix.
    err_y = np.sqrt(covs_array[:, idx, idx])
    
    n_points = len(x_vals)
    x_arr = array('d', x_vals.tolist())
    y_arr = array('d', y_vals.tolist())
    ex_arr = array('d', [0.0]*n_points)  # no error in x.
    err_arr = array('d', err_y.tolist())
    
    c_corr = ROOT.TCanvas(f"c_corr_{var}", "", 600, 600)
    gr = ROOT.TGraphErrors(n_points, x_arr, y_arr, ex_arr, err_arr)
    gr.SetTitle(f"Correlation for {var};{var}_true;{var}_measured")
    gr.SetMarkerStyle(20)
    gr.Draw("AP")
    # Set equal axis limits to force a 1:1 aspect.
    gr.GetXaxis().SetLimits(axis_min, axis_max)
    gr.GetHistogram().SetMinimum(axis_min)
    gr.GetHistogram().SetMaximum(axis_max)
    c_corr.Update()
    
    c_corr.Print(os.path.join(dir_, f"corr_{var}.png"))
    c_corr.Print(os.path.join(dir_, f"corr_{var}.pdf"))
    
    # 3) TH2F COLZ plot:
    c_corr2 = ROOT.TCanvas(f"c_corr2_{var}", "", 600, 600)
    c_corr2.SetRightMargin(0.15)
    # Create a 2D histogram with 50 bins in both directions and fixed axis ranges.
    h2 = ROOT.TH2F(f"h2_{var}", f"2D Correlation for {var};{var}_true;{var}_measured", nbins, axis_min, axis_max, nbins, axis_min, axis_max)
    for i in range(n_points):
        h2.Fill(x_vals[i], y_vals[i])
    h2.Draw("COLZ")
    c_corr2.Update()
    c_corr2.SetLogz(1)
    c_corr2.Print(os.path.join(dir_, f"corr2_{var}.png"))
    c_corr2.Print(os.path.join(dir_, f"corr2_{var}.pdf"))

common.syncer.sync()

