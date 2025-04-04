#!/usr/bin/env python3
import argparse
import numpy as np
import ROOT

from array import array
import os, glob, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')
import common.user as user
import common.syncer

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Parse command line arguments for unbinned and binned directories.
parser = argparse.ArgumentParser(
    description="Plot mu_fit vs. f_min using PyROOT TGraph from npz files for unbinned and binned likelihood fits"
)
parser.add_argument("--unbinned", help="Directory containing the unbinned npz files")
parser.add_argument("--binned", help="Directory containing the binned npz files")
args = parser.parse_args()

# Function to load and sort data from a directory of npz files by mu_fit.
def load_data(directory):
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    if not npz_files:
        print("No npz files found in directory: {}".format(directory))
        sys.exit(1)
    mu_fit_vals = []
    f_min_vals = []
    for file in npz_files:
        data = np.load(file)
        mu_fit = data['mu_fit']
        f_min = data['f_min']
        # Convert one-element numpy arrays to scalars if necessary.
        if isinstance(mu_fit, np.ndarray) and mu_fit.size == 1:
            mu_fit = mu_fit.item()
        if isinstance(f_min, np.ndarray) and f_min.size == 1:
            f_min = f_min.item()
        mu_fit_vals.append(mu_fit)
        f_min_vals.append(f_min)
    # Sort the data by mu_fit
    sorted_indices = sorted(range(len(mu_fit_vals)), key=lambda i: mu_fit_vals[i])
    sorted_mu_fit = [mu_fit_vals[i] for i in sorted_indices]
    sorted_f_min = [f_min_vals[i] for i in sorted_indices]
    return sorted_mu_fit, sorted_f_min

# Load unbinned and binned data.
mu_fit_unbinned, f_min_unbinned = load_data(args.unbinned)
mu_fit_binned, f_min_binned = load_data(args.binned)

# Create C-style arrays for ROOT TGraph.
n_unbinned = len(mu_fit_unbinned)
n_binned   = len(mu_fit_binned)

x_unbinned = array('d', mu_fit_unbinned)
y_unbinned = array('d', f_min_unbinned)
x_binned   = array('d', mu_fit_binned)
y_binned   = array('d', f_min_binned)

# Create TGraphs.
graph_unbinned = ROOT.TGraph(n_unbinned, x_unbinned, y_unbinned)
graph_unbinned.SetTitle(";Signal strength #mu;-2#Delta log L")
graph_unbinned.SetMarkerStyle(20)
graph_unbinned.SetMarkerColor(ROOT.kBlue)
graph_unbinned.SetLineColor(ROOT.kBlue)
graph_unbinned.SetLineWidth(2)
graph_unbinned.SetMarkerSize(1)

graph_binned = ROOT.TGraph(n_binned, x_binned, y_binned)
graph_binned.SetMarkerStyle(20)
graph_binned.SetMarkerColor(ROOT.kGray)
graph_binned.SetLineColor(ROOT.kGray)
graph_binned.SetLineWidth(2)
graph_binned.SetMarkerSize(1)

# Determine overall x and y ranges.
all_mu = mu_fit_unbinned + mu_fit_binned
all_fmin = f_min_unbinned + f_min_binned
x_min = min(all_mu)
x_max = max(all_mu)
# Force the y-axis to start at 0.
y_min = 0  
y_max = 30 #max(all_fmin) * 1.1  # add a 10% margin

# Create a canvas and draw the graphs.
c = ROOT.TCanvas("c", "Likelihood Fit", 800, 600)

# Move the frame more to the left by increasing the left margin.
c.SetLeftMargin(0.15)
c.SetRightMargin(0.1)

# Move the y-axis labels and title closer to the axis.
graph_unbinned.GetYaxis().SetLabelOffset(0.005)
graph_unbinned.GetYaxis().SetTitleOffset(0.9)

# Draw unbinned graph first (with axes) then overlay binned points.
graph_unbinned.Draw("AL")
graph_unbinned.GetXaxis().SetLimits(x_min, x_max)
graph_unbinned.GetYaxis().SetRangeUser(y_min, y_max)
graph_binned.Draw("L SAME")

# Add a legend.
legend = ROOT.TLegend(0.25, 0.68, 0.48, 0.81)
legend.AddEntry(graph_unbinned, "Unbinned", "p")
legend.AddEntry(graph_binned, "Binned", "p")
legend.SetBorderSize(0)
legend.SetShadowColor(0)
legend.Draw()

# Compute the global best-fit (minimum f_min) among all points.
global_fmin = min(all_fmin)

# Draw horizontal lines for 1, 2, 3, 4, 5 sigma thresholds.
# For a chi2 with 1 dof, the thresholds are: global_fmin + sigma^2.
stuff=[]
for sigma in range(1, 6):
    threshold = global_fmin + sigma**2
    line = ROOT.TLine(x_min, threshold, x_max, threshold)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("same")
    # Optional: label each line with its sigma value.
    latex = ROOT.TLatex()
    latex.SetTextSize(0.03)
    latex.DrawLatex(x_max, threshold, f"{sigma}#sigma")
    stuff.append(line)

c.Update()

# Get subdirectory names from input directories for naming the output file.
subdir_unbinned = os.path.basename(os.path.normpath(args.unbinned))
subdir_binned   = os.path.basename(os.path.normpath(args.binned))
output_png = os.path.join(user.plot_directory, "asimov_fit", f"likelihood_unbinned_{subdir_unbinned}_binned_{subdir_binned}_fit.png")
output_pdf = os.path.join(user.plot_directory, "asimov_fit", f"likelihood_unbinned_{subdir_unbinned}_binned_{subdir_binned}_fit.pdf")

# Save the canvas as PNG and PDF.
c.SaveAs(output_png)
c.SaveAs(output_pdf)

common.syncer.sync()
