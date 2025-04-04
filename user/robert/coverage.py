#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os
import glob
import ROOT

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform coverage tests on uncertainty predictions from NPZ files in a directory."
    )
    parser.add_argument(
        "--npzdir",
        default = "/scratch-cbe/users/robert.schoefbeck/Challenge/output/toyFits/v5_train/",
        type=str,
        help="Input directory containing NPZ files with 'mu_measured_down', 'mu_measured_up', and 'mu_true'."
    )
    parser.add_argument( "--subdir", default = "", type=str,  )
    parser.add_argument( "--Ntest", default = 1000, type=int,  )
    args = parser.parse_args()

    args.subdir += f"_Ntest{args.Ntest}"

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

    Nentries= 10000

    for inflate in np.linspace(1., 1.3, 76):

        mu_measured_up_inf   = mu_measured   + inflate*(mu_measured_up-mu_measured)
        mu_measured_down_inf = mu_measured - inflate*(mu_measured-mu_measured_down) 

        inside = (mu_true >= mu_measured_down_inf) & (mu_true <= mu_measured_up_inf)
        width  = mu_measured_up_inf - mu_measured_down_inf

        score_list = []
        width_list = []
        cov_list   = []
        penalty_list = []

        # Loop over the array in chunks of Ntest elements
        while True:
            # raudnom shffling
            indices = np.random.permutation(inside.shape[0])
            inside  = inside[indices]
            width   = width[indices]

            for i_batch in range(0, inside.shape[0], args.Ntest):
                batch_inside = inside[i_batch:i_batch+args.Ntest]
                batch_width  = width[i_batch:i_batch+args.Ntest]

                coverage = batch_inside.mean()
                avg_width= batch_width.mean()

                penalty  = calc_penalty( coverage, args.Ntest )
                epsilon = 0.01
                score = -np.log( (avg_width+epsilon)*penalty )
                score_list.append(score)
                width_list.append( avg_width )
                cov_list.append( coverage )
                penalty_list.append( penalty )
                #print (f"#{i_batch//1000:3} width: {avg_width:.5f} cov: {coverage:.5f} penalty: {penalty:.5f} score: {score:+.5f}")

            if len(score_list)>Nentries: break

        score_quantiles   = np.quantile(score_list, quantiles )
        width_quantiles   = np.quantile(width_list, quantiles )
        cov_quantiles     = np.quantile(cov_list, quantiles )
        penalty_quantiles = np.quantile(penalty_list, quantiles )

        # Add a point (inflate, quantile value) to each corresponding TGraph.
        for j, q in enumerate(quantiles):
            score_graphs[q].SetPoint(score_graphs[q].GetN(), inflate, score_quantiles[j])
            width_graphs[q].SetPoint(width_graphs[q].GetN(), inflate, width_quantiles[j])
            cov_graphs[q].SetPoint(cov_graphs[q].GetN(), inflate, cov_quantiles[j])
            penalty_graphs[q].SetPoint(penalty_graphs[q].GetN(), inflate, penalty_quantiles[j])

def draw_canvas(canvas_name, title, graph_dict, y_range=None):
    c = ROOT.TCanvas(canvas_name, title, 800, 600)
    
   # Determine the overall y-axis minimum and maximum over all graphs.
    if y_range is None:
        overall_min = float('inf')
        overall_max = float('-inf')
        for q in quantiles:
            gr = graph_dict[q]
            n_points = gr.GetN()
            y_array = gr.GetY()  # Get the pointer to the array of y-values.
            for i in range(n_points):
                y_val = y_array[i]
                overall_min = min(overall_min, y_val)
                overall_max = max(overall_max, y_val)
    else:
        overall_min, overall_max = y_range
 
    first = True
    leg = ROOT.TLegend(0.65, 0.15, 0.88, 0.38)
    for q in quantiles:
        draw_option = "AL" if first else "L same"
        gr = graph_dict[q]
        gr.GetXaxis().SetTitle("inflate")
        gr.GetYaxis().SetTitle(title)
        if first:
            # Set the overall y-axis range on the first graph.
            gr.GetYaxis().SetRangeUser(overall_min, overall_max)
        gr.Draw(draw_option)
        leg.AddEntry(gr, f"q = {q}", "l")
        first = False
    leg.Draw()
    c.Update()

    plot_directory = os.path.join(user.plot_directory, "quantiles", args.subdir)
    common.helpers.copyIndexPHP(plot_directory)
    c.Print(os.path.join( plot_directory, canvas_name + '.png'))


c_score   = draw_canvas("c_score", "Score Quantiles", score_graphs, y_range = (-2, 1) )
c_width   = draw_canvas("c_width", "Width Quantiles", width_graphs )
c_cov     = draw_canvas("c_cov", "Coverage Quantiles", cov_graphs)
c_penalty = draw_canvas("c_penalty", "Penalty Quantiles", penalty_graphs)

