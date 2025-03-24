#!/usr/bin/env python

import sys, os
import numpy as np
import ROOT
from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer
import common.helpers as helpers
import common.data_structure as data_structure

# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
argParser.add_argument("--modelDir", action="store", default="/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6",  help="Directory containing the trained TFMC model.")
argParser.add_argument("--ratio", action="store_true", default=False, help="Plot ratio?")
args = argParser.parse_args()


# let us load a classifier! 
from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load(args.modelDir)

import common.datasets_hephy as datasets_hephy

# Load the data
data_loader = {}
data_loader['nominal'] = datasets_hephy.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split)

# systmatics data_loader: values = (tes, jes, nu) where for jes/tes we have 1.01 is +1 sigma, 0.99 is -1 sigma, etc. and for MET, the value directly is the sigma (between 0 and 3)
#datasets_hephy.get_data_loader( selection='lowMT_VBFJet', values=(1.01,1.01,0))
# all available combinations are here: ls /scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/lowMT_VBFJet

variations = ['nominal'] + [f"{var}_{direction}" for var in data_structure.systematics for direction in ["up", "down"]]
data_loader['tes_up'] = datasets_hephy.get_data_loader( selection=args.selection, values=(1.01,1.0,0),selection_function=None, n_split=args.n_split)
data_loader['tes_down'] = datasets_hephy.get_data_loader( selection=args.selection, values=(0.99,1.0,0),selection_function=None, n_split=args.n_split)
data_loader['jes_up'] = datasets_hephy.get_data_loader( selection=args.selection, values=(1.0,1.01,0),selection_function=None, n_split=args.n_split)
data_loader['jes_down'] = datasets_hephy.get_data_loader( selection=args.selection, values=(1.0,0.99,0),selection_function=None, n_split=args.n_split)
data_loader['met_up'] = datasets_hephy.get_data_loader( selection=args.selection, values=(1.0,1.0,1),selection_function=None, n_split=args.n_split)
data_loader['met_down'] = datasets_hephy.get_data_loader( selection=args.selection, values=(1.0,1.0,0),selection_function=None, n_split=args.n_split)

# This is how you change the normalization uncertainty to +1 sigma: Example for ttbar
#nu_tt = 1
#nu_diboson = 1
#nu_bkg = 1
#alpha_bkg = 0.001
#alpha_tt = 0.02
#alpha_diboson = 0.25
#weights[labels==data_structure.label_encoding['ttbar']] = weights[labels==data_structure.label_encoding['ttbar']]*(1+self.alpha_tt)**nu_tt


weight_lib = {
    "norm_ttbar": {
        "condition": lambda labels, ds: labels == ds.label_encoding["ttbar"],
        "weight": lambda weights, alpha, nu: weights * (1 + alpha) ** nu,
        "process": "ttbar",
        "alpha":0.02,
        "nu":1
    },
    "norm_diboson": {
        "condition": lambda labels, ds: labels == ds.label_encoding["diboson"],
        "weight": lambda weights, alpha, nu: weights * (1 + alpha) ** nu,
        "process": "diboson",
        "alpha":0.25,
        "nu":1
    },
    "norm_bkg": {
        "condition": lambda labels, ds: labels != ds.label_encoding["htautau"],
        "weight": lambda weights, alpha, nu: weights * (1 + alpha) ** nu,
        "process": "bkg",
        "alpha":0.001,
        "nu":1
    },
}

max_batch = 1 if args.small else -1

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "paper_plots", args.selection)
if args.small: plot_directory = os.path.join(user.plot_directory, "paper_plots", args.selection+"_small")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Initialize histogram with zero (one per label and variable)
histograms = {}
for var, options in data_structure.plot_options.items():
    histograms[var] = {}
    binning = options['binning']
    options["bin_edges"] = np.linspace(binning[1], binning[2], binning[0] + 1)
    for label in data_structure.labels:
        histograms[var][label] = {}
        for variation in variations:
            histograms[var][label][variation] = np.zeros(binning[0])
        for weight_name in weight_lib: histograms[var][label][weight_name] = np.zeros(binning[0])
    for direction in ['up','down']:
        histograms[var]['sb'] = {}
        histograms[var]['sb'][direction] = np.zeros(binning[0])
        histograms[var]['calib'] = {}
        histograms[var]['calib'][direction] = np.zeros(binning[0])
        histograms[var]['norm'] = {}
        histograms[var]['norm'][direction] = np.zeros(binning[0])

        histograms[var]['s'] = {}
        histograms[var]['s'][direction] = np.zeros(binning[0])
        histograms[var]['s_calib'] = {}
        histograms[var]['s_calib'][direction] = np.zeros(binning[0])
        histograms[var]['s_norm'] = {}
        histograms[var]['s_norm'][direction] = np.zeros(binning[0])

        histograms[var]['b'] = {}
        histograms[var]['b'][direction] = np.zeros(binning[0])
        histograms[var]['b_calib'] = {}
        histograms[var]['b_calib'][direction] = np.zeros(binning[0])
        histograms[var]['b_norm'] = {}
        histograms[var]['b_norm'][direction] = np.zeros(binning[0])


# Loop over data batches and calculate predictions
total_batches = len(data_loader['nominal'])
for variation, data_var in data_loader.items():
    for i_batch, batch in enumerate(tqdm(data_var, total=total_batches, desc=f"Batches ({variation})")):
        features, weights, labels = data_var.split(batch)
        predictions = tfmc.predict(features, ic_scaling=False)
    
        # Accumulate predicted probabilities for each class and label
        for label_idx, label in enumerate(data_structure.labels):
            true_indices = (labels == label_idx)
            for i_var, var in enumerate( data_structure.feature_names ):
                if np.any(true_indices):
                    histogram_batch, _ = np.histogram(
                        features[true_indices, i_var],
                        bins=data_structure.plot_options[var]['bin_edges'],
                        weights=weights[true_indices]
                    )
                    histograms[var][label][variation] += histogram_batch 
        
        # do another loop for weights 
        if variation == 'nominal':
          for weight_name, weight_info in weight_lib.items():
            #weights[labels==data_structure.label_encoding['ttbar']] = weights[labels==data_structure.label_encoding['ttbar']]*(1+alpha)**nu
            condition_mask = weight_info["condition"](labels, data_structure)
            alpha, nu = weight_info["alpha"], weight_info["nu"]
            weights[condition_mask] = weight_info["weight"](weights[condition_mask], alpha, nu)
            for label_idx, label in enumerate(data_structure.labels):
                true_indices = (labels == label_idx)
                for i_var, var in enumerate( data_structure.feature_names ):
                    if np.any(true_indices):
                        histogram_batch, _ = np.histogram(
                            features[true_indices, i_var],
                            bins=data_structure.plot_options[var]['bin_edges'],
                            weights=weights[true_indices]
                        )
                        histograms[var][label][weight_name] += histogram_batch 
    
        if max_batch > 0 and i_batch + 1 >= max_batch:
            break

def getRatioLine( h1):
        line = h1.Clone()
        Nbins = line.GetSize()-2
        for i in range(Nbins):
            bin=i+1
            line.SetBinContent(bin,0.0)
            line.SetBinError(bin,0.0)
            line.SetFillColor(0)
            line.SetLineColor(13)
            line.SetLineWidth(2)
        return line

def setLabel(text):

    cmsText = ROOT.TLatex(0.2, 0.85, text)
    cmsText.SetNDC(True)
    cmsText.SetTextAlign(13)
    cmsText.SetTextFont(62)
    cmsText.SetTextSize(0.04)
    cmsText.Draw()
    return cmsText

# systematic deltas
for i_var, var in enumerate( data_structure.feature_names ):
    nominal_ref = histograms[var][data_structure.labels[0]]['nominal']
    up_err2 = np.zeros_like(nominal_ref)
    down_err2 = np.zeros_like(nominal_ref)
    calib_up_err2 = np.zeros_like(nominal_ref)
    calib_down_err2 = np.zeros_like(nominal_ref)
    norm_up_err2 = np.zeros_like(nominal_ref)
    norm_down_err2 = np.zeros_like(nominal_ref)
    
    s_up_err2 = np.zeros_like(nominal_ref)
    s_down_err2 = np.zeros_like(nominal_ref)
    s_calib_up_err2 = np.zeros_like(nominal_ref)
    s_calib_down_err2 = np.zeros_like(nominal_ref)
    s_norm_up_err2 = np.zeros_like(nominal_ref)
    s_norm_down_err2 = np.zeros_like(nominal_ref)

    b_up_err2 = np.zeros_like(nominal_ref)
    b_down_err2 = np.zeros_like(nominal_ref)
    b_calib_up_err2 = np.zeros_like(nominal_ref)
    b_calib_down_err2 = np.zeros_like(nominal_ref)
    b_norm_up_err2 = np.zeros_like(nominal_ref)
    b_norm_down_err2 = np.zeros_like(nominal_ref)

    for sys in data_structure.systematics:
        shift_up = np.zeros_like(nominal_ref)
        shift_down = np.zeros_like(nominal_ref)
        s_shift_up = np.zeros_like(nominal_ref)
        s_shift_down = np.zeros_like(nominal_ref)
        b_shift_up = np.zeros_like(nominal_ref)
        b_shift_down = np.zeros_like(nominal_ref)
        for label_idx, label in enumerate(data_structure.labels):
           initial_up = histograms[var][label][f'{sys}_up'] - histograms[var][label]['nominal']
           initial_down = histograms[var][label][f'{sys}_down'] - histograms[var][label]['nominal']
           
           shift_up += np.where(initial_up >= 0, initial_up, 0) + np.where(initial_down >= 0, initial_down, 0)
           shift_down += np.where(initial_up < 0, initial_up, 0) + np.where(initial_down < 0, initial_down, 0)

           if label == 'htautau':
               s_shift_up += np.where(initial_up >= 0, initial_up, 0) + np.where(initial_down >= 0, initial_down, 0)
               s_shift_down += np.where(initial_up < 0, initial_up, 0) + np.where(initial_down < 0, initial_down, 0)
           else :
               b_shift_up += np.where(initial_up >= 0, initial_up, 0) + np.where(initial_down >= 0, initial_down, 0)
               b_shift_down += np.where(initial_up < 0, initial_up, 0) + np.where(initial_down < 0, initial_down, 0)
        up_err2   += np.power(shift_up,2)
        down_err2 += np.power(shift_down,2)
        calib_up_err2   += np.power(shift_up,2)
        calib_down_err2 += np.power(shift_down,2)
        b_up_err2   += np.power(b_shift_up,2)
        b_down_err2 += np.power(b_shift_down,2)
        b_calib_up_err2   += np.power(b_shift_up,2)
        b_calib_down_err2 += np.power(b_shift_down,2)
        s_up_err2   += np.power(s_shift_up,2)
        s_down_err2 += np.power(s_shift_down,2)
        s_calib_up_err2   += np.power(s_shift_up,2)
        s_calib_down_err2 += np.power(s_shift_down,2)
    for weight_name in weight_lib:
        shift_up = np.zeros_like(nominal_ref)
        shift_down = np.zeros_like(nominal_ref)
        s_shift_up = np.zeros_like(nominal_ref)
        s_shift_down = np.zeros_like(nominal_ref)
        b_shift_up = np.zeros_like(nominal_ref)
        b_shift_down = np.zeros_like(nominal_ref)
        for label_idx, label in enumerate(data_structure.labels):
           initial = histograms[var][label][weight_name] - histograms[var][label]['nominal']
           shift_up += np.where(initial >= 0, initial, 0) 
           shift_down += np.where(initial < 0, initial, 0)
           if label == 'htautau':
               s_shift_up += np.where(initial >= 0, initial, 0) 
               s_shift_down += np.where(initial < 0, initial, 0)
           else:
               b_shift_up += np.where(initial >= 0, initial, 0) 
               b_shift_down += np.where(initial < 0, initial, 0)

        up_err2   += np.power(shift_up,2)
        down_err2 += np.power(shift_down,2)
        norm_up_err2   += np.power(shift_up,2)
        norm_down_err2 += np.power(shift_down,2)
        b_up_err2   += np.power(b_shift_up,2)
        b_down_err2 += np.power(b_shift_down,2)
        b_norm_up_err2   += np.power(b_shift_up,2)
        b_norm_down_err2 += np.power(b_shift_down,2)
        s_up_err2   += np.power(s_shift_up,2)
        s_down_err2 += np.power(s_shift_down,2)
        s_norm_up_err2   += np.power(s_shift_up,2)
        s_norm_down_err2 += np.power(s_shift_down,2)
        
    histograms[var]['sb']['up'] = np.sqrt(up_err2)
    histograms[var]['sb']['down'] = np.sqrt(down_err2)
    histograms[var]['calib']['up'] = np.sqrt(calib_up_err2)
    histograms[var]['calib']['down'] = np.sqrt(calib_down_err2)
    histograms[var]['norm']['up'] = np.sqrt(norm_up_err2)
    histograms[var]['norm']['down'] = np.sqrt(norm_down_err2)

    histograms[var]['s']['up'] = np.sqrt(s_up_err2)
    histograms[var]['s']['down'] = np.sqrt(s_down_err2)
    histograms[var]['s_calib']['up'] = np.sqrt(s_calib_up_err2)
    histograms[var]['s_calib']['down'] = np.sqrt(s_calib_down_err2)
    histograms[var]['s_norm']['up'] = np.sqrt(s_norm_up_err2)
    histograms[var]['s_norm']['down'] = np.sqrt(s_norm_down_err2)

    histograms[var]['b']['up'] = np.sqrt(b_up_err2)
    histograms[var]['b']['down'] = np.sqrt(b_down_err2)
    histograms[var]['b_calib']['up'] = np.sqrt(b_calib_up_err2)
    histograms[var]['b_calib']['down'] = np.sqrt(b_calib_down_err2)
    histograms[var]['b_norm']['up'] = np.sqrt(b_norm_up_err2)
    histograms[var]['b_norm']['down'] = np.sqrt(b_norm_down_err2)

# Plot the predicted probabilities for each predicted class using ROOT
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Combined library for LaTeX labels and colors
lib = {
    'htautau': {"latex": r'H #rightarrow #tau^{+}#tau^{-}', "color": ROOT.TColor.GetColor("#F28E2B")},
    'ztautau': {"latex": r'Z #rightarrow #tau^{+}#tau^{-}', "color": ROOT.TColor.GetColor("#59A14F")},
    'ttbar':   {"latex": r't#bar{t}',              "color": ROOT.TColor.GetColor("#E15759")},
    'diboson': {"latex": r'VV',                    "color": ROOT.TColor.GetColor("#4E79A7")},
}

# Loop over variables 
for i_var, var in enumerate( data_structure.feature_names ):
    canvas = ROOT.TCanvas("","", 800, 600)
    if args.ratio:
        canvas.SetCanvasSize(800,800)
        yRatio = 0.31
        canvas.Divide(1,2,0,0)
        topPad = canvas.cd(1)
        topPad.SetBottomMargin(0)
        topPad.SetLeftMargin(0.15)
        topPad.SetTopMargin(0.07)
        topPad.SetRightMargin(0.05)
        topPad.SetPad(topPad.GetX1(), yRatio , topPad.GetX2(), topPad.GetY2())

        bottomPad = canvas.cd(2)
        bottomPad.SetPad(bottomPad.GetX1(), bottomPad.GetY1(), bottomPad.GetX2(), yRatio)
        bottomPad.SetTopMargin(0)
        bottomPad.SetRightMargin(0.05)
        bottomPad.SetLeftMargin(0.15)
        bottomPad.SetBottomMargin(0.13*3.5)
    else:
        topPad = canvas
        topPad.SetBottomMargin(0.13)
        topPad.SetLeftMargin(0.15)
        topPad.SetTopMargin(0.07)
        topPad.SetRightMargin(0.05)

    # make the stack plot first
    topPad.cd()
    log = data_structure.plot_options[var]['logY']
    topPad.SetLogy(log)

    # Create a histogram stack for visualization
    max_y = 0  # To set uniform Y-axis limits
    histograms_for_class = {}
    hist_stack = ROOT.THStack("hist_stack", var)
    hist_all_stack = ROOT.THStack("hist_all_stack", var)

    for label in data_structure.labels:
        h = ROOT.TH1F(f"{var}_{label}", var, *data_structure.plot_options[var]['binning'])
        for i, val in enumerate(histograms[var][label]['nominal']):
            h.SetBinContent(i + 1, val)
        
        # Retrieve LaTeX and color info, or fall back to defaults
        props = lib.get(label, {"latex": label, "color": ROOT.kBlack})
        h.SetFillColor(props["color"])
        h.SetLineColor(props["color"])
        h.SetLineWidth(2)
        
        histograms_for_class[label] = {
            "hist": h,
            "integral": h.Integral(),
            "latex": props["latex"]
        }
    
    max_y = max(item["hist"].GetMaximum() for item in histograms_for_class.values())
    
    # Sort histograms by their integral (ascending order)
    sorted_histograms = sorted(histograms_for_class.items(), key=lambda item: item[1]['integral'])
    
    # Add histograms to stacks in sorted order
    for label, data in sorted_histograms:
        h = data["hist"]
        if "htautau" not in h.GetName():
            hist_stack.Add(h)
        hist_all_stack.Add(h)    

    #make the systematic band
    n_bins = data_structure.plot_options[var]['binning'][0]
    bin_edges = data_structure.plot_options[var]['bin_edges']
    bin_width = bin_edges[1]-bin_edges[0]
    x_values = np.array([(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)])
    hist_sum = hist_stack.GetStack().Last()
    y_values = np.array([hist_sum.GetBinContent(i+1) for i in range(n_bins)])
    total_unc = ROOT.TGraphAsymmErrors(n_bins, x_values, y_values)
    calib_unc = ROOT.TGraphAsymmErrors(n_bins, x_values, y_values)
    norm_unc = ROOT.TGraphAsymmErrors(n_bins, x_values, y_values)
    for i_bin in range(n_bins):
        total_unc.SetPointError(i_bin, bin_width/2, bin_width/2,
            histograms[var]['b']['down'][i_bin],histograms[var]['b']['up'][i_bin])
        calib_unc.SetPointError(i_bin, bin_width/2, bin_width/2,
            histograms[var]['b_calib']['down'][i_bin],histograms[var]['b_calib']['up'][i_bin])
        norm_unc.SetPointError(i_bin, bin_width/2, bin_width/2,
            histograms[var]['b_norm']['down'][i_bin],histograms[var]['b_norm']['up'][i_bin])
    total_unc.SetFillStyle(3005)
    total_unc.SetFillColor(13)
    total_unc.SetLineWidth(0)
    total_unc.SetMarkerStyle(0)
    total_unc.SetMarkerSize(0)
    calib_unc.SetFillStyle(3004)
    calib_unc.SetFillColor(ROOT.TColor.GetColor("#0072B2"))
    calib_unc.SetLineWidth(0)
    calib_unc.SetMarkerStyle(0)
    calib_unc.SetMarkerSize(0)
    norm_unc.SetFillStyle(3354)
    norm_unc.SetFillColor(ROOT.TColor.GetColor("#D55E00"))
    norm_unc.SetLineWidth(0)
    norm_unc.SetMarkerStyle(0)
    norm_unc.SetMarkerSize(0)

    #set drawing options
    hist_sum.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
    hist_sum.GetYaxis().SetTitle("Events")
    hist_sum.SetTitle(data_structure.plot_options[var]['tex'])
    hist_sum.GetXaxis().SetTitleFont(43)
    hist_sum.GetYaxis().SetTitleFont(43)
    hist_sum.GetXaxis().SetLabelFont(43)
    hist_sum.GetYaxis().SetLabelFont(43)
    hist_sum.GetXaxis().SetTitleSize(28)
    hist_sum.GetYaxis().SetTitleSize(28)
    hist_sum.GetXaxis().SetLabelSize(24)
    hist_sum.GetYaxis().SetLabelSize(24)
    hist_sum.GetYaxis().SetTitleOffset(1.3 if not args.ratio else 1.6)
    y_max = hist_sum.GetBinContent(hist_sum.GetMaximumBin())
    y_factor = 1.3 
    if log:  y_factor = 1.3 * 100
    hist_sum.GetYaxis().SetRangeUser(1e-1,y_max*y_factor)
    ROOT.gPad.Modified()  # Ensures updated axis titles appear
    ROOT.gPad.Update()
    hist_sum.Draw()

    #draw the stacked histograms
    hist_all_stack.Draw("HIST same")

    #draw the uncertainty band
    total_unc.Draw("E2 hist same")
    #calib_unc.Draw("E2 hist same")
    #norm_unc.Draw("E2 hist same")

    latex_lib = {
    'htautau': r'H #rightarrow #tau^{+}#tau^{-}',
    'ztautau': r'Z #rightarrow #tau^{+}#tau^{-}',
    'ttbar': r't#bar{t}',
    'diboson': r'VV'
    } 


    # Add legend
    legend = ROOT.TLegend(0.5, 0.65, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetShadowColor(0)
    legend.SetNColumns(2)
    for i , i_item in histograms_for_class.items():
      legend.AddEntry(
            i_item['hist'],
            i_item['latex'],
            "f"
        )
    legend.AddEntry(total_unc, "Bkg Systematic Uncertainty")
    #legend.AddEntry(calib_unc, "Calibration Uncertainty")
    #legend.AddEntry(norm_unc, "Normalization Uncertainty")
    ROOT.gPad.RedrawAxis()
    legend.Draw()

    #selection_label = setLabel(args.selection)
    ROOT.gPad.Modified()  # Ensures updated axis titles appear
    ROOT.gPad.Update()

    if args.ratio:
        bottomPad.cd()
        ratio_unc_hist = ROOT.TH1F( var+"_ratio", var+"_ratio", *data_structure.plot_options[var]['binning'])
        ratio_calib_unc = ROOT.TH1F( var+"_ratio_calib", var+"_ratio", *data_structure.plot_options[var]['binning'])
        ratio_norm_unc = ROOT.TH1F( var+"_ratio_norm", var+"_ratio", *data_structure.plot_options[var]['binning'])
        ratio_sig_hist = ROOT.TH1F( var+"_signal", var+"_signal", *data_structure.plot_options[var]['binning'])
        for j_bin in range(n_bins):
            ratio_unc_hist.SetBinContent(j_bin+1, ((histograms[var]['b']['up'][j_bin]+histograms[var]['b']['down'][j_bin])/(2*hist_sum.GetBinContent(j_bin+1)) if hist_sum.GetBinContent(j_bin+1)!=0 else 0))
            ratio_calib_unc.SetBinContent(j_bin+1, ((histograms[var]['b_calib']['up'][j_bin]+histograms[var]['b_calib']['down'][j_bin])/(2*hist_sum.GetBinContent(j_bin+1))if hist_sum.GetBinContent(j_bin+1)!=0 else 0))
            ratio_norm_unc.SetBinContent(j_bin+1, ((histograms[var]['b_norm']['up'][j_bin]+histograms[var]['b_norm']['down'][j_bin])/(2*hist_sum.GetBinContent(j_bin+1))if hist_sum.GetBinContent(j_bin+1)!=0 else 0))
            ratio_sig_hist.SetBinContent(j_bin+1, (histograms[var]['htautau']['nominal'][j_bin]/hist_sum.GetBinContent(j_bin+1) if hist_sum.GetBinContent(j_bin+1)!=0 else 0))
        s_y_values = np.array([ratio_sig_hist.GetBinContent(a+1) for a in range(n_bins)])
        s_unc = ROOT.TGraphAsymmErrors(n_bins, x_values, s_y_values)
        for s_bin in range(n_bins):
            s_unc.SetPointError(s_bin, bin_width/2, bin_width/2,
            (histograms[var]['s']['down'][s_bin]/hist_sum.GetBinContent(j_bin+1) if hist_sum.GetBinContent(j_bin+1)!=0 else 0),(histograms[var]['s']['up'][s_bin]/hist_sum.GetBinContent(j_bin+1) if hist_sum.GetBinContent(j_bin+1)!=0 else 0))
    
            # Add legend
        r_legend = ROOT.TLegend(0.2, 0.7, 0.4, 0.95)
        r_legend.SetBorderSize(0)
        r_legend.SetShadowColor(0)
        r_legend.SetFillColorAlpha(ROOT.kWhite, 0.5)

        r_legend.AddEntry(ratio_sig_hist,r'H #rightarrow #tau^{+}#tau^{-} / Bkg',"l")

        #set draw options
        ratioline = getRatioLine(ratio_unc_hist)
        ratioline.GetYaxis().SetRangeUser(0,0.12)
        ratio_unc_hist.SetFillStyle(3005)
        ratio_unc_hist.SetFillColor(13)      
        ratio_unc_hist.SetLineWidth(0)
        ratio_unc_hist.SetMarkerStyle(0)   
        ratioline.SetTitle('')
        ratioline.GetYaxis().SetTitle("Frac over Bkg")
        ratioline.GetYaxis().SetNdivisions(505)
        ratioline.GetYaxis().CenterTitle()
        ratioline.GetYaxis().SetTitleSize(24)
        ratioline.GetYaxis().SetTitleFont(43)
        ratioline.GetYaxis().SetTitleOffset(2.2)
        ratioline.GetYaxis().SetLabelFont(43)
        ratioline.GetYaxis().SetLabelSize(24)
        ratioline.GetYaxis().SetLabelOffset(0.009)
        ratioline.GetXaxis().SetLabelOffset(0.035)
        ratioline.GetYaxis().SetLabelOffset(0.009)
        ratioline.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
        ratio_sig_hist.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
        ratio_calib_unc.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
        ratioline.GetXaxis().SetTickLength(0.07)
        ratioline.GetXaxis().SetTitleSize(28)
        ratioline.GetXaxis().SetTitleFont(43)
        ratioline.GetXaxis().SetTitleOffset(1.2)
        ratioline.GetXaxis().SetLabelFont(43)
        ratioline.GetXaxis().SetLabelSize(24)
        ratioline.GetXaxis().SetLabelOffset(0.035)
        ratio_calib_unc.SetFillStyle(3004)
        ratio_calib_unc.SetFillColor(4)
        ratio_calib_unc.SetLineWidth(0)
        ratio_calib_unc.SetMarkerStyle(0)
        ratio_calib_unc.SetMarkerSize(0)
        ratio_norm_unc.SetFillStyle(3005)
        ratio_norm_unc.SetFillColor(2)
        ratio_norm_unc.SetLineWidth(0)
        ratio_norm_unc.SetMarkerStyle(0)
        ratio_norm_unc.SetMarkerSize(0)
        s_unc.SetFillStyle(3004)
        s_unc.SetFillColor(lib['htautau']['color'])
        s_unc.SetLineWidth(0)
        s_unc.SetMarkerStyle(0)
        s_unc.SetMarkerSize(0)
        ratio_sig_hist.SetLineColor(data_structure.plot_styles['htautau']['fill_color'])
        ratio_sig_hist.SetFillStyle(0)
   
        #now draw things
        ROOT.gPad.Modified()  # Ensures updated axis titles appear
        ROOT.gPad.Update()
        ratioline.Draw("hist")
        ratio_unc_hist.Draw("hist same")  
        #ratio_calib_unc.Draw("same")  
        #ratio_norm_unc.Draw("same")  
        ratio_sig_hist.Draw("hist same")
        s_unc.Draw("E2 hist same")
        r_legend.Draw()
        
        ROOT.gPad.RedrawAxis()

    
    # Save the canvas  
    for ext in [".png",".pdf"]:
        output_file = os.path.join(plot_directory, f"{var}"+ext)
        canvas.SaveAs(output_file)
    print(f"Saved plot for predicted class {output_file}")

common.syncer.sync()
