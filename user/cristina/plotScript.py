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
import common.datasets_hephy as datasets_hephy
import common.logger as _logger

# argparser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument('--logLevel',    action='store',         nargs='?',  choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'],   default='INFO', help="Log level for logging" )
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
argParser.add_argument('--var', action='store', default=None, help="Specify single variable to plot")
argParser.add_argument('--predict', action='store_true', default=False, help='import ML model for prediction?')
argParser.add_argument("--modelDir", action="store", default="/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6",  help="Directory containing the trained TFMC model.")
argParser.add_argument("--plot_scores", action="store_true", default =False)
argParser.add_argument("--bottom_pad", action="store_true", default=False)
args = argParser.parse_args()

# logger for debugging/info statements
logger  = _logger.get_logger(args.logLevel, logFile = None)

# some helpers for cleaner reading
def get_bin_edges(var, selection):
    if var == "score" and selection in nice_labels_regions:
        return np.array(nice_labels_regions[selection]["binning"])
    else:
        binning = data_structure.plot_options[var]['binning']
        return np.linspace(binning[1], binning[2], binning[0] + 1)
    
def get_many_loaders(list_variations, selection, n_split=10, batch_size=None):
    """ list of data loaders in this fixed order """

    loader_list = []
    for var_name,values in list_variations:
        loader = datasets_hephy.get_data_loader(selection=selection, values=values, n_split=n_split, batch_size=batch_size)
        loader_list.append((var_name, loader))
    
    return loader_list

nice_labels_regions = {"lowMT_VBFJet": {"tex": "#scale[1.0]{#font[12]{lowMT_VBFJet}}", "binning": [0, 0.025, 0.095, 0.295, 0.625, 0.89, 0.965, 1]},
                       "lowMT_noVBFJet_ptH100": {"tex": "#scale[1.0]{#font[12]{lowMT_noVBFJet_ptH100}}", "binning": [0, 0.06, 0.17, 0.405, 0.72, 0.99, 1]},
                       "highMT_VBFJet": {"tex": "#scale[1.0]{#font[12]{highMT_VBFJet}}", "binning": [0, 0.045, 0.32, 0.665, 0.875, 1]}
}
# n_bins_score = len(nice_labels_regions[args.selection]["binning"])-1
# print(n_bins_score)
bins_score = get_bin_edges("score", args.selection)
#scale[1.0]{#font[12]{H#rightarrow#tau^{+}#tau^{-}}}
print(data_structure.plot_options)

from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load(args.modelDir)

# shape variations (tes, jes, met)
systematic_variations = [
    ("nominal",     (1.00, 1.00, 0)),
    ("TESUp",       (1.01, 1.00, 0)),  # TES+1, JES nominal, MET nominal
    ("TESDown",     (0.99, 1.00, 0)),  # TES-1, JES nominal, MET nominal
    ("JESUp",       (1.00, 1.01, 0)),  # TES nominal, JES+1, MET nominal
    ("JESDown",     (1.00, 1.01, 0)),  # TES nominal, JES-1, MET nominal
    ("METUp",       (1.00, 1.00, 1))   # TES nominal, JES nominal, MET+1
]

# norm uncertainties
nu = 1
alpha_bkg = 0.001
alpha_tt = 0.02
alpha_diboson = 0.25

# norm variations dictionary
norm_variations_dict = {
    "normTTUp":         {"ttbar":   {"selector": lambda label, data_structure: label==data_structure.label_encoding["ttbar"],       "weights": (1 + alpha_tt)**nu}},
    "normTTDown":       {"ttbar":   {"selector": lambda label, data_structure: label==data_structure.label_encoding["ttbar"],       "weights": (1 + alpha_tt)**(-nu)}},
    "normDibosonUp":    {"diboson": {"selector": lambda label, data_structure: label==data_structure.label_encoding["diboson"],     "weights":(1 + alpha_diboson)**nu}},
    "normDibosonDown":  {"diboson": {"selector": lambda label, data_structure: label==data_structure.label_encoding["diboson"],     "weights": (1 + alpha_diboson)**(-nu)}},
    "normTotUp":        {"tot_bkg": {"selector": lambda label, data_structure: label!=data_structure.label_encoding["htautau"],     "weights":(1 + alpha_bkg)**nu}},
    "normTotDown":      {"tot_bkg": {"selector": lambda label, data_structure: label!=data_structure.label_encoding["htautau"],     "weights":(1 + alpha_bkg)**(-nu)}}
}

model_dict = {}

max_batch = 1 if args.small else -1

# get all the data loaders
data_loaders = get_many_loaders(systematic_variations, selection=args.selection, n_split=args.n_split)

# plot directory
plot_directory = os.path.join(user.plot_directory, "paper_plots", args.selection)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# utils for plotting speed (only one variable selected)
to_plot = [args.var] if args.var else data_structure.plot_options.keys()

if args.plot_scores:
    new_data_structure = data_structure.feature_names+["score"]
    names_to_plot = [args.var] if args.var else new_data_structure
else:
    names_to_plot = [args.var] if args.var else data_structure.feature_names
# names_to_plot = [args.var] if args.var else data_structure.feature_names

# create the histograms dictionary (to fill later)
histograms = {}
histograms_by_thr = {}


data_structure.plot_options["score"] = {
    "binning": bins_score,  # uniform bins from 0..1
    "tex": "Score"
}

for var in to_plot:
    options = data_structure.plot_options[var]
    binning = options['binning']
    bin_edges = get_bin_edges(var, args.selection)
    histograms[var] = {}
    for label in data_structure.labels:
        histograms[var][label] = {}
        for var_name, _ in systematic_variations:
            
            if var=="score":
                histograms[var][label][var_name] = np.zeros(len(nice_labels_regions[args.selection]["binning"])-1)
            else:
                histograms[var][label][var_name] = np.zeros(binning[0])
        for nv in norm_variations_dict.keys():
            if var=="score":
                histograms[var][label][var_name] = np.zeros(len(nice_labels_regions[args.selection]["binning"])-1)
            else:
                histograms[var][label][nv] = np.zeros(binning[0])

logger.debug(f" Is the histogram dictionary initialized in the correct way? \n {histograms}")



all_nominal_data = [] # for the uncertainty on the signal fraction

# dataLoaders loop to fill histograms (nominal + shape + norm)
for loader_name, data_loader in data_loaders:
    # for the score plotting, define toggle
    # but if it does it only for the nominal the uncertainties ???
    globalScoreToggle = args.plot_scores and loader_name=="nominal"

    total_batches = len(data_loader)
    logger.info(f"\nProcessing {loader_name} variation...")
    for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc=f"Batches ({loader_name})")):
        features, weights, labels = data_loader.split(batch)
        predictions = tfmc.predict(features, ic_scaling=False)
        print(predictions.shape)
        score = predictions[:, 0]
        print(f"First event predictions:{predictions[0]}")

        predictions = tfmc.predict(features, ic_scaling=False)  # shape (batch_size, n_classes)
        # let's pick the signal node index 0
        score = predictions[:, 0]
        logger.info(f"Score stats: min={score.min()}, max={score.max()}, mean={score.mean()}")
        if "score" in to_plot:
            for label_idx, label_str in enumerate(data_structure.labels):
                mask = (labels == label_idx)
                if np.any(mask):
                    score_values = predictions[mask, 0]  # signal node
                    bin_edges = get_bin_edges("score", args.selection)    # 50 bins from 0..1
                    histogram_batch, _ = np.histogram(
                        score_values,
                        bins=np.array(nice_labels_regions[args.selection]["binning"], dtype=np.float64),
                        weights=weights[mask]
                    )
                    histograms["score"][label_str][loader_name] += histogram_batch

        
        # fill the shape variations
        for label_idx, label_str in enumerate(data_structure.labels):
            label_mask = (labels == label_idx)
            if np.any(label_mask):
                to_plot = [args.var] if args.var else data_structure.plot_options.keys()
                for i_var, var in enumerate(data_structure.feature_names):
                    bin_edges = get_bin_edges(var, args.selection)
                    values = features[label_mask, i_var]
                    histogram_batch, _ = np.histogram(
                        values,
                        #features[label_mask, i_var],
                        bins=bin_edges,
                        weights=weights[label_mask]
                    )
                    # logger.info("Fillin the shape variations")
                    if var in histograms and label in histograms[var]:
                        histograms[var][label_str][loader_name] += histogram_batch

        if loader_name=="nominal":
            for norm_var, process_dict in norm_variations_dict.items():
                w_copy = weights.copy()
                for process_key, info in process_dict.items():
                    print(process_key)
                    # print(w_copy)
                    # print("Label encoding:", data_structure.label_encoding)
                    # print(process_key)
                    mask = info["selector"](labels, data_structure)
                    logger.debug(f"{norm_var}: process={process_key}, # events = {mask.sum()}")
                    logger.debug(f"Actual mask for {norm_var}: {mask}")
                    
                    w_copy[mask] *= info["weights"]

                for label_idx, label in enumerate(data_structure.labels):
                    label_mask = (labels == label_idx)
                    if np.any(label_mask):
                        to_plot = [args.var] if args.var else data_structure.plot_options.keys()
                        for i_var, var in enumerate(data_structure.feature_names):
                            # if np.any(true_indices):
                            bin_edges = get_bin_edges(var, args.selection)
                            # if var == "score":
                            #     values = score[label_mask]
                            # else:
                            values = features[label_mask, i_var]
                            print(values)
                            histogram_batch, _ = np.histogram(
                                values,
                                # features[label_mask, i_var],
                                bins=bin_edges,#data_structure.plot_options[var]['bin_edges'],
                                weights=w_copy[label_mask]
                            )
                            # fill rate
                            # logger.info("Fillin the rate variations")
                            if var in histograms and label in histograms[var]:
                                histograms[var][label][norm_var] += histogram_batch
            


        if loader_name == "nominal":
            # Save a copy of the arrays for each batch
            all_nominal_data.append((features.copy(), weights.copy(), labels.copy()))

        if max_batch > 0 and i_batch + 1 >= max_batch:
            break

logger.debug(f" Are the histograms filled correctly? \n {histograms}")


all_systematics = ["TESUp", "TESDown", "JESUp", "JESDown", "METUp"]+list(norm_variations_dict.keys())

# uncertainty computation loop
total_histograms = {}
signal_only_uncertainties = {}
for var, label_histograms in histograms.items():
    # print(var, label_histograms)
    n_bins = len(next(iter(label_histograms.values()))['nominal'])
    total_nominal = np.zeros(n_bins)
    # total up/down variances
    total_up_variance = np.zeros(n_bins)
    total_down_variance = np.zeros(n_bins)
    # only signal
    signal_up_variance = np.zeros(n_bins)
    signal_down_variance = np.zeros(n_bins)

    for label, histogram_data in label_histograms.items():
        print(f"Processing the nominal of: {label}")
        nominal = histogram_data["nominal"]
        logger.debug(f"[DEBUG] {var} label={label}: nominal bin contents: {nominal}")
        total_nominal += nominal

        # loop over variations
        for var_name in all_systematics:
            variation = histogram_data[var_name]
            delta = variation - nominal
            # print(f"Variation: {var_name}, Process: {label}")
            # print(f"Nominal bin contents: {nominal}")
            # print(f"Variation bin contents: {variation}")
            # print(f"Delta: {delta}\n")
            # if var_name in ["TESDown","normTotDown","..."]:
            #     print(f"Bin {i}: nominal={nominal[i]}, variation={variation[i]}, delta={delta[i]}")
            up_contrib = np.where(delta > 0, delta, 0)
            down_contrib = np.where(delta < 0, delta, 0)
            # logger.debug(f"[{var_name} | {label}] nominal = {nominal}")
            # logger.debug(f"[{var_name} | {label}] variation = {variation}")
            # logger.debug(f"[{var_name} | {label}] delta = {delta}")
            # logger.debug(f"[{var_name} | {label}] up_contrib = {up_contrib}")
            # logger.debug(f"[{var_name} | {label}] down_contrib = {down_contrib}")

            total_up_variance += up_contrib ** 2
            total_down_variance += down_contrib ** 2
            # logger.debug(f"total up var {total_up_variance} for {var_name}")
            # logger.debug(f"total down var {total_down_variance} for {var_name}")

            if label == 'htautau':
                signal_up_variance += up_contrib **2
                signal_down_variance += down_contrib **2

            # print(f"Signal down variance BEFORE sqrt: {signal_down_variance}")
            # print(f"Any negatives? {np.any(signal_down_variance < 0)}")

    total_up_uncertainty = np.sqrt(total_up_variance)
    logger.debug(f"total up unc {total_up_uncertainty}")

    total_down_uncertainty = np.sqrt(total_down_variance)
    logger.debug(f"total down unc {total_down_uncertainty}")
    signal_up_uncertainty = np.sqrt(signal_up_variance)
    signal_down_uncertainty = np.sqrt(signal_down_variance)

    total_histograms[var] = {
        "nominal": total_nominal,
        "up_uncertainty": total_up_uncertainty,
        "down_uncertainty": total_down_uncertainty
    }
    signal_only_uncertainties[var] = {
        "up_uncertainty": signal_up_uncertainty,
        "down_uncertainty": signal_down_uncertainty
    }

# ==============================
# plotting
# ==============================
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

for var in names_to_plot:
    canvas = ROOT.TCanvas(f"c_{var}", "", 1000, 1000)
    if args.bottom_pad:
        pad1 = ROOT.TPad("pad1", "Top pad", 0, 0.35, 1, 1)
        pad1.SetBottomMargin(0.05)
        pad1.SetLeftMargin(0.13)
        pad1.SetRightMargin(0.04)
        pad1.SetTopMargin(0.07)
        pad1.SetTickx()
        pad1.SetTicky()
        pad1.SetLogy()
        pad1.Draw()

        canvas.cd()
        pad2 = ROOT.TPad("pad2", "Bottom pad", 0, 0, 1, 0.35)
        pad2.SetTopMargin(0.07)
        pad2.SetBottomMargin(0.25)
        pad2.SetLeftMargin(0.13)
        pad2.SetRightMargin(0.04)
        pad2.SetTickx()
        pad2.SetTicky()
        pad2.Draw()

    else:
        pad1 = ROOT.TPad("pad1", "Full pad", 0, 0, 1, 1)
        pad1.SetBottomMargin(0.13)
        pad1.SetTopMargin(0.07)
        pad1.SetLeftMargin(0.13)
        pad1.SetRightMargin(0.04)
        pad1.SetTickx()
        pad1.SetTicky()
        pad1.SetLogy()
        pad1.Draw()
    
    pad1.cd()
    # --- Stack Plot ---
    stack = ROOT.THStack(var, var)
    max_y = 0
    custom_order = [0, 3, 2, 1]  # htautau, diboson, ttbar, ztautau
    legend_entries = []

    if args.plot_scores:
        score_edges = np.array(nice_labels_regions[args.selection]["binning"], dtype=np.double)
        nbins_score = len(score_edges) - 1
        h_total_fraction = ROOT.TH1F(f"{var}_total_fraction", var, nbins_score, score_edges)
        h_htautau_fraction = ROOT.TH1F(f"{var}_htautau_fraction", var, nbins_score, score_edges)
    else:
        h_total_fraction = ROOT.TH1F(f"{var}_total_fraction", var, *data_structure.plot_options[var]['binning'])
        h_htautau_fraction = ROOT.TH1F(f"{var}_htautau_fraction", var, *data_structure.plot_options[var]['binning'])

    for order_idx in custom_order:
        label = data_structure.labels[order_idx]
        if args.plot_scores:
            score_edges = np.array(nice_labels_regions[args.selection]["binning"], dtype=np.double)
            nbins_score = len(score_edges) - 1
            h = ROOT.TH1F(f"{var}_{label}", var,  nbins_score, score_edges)
            h_fraction = ROOT.TH1F(f"{var}_{label}_fraction", var,  nbins_score, score_edges)
        else:
            h = ROOT.TH1F(f"{var}_{label}", var, *data_structure.plot_options[var]['binning'])
            h_fraction = ROOT.TH1F(f"{var}_{label}_fraction", var, *data_structure.plot_options[var]['binning'])

        for i, value in enumerate(histograms[var][label]["nominal"]):
            h.SetBinContent(i + 1, value)
            h_fraction.SetBinContent(i + 1, value)

        fill_color = data_structure.plot_styles[label]['fill_color']
        h.SetFillColorAlpha(fill_color, 1.0)
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        stack.Add(h)

        # Solid clone for legend
        h_legend = h.Clone()
        h_legend.SetFillColor(fill_color)
        legend_entries.append((h_legend, label))

        h_total_fraction.Add(h_fraction)
        if label == 'htautau':
            h_htautau_fraction.Add(h_fraction)

        max_y = max(max_y, h.GetMaximum())

    # Draw stack
    stack.Draw("HIST")
    stack.GetYaxis().SetTitle("Events")
    stack.GetYaxis().SetTitleSize(0.05)
    stack.GetYaxis().SetTitleOffset(1.2)
    stack.GetYaxis().SetLabelSize(0.045)
    stack.GetXaxis().SetLabelSize(0.045)
    stack.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
    stack.SetMaximum(1.5 * max_y)

    # --- Systematic Uncertainty Band (Top Pad) ---
    total = total_histograms[var]
    binning = data_structure.plot_options[var]['binning']
    if args.plot_scores:
        bin_edges = np.array(nice_labels_regions[args.selection]["binning"], dtype=np.double)
        n_bins = len(bin_edges)-1
    else:
        bin_edges = np.linspace(binning[1], binning[2], binning[0] + 1)
        n_bins = binning[0]

    x_vals = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)], dtype=np.float64)
    y_vals = total['nominal']
    x_errors = np.array([(bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(n_bins)], dtype=np.float64)

    syst_band_top = ROOT.TGraphAsymmErrors(n_bins, x_vals, y_vals, x_errors, x_errors, total['down_uncertainty'], total['up_uncertainty'])
    syst_band_top.SetFillColor(ROOT.kBlack)
    syst_band_top.SetFillStyle(3354)
    syst_band_top.SetLineColor(ROOT.kBlack)
    syst_band_top.SetLineWidth(1)
    syst_band_top.Draw("E2 SAME")

    # --- Legend ---
    legend = ROOT.TLegend(0.65, 0.55, 0.88, 0.9)
    legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    legend.SetMargin(0.4)
    legend.SetFillStyle(0) 
    legend.SetFillColor(0)
    legend.SetEntrySeparation(0.6)
    for h_legend, label in legend_entries:
        h_legend.SetMarkerSize(2.0)
        latex_label = data_structure.latex_labels[label]
        legend.AddEntry(h_legend, latex_label, "f")
    legend.AddEntry(syst_band_top, "#scale[1.0]{Syst. Uncertainties}", "f")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(13)
    # latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    tex_label = nice_labels_regions[args.selection]['tex']
    latex.DrawLatex(0.8, 0.96, tex_label)

    # signal fraction (linea arancione)
    if args.bottom_pad:
        signal_fraction_legend = ROOT.TLine()
        signal_fraction_legend.SetLineColor(ROOT.kOrange-2)
        signal_fraction_legend.SetLineWidth(2)

        # Ora aggiungi questi elementi alla legenda
        # legend.AddEntry(signal_uncertainty_legend, "#scale[1.0]{Signal Stat. Unc.}", "f")
        legend.AddEntry(signal_fraction_legend, "#scale[1.0]{Signal Fraction}", "l")
    
    legend.Draw()
    pad1.Update()

    # --- Bottom pad: Fraction ---
    
    # pad1.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])

    if args.bottom_pad:
        pad2.cd()
        frame = pad2.DrawFrame(bin_edges[0], -0.8, bin_edges[-1], 0.8)
        frame.GetYaxis().SetTitle("#DeltaB/B")
        frame.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
        frame.GetYaxis().SetTitleSize(0.09)
        frame.GetYaxis().SetLabelSize(0.07)
        frame.GetXaxis().SetTitleSize(0.09)
        frame.GetXaxis().SetLabelSize(0.07)
        frame.GetYaxis().SetTitleOffset(0.6)
    # else:
    #     pad1.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
    #     pad1.Update()

        # calculate fraction S/B
        h_fraction_result = h_htautau_fraction.Clone(f"{var}_fraction_htautau")
        background_only = h_total_fraction.Clone()
        background_only.Add(h_htautau_fraction, -1)

        h_fraction_result.Divide(background_only)

        h_fraction_result.SetLineColor(ROOT.kOrange - 4)
        h_fraction_result.SetLineWidth(2)
        h_fraction_result.SetTitle("")
        h_fraction_result.GetYaxis().SetTitle("S/B")
        h_fraction_result.GetYaxis().SetNdivisions(505)
        h_fraction_result.GetYaxis().SetTitleSize(0.12)
        h_fraction_result.GetYaxis().SetTitleOffset(0.5)
        h_fraction_result.GetYaxis().SetLabelSize(0.1)
        h_fraction_result.GetXaxis().SetTitle(data_structure.plot_options[var]['tex'])
        h_fraction_result.GetXaxis().SetTitleSize(0.12)
        h_fraction_result.GetXaxis().SetLabelSize(0.1)
        # h_fraction_result.SetMinimum(0)
        # h_fraction_result.SetMaximum(0.08)
        h_fraction_result.Draw("HIST SAME")

        line_zero = ROOT.TLine(bin_edges[0], 0, bin_edges[-1], 0)
        line_zero.SetLineColor(ROOT.kBlack)
        line_zero.SetLineStyle(1)  # linea continua
        # line_zero.SetLineWidth(2)
        line_zero.Draw("same")

        # --- Total Uncertainty Band (Bottom Pad using TBox) ---
        # --- Background-only Uncertainty Band (Bottom Pad using TBox) ---

        tboxes_unc = []

        for i in range(n_bins):
            B = sum(histograms[var][bkg]['nominal'][i] 
                    for bkg in data_structure.labels if bkg != 'htautau')

            dB_up_sq, dB_down_sq = 0, 0
            for bkg in data_structure.labels:
                if bkg == 'htautau':
                    continue
                nom_bkg = histograms[var][bkg]['nominal'][i]
                for var_name in ["TESUp", "TESDown", "JESUp", "JESDown", "METUp"]:
                    variation = histograms[var][bkg][var_name][i]
                    delta = variation - nom_bkg
                    if delta > 0:
                        dB_up_sq += delta ** 2
                    else:
                        dB_down_sq += delta ** 2

            dB_up = np.sqrt(dB_up_sq)
            dB_down = np.sqrt(dB_down_sq)

            # Calcola incertezza relativa background (simmetrica attorno a 0)
            if B > 0:
                rel_up = dB_up / B
                rel_down = dB_down / B
            else:
                rel_up = rel_down = 0

            # Usa la più grande delle due per visualizzare simmetricamente (opzionale)
            rel_err = max(rel_up, rel_down)

            # print(f"Bin {i+1}: B={B:.3f}, Rel. uncertainty=±{rel_err:.3f}")

            bin_low = bin_edges[i]
            bin_high = bin_edges[i+1]

            # UNA SOLA TBox centrata a 0, che copre sia valori positivi che negativi?
            box = ROOT.TBox(bin_low, -rel_down, bin_high, rel_up)
            # box = ROOT.TBox(bin_low, -rel_err, bin_high, rel_err)
            box.SetFillColorAlpha(ROOT.kBlack, 1.0)
            box.SetFillStyle(3354)
            box.Draw("same")
            tboxes_unc.append(box)

        
        pad2.RedrawAxis()
        pad2.Update()

        signal_unc = signal_only_uncertainties[var]
        h_unc_band = h_fraction_result.Clone(f"{var}_unc_band")
        h_unc_band.SetFillStyle(3354)
        h_unc_band.SetFillColor(ROOT.kBlack)
        h_unc_band.SetMarkerSize(0)
        h_unc_band.SetLineColor(ROOT.kBlack)

        for i in range(1, h_unc_band.GetNbinsX() + 1):
            signal = h_htautau_fraction.GetBinContent(i)
            background = background_only.GetBinContent(i)

            if background > 0:
                signal_up = signal_unc['up_uncertainty'][i - 1]
                signal_down = signal_unc['down_uncertainty'][i - 1]

                rel_up = signal_up / background
                rel_down = signal_down / background

      
                avg_err = 0.5 * (abs(rel_up) + abs(rel_down))

                h_unc_band.SetBinError(i, avg_err)

        h_unc_band.Draw("E2 SAME")

    
        pad2.Update()


    

    # box.Draw("same")
    canvas.Update()

    output_file = os.path.join(plot_directory, f"{var}.png")
    canvas.SaveAs(output_file)
    print(f"Saved plot: {output_file}")





common.syncer.sync()
