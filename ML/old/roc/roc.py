#!/usr/bin/env python

import sys, os
import numpy as np
import ROOT
from tqdm import tqdm
import importlib

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
argParser.add_argument("--modelDirs", nargs="+", required=True, help="Directories containing the trained TFMC models.")
argParser.add_argument("--filename", default = "ROC",  help="The filename.")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
argParser.add_argument('--test', action='store_true', help="Test data?")
args = argParser.parse_args()

if args.test:
    import common.test_datasets as datasets_hephy
else:
    import common.datasets_hephy as datasets_hephy

# Load the data
data_loader = datasets_hephy.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split)

max_batch = 1 if args.small else -1

# Load the TFMC models
models = {}
for model_dir in args.modelDirs:
    if '/TFMC/' in model_dir:
        from ML.TFMC.TFMC import TFMC
        models[model_dir] = TFMC.load(model_dir)
    elif '/XGBMC/' in model_dir:
        from ML.XGBMC.XGBMC import XGBMC
        models[model_dir] = XGBMC.load(model_dir)

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "roc", args.selection, "ROC_curves")
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Prepare for ROC computation
signal_class = "htautau"
signal_idx = data_structure.labels.index(signal_class)

# Loop through models and calculate predictions
roc_curves = {}

for model_name, model in models.items():
    y_true = []
    y_scores = []

    print(f"Evaluating model: {model_name}")
    for i_batch, batch in enumerate(tqdm(data_loader, desc=f"Processing batches for {model_name}")):
        features, weights, labels = data_loader.split(batch)
        predictions = model.predict(features, ic_scaling=False)

        # True labels: 1 for signal, 0 for background
        y_true.extend((labels == signal_idx).astype(int))
        # Scores: predicted probabilities for signal class
        y_scores.extend(predictions[:, signal_idx])

        if max_batch > 0 and i_batch + 1 >= max_batch:
            break

    # Compute ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_curves[model_name] = (fpr, tpr)

# Plot ROC curves using ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta, ROOT.kCyan, ROOT.kPink]

canvas = ROOT.TCanvas("c_ROC", "ROC Curves", 800, 600)
canvas.SetTicks(1, 1)
frame = ROOT.TH2F("frame", ";False Positive Rate;True Positive Rate", 100, 0, 1, 100, 0, 1)
frame.Draw()

# Main legend
legend = ROOT.TLegend(0.45, 0.2, 0.9, 0.45)  # Adjusted position
legend.SetBorderSize(0)
legend.SetShadowColor(0)
stuff = []

# Add ROC curves
for i, (model_name, (fpr, tpr)) in enumerate(roc_curves.items()):
    graph = ROOT.TGraph(len(fpr), np.array(fpr, dtype=float), np.array(tpr, dtype=float))
    graph.SetLineColor(colors[i % len(colors)])
    graph.SetLineWidth(2)
    graph.Draw("L SAME")
    stuff.append(graph)
    auc = np.trapz(tpr, x=fpr)
    label = "/".join(model_name.rstrip("/").split("/")[-2:])
    legend.AddEntry(graph, f"{label} (AUC: {auc:.3f})", "l")

legend.Draw()

# Inset for zoomed ROC
inset = ROOT.TPad("inset", "Inset", 0.6, 0.6, 0.9, 0.9)  # Adjust position
inset.SetFillStyle(4000)  # Transparent background
inset.SetBorderSize(1)
inset.SetTicks(1, 1)
inset.Draw()
inset.cd()

# Inset frame
inset_frame = ROOT.TH2F("inset_frame", "", 100, 0, 0.01, 100, 0, 0.2)  # Focused on low FPR/TPR
inset_frame.GetXaxis().SetTitle("False Positive Rate")
inset_frame.GetYaxis().SetTitle("True Positive Rate")
inset_frame.GetXaxis().SetTitleSize(0.08)
inset_frame.GetYaxis().SetTitleSize(0.08)
inset_frame.GetXaxis().SetLabelSize(0.07)
inset_frame.GetYaxis().SetLabelSize(0.07)
inset_frame.GetXaxis().SetTitleOffset(1.2)
inset_frame.GetYaxis().SetTitleOffset(1.0)
inset_frame.Draw()

# Add ROC curves to inset
for i, (model_name, (fpr, tpr)) in enumerate(roc_curves.items()):
    zoom_fpr = np.array([x for x in fpr if x <= 0.1], dtype=float)
    zoom_tpr = np.array(tpr[:len(zoom_fpr)], dtype=float)
    inset_graph = ROOT.TGraph(len(zoom_fpr), zoom_fpr, zoom_tpr)
    inset_graph.SetLineColor(colors[i % len(colors)])
    inset_graph.SetLineWidth(2)
    inset_graph.Draw("L SAME")
    stuff.append(inset_graph)

# Back to main canvas
canvas.cd()

postfix = "test" if args.test else "train"

# Save the canvas
output_file = os.path.join(plot_directory, f"{args.filename}_{postfix}.png")
canvas.SaveAs(output_file)
output_file = os.path.join(plot_directory, f"{args.filename}_{postfix}.pdf")
canvas.SaveAs(output_file)
print(f"Saved ROC curves to {output_file}")

common.syncer.sync()

