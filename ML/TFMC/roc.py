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
import common.datasets as datasets
# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--modelDirs", nargs="+", required=True, help="Directories containing the trained TFMC models.")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
args = argParser.parse_args()

# Load the data
data_loader = datasets.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split)

max_batch = 1 if args.small else -1

# Load the TFMC models
from TFMC import TFMC
models = {model_dir: TFMC.load(model_dir) for model_dir in args.modelDirs}

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "TFMC", args.selection, "ROC_curves")
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

legend = ROOT.TLegend(0.4, 0.15, 0.85, 0.4)
legend.SetBorderSize(0)
legend.SetShadowColor(0)
stuff = []
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

# Save the canvas
output_file = os.path.join(plot_directory, "ROC_curves.png")
canvas.SaveAs(output_file)
output_file = os.path.join(plot_directory, "ROC_curves.pdf")
canvas.SaveAs(output_file)
print(f"Saved ROC curves to {output_file}")

common.syncer.sync()

