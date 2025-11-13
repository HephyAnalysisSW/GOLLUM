import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")
import common.helpers as helpers

import os
import numpy as np
import pickle
import argparse
import time
import yaml
import copy
import common.syncer
from tqdm import tqdm

# ROOT imports
import ROOT
ROOT.gROOT.SetBatch(True)  # Run in batch mode so we don't pop up windows.
dir_path = os.path.dirname(os.path.realpath(__file__))

ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))

ROOT.setTDRStyle()
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kTemperatureMap)

#from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user
import common.data_structure as data_structure

def update_dict(d, keys, value):
    """Recursively update a nested dictionary."""
    key = keys[0]
    if len(keys) == 1:
        # Convert value to appropriate type
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        d[key] = value
    else:
        d = d.setdefault(key, {})
        update_dict(d, keys[1:], value)

#import array
#
## Number of color levels
#NCONT = 255
#ROOT.gStyle.SetNumberContours(NCONT)
#
## Define a simple 2-stop gradient from blue to white
#stops = array.array('d', [0.0,   0.3,  0.6,  0.8,  1.0])
#red   = array.array('d', [0.0,   0.0,  0.0,  1.0,  1.0])
#green = array.array('d', [0.0,   1.0,  1.0,  1.0,  1.0])
#blue  = array.array('d', [1.0,   1.0,  0.0,  0.0,  1.0])
#ROOT.TColor.CreateGradientColorTable(5, stops, red, green, blue, NCONT)
#
#ROOT.gStyle.SetPaintTextFormat(".0f")

# Argument parser setup
parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
parser.add_argument("--config", default="../../Workflow/configs/config_reference_v5-v2.yaml", help="Path to the config file.")
parser.add_argument("--small", action="store_true", help="Run a subset.")
parser.add_argument("--asimov_mu", type=float, default=None, help="Modify asimov weights according to mu.")
parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify asimov weights according to nu_bkg.")
parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Modify asimov weights according to nu_ttbar.")
parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify asimov weights according to nu_diboson.")
parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
parser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
parser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
parser.add_argument("--n_bins", action="store", default=20, type=int, help="How many batches?")
parser.add_argument("--var1", default="mu", help="Path to the config file.")
parser.add_argument("--var2", default="mu", help="Path to the config file.")

args = parser.parse_args()

from common.logger import get_logger
logger = get_logger(args.logLevel, logFile = None)

with open(args.config) as f:
    cfg = yaml.safe_load(f)
logger.info("Config loaded from {}".format(args.config))

# Process modifications
if args.modify:
    for mod in args.modify:
        if "=" not in mod:
            raise ValueError(f"Invalid modify argument: {mod}. Must be in 'key=value' format.")
        key, value = mod.split("=", 1)
        logger.warning( "Updating cfg with: %s=%r"%( key, value) )
        key_parts = key.split(".")
        update_dict(cfg, key_parts, value)

# Construct postfix for filenames based on asimov parameters
postfix = [cfg["Toy_name"]]
if args.asimov_mu is not None:
    postfix.append(f"mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p"))
if args.asimov_nu_bkg is not None:
    postfix.append(f"nu_bkg_{args.asimov_nu_bkg:.3f}".replace("-", "m").replace(".", "p"))
if args.asimov_nu_tt is not None:
    postfix.append(f"nu_ttbar_{args.asimov_nu_tt:.3f}".replace("-", "m").replace(".", "p"))
if args.asimov_nu_diboson is not None:
    postfix.append(f"nu_diboson_{args.asimov_nu_diboson:.3f}".replace("-", "m").replace(".", "p"))

if args.postfix is not None:
    postfix.append( args.postfix )

postfix = "_".join( postfix )

# Define output directory
config_name = os.path.basename(args.config).replace(".yaml", "")
output_directory = os.path.join ( user.output_directory, config_name)

# Define output directory
plot_directory   = os.path.join(user.plot_directory,   "fisher", config_name)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
os.makedirs(fit_directory, exist_ok=True)
cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data{'_small' if args.small else ''}" )
os.makedirs(cfg['tmp_path'], exist_ok=True)

infer = Inference(cfg, small=args.small, overwrite=False)

max_batch = 1 if args.small else -1

# Get the list of keys.
keys = list(data_structure.plot_options.keys())

# Prepare dictionaries to hold the numpy histogram accumulators and the corresponding TH2F histograms.
h2_accum = {}  # (key_x, key_y) -> np.zeros((n_bins, n_bins))
h2_dict = {}   # (key_x, key_y) -> ROOT.TH2F

for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        key_x = keys[i]
        key_y = keys[j]
        # Extract x-axis range from plot_options: lower and upper.
        binning_x = data_structure.plot_options[key_x]['binning']
        x_low = binning_x[1]
        x_high = binning_x[2]
        n_bins_x = binning_x[0] if binning_x[0]<args.n_bins else args.n_bins
        # Extract y-axis range.
        binning_y = data_structure.plot_options[key_y]['binning']
        y_low = binning_y[1]
        y_high = binning_y[2]
        n_bins_y = binning_y[0] if binning_y[0]<args.n_bins else args.n_bins
        # Create a numpy accumulator for the 2D histogram.
        h2_accum[(key_x, key_y)] = np.zeros((n_bins_x, n_bins_y))
        # Create a TH2F with 20 bins on each axis and appropriate ranges.
        hist_name = "h2_%s_%s" % (key_x, key_y)
        title = "%s vs %s" % (data_structure.plot_options[key_x]['tex'], data_structure.plot_options[key_y]['tex'])
        h2 = ROOT.TH2F(hist_name, title, n_bins_x, x_low, x_high, n_bins_y, y_low, y_high)
        h2.GetXaxis().SetTitle(data_structure.plot_options[key_x]['tex'])
        h2.GetYaxis().SetTitle(data_structure.plot_options[key_y]['tex'])
        h2_dict[(key_x, key_y)] = h2

import common.datasets_hephy as datasets_hephy
# Load the data
data_loader = datasets_hephy.get_data_loader(
    selection=args.selection, selection_function=None, n_split=args.n_split if not args.small else 100)

# Loop over data batches and calculate predictions
total_batches = len(data_loader)
for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
    features, weights, labels = data_loader.split(batch)
    #predictions = tfmc.predict(features, ic_scaling=False)

    g     = infer.models["MultiClassifier"]["lowMT_VBFJet"].predict(features) 
    g_sum = g.sum(axis=1)

    d_R = {}
    if "mu" in [args.var1, args.var2]:
        d_R["mu"]          = g[:, data_structure.label_encoding['htautau']]/g_sum
    if "nu_bkg" in [args.var1, args.var2]:
        d_R["nu_bkg"]      = (1-g[:, data_structure.label_encoding['htautau']])/g_sum*np.log1p(infer.alpha_bkg)
    if "nu_tt" in [args.var1, args.var2]:
        d_R["nu_tt"]       = g[:, data_structure.label_encoding['ttbar']]/g_sum*np.log1p(infer.alpha_tt)
    if "nu_diboson" in [args.var1, args.var2]:
        d_R["nu_diboson"]  = g[:, data_structure.label_encoding['diboson']]/g_sum*np.log1p(infer.alpha_diboson)

    DeltaA = {}
    for c in ['htautau', 'ztautau', 'ttbar', 'diboson']:
        DeltaA[c] = {}
        dA =  infer.models[c][args.selection].get_DeltaA( features )
        if "nu_jes" in [args.var1, args.var2]:
            DeltaA[c]['nu_jes'] = dA[:, infer.models[c][args.selection].combinations.index(('nu_jes',))].numpy()
            DeltaA[c]['nu_jes'] += infer.icps[c][args.selection].DeltaA[infer.icps[c][args.selection].combinations.index(('nu_jes',))]
        if "nu_tes" in [args.var1, args.var2]:
            DeltaA[c]['nu_tes'] = dA[:, infer.models[c][args.selection].combinations.index(('nu_tes',))].numpy()
            DeltaA[c]['nu_tes'] += infer.icps[c][args.selection].DeltaA[infer.icps[c][args.selection].combinations.index(('nu_tes',))]
        if "nu_met" in [args.var1, args.var2]:
            DeltaA[c]['nu_met'] = dA[:, infer.models[c][args.selection].combinations.index(('nu_met',))].numpy()
            DeltaA[c]['nu_met'] += infer.icps[c][args.selection].DeltaA[infer.icps[c][args.selection].combinations.index(('nu_met',))]

    if "nu_jes" in [args.var1, args.var2]:
        d_R["nu_jes"] = np.sum([ DeltaA[c]['nu_jes']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum
    if "nu_tes" in [args.var1, args.var2]:
        d_R["nu_tes"] = np.sum([ DeltaA[c]['nu_tes']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum
    if "nu_met" in [args.var1, args.var2]:
        d_R["nu_met"] = np.sum([ DeltaA[c]['nu_met']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum

    # Compute the overall weight for this batch: weights multiplied by (d_R["mu"])².
    weight_batch = weights * d_R[args.var1]*d_R[args.var2]

    # For each unique pair, use np.histogram2d to compute the bin counts.
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key_x = keys[i]
            key_y = keys[j]
            # Get the x and y ranges from plot_options.
            binning_x = data_structure.plot_options[key_x]['binning']
            x_range = (binning_x[1], binning_x[2])
            n_bins_x = binning_x[0] if binning_x[0]<args.n_bins else args.n_bins
            binning_y = data_structure.plot_options[key_y]['binning']
            y_range = (binning_y[1], binning_y[2])
            n_bins_y = binning_y[0] if binning_y[0]<args.n_bins else args.n_bins
            # Compute the 2D histogram for this batch.
            H, xedges, yedges = np.histogram2d(features[:,i], features[:,j],
                                               bins=[n_bins_x, n_bins_y], range=[x_range, y_range],
                                               weights=weight_batch)
            # Accumulate the result.
            h2_accum[(key_x, key_y)] += H
 
    if max_batch > 0 and i_batch + 1 >= max_batch:
        break

# --- After the Event Loop: Fill the TH2F histograms with the accumulated counts ---
for key_pair, H_acc in h2_accum.items():
    h2 = h2_dict[key_pair]
    # Fill the TH2F: TH2F bins are 1-indexed.
    for ix in range(h2.GetNbinsX()):
        for iy in range(h2.GetNbinsY()):
            h2.SetBinContent(ix+1, iy+1, H_acc[ix, iy])
    # Optionally, draw the histogram with COLZ and save to file.
    c = ROOT.TCanvas("c_%s_%s" % key_pair, "%s vs %s" % key_pair, 600, 600)
    c.SetRightMargin(0.18)
    c.SetTopMargin(0.1)
    h2.Draw("COLZ")
    c.SetLogz()
    out_name = "2D_%s_vs_%s.pdf" % key_pair
    c.Print(os.path.join(user.plot_directory, "Fisher", f"2D_{args.var1}_{args.var2}", out_name))
    out_name = "2D_%s_vs_%s.png" % key_pair
    c.Print(os.path.join(user.plot_directory, "Fisher", f"2D_{args.var1}_{args.var2}", out_name))

helpers.copyIndexPHP(os.path.join(user.plot_directory, "Fisher", f"2D_{args.var1}_{args.var2}"))
common.syncer.sync()

