import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")
import common.helpers as helpers

import os
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200,
                        formatter={'float': '{:6.3f}'.format})
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
#parser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
parser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")

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

parameters = ["mu", "nu_bkg", "nu_tt", "nu_diboson", "nu_tes", "nu_jes", "nu_met"]
FI_unbinned = np.zeros((7, 7))

import common.datasets_hephy as datasets_hephy

for selection in cfg["Selections"]:
    # Load the data
    data_loader = datasets_hephy.get_data_loader(
        selection=selection, selection_function=None, n_split=args.n_split if not args.small else 100)

    # Loop over data batches and calculate predictions
    total_batches = len(data_loader)
    for i_batch, batch in enumerate(tqdm(data_loader, total=total_batches, desc="Batches")):
        features, weights, labels = data_loader.split(batch)
        #predictions = tfmc.predict(features, ic_scaling=False)

        g     = infer.models["MultiClassifier"]["lowMT_VBFJet"].predict(features) 
        g_sum = g.sum(axis=1)

        d_R = {}
        d_R["mu"]          = g[:, data_structure.label_encoding['htautau']]/g_sum
        d_R["nu_bkg"]      = (1-g[:, data_structure.label_encoding['htautau']])/g_sum*np.log1p(infer.alpha_bkg)
        d_R["nu_tt"]       = g[:, data_structure.label_encoding['ttbar']]/g_sum*np.log1p(infer.alpha_tt)
        d_R["nu_diboson"]  = g[:, data_structure.label_encoding['diboson']]/g_sum*np.log1p(infer.alpha_diboson)

        DeltaA = {}
        for c in ['htautau', 'ztautau', 'ttbar', 'diboson']:
            DeltaA[c] = {}
            dA =  infer.models[c][selection].get_DeltaA( features )
            DeltaA[c]['nu_jes'] = dA[:, infer.models[c][selection].combinations.index(('nu_jes',))].numpy()
            DeltaA[c]['nu_jes'] += infer.icps[c][selection].DeltaA[infer.icps[c][selection].combinations.index(('nu_jes',))]
            DeltaA[c]['nu_tes'] = dA[:, infer.models[c][selection].combinations.index(('nu_tes',))].numpy()
            DeltaA[c]['nu_tes'] += infer.icps[c][selection].DeltaA[infer.icps[c][selection].combinations.index(('nu_tes',))]
            DeltaA[c]['nu_met'] = dA[:, infer.models[c][selection].combinations.index(('nu_met',))].numpy()
            DeltaA[c]['nu_met'] += infer.icps[c][selection].DeltaA[infer.icps[c][selection].combinations.index(('nu_met',))]

        d_R["nu_jes"] = np.sum([ DeltaA[c]['nu_jes']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum
        d_R["nu_tes"] = np.sum([ DeltaA[c]['nu_tes']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum
        d_R["nu_met"] = np.sum([ DeltaA[c]['nu_met']*g[:,data_structure.label_encoding[c]] for c in ['htautau', 'ztautau', 'ttbar', 'diboson']], axis=0)/g_sum

        # For each pair (var1, var2) in the parameters list, update the corresponding matrix entry.
        # Here we sum over the batch (using np.sum) to get a scalar contribution per batch.
        for i, var1 in enumerate(parameters):
            for j, var2 in enumerate(parameters):
                weight_batch = weights * d_R[var1] * d_R[var2]
                FI_unbinned[i, j] += np.sum(weight_batch)
                        
        if max_batch > 0 and i_batch + 1 >= max_batch:
            break

FI_binned = np.zeros((7, 7))

for _, poisson in infer.poisson.items():
        d_lambda = {}

        d_lambda["mu"]          = poisson["IC"].weight_sums[data_structure.label_encoding['htautau']] 
        d_lambda["nu_bkg"]      = np.sum( [poisson["IC"].weight_sums[data_structure.label_encoding[c]] for c in ['ztautau', 'ttbar', 'diboson']])*np.log1p(infer.alpha_bkg) 
        d_lambda["nu_tt"]       = poisson["IC"].weight_sums[data_structure.label_encoding['ttbar']]*np.log1p(infer.alpha_tt)
        d_lambda["nu_diboson"]  = poisson["IC"].weight_sums[data_structure.label_encoding['diboson']]*np.log1p(infer.alpha_diboson)
        d_lambda["nu_tes"]      = np.sum([poisson["IC"].weight_sums[data_structure.label_encoding[p]]*poisson["ICP"][p].DeltaA[poisson["ICP"][p].combinations.index(('nu_tes',))] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']])
        d_lambda["nu_jes"]      = np.sum([poisson["IC"].weight_sums[data_structure.label_encoding[p]]*poisson["ICP"][p].DeltaA[poisson["ICP"][p].combinations.index(('nu_jes',))] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']])
        d_lambda["nu_met"]      = np.sum([poisson["IC"].weight_sums[data_structure.label_encoding[p]]*poisson["ICP"][p].DeltaA[poisson["ICP"][p].combinations.index(('nu_met',))] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']])

        total = np.sum( [poisson["IC"].weight_sums[data_structure.label_encoding[p]] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']])

        for i, var1 in enumerate(parameters):
            for j, var2 in enumerate(parameters):
                FI_binned[i, j] += 1./total*d_lambda[var1]*d_lambda[var2] 


FI_penalty = np.eye(7)
FI_penalty[0,0]=0

FI = FI_binned + FI_unbinned + FI_penalty 



# Compute eigenvalues and eigenvectors.
# Using np.linalg.eigh because FI is symmetric.
eig_vals, eig_vecs = np.linalg.eigh(FI)

# Optionally, sort the eigenvalues (and corresponding eigenvectors) in descending order.
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

print("Eigenvalues:")
print(eig_vals)
print("\nEigenvectors (columns correspond to eigenvectors):")
print(eig_vecs)


# Partition the Fisher matrix:
I_aa = FI[0, 0]             # Scalar: information for mu
I_ab = FI[0, 1:].reshape(1, -1)  # 1 x 6: cross terms between mu and nuisance parameters
I_ba = FI[1:, 0].reshape(-1, 1)  # 6 x 1: transpose of I_ab
I_bb = FI[1:, 1:]           # 6 x 6: information for the nuisance parameters

# Compute the profiled Fisher information using the Schur complement:
I_profiled = I_aa - I_ab @ np.linalg.inv(I_bb) @ I_ba

# Convert to a scalar value
I_profiled = I_profiled.item()

print( "I_profiled", I_profiled )


# Create a canvas and set y-axis to logarithmic scale.
c = ROOT.TCanvas("c", "Fisher Eigensystem", 800, 600)
c.SetLogy()

# Create an empty histogram to establish the axes.
# x-axis from 0 to 1; y-axis range is set according to the eigenvalues.
h = ROOT.TH1F("h", "Fisher Eigensystem Composition;Fraction;Eigenvalue", 10, 0, 1)
min_val = np.min(eig_vals)
max_val = np.max(eig_vals)
# Set a y-range a bit below and above the min and max eigenvalues.

if args.small:
    h.GetYaxis().SetRangeUser(min_val*0.5, max_val*1.5)
else:
    h.GetYaxis().SetRangeUser(0.7, 5*10**3)
    
h.Draw()

# Define a list of colors for the 7 eigenvector components.
colors = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kOrange, ROOT.kMagenta+2, ROOT.kCyan-2, ROOT.kGray]
parameters = ["mu", "nu_bkg", "nu_tt", "nu_diboson", "nu_tes", "nu_jes", "nu_met"]

# Draw a horizontal stacked bar for each eigenvalue/eigenvector.
stuff = []
for i in range(len(eig_vals)):
    y_val = eig_vals[i]
    # Define a small vertical window around the eigenvalue.
    # (In a log-scale, a fixed percentage offset works better than a constant offset.)
    y_low = y_val * 0.95
    y_high = y_val * 1.05

    x_start = 0.0
    # Compute the normalization factor using the absolute values of the eigenvector components.
    total = sum(abs(eig_vecs[i, j]) for j in range(7))
    
    # Loop over the 7 components of the eigenvector.
    for j in range(7):
        # Take the absolute value and normalize to get a fraction.
        fraction = abs(eig_vecs[i, j]) / total
        x_end = x_start + fraction
        # Create a box representing this contribution.
        box = ROOT.TBox(x_start, y_low, x_end, y_high)
        box.SetFillColor(colors[j])
        box.Draw("same")
        stuff.append(box)
        x_start = x_end

config_name = os.path.basename(os.path.normpath(args.config)).replace('.yaml','') 

# Update canvas and optionally save to file.
c.Update()
c.Print(os.path.join( user.plot_directory, "Fisher", "eig"+("_small" if args.small else ""), config_name, "fisher_eigensystem.png"))
c.Print(os.path.join( user.plot_directory, "Fisher", "eig"+("_small" if args.small else ""), config_name, "fisher_eigensystem.pdf"))

helpers.copyIndexPHP(os.path.join(user.plot_directory, "Fisher", "eig"+("_small" if args.small else ""), config_name))
common.syncer.sync()

