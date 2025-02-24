import sys
sys.path.insert(0, "..")
import common.helpers as helpers

from common.logger import get_logger
import os
import numpy as np
import pickle
import argparse
import time
import yaml
import copy
import common.syncer

# ROOT imports
import ROOT
ROOT.gROOT.SetBatch(True)  # Run in batch mode so we don't pop up windows.

#from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user

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

import array

# Number of color levels
NCONT = 255
ROOT.gStyle.SetNumberContours(NCONT)

# Define a simple 2-stop gradient from blue to white
stops = array.array('d', [0.0,   0.3,  0.6,  0.8,  1.0])
red   = array.array('d', [0.0,   0.0,  0.0,  1.0,  1.0])
green = array.array('d', [0.0,   1.0,  1.0,  1.0,  1.0])
blue  = array.array('d', [1.0,   1.0,  0.0,  0.0,  1.0])
ROOT.TColor.CreateGradientColorTable(5, stops, red, green, blue, NCONT)

ROOT.gStyle.SetPaintTextFormat(".0f")

# Argument parser setup
parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
parser.add_argument("-c", "--config", help="Path to the config file.")
parser.add_argument("--small", action="store_true", help="Run a subset.")
parser.add_argument("--asimov_mu", type=float, default=None, help="Modify asimov weights according to mu.")
parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify asimov weights according to nu_bkg.")
parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Modify asimov weights according to nu_ttbar.")
parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify asimov weights according to nu_diboson.")
parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
parser.add_argument("--CSI", nargs="+", default = [], help="Make only those CSIs")
parser.add_argument("--toy", default = None, type=str,  help="Specify toy with path to h5 file.")

# The user-specific parameter scanning
parser.add_argument("--var1",
                    type=str,
                    default="mu",
                    choices=["mu","nu_bkg","nu_tt","nu_diboson","nu_tes","nu_jes","nu_met"],
                    help="Which parameter to vary.")
parser.add_argument("--range1",
                    nargs=3,
                    type=float,
                    default=[0, 3, 0.25],
                    help="start end step for scanning the chosen parameter.")
parser.add_argument("--var2",
                    type=str,
                    default="nu_tes",
                    choices=["mu","nu_bkg","nu_tt","nu_diboson","nu_tes","nu_jes","nu_met"],
                    help="Which parameter to vary.")
parser.add_argument("--range2",
                    nargs=3,
                    type=float,
                    default=[-1, 1, .2],
                    help="start end step for scanning the chosen parameter.")

args = parser.parse_args()

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
plot_directory   = os.path.join(user.plot_directory,   "LLplots2D", config_name)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
os.makedirs(fit_directory, exist_ok=True)
cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data{'_small' if args.small else ''}" )
os.makedirs(cfg['tmp_path'], exist_ok=True)

# Initialize inference object
toy_origin = "config"
toy_path = None
toy_from_memory = None
if args.toy is not None:
    toy_origin = "path"
    toy_path = args.toy

infer = Inference(cfg, small=args.small, overwrite=False, toy_origin=toy_origin, toy_path=toy_path, toy_from_memory=toy_from_memory)

# Define the likelihood function
likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
    infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                  nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                  asimov_mu=args.asimov_mu, \
                  asimov_nu_bkg=args.asimov_nu_bkg, \
                  asimov_nu_tt=args.asimov_nu_tt, \
                  asimov_nu_diboson=args.asimov_nu_diboson)


# Best-fit reference (example values from your snippet)
default_param = {
  'mu': 1,
  'nu_bkg': 0,
  'nu_tt': 0,
  'nu_diboson': 0,
  'nu_tes': 0,
  'nu_jes': 0,
  'nu_met': 0,
}

# ------------------------------------------------------------
#  2D scanning part
# ------------------------------------------------------------
var1_to_scan = args.var1
start1, end1, step1 = args.range1
var2_to_scan = args.var2
start2, end2, step2 = args.range2

# Calculate number of bins from (start, end, step)
nbinsX = int(round((end1 - start1) / step1))
nbinsY = int(round((end2 - start2) / step2))

h2 = ROOT.TH2D(
    "h2", 
    f"-2 lnL vs. {var1_to_scan} and {var2_to_scan};{var1_to_scan};{var2_to_scan}", 
    nbinsX, start1, end1, 
    nbinsY, start2, end2
)
h2.SetStats(False)

# Copy best-fit point so we can modify
p = copy.deepcopy(default_param)

# Loop over grid in (var1, var2)
# Here, to fill each bin, we choose the "center" of the bin. Alternatively,
# you could do start1 + i*step1 if you want the left edge, etc.
for i in range(nbinsX):
    x_val = start1 + (i + 0.5)*step1  # bin center in x
    for j in range(nbinsY):
        y_val = start2 + (j + 0.5)*step2  # bin center in y
        p[var1_to_scan] = x_val
        p[var2_to_scan] = y_val

        # Evaluate log-likelihoods
        lnL   = (100 if args.small else 1)*likelihood_function(**p)

        # Fill histograms
        h2.Fill(x_val, y_val, lnL)

z_min = h2.GetMinimum()

nbinsX = h2.GetNbinsX()
nbinsY = h2.GetNbinsY()
for i in range(1, nbinsX + 1):
    for j in range(1, nbinsY + 1):
        old_val = h2.GetBinContent(i, j)
        new_val = old_val - z_min
        # If new_val is very close to 0 (e.g. negative by tiny floating error):
        if abs(new_val) < 1e-5:
            new_val = 0.0
        h2.SetBinContent(i, j, new_val)
h2.GetZaxis().SetRangeUser(-0.01, 9)

# ------------------------------------------------------------
#  Plot and save
# ------------------------------------------------------------

c = ROOT.TCanvas("c","-2LogL 2D",800,600)
h2.Draw("COLZ TEXT")
h2.SetMarkerSize(1.0)
c.SetRightMargin(0.15)
c.SetGrid()
outname = f"h2_{var1_to_scan}_{var2_to_scan}"
if postfix:
    outname += f"_{postfix}"
outname += ".png"
c.SaveAs(os.path.join(plot_directory, outname))

print(f"Saved 2D plots in {os.path.abspath('plots')}")

# Optional: sync if you use some custom sync or display

common.syncer.sync()
