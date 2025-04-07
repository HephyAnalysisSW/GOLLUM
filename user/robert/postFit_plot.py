import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os
import numpy as np
import pickle
import argparse
import time
import yaml
from tqdm import tqdm
#from common.likelihoodFit import likelihoodFit
from Workflow.Inference import Inference
import common.user as user
import common.data_structure as data_structure
import common.helpers as helpers
import common.selections as selections
import common.syncer

import ROOT
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

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
parser.add_argument("--config", help="Path to the config file.")
parser.add_argument("--impacts", action="store_true", help="Run post-fit uncertainties.")
parser.add_argument("--scan", action="store_true", help="Run likelihood scan.")
parser.add_argument("--small", action="store_true", help="Run a subset.")
parser.add_argument("--doHesse", action="store_true", help="Run Hesse after Minuit?")
parser.add_argument("--minimizer", type=str, default="minuit", choices=["minuit", "bfgs", "robust"], help="Which minimizer?")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
parser.add_argument("--asimov_mu", type=float, default=None, help="Modify asimov weights according to mu.")
parser.add_argument("--start_mu", type=float, default=1.0, help="Modify asimov weights according to mu.")
parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify asimov weights according to nu_bkg.")
parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Modify asimov weights according to nu_ttbar.")
parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify asimov weights according to nu_diboson.")
parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
parser.add_argument("--selection", default = "lowMT_VBFJet", type=str,  help="Which selection")
parser.add_argument("--CSI", nargs="+", default = [], help="Make only those CSIs")
parser.add_argument("--toy", default = None, type=str,  help="Specify toy with path to h5 file.")
parser.add_argument("--var", default = "DER_pt_h", type=str,  help="Which feature to plot")

parser.add_argument("--prefit", action="store_true", help="Use prefit?")

args = parser.parse_args()

from common.logger import get_logger
logger = get_logger(args.logLevel, logFile = None)

# Construct postfix for filenames based on asimov parameters
postfix = []
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

## Define output directory
config_name = os.path.basename(args.config).replace(".yaml", "")
output_directory = os.path.join ( user.output_directory, config_name)

#fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
#os.makedirs(fit_directory, exist_ok=True)
cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data" )

from common.likelihoodFit import likelihoodFit

# Initialize inference object
toy_origin = "config"
toy_path = None
toy_from_memory = None
if args.toy is not None:
    toy_origin = "path"
    toy_path = args.toy

infer = Inference(cfg, small=False, overwrite=args.overwrite, toy_origin=toy_origin, toy_path=toy_path, toy_from_memory=toy_from_memory)

# Define the likelihood function
likelihood_function = lambda mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met: \
    infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                  nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                  asimov_mu=args.asimov_mu, \
                  asimov_nu_bkg=args.asimov_nu_bkg, \
                  asimov_nu_tt=args.asimov_nu_tt, \
                  asimov_nu_diboson=args.asimov_nu_diboson)

# Perform global fit
logger.info("Start global fit.")
fit = likelihoodFit(likelihood_function, doHesse=args.doHesse)

#profiler = cProfile.Profile()
#profiler.enable()

q_mle, parameters_mle, cov, limits = fit.fit(start_mu=args.start_mu)
logger.info("Fit done.")

n_split = 100

# Decide how we get the data: from training or from a toy file
data_input = infer.training_data_loader(args.selection, n_split)

# Translate to numpy
parameters = [name for name, _ in sorted(cov._var2pos.items(), key=lambda x: x[1])]
if args.prefit:
    params_mle = {p:0 for p in parameters}
    params_mle['mu'] = 1
    cov       = np.eye(7)
    cov[0,0]  = 0
else:
    params_mle = {p:parameters_mle[p] for p in parameters}
    cov       = np.array(cov)

# Output directory for plots
plot_directory = os.path.join(user.plot_directory, "postfit", ("small_" if args.small else "") + config_name, args.selection)
os.makedirs(plot_directory, exist_ok=True)
helpers.copyIndexPHP(plot_directory)

# Initialize histogram with zero (one per label and variable)
histograms = {}
feature_index = data_structure.feature_names.index(args.var)
for label in data_structure.labels:
    binning = data_structure.plot_options[args.var]['binning'] 
    bin_edges = np.linspace(binning[1], binning[2], binning[0] + 1)
    histograms[label] = np.zeros(binning[0])


data_histo = np.histogram( getattr(selections, args.selection)(infer.h5s['Toy']['lowMT_VBFJet']['features'])[:, feature_index], bins=bin_edges )

diff_lambda = np.zeros((binning[0], cov.shape[0]))

# Loop over batches of data and store the results incrementally
with tqdm(total=len(data_input), desc="Processing batches") as pbar:
    for i_batch, batch in enumerate(data_input):
        features, weights, labels = data_input.split(batch)

        g     = infer.models["MultiClassifier"]["lowMT_VBFJet"].predict(features)
        g_sum = g.sum(axis=1)
        g/=g_sum[:,np.newaxis]
       
        gH  = g[:, data_structure.label_encoding['htautau']] 
        gZ  = g[:, data_structure.label_encoding['ztautau']] 
        gtt = g[:, data_structure.label_encoding['ttbar']] 
        gVV = g[:, data_structure.label_encoding['diboson']] 
        
        logS = {}
        logS_diff = {}
        for process in ['htautau', 'ztautau', 'ttbar', 'diboson']:

            delta_A = infer.models[process][args.selection].get_DeltaA( features )
            nu_A    = infer.models[process][args.selection].nu_A(tuple([params_mle[p] for p in infer.models[process][args.selection].parameters]))
            nu_A_diff_tes    = infer.models[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.models[process][args.selection].parameters]), "nu_tes")
            nu_A_diff_jes    = infer.models[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.models[process][args.selection].parameters]), "nu_jes")
            nu_A_diff_met    = infer.models[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.models[process][args.selection].parameters]), "nu_met")

            delta_A_icp = infer.icps[process][args.selection].DeltaA
            nu_A_icp    = infer.icps[process][args.selection].nu_A(tuple([params_mle[p] for p in infer.icps[process][args.selection].parameters]))
            nu_A_icp_diff_tes    = infer.icps[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.icps[process][args.selection].parameters]), "nu_tes")
            nu_A_icp_diff_jes    = infer.icps[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.icps[process][args.selection].parameters]), "nu_jes")
            nu_A_icp_diff_met    = infer.icps[process][args.selection].nu_A_diff(tuple([params_mle[p] for p in infer.icps[process][args.selection].parameters]), "nu_met")

            logS[process] = np.dot( delta_A, nu_A) + np.dot( delta_A_icp, nu_A_icp )

            logS_diff[process] = {
                'nu_tes': np.dot( delta_A, nu_A_diff_tes) + np.dot(  delta_A_icp, nu_A_icp_diff_tes ),
                'nu_jes': np.dot( delta_A, nu_A_diff_jes) + np.dot(  delta_A_icp, nu_A_icp_diff_jes ),
                'nu_met': np.dot( delta_A, nu_A_diff_met) + np.dot(  delta_A_icp, nu_A_icp_diff_met ),
                }

        mu = params_mle['mu'] 
        nu_bkg     = params_mle['nu_bkg'] 
        nu_tt      = params_mle['nu_tt'] 
        nu_diboson = params_mle['nu_diboson'] 

        fac_bkg    = np.exp(nu_bkg * np.log1p(infer.alpha_bkg))
        fac_tt     = np.exp(nu_tt  * np.log1p(infer.alpha_tt))
        fac_VV     = np.exp(nu_diboson * np.log1p(infer.alpha_diboson))

        R = { 'htautau': mu*gH*np.exp(logS['htautau']),
              'ztautau': fac_bkg*gZ*np.exp(logS['ztautau']),
              'ttbar':   fac_bkg*gtt*fac_tt*np.exp(logS['ttbar']),
              'diboson': fac_bkg*gVV*fac_VV*np.exp(logS['diboson']),
                   }
        R_ = np.column_stack( [R[p] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']]) 
        R_diff = np.column_stack( [
          gH*np.exp(logS['htautau']),
          np.log1p(infer.alpha_bkg)*fac_bkg*( gZ*np.exp(logS['ztautau']) + gtt*fac_tt*np.exp(logS['ttbar']) + gVV*fac_VV*np.exp(logS['diboson'])),
          fac_bkg*( np.log1p(infer.alpha_tt)*gtt*fac_tt*np.exp(logS['ttbar']) ),
          fac_bkg*( np.log1p(infer.alpha_diboson)*gVV*fac_VV*np.exp(logS['diboson']) ),
          mu*gH*np.exp(logS['htautau'])*logS_diff['htautau']['nu_tes'] + fac_bkg*( gZ*np.exp(logS['ztautau'])*logS_diff['ztautau']['nu_tes'] + gtt*fac_tt*np.exp(logS['ttbar'])*logS_diff['ttbar']['nu_tes'] + gVV*fac_VV*np.exp(logS['diboson'])*logS_diff['diboson']['nu_tes']),
          mu*gH*np.exp(logS['htautau'])*logS_diff['htautau']['nu_jes'] + fac_bkg*( gZ*np.exp(logS['ztautau'])*logS_diff['ztautau']['nu_jes'] + gtt*fac_tt*np.exp(logS['ttbar'])*logS_diff['ttbar']['nu_jes'] + gVV*fac_VV*np.exp(logS['diboson'])*logS_diff['diboson']['nu_jes']),
          mu*gH*np.exp(logS['htautau'])*logS_diff['htautau']['nu_met'] + fac_bkg*( gZ*np.exp(logS['ztautau'])*logS_diff['ztautau']['nu_met'] + gtt*fac_tt*np.exp(logS['ttbar'])*logS_diff['ttbar']['nu_met'] + gVV*fac_VV*np.exp(logS['diboson'])*logS_diff['diboson']['nu_met']),
            ])

        bin_indices = np.digitize(features[:, feature_index], bin_edges) - 1
        weights[bin_indices == binning[0]] = 0   # Do NOT show overflow 
        bin_indices[bin_indices == binning[0]] = binning[0] - 1  # ensure valid indices

        # Accumulate weighted yields in each bin for each key
        for b in range(binning[0]):
            in_bin = (bin_indices == b)

            #exp_yield = np.einsum("i,i->", weights, R)
            #g = np.einsum("i,ij->j", weights[in_bin], R_diff[in_bin])
            #histograms["variance"][b] += np.einsum("i,ij,j->", g, cov, g)  

            #diff_lambda =  weights[in_bin] @ R_diff[in_bin]

            #histograms["variance"][b] += diff_lambda @ cov @ diff_lambda 

            #rel_unc = np.sqrt(diff_lambda @ cov @ diff_lambda)/np.sum(weights[in_bin] @ R_[in_bin])

            for key in R:
                histograms[key][b] += np.sum(weights[in_bin] @ R[key][in_bin])

            diff_lambda[b] += weights[in_bin] @ R_diff[in_bin]

            #print( "bin", b, "rel_unc", rel_unc,"var",histograms["variance"][b], "yield",np.sum([histograms[p][b] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']]), "current rel:", np.sqrt(histograms["variance"][b])/np.sum([histograms[p][b] for p in ['htautau', 'ztautau', 'ttbar', 'diboson']]) )

        if args.small: break

        pbar.update(1)

histograms['variance'] = np.diag(((diff_lambda @ cov) @ diff_lambda.transpose()))

color = {
    'htautau': ROOT.TColor.GetColor("#F28E2B"),
    'ztautau': ROOT.TColor.GetColor("#59A14F"),
    'ttbar':   ROOT.TColor.GetColor("#E15759"),
    'diboson': ROOT.TColor.GetColor("#4E79A7"),
}

root_hists = {}
for key in ['htautau', 'ztautau', 'ttbar', 'diboson']:
    h = ROOT.TH1F("h_" + key, key, len(bin_edges) - 1, bin_edges[0], bin_edges[-1])
    # Fill each bin (ROOT bins are 1-indexed)
    for i, content in enumerate(histograms[key]):
        h.SetBinContent(i+1, content)

    h.SetFillColor(color[key])
    h.SetLineColor(color[key])

    root_hists[key] = h

#--- Determine stacking order ---
# We stack by total yield (lowest total yield at the bottom)
#stack_order = sorted(root_hists.keys(), key=lambda k: root_hists[k].Integral())
stack_order = ['htautau', 'diboson', 'ttbar', 'ztautau'] 

#--- Create a THStack and add histograms in that order ---#
stack = ROOT.THStack("stack", "Stacked Yield;%s;Events"%data_structure.plot_options[args.var]['tex'])
for key in stack_order:
    stack.Add(root_hists[key])

#--- Create a total yield histogram (sum of all components) ---#
h_total = root_hists[stack_order[0]].Clone("h_total")
h_total.Reset()
for key in root_hists:
    h_total.Add(root_hists[key])

#--- Build a TGraphAsymmErrors for the uncertainty band ---
# Use the 'variance' array to compute the uncertainty per bin (error = sqrt(variance))
total_unc = np.sqrt(histograms['variance'])
n_bins = h_total.GetNbinsX()
x = []
y = []
ex_low = []
ex_high = []
ey_low = []
ey_high = []

for i in range(1, n_bins+1):
    xc = h_total.GetBinCenter(i)
    bw = h_total.GetBinWidth(i)
    content = h_total.GetBinContent(i)
    error = total_unc[i-1]  # assuming same bin ordering as the np arrays
    x.append(xc)
    y.append(content)
    ex_low.append(bw/2.)
    ex_high.append(bw/2.)
    ey_low.append(error)
    ey_high.append(error)

unc_graph = ROOT.TGraphAsymmErrors(n_bins,
                                   np.array(x, dtype=float),
                                   np.array(y, dtype=float),
                                   np.array(ex_low, dtype=float),
                                   np.array(ex_high, dtype=float),
                                   np.array(ey_low, dtype=float),
                                   np.array(ey_high, dtype=float))
# Style for uncertainty band
unc_graph.SetFillColor(ROOT.kGray+2)
unc_graph.SetFillStyle(3004)
unc_graph.SetLineColor(ROOT.kGray+2)
unc_graph.SetMarkerSize(0)

# data histo
data_histo_ = helpers.make_TH1F( data_histo )

data_histo_.SetLineColor( ROOT.kBlack )
data_histo_.SetMarkerSize( 1 )
data_histo_.SetMarkerStyle( 20 )
data_histo_.SetMarkerColor( ROOT.kBlack )
data_histo_.SetLineWidth( 1 )

#--- Draw on a canvas with two pads (no gap between pads and larger bottom pad labels) ---#
canvas = ROOT.TCanvas("c", "Stacked Histogram with Uncertainty", 800, 600)

# Upper pad: main stack, data, and uncertainty band
pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1)
pad1.SetBottomMargin(0.0)  # Remove gap by setting bottom margin to 0
pad1.SetLogy(data_structure.plot_options[args.var]['logY'])
pad1.Draw()
pad1.cd()

stack.Draw("HIST")
stack.GetXaxis().SetTitle(data_structure.plot_options[args.var]['tex'])
unc_graph.Draw("E2 SAME")
data_histo_.Draw("e1 SAME")
pad1.Update()

# Lower pad: difference (data minus background) vs. signal prediction with uncertainty band
canvas.cd()
pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
pad2.SetTopMargin(0.0)   # Remove gap by setting top margin to 0
pad2.SetBottomMargin(0.3)
pad2.Draw()
pad2.cd()

# Build background sum histogram (ztautau, ttbar, diboson)
h_bkg = root_hists["ztautau"].Clone("h_bkg")
h_bkg.Add(root_hists["ttbar"])
h_bkg.Add(root_hists["diboson"])

# Compute subtracted data: data - background (only central values)
h_subtracted_data = data_histo_.Clone("h_subtracted_data")
h_subtracted_data.Add(h_bkg, -1)

# Compute subtracted signal: signal - background
h_subtracted_signal = root_hists["htautau"].Clone("h_subtracted_signal")

# Build an uncertainty band for the subtracted signal.
# (Here we reuse the total_unc array as an approximation for the signal uncertainty.)
n_bins_diff = h_subtracted_signal.GetNbinsX()
signal_x = []
signal_y = []
ex_low = []
ex_high = []
signal_unc = []
for i in range(1, n_bins_diff+1):
    xc = h_subtracted_signal.GetBinCenter(i)
    bw = h_subtracted_signal.GetBinWidth(i)
    # Central value of the subtracted signal
    content = h_subtracted_signal.GetBinContent(i)
    # Use the same uncertainty as computed earlier (approximate)
    error = total_unc[i-1]

    signal_x.append(xc)
    signal_y.append(content)
    ex_low.append(bw/2.)
    ex_high.append(bw/2.)
    signal_unc.append(error)

    h_subtracted_data.SetBinError(i, data_histo_.GetBinError( i ) )

signal_graph = ROOT.TGraphAsymmErrors(n_bins_diff,
                                      np.array(signal_x, dtype=float),
                                      np.array(signal_y, dtype=float),
                                      np.array(ex_low, dtype=float),
                                      np.array(ex_high, dtype=float),
                                      np.array(signal_unc, dtype=float),
                                      np.array(signal_unc, dtype=float))
signal_graph.SetFillColor(ROOT.kGray+2)
signal_graph.SetFillStyle(3004)
signal_graph.SetLineColor(ROOT.kGray+2)
signal_graph.SetMarkerSize(0)

h_subtracted_signal.SetLineColor(color['htautau'])
h_subtracted_signal.SetLineWidth(2)
h_subtracted_signal.SetFillStyle(0)

# Draw the difference pad: set axis titles and ranges as needed
h_subtracted_data.GetXaxis().SetTitle(data_structure.plot_options[args.var]['tex'])
h_subtracted_data.Draw("e1")
h_subtracted_data.GetYaxis().SetTitle("Data - Bkg.")
h_subtracted_data.GetYaxis().SetLabelSize(0.12)
h_subtracted_data.GetYaxis().SetTitleSize(0.12)

signal_graph.Draw("E2 SAME")
h_subtracted_data.Draw("e1 SAME")
h_subtracted_signal.Draw("same")

# Increase label and title sizes on the bottom pad for clarity
h_subtracted_data.GetXaxis().SetLabelSize(0.12)
h_subtracted_data.GetXaxis().SetTitleSize(0.12)
h_subtracted_data.GetYaxis().SetLabelSize(0.12)
h_subtracted_data.GetYaxis().SetTitleSize(0.12)

pad2.Update()
canvas.cd()
canvas.Update()

#--- Save the canvas ---#
canvas.SaveAs(os.path.join(plot_directory, ("prefit_" if args.prefit else "postfit_")+f"{args.var}.png"))
canvas.SaveAs(os.path.join(plot_directory, ("prefit_" if args.prefit else "postfit_")+f"{args.var}.pdf"))

common.syncer.sync()

