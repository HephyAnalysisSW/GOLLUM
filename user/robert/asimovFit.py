import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os
import numpy as np
import pickle
import argparse
import time
import yaml
from Workflow.Inference import Inference
import common.user as user
from iminuit import Minuit

import cProfile, pstats


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
parser.add_argument("--small", action="store_true", help="Run a subset.")
parser.add_argument("--doHesse", action="store_true", help="Run Hesse after Minuit?")
parser.add_argument("--minimizer", type=str, default="minuit", choices=["minuit", "bfgs", "robust"], help="Which minimizer?")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
parser.add_argument("--asimov_mu", type=float, default=1.0, help="Modify asimov weights according to mu.")
parser.add_argument("--fixed_mu", type=float, default=None, help="Modify asimov weights according to mu.")
parser.add_argument("--asimov_nu_bkg", type=float, default=None, help="Modify asimov weights according to nu_bkg.")
parser.add_argument("--asimov_nu_tt", type=float, default=None, help="Modify asimov weights according to nu_ttbar.")
parser.add_argument("--asimov_nu_diboson", type=float, default=None, help="Modify asimov weights according to nu_diboson.")
parser.add_argument("--modify", nargs="+", help="Key-value pairs to modify, e.g., CSI.save=true.")
parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")
parser.add_argument("--CSI", nargs="+", default = [], help="Make only those CSIs")
parser.add_argument("--toy", default = None, type=str,  help="Specify toy with path to h5 file.")

args = parser.parse_args()

from common.logger import get_logger
logger = get_logger(args.logLevel, logFile = None)

# Construct postfix for filenames based on asimov parameters
postfix = []
if args.fixed_mu is not None:
    postfix.append(f"fixed_mu_{args.fixed_mu:.3f}".replace("-", "m").replace(".", "p"))
if args.asimov_mu is not None:
    postfix.append(f"asimov_mu_{args.asimov_mu:.3f}".replace("-", "m").replace(".", "p"))
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

# Define output directory
config_name = os.path.basename(args.config).replace(".yaml", "")
output_directory = os.path.join ( user.output_directory, config_name)

fit_directory = os.path.join( output_directory, f"fit_data{'_small' if args.small else ''}" )
os.makedirs(fit_directory, exist_ok=True)
cfg['tmp_path'] = os.path.join( output_directory, f"tmp_data{'_small' if args.small else ''}" )
os.makedirs(cfg['tmp_path'], exist_ok=True)

if args.minimizer=="bfgs":
    from common.likelihoodFit_BFGS import likelihoodFit
elif args.minimizer=="robust":
    from common.likelihoodFit2 import likelihoodFit
else:
    from common.likelihoodFit import likelihoodFit

# Initialize inference object
toy_origin = "config"
toy_path = None
toy_from_memory = None
if args.toy is not None:
    toy_origin = "path"
    toy_path = args.toy

parameterBoundaries = {
    "mu": (0.0, None),
    "nu_bkg": (-10., 10.),
    "nu_tt": (-10., 10.),
    "nu_diboson": (-4., 4.),
    "nu_jes": (-10., 10.),
    "nu_tes": (-10., 10.),
    "nu_met": (0., 5.),
}

def within_boundaries(params):
    for param, (lower, upper) in parameterBoundaries.items():
        value = params.get(param)
        if value is None:
            # Parameter not provided; consider it out-of-bound.
            return False
        if lower is not None and value < lower:
            return False
        if upper is not None and value > upper:
            return False
    return True

infer = Inference(cfg, small=args.small, overwrite=False, toy_origin=toy_origin, toy_path=toy_path, toy_from_memory=toy_from_memory)

# Define the likelihood function

if args.fixed_mu is None:
    def function(mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met): 
        if not within_boundaries({'mu':mu, 'nu_bkg':nu_bkg, 'nu_tt':nu_tt, 'nu_diboson':nu_diboson, 'nu_tes':nu_tes, 'nu_jes':nu_jes, 'nu_met':nu_met}):
            logger.warning("Hit boundaries: %r", {'mu':mu, 'nu_bkg':nu_bkg, 'nu_tt':nu_tt, 'nu_diboson':nu_diboson, 'nu_tes':nu_tes, 'nu_jes':nu_jes, 'nu_met':nu_met})
            return 999
        return infer.predict(mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                      nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                      asimov_mu=args.asimov_mu, \
                      asimov_nu_bkg=args.asimov_nu_bkg, \
                      asimov_nu_tt=args.asimov_nu_tt, \
                      asimov_nu_diboson=args.asimov_nu_diboson)
else:
    def function(nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met): 
        if not within_boundaries({'nu_bkg':nu_bkg, 'nu_tt':nu_tt, 'nu_diboson':nu_diboson, 'nu_tes':nu_tes, 'nu_jes':nu_jes, 'nu_met':nu_met}):
            logger.warning("Hit boundaries: %r", {'nu_bkg':nu_bkg, 'nu_tt':nu_tt, 'nu_diboson':nu_diboson, 'nu_tes':nu_tes, 'nu_jes':nu_jes, 'nu_met':nu_met})
            return 999
        return infer.predict(mu=args.fixed_mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, \
                      nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met, \
                      asimov_mu=args.asimov_mu, \
                      asimov_nu_bkg=args.asimov_nu_bkg, \
                      asimov_nu_tt=args.asimov_nu_tt, \
                      asimov_nu_diboson=args.asimov_nu_diboson)

strategy        = 0 # default 1
tolerance       = 0.1 # default 0.1
eps             = 0.1 # default
q_mle           = None
parameters_mle  = None
doHesse         = True

# function to find the global minimum, minimizing mu and nus
logger.info("Fit global minimum")
if args.fixed_mu is None:
    m = Minuit(function, mu=1., nu_bkg=0., nu_tt=0., nu_diboson=0., nu_jes=0., nu_tes=0., nu_met=0.)
    parameters = [ "mu", "nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]
else:
    m = Minuit(function, nu_bkg=0., nu_tt=0., nu_diboson=0., nu_jes=0., nu_tes=0., nu_met=0.)
    parameters = [ "nu_bkg", "nu_tt", "nu_diboson", "nu_jes", "nu_tes", "nu_met"]
    
m.errordef = Minuit.LIKELIHOOD

for nuname in parameters:
    m.limits[nuname] = parameterBoundaries[nuname]

m.strategy = strategy
m.tol      = tolerance

for param in m.parameters:
    m.errors[param] = eps  # Set the step size for all parameters

np.set_printoptions(precision=4, suppress=True, linewidth=200, 
                        formatter={'float': '{:6.3f}'.format})

if args.fixed_mu is None:
    bounds=[
            parameterBoundaries["mu"],
            parameterBoundaries["nu_bkg"],
            parameterBoundaries["nu_tt"],
            parameterBoundaries["nu_diboson"],
            parameterBoundaries["nu_jes"],
            parameterBoundaries["nu_tes"],
            parameterBoundaries["nu_met"],
        ]
else:
    bounds=[
            #parameterBoundaries["mu"],
            parameterBoundaries["nu_bkg"],
            parameterBoundaries["nu_tt"],
            parameterBoundaries["nu_diboson"],
            parameterBoundaries["nu_jes"],
            parameterBoundaries["nu_tes"],
            parameterBoundaries["nu_met"],
        ]

from common.approximate_hessian import approximate_hessian
wrapped_func = lambda x: function(*x)

c_name = os.path.splitext(os.path.basename(args.config))[0]
out_dir = os.path.join(user.output_directory, "asimovFits", c_name)
if not os.path.exists(out_dir):
    os.makedirs( out_dir, exist_ok=True)

filename = os.path.join(out_dir, f"output_{postfix}.npz")

if not os.path.exists( filename ) or args.overwrite:
    m.migrad()
    logger.info("Before 'm.hesse().")
    print(m)
    if(doHesse):
        m.hesse()
        logger.info("After 'm.hesse()'")
        print(m)

    hess = approximate_hessian(
        wrapped_func,
        np.array(m.values),
        step=0.1,
        bounds=bounds
    )

    f_min = m.fmin.fval
    mu_asimov=args.asimov_mu
    mu_fit=args.fixed_mu
    m_cov = np.array(m.covariance)
    m_val = np.array(m.values)
    
    np.savez(filename,
        f_min = f_min,
        mu_asimov=args.asimov_mu,
        mu_fit=mu_fit,
        m_cov = m_cov,
        m_val = m_val,
        approx_hess=hess,
        )
    logger.info("Saved to file: %s", filename)
else:
    data = np.load(filename, allow_pickle=True)

    f_min      = data['f_min']
    mu_asimov  = data['mu_asimov']
    mu_fit     = data['mu_fit']
    m_cov      = data['m_cov']
    m_val      = data['m_val']
    hess = data['approx_hess']
    logger.info("Loaded from file: %s", filename)

# Print the matrix
print("Approximate covariance:")
print(np.linalg.inv(hess))

if args.fixed_mu is None:
    import ROOT
    import common.user as user
    import common.helpers
    import common.syncer
    ROOT.gStyle.SetOptStat(0)
    plot_directory = os.path.join(user.plot_directory, "impacts", c_name)
    os.makedirs(plot_directory, exist_ok=True)
    common.helpers.copyIndexPHP(plot_directory)

    # Invert the Hessian to get the covariance matrix.
    cov = np.linalg.inv(hess)
    nparameters = len(parameters)

    # Compute the correlation matrix.
    correlations = np.zeros((nparameters, nparameters))
    for i in range(nparameters):
        for j in range(nparameters):
            correlations[i, j] = cov[i, j] / (np.sqrt(cov[i, i] * cov[j, j]))

    # Define parameter names.
    fitResult = {k:m_val[i_k] for i_k, k in enumerate(parameters)}

    ## GAUSSIAN Compute impacts on mu for each nuisance parameter.
    #impacts = {}
    #for i in range(1, len(parameters)):
    #    # Impact = - (cov[mu, nu] / cov[mu, mu]) * sqrt(cov[nu, nu])
    #    impacts[parameters[i]] = - (cov[0, i] / cov[0, 0]) * np.sqrt(cov[i, i])
    #    print("Impact on mu from %s: %.3f" % (parameters[i], impacts[parameters[i]]))

    # Compute non-Gaussian impacts on μ by stepping each nuisance parameter until the function (−2 ln L)
    # increases by 1 unit relative to the best-fit value, and then refitting μ with that nuisance fixed.

    # Best-fit parameters and target function value.
    #best_fit = copy.deepcopy(m_val)   # dictionary of best-fit parameter values (keys: "mu", "nu_bkg", etc.)
    f_target = float(f_min) + 1.0      # target: increase of 1 unit in −2 ln L
    step = {p:0.1*np.sqrt(cov[i_p,i_p]) for i_p, p in enumerate(parameters)}                 # step size for nuisance parameter variation

    # Define the list of nuisance parameters (excluding μ)
    nuisances = parameters[1:] 

    from scipy.optimize import brentq, minimize_scalar

    # f_target is the target function value: best-fit + 1.
    f_target = float(f_min) + 1.0  

    # Dictionary to hold numerical impacts on μ and the corresponding nuisance values.
    impacts_numerical = {}
    nuisance_values = {}

    # Loop over each nuisance (exclude the first, "mu").
    for nu in parameters[1:]:
        best_nu = fitResult[nu]

        # Define a function of the nuisance parameter with μ fixed at best-fit.
        # It returns (function(parameters) - f_target).
        def g_nu(nu_val):
            p = fitResult.copy()
            p[nu] = nu_val
            return function(**p) - f_target

        # --- Find positive variation ---
        a = best_nu
        upper_bound = parameterBoundaries[nu][1]
        # If no upper bound is set, choose an arbitrary upper value.
        b = upper_bound if upper_bound is not None else best_nu + 1.0
        try:
            nu_val_plus = brentq(g_nu, a, b)
        except ValueError:
            nu_val_plus = best_nu  # If no crossing is found, default to best-fit.

        # --- Find negative variation ---
        lower_bound = parameterBoundaries[nu][0]
        a_neg = lower_bound if lower_bound is not None else best_nu - 1.0
        b_neg = best_nu
        try:
            nu_val_minus = brentq(g_nu, a_neg, b_neg)
        except ValueError:
            nu_val_minus = best_nu

        # Save the nuisance parameter values.
        nuisance_values[nu] = {'up': nu_val_plus, 'down': nu_val_minus}

        # Now, refit μ with the nuisance fixed at the found values.
        def f_mu(mu):
            p = fitResult.copy()
            p["mu"] = mu
            p[nu] = nu_val_plus
            return function(**p)
        res_plus = minimize_scalar(f_mu, bounds=(fitResult["mu"] - 0.5, fitResult["mu"] + 0.5), method='bounded')
        mu_plus = res_plus.x

        def f_mu_neg(mu):
            p = fitResult.copy()
            p["mu"] = mu
            p[nu] = nu_val_minus
            return function(**p)
        res_minus = minimize_scalar(f_mu_neg, bounds=(fitResult["mu"] - 0.5, fitResult["mu"] + 0.5), method='bounded')
        mu_minus = res_minus.x

        # Define impacts as the difference relative to the best-fit μ.
        impact_plus = mu_plus - fitResult["mu"]
        impact_minus = mu_minus - fitResult["mu"]
        impacts_numerical[nu] = {'down': impact_minus, 'up': impact_plus}
        
        logger.info("Nuisance %s: non-Gaussian impact: - = %.4f, + = %.4f" % (nu, impact_minus, impact_plus))

    logger.info( "nuisance values %r impacts_numerical %r", nuisance_values, impacts_numerical )

    # --------------------
    # IMPACT PLOT BLOCK (Spaghetti)
    # --------------------
    # This block assumes that 'fitResult' is a dictionary with keys for each nuisance parameter 
    # containing tuples (mle, lower, upper) for the post-fit values, that 'impacts' is a dictionary 
    # with the computed impacts on μ, and that 'plot_directory' is defined.
    # The nuisance parameters are:
    # Canvas and pad settings.
    xboundary = 0.7
    separation = 0.02
    TopMargin = 0.02
    LeftMargin = 0.35
    RightMargin = 0.01
    BottomMargin = 0.1
    xmin, xmax = -2.5, 2.5
    ymin, ymax = 0, 7.5
    stuff = []
    # Helper function: Dummy graph for setting up axes.
    def getDummy(title, xmin, xmax, ymin, ymax, factor=1.0):
        g = ROOT.TGraph(2)
        g.SetPoint(0, xmin, ymin)
        g.SetPoint(1, xmax, ymax)
        g.SetMarkerSize(0.0)
        g.SetTitle('')
        g.GetXaxis().SetTitle(title)
        g.GetYaxis().SetTitle("")
        g.GetXaxis().SetRangeUser(xmin, xmax)
        g.GetYaxis().SetRangeUser(ymin, ymax)
        g.GetYaxis().SetLabelSize(0.0)
        g.GetYaxis().SetTickLength(0.0)
        g.GetXaxis().SetTitleOffset(0.8/factor)
        g.GetXaxis().SetLabelOffset(0.001/factor)
        g.GetXaxis().SetTitleSize(0.05*factor)
        g.GetXaxis().SetLabelSize(0.03*factor)
        g.GetXaxis().SetNdivisions(505)
        return g

    def calcPosition(x,y):
        axis_length_x = xmax - xmin
        axis_length_canvas_x = 1-RightMargin-LeftMargin
        x_canvas = LeftMargin + (x/axis_length_x) * axis_length_canvas_x

        axis_length_y = ymax - ymin
        axis_length_canvas_y = 1-TopMargin-BottomMargin
        y_canvas = BottomMargin + (y/axis_length_y) * axis_length_canvas_y
        return x_canvas, y_canvas


    # Helper to add text using TLatex.
    def addText(x, y, text, font=43, size=16, color=ROOT.kBlack):
        latex = ROOT.TLatex(x, y, text)
        latex.SetNDC()
        latex.SetTextAlign(12)
        latex.SetTextFont(font)
        latex.SetTextSize(size)
        latex.SetTextColor(color)
        return latex

    # Helper to create post-fit and pre-fit graphs.
    def createGraph(fitResult, nuisance_values, parameters):
        # For each nuisance parameter, fitResult[n] returns a tuple: (mle, lower, upper)
        Npoints = len(parameters)
        g = ROOT.TGraphAsymmErrors(Npoints)
        g_prefit = ROOT.TGraphAsymmErrors(Npoints)
        labels = []
        for i, n in enumerate(parameters):
            mle = fitResult[n]
            #unc = np.sqrt(cov[i,i])
            if n=="mu":
                err_down, err_up = 0., 0.
            else:
                err_down, err_up = nuisance_values[n]['down']-fitResult[n], nuisance_values[n]['up']-fitResult[n]
            g.SetPoint(i, mle, Npoints - i)
            g.SetPointError(i, abs(err_down), abs(err_up), 0.0, 0.0)
            g_prefit.SetPoint(i, 0.0, Npoints - i)
            g_prefit.SetPointError(i, 1.0, 1.0, 0.2, 0.2)
            _, ypos = calcPosition(0, Npoints - i)
            labels.append(addText(0.01, ypos, n))
        return g, g_prefit, labels

    # Helper to create impact graphs.
    def createGraph_impacts(impacts_numerical, parameters):
        Npoints = len(parameters)
        g_plus = ROOT.TGraphAsymmErrors(Npoints)
        g_minus = ROOT.TGraphAsymmErrors(Npoints)
        for i, n in enumerate(parameters):
            g_plus.SetPoint(i, 0, Npoints - i)
            g_minus.SetPoint(i, 0, Npoints - i)
            if n == 'mu':
                g_plus.SetPointError(i, 0, 0, 0.2, 0.2)
                g_minus.SetPointError(i, 0, 0, 0.2, 0.2)
                continue

            if impacts_numerical[n]["up"] < 0:
                g_plus.SetPointError(i, abs(impacts_numerical[n]["up"]), 0, 0.2, 0.2)
            else:
                g_plus.SetPointError(i, 0, abs(impacts_numerical[n]["up"]), 0.2, 0.2)
            if impacts_numerical[n]["down"] < 0:
                g_minus.SetPointError(i, abs(impacts_numerical[n]["down"]), 0, 0.2, 0.2)
            else:
                g_minus.SetPointError(i, 0, abs(impacts_numerical[n]["down"]), 0.2, 0.2)
        return g_plus, g_minus

    # Helper to get reference lines.
    def getLines():
        lines = []
        lines.append(ROOT.TLine(0, ymin, 0, ymax-1.))
        lines.append(ROOT.TLine(1., ymin, 1., ymax-1.))
        lines.append(ROOT.TLine(-1., ymin, -1., ymax-1.))
        for l in lines:
            l.SetLineStyle(2)
            l.SetLineWidth(2)
        return lines

    # Determine x-axis range for the impacts plot based on maximum absolute impact.
    maxImpact = np.max(np.abs(np.array(list(map( lambda f:list(f.values()), list(impacts_numerical.values()))))))
    xmin2 = -1.1 * maxImpact
    xmax2 =  1.1 * maxImpact

    # Create the main canvas.
    c_imp = ROOT.TCanvas("postFitUncerts", "", 600, 600)
    # Create left pad (for post-fit uncertainties).
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.0, xboundary, 1.0)
    pad1.SetTopMargin(TopMargin)
    pad1.SetLeftMargin(LeftMargin)
    pad1.SetRightMargin(separation/2)
    pad1.SetBottomMargin(BottomMargin)
    pad1.Draw()

    # Create right pad (for impacts).
    pad2 = ROOT.TPad("pad2", "pad2", xboundary, 0.0, 1.0, 1.0)
    pad2.SetTopMargin(TopMargin)
    pad2.SetLeftMargin(separation/2)
    pad2.SetRightMargin(RightMargin)
    pad2.SetBottomMargin(BottomMargin)
    pad2.Draw()

    # --- Left Pad: Post-Fit Uncertainties ---
    pad1.cd()
    dummy_left = getDummy("#nu", xmin, xmax, ymin, ymax)
    dummy_left.Draw("AP")
    g_unc, g_prefit, labels = createGraph(fitResult, nuisance_values, parameters)
    g_prefit.SetFillColor(17)
    g_prefit.Draw("E2 SAME")
    g_unc.SetMarkerStyle(20)
    g_unc.Draw("P SAME")
    for lab in labels:
        lab.Draw()
    for line in getLines():
        line.Draw("SAME")
        stuff.append(line)
    leg = ROOT.TLegend(LeftMargin+0.1, 1-TopMargin-0.1, 1-0.1-separation, 1-TopMargin-0.01)
    leg.SetNColumns(2)
    leg.AddEntry(g_unc, "PostFit", "pl")
    leg.AddEntry(g_prefit, "PreFit", "f")
    leg.Draw()
    ROOT.gPad.RedrawAxis()

    # --- Right Pad: Impacts on #mu ---
    pad2.cd()
    dummy_right = getDummy("#Delta #mu", xmin2, xmax2, ymin, ymax, factor=2)
    dummy_right.Draw("AP")
    g_plus, g_minus = createGraph_impacts(impacts_numerical, parameters)
    g_plus.SetMarkerSize(0)
    g_minus.SetMarkerSize(0)
    g_plus.SetFillColor(ROOT.kRed)
    g_minus.SetFillColor(ROOT.kBlue)
    g_plus.Draw("E2 SAME")
    g_minus.Draw("E2 SAME")
    zero_line = ROOT.TLine(0, ymin, 0, ymax-1.)
    zero_line.SetLineStyle(2)
    zero_line.SetLineWidth(2)
    zero_line.Draw("SAME")
    mu_text = addText(0.2, 0.9, "#mu = %.2f #pm %.2f" % (fitResult["mu"], np.sqrt(cov[0,0])), 43, 16, ROOT.kBlack)
    mu_text.Draw()

    # Update and save the canvas.
    c_imp.Update()
    c_imp.Print(os.path.join(plot_directory, "impacts.pdf"))
    c_imp.Print(os.path.join(plot_directory, "impacts.png"))

    # --------------------
    # Correlation Plot
    # --------------------
    # Create a TH2F for the correlation matrix.
    h_cor = ROOT.TH2F("h_cor", "Correlation Matrix;Parameter;Parameter", nparameters, 0, nparameters, nparameters, 0, nparameters)
    for i in range(nparameters):
        for j in range(nparameters):
            h_cor.SetBinContent(i+1, j+1, correlations[i, j])
    # Set bin labels.
    for i in range(nparameters):
        h_cor.GetXaxis().SetBinLabel(i+1, parameters[i])
        h_cor.GetYaxis().SetBinLabel(i+1, parameters[i])
    h_cor.GetZaxis().SetRangeUser(-1, 1)
    c_cor = ROOT.TCanvas("c_cor", "Correlations", 600, 600)
    h_cor.Draw("COLZ")
    c_cor.Update()
    c_cor.Print(os.path.join(plot_directory, "correlations.pdf"))
    c_cor.Print(os.path.join(plot_directory, "correlations.png"))

    common.syncer.sync()
