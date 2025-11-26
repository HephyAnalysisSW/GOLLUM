#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import random
import cProfile
import time
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from math import log, exp, sin, cos, sqrt, pi
import copy
import pickle
import itertools

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.gROOT.SetBatch(True)          
ROOT.TH1.AddDirectory(False)       

from   tools import helpers
import tools.syncer as syncer

# BIT
from BIT.MultiBoostedInformationTree import MultiBoostedInformationTree

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="PDFBIT_v1",                 help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default=None, type=str,  help="prefix")
argParser.add_argument("--model",              action="store",      default="TT2l_PDF", type=str,  help="model?")
argParser.add_argument("--nTraining",          action="store",      default=-1,        type=int,  help="number of training events")
argParser.add_argument("--plot_iterations",    action="store",      default=None,          nargs="*", type=int, help="Certain iterations to plot? If first iteration is -1, plot only list provided.")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--auto_clip',          action='store',      default=None, type=float, help="Remove quantiles of the training variable?")
argParser.add_argument('--no_orth',            action='store_true', help="Don't rotate basis.")

argParser.add_argument('--top_kinematics',     action='store_true')
argParser.add_argument('--lepton_kinematics',  action='store_true')
argParser.add_argument('--asymmetry',          action='store_true')
argParser.add_argument('--spin_correlation',   action='store_true')

argParser.add_argument("--red",         action="store",      default=-1,        type=int,  help="Reduction facto")

args, extra = argParser.parse_known_args(sys.argv[1:])

def parse_value( s ):
    try:
        r = int( s )
    except ValueError:
        try:
            r = float(s)
        except ValueError:
            r = s
    return r

extra_args = {}
key        = None
for arg in extra:
    if arg.startswith('--'):
        # previous no value? -> Interpret as flag
        #if key is not None and extra_args[key] is None:
        #    extra_args[key]=True
        key = arg.lstrip('-')
        extra_args[key] = True # without values, interpret as flag
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.items():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

exec("import data_models.%s as model"%args.model)
from data_models.plot_options import plot_options

model.multi_bit_cfg.update( extra_args )
data_model = model.DataModel(
        top_kinematics      =   args.top_kinematics, 
        lepton_kinematics   =   args.lepton_kinematics, 
        asymmetry           =   args.asymmetry, 
        spin_correlation    =   args.spin_correlation
    )

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.model )
os.makedirs( plot_directory, exist_ok=True)

training_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "training_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(training_data_filename):
    training_features, training_weights, training_observers = data_model.getEvents(args.nTraining, return_observers=True)
    print ("Created data set of size %i" % len(training_features) )
    if not os.path.exists(os.path.dirname(training_data_filename)):
        os.makedirs(os.path.dirname(training_data_filename))
    with open( training_data_filename, 'wb' ) as _file:
        pickle.dump( [training_features, training_weights, training_observers], _file )
        print ("Written training data to", training_data_filename)
else:
    with open( training_data_filename, 'rb') as _file:
        training_features, training_weights, training_observers= pickle.load( _file )
        print ("Loaded training data from ", training_data_filename, "with size", len(training_features))

if args.auto_clip is not None:
    len_before = len(training_features)
    training_features, training_weights = helpers.clip_quantile(training_features, args.auto_clip, training_weights )
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( args.auto_clip, len(training_features)/len_before ) )

# reduce training data 
if args.red>0:
    oldlen_ = training_features.shape[0] 
    len_ = int(training_features.shape[0]/args.red)
    training_features = training_features[:len_]
    training_weights = {key:val[:len_] for key, val in training_weights.items()}
    print("Reducing training from %i to %i"%( oldlen_, len_) )

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

postfix = ""
if args.red>0:
    postfix += "_red%00i"%args.red

base_points = []
for comb in list(itertools.combinations_with_replacement(model.pdf.variables,1))+list(itertools.combinations_with_replacement(model.pdf.variables,2)):
    base_points.append( {c:comb.count(c) for c in model.pdf.variables} )
if args.prefix == None:
    bit_name = "PDFBIT_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, "_".join(model.pdf.variables), args.nTraining, model.multi_bit_cfg["n_trees"])
else:
    bit_name = "PDFBIT_%s_%s_%s_coeffs_%s_nTraining_%i_nTrees_%i"%(args.model+postfix, data_model.name, args.prefix, "_".join(model.pdf.variables), args.nTraining, model.multi_bit_cfg["n_trees"])

filename = os.path.join(user.model_directory, bit_name)+'.pkl'
try:
    print ("Loading %s for %s"%(bit_name, filename))
    bit = MultiBoostedInformationTree.load(filename)
except (IOError, EOFError, ValueError):
    bit = None

if bit is None or args.overwrite in ["all", "training"]:
    raise NotImplementedError("I don't want to boost here, but if you uncomment the following you will.")

test_data_filename = os.path.join(user.data_directory, args.model, data_model.name, "test_%i"%args.nTraining)+'.pkl'
if args.overwrite in ["all", "data"] or not os.path.exists(test_data_filename):
    test_features, test_weights, test_observers = data_model.getEvents(args.nTraining, return_observers=True)
    print ("Created data set of size %i" % len(test_features) )
    if not os.path.exists(os.path.dirname(test_data_filename)):
        os.makedirs(os.path.dirname(test_data_filename), exist_ok=True)
    with open( test_data_filename, 'wb' ) as _file:
        pickle.dump( [test_features, test_weights, test_observers], _file )
        print ("Written test data to", test_data_filename)
else:
    with open( test_data_filename, 'rb') as _file:
        test_features, test_weights, test_observers = pickle.load( _file )
        print ("Loaded test data from ", test_data_filename, "with size", len(test_features))

if args.auto_clip is not None:
    len_before = len(test_features)

    selected = helpers.clip_quantile(test_features, args.auto_clip, return_selection = True)
    test_features = test_features[selected]
    test_weights = {k:test_weights[k][selected] for k in test_weights.keys()}
    if test_observers.size:
        test_observers = test_observers[selected] 
    print ("Auto clip efficiency (test) %4.3f is %4.3f"%( args.auto_clip, len(test_features)/len_before ) )

# delete coefficients we don't need
if model.pdf.variables is not None:
    for key in list(test_weights.keys()):
        if not all( [k in model.pdf.variables for k in key]):
            del test_weights[key]

# ==== Fisher-space basis, rate split, checks, adaptive steps, plotting ====

# --- Collect linear derivative weights w^(a) and nominal w^(0) ---
var_names = [d[0] for d in bit.derivatives if len(d) == 1]  # e.g. ('c0',), ('c1',), ...
n_params  = len(var_names)

w0  = test_weights[()]                                   # w^(0)
eps = 1e-30
w0s = np.maximum(w0, eps)

def get_lin(nm):
    # nm is a 1-tuple like ('c0',), ('c1',) ...
    return test_weights.get(nm, np.zeros_like(w0))

# Ratios r_a = w^(a)/w^(0). Shape-only objects.
r_lin = np.stack([ get_lin(tuple([nm])) / w0s for nm in var_names ], axis=1)   # (N, n_params)

# Cross-section derivatives s_a = sum_i w^(a)_i
s = (w0[:, None] * r_lin).sum(axis=0)                                  # (n_params,)

# Fisher per-L: M_ab = sum_i w0_i r_a r_b  (== sum_i w^(a) w^(b) / w0)
M = r_lin.T @ (w0[:, None] * r_lin)                                    # (n_params, n_params)

# --- Rate direction: v_rate = M^{-1}s / sqrt(s^T M^{-1} s) ---
Minv = np.linalg.pinv(M, rcond=1e-12)
num  = Minv @ s
den2 = float(s @ (Minv @ s))
if den2 <= 0:
    raise RuntimeError("Non-positive s^T M^{-1} s; check inputs.")
v_rate = num / np.sqrt(den2)                                           # (n_params,)

# Check: unit norm in M, and alignment with s
norm_rate = float(v_rate @ (M @ v_rate))
align     = float(s @ v_rate)                                          # expected sqrt(s^T M^{-1} s)

print("\n=== Fisher-basis (per-L) summary ===")
print(f"||v_rate||_M^2 = {norm_rate:.6f} (should be ~1)")
print(f"s · v_rate     = {align:.6f} (should be sqrt(s^T M^-1 s) = {np.sqrt(den2):.6f})")

# --- Projector onto M-orthogonal complement of v_rate ---
# For any vector u, u_perp = u - (v_rate^T M u) * v_rate
def M_inner(u, v):     return float(u @ (M @ v))
def M_proj_perp(u):    return u - M_inner(v_rate, u) * v_rate

# Build an M-orthonormal basis in the orthogonal subspace (Gram–Schmidt in the Fisher metric)
V = [v_rate.copy()]  # first column = rate direction

# Start from canonical basis; orthonormalize wrt M & v_rate
for k in range(n_params):
    e = np.zeros(n_params); e[k] = 1.0
    u = e.copy()
    # remove components along already-built columns (Fisher metric)
    for v in V:
        u -= M_inner(v, u) * v
    nu2 = M_inner(u, u)
    if nu2 > 1e-12:
        V.append(u / np.sqrt(nu2))

# If numerical rank < n_params, V may be shorter; keep as many as we got
V = np.stack(V, axis=1)                            # shape (n_params, m_cols)
m_cols = V.shape[1]
print(f"#basis vectors built (incl. rate) = {m_cols}/{n_params}")

# Orthogonality check: V^T M V ≈ I
G = V.T @ (M @ V)
off = G - np.eye(m_cols)
print("max|offdiag(V^T M V)| =", np.max(np.abs(off - np.diag(np.diag(off)))))

# --- Diagonalize M within the orthogonal subspace for nicer “shape-k” ordering ---
# Extract the block excluding the first (rate) vector
if m_cols > 1:
    V_perp = V[:, 1:]                                 # (n_params, m_cols-1)
    M_perp = V_perp.T @ (M @ V_perp)                  # Fisher in that subspace
    lam, U = np.linalg.eigh(M_perp)                   # sorted ascending
    # Map back: shape basis vectors = V_perp @ U
    V_shapes = V_perp @ U                             # (n_params, m_cols-1)
    # re-assemble full basis with rate first
    V_new = np.column_stack([v_rate, V_shapes])       # (n_params, m_cols)
    # check orthonormality again
    G_new = V_new.T @ (M @ V_new)
    print("max|offdiag(V_new^T M V_new)| =", np.max(np.abs(G_new - np.eye(m_cols))))
    # Print eigenvalues (shape directions) per-L
    print("shape eigenvalues (per-L) =", ", ".join(f"{x:.6e}" for x in lam))
else:
    V_new = V
    lam   = np.array([])

# Label basis vectors
labels = ["rate"] + [f"shape-{i+1}" for i in range(m_cols-1)]

# --- Print the eigensystem cleanly: split normalization vs shapes ---
print("\n=== Eigensystem summary (per-L) ===")
print(f"Rate direction first.\nV_new (columns) in parameter order {var_names}:")
np.set_printoptions(precision=4, suppress=True)
for j, lab in enumerate(labels):
    print(f"{lab:>8s}: {V_new[:, j]}")

if lam.size:
    print("Shape eigenvalues (per-L):", lam)

# --- Plots: step +alpha along each basis vector; show yield & shapes, with legend mapping ---

# Colors for rate and shape-k (distinct palette, will cycle if needed)
rate_color = ROOT.kMagenta + 1
_shape_palette = [
    ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2, ROOT.kOrange + 1,
    ROOT.kCyan + 2, ROOT.kViolet + 1, ROOT.kTeal + 1, ROOT.kPink + 1,
    ROOT.kAzure + 2, ROOT.kSpring + 5, ROOT.kYellow + 2, ROOT.kGray + 1
]
shape_colors = [_shape_palette[i % len(_shape_palette)] for i in range(max(1, m_cols - 1))]

# Prebuild a legend-only canvas
leg_canvas = ROOT.TCanvas("legend_only", "legend_only", 800, 260)
legend_only = ROOT.TLegend(0.05, 0.20, 0.95, 0.85)
legend_only.SetNColumns(min(3, 1 + (m_cols - 1)))
legend_only.SetFillStyle(0); legend_only.SetShadowColor(ROOT.kWhite); legend_only.SetBorderSize(0)

_dummy_h = []
h = ROOT.TH1F("h_rate_legend_dummy", "", 1, 0, 1); h.SetLineColor(rate_color);  h.SetLineWidth(3); h.SetMarkerStyle(0)
_dummy_h.append(h); legend_only.AddEntry(h, "rate", "l")
for i in range(m_cols - 1):
    hh = ROOT.TH1F(f"h_shape{i+1}_legend_dummy", "", 1, 0, 1)
    hh.SetLineColor(shape_colors[i]); hh.SetLineWidth(2); hh.SetMarkerStyle(0)
    _dummy_h.append(hh); legend_only.AddEntry(hh, f"shape-{i+1}", "l")

legend_only.Draw()
plot_directory_ = os.path.join(plot_directory, "basis_legend")
os.makedirs(plot_directory_, exist_ok=True)
helpers.copyIndexPHP(plot_directory_)
leg_canvas.Print(os.path.join(plot_directory_, "legend.pdf"))
leg_canvas.Print(os.path.join(plot_directory_, "legend.png"))
leg_canvas.Close()

# helper: 1D hist for a feature with given weights
def make_hist_feature(weights, feature_idx, feature_name):
    bins = plot_options[feature_name]['binning']
    arr  = test_features[:, feature_idx]
    return helpers.make_TH1F(
        np.histogram(arr, np.linspace(bins[1], bins[2], bins[0] + 1), weights=weights)
    )

# helper: safe ratio H_var / H_nom (bin by bin)
def make_ratio_hist(H_var, H_nom, name_suffix="__ratio"):
    R = H_var.Clone(f"{H_var.GetName()}{name_suffix}")
    R.SetDirectory(0)
    for b in range(1, H_nom.GetNbinsX() + 1):
        n = H_var.GetBinContent(b)
        d = H_nom.GetBinContent(b)
        if d > 0:
            R.SetBinContent(b, n / d)
            R.SetBinError(b, 0.0)
        else:
            R.SetBinContent(b, 1.0 if n == 0 else 0.0)
            R.SetBinError(b, 0.0)
    return R

plot_postfix = "_no_orth" if args.no_orth else ""
if args.no_orth:
    V_new = V_new[0,0]*np.eye(V_new.shape[0])
    labels = ["c%i"%i for i in range(m_cols)]

steps = [500 for _ in range(m_cols)]
stuff=[]
# Now draw, per feature, the nominal and the stepped variants (including quadratic term)
for feature_idx, feature_name in enumerate(data_model.feature_names):

    # Nominal (SM) yield and shape
    h_nom = make_hist_feature(w0, feature_idx, feature_name)
    h_nom.SetLineColor(ROOT.kGray + 2); h_nom.SetMarkerStyle(0); h_nom.SetLineWidth(2)

    # Helpers to access learned derivatives as event-wise ratios
    def r_lin(a):
        return test_weights[tuple([var_names[a]])] / w0

    def r_quad(a, b):
        key = tuple(sorted([var_names[a], var_names[b]]))
        return test_weights[key] / w0

    n_params = len(var_names)

    # Build the stepped weights for each direction (use V_new columns; step sizes in `steps`)
    h_dirs = []
    for j in range(m_cols):
        alpha = steps[j]

        if alpha == 0.0:
            w_step = w0.copy()
        else:
            # Linear:  Σ_a v_a r_a(x)
            lin_proj = np.zeros_like(w0, dtype=float)
            for a in range(n_params):
                lin_proj += V_new[a, j] * r_lin(a)

            # Quadratic: (1/2) Σ_{a,b} v_a v_b r_ab(x)
            quad_proj = np.zeros_like(w0, dtype=float)
            for a in range(n_params):
                va = V_new[a, j]
                quad_proj += 0.5 * (va * va) * r_quad(a, a)
                for b in range(a + 1, n_params):
                    vb = V_new[b, j]
                    quad_proj += (va * vb) * r_quad(a, b)

            poly = 1.0 + alpha * lin_proj + (alpha ** 2) * quad_proj
            w_step = w0 * poly

        h_j = make_hist_feature(w_step, feature_idx, feature_name)
        if j == 0:
            h_j.SetLineColor(rate_color);  h_j.SetLineWidth(3)
        else:
            h_j.SetLineColor(shape_colors[j - 1]); h_j.SetLineWidth(2)
        h_j.SetMarkerStyle(0)
        h_dirs.append(h_j)

    # 1) Yield overlays (linear/log) + per-plot legend (3 columns, compact top band)
    for logY in (False, True):
        c = ROOT.TCanvas(f"c_yield_{feature_name}_{int(logY)}", "", 800, 650)
        leg = ROOT.TLegend(0.20, 0.78, 0.90, 0.9)  # top band
        leg.SetNColumns(3); leg.SetFillStyle(0); leg.SetShadowColor(ROOT.kWhite); leg.SetBorderSize(0)

        h_nom.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
        h_nom.GetYaxis().SetTitle("events")
        h_nom.Draw("hist")
        for j, hj in enumerate(h_dirs):
            hj.Draw("histsame")
            leg.AddEntry(hj, labels[j], "l")
        leg.AddEntry(h_nom, "nominal", "l")

        stuff.append(leg); leg.Draw()
        ROOT.gPad.SetLogy(logY); c.Update()
        outdir = os.path.join(plot_directory, "basis_yield"+plot_postfix, "log" if logY else "lin")
        os.makedirs(outdir, exist_ok=True); helpers.copyIndexPHP(outdir)
        c.Print(os.path.join(outdir, f"{feature_name}.png"))
        c.Print(os.path.join(outdir, f"{feature_name}.pdf"))
        c.Close()

    # 2) Ratio-to-nominal overlays + per-plot legend (3 columns, compact top band)
    def clone_and_divide(num, den):
        h = num.Clone(f"{num.GetName()}_ratio")
        h.Divide(den)
        return h

    # for logY in (False, True):
    for logY in (False, ):
        c = ROOT.TCanvas(f"c_ratio_{feature_name}_{int(logY)}", "", 800, 650)
        leg = ROOT.TLegend(0.20, 0.78, 0.90, 0.9)
        leg.SetNColumns(3); leg.SetFillStyle(0); leg.SetShadowColor(ROOT.kWhite); leg.SetBorderSize(0)

        # unity and dirs
        h_unity = clone_and_divide(h_nom, h_nom)
        h_unity.SetLineColor(ROOT.kGray + 1); h_unity.SetLineStyle(2); h_unity.SetLineWidth(2)
        h_unity.GetXaxis().SetTitle(plot_options[feature_name]['tex'])

        # --- draw an explicit frame to control Y range ---
        xmin = h_unity.GetXaxis().GetXmin()
        xmax = h_unity.GetXaxis().GetXmax()
        frame = c.DrawFrame(xmin, 0.4, xmax, 1.6)  # <-- Y-range fixed here
        frame.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
        frame.GetYaxis().SetTitle("variation / nominal")

        # now draw histograms on top
        h_unity.Draw("hist same")
        leg.AddEntry(h_unity, "nominal", "l")

        for j, hj in enumerate(h_dirs):
            h_ratio = clone_and_divide(hj, h_nom)
            h_ratio.SetLineColor(rate_color if j == 0 else shape_colors[j - 1])
            h_ratio.SetLineWidth(3 if j == 0 else 2)
            h_ratio.SetMarkerStyle(0)
            h_ratio.Draw("hist same")
            stuff.append(h_ratio)
            leg.AddEntry(h_ratio, labels_d[j], "l")

        stuff.append(leg); leg.Draw()
        ROOT.gPad.SetLogy(logY)
        c.Modified(); c.Update()

        outdir = os.path.join(plot_directory, "basis_ratio"+plot_postfix, "log" if logY else "lin")
        os.makedirs(outdir, exist_ok=True); helpers.copyIndexPHP(outdir)
        c.Print(os.path.join(outdir, f"{feature_name}.png"))
        c.Print(os.path.join(outdir, f"{feature_name}.pdf"))
        c.Close()


print("\nDone: legends added per-plot, discriminative colors applied, ratio-to-nominal plots written.")
