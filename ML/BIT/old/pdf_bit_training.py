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
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.gROOT.SetBatch(True)          
ROOT.TH1.AddDirectory(False)       

# BIT
from MultiBoostedInformationTree import MultiBoostedInformationTree

# User
import common.user as user
import common.syncer
import common.helpers as helpers

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--training",           action="store",      default="v1",                     help="Training version")
argParser.add_argument("--model",              action="store",      default="TT2l_PDF", type=str,  help="model?")
argParser.add_argument('--overwrite',          action='store',      default=None, choices = [None, "training", "data", "all"],  help="Overwrite output?")
argParser.add_argument('--training_plots',     action='store_true', help="Make training plots?")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")

argParser.add_argument("--red",         action="store",      default=-1,        type=int,  help="Reduction facto")
argParser.add_argument('--nJobs',       action='store',         nargs='?',  type=int, default=0,                                    help="Bootstrapping total number" )
argParser.add_argument('--job',         action='store',                     type=int, default=0,                                    help="Bootstrepping iteration" )

args = argParser.parse_args()

exec("import data_models.%s as model"%args.model)
from data_models.plot_options import plot_options

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

# Resample for bootstrapping
if args.nJobs>0:
    from sklearn.utils import resample
    rs_mask = resample(range(training_features.shape[0]))
    training_features = training_features[rs_mask]
    training_weights = {key:val[rs_mask] for key, val in training_weights.items()}
    print("Bootstrapping training data for job %i/%i"%( args.job, args.nJobs) )

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

###############
## Plot Model #
###############

stuff = []
if args.feature_plots and hasattr( model, "pdf_plot_points"):
    h    = {}
    h_obs= {}
    for i_pdf, pdf_plot_point in enumerate(model.pdf_plot_points):
        pdf = pdf_plot_point['pdf']

        if i_pdf == 0:
            pdf_sm     = pdf

        name = ''
        name= '_'.join( [ (coeff+'_%3.2f'%pdf[coeff]).replace('.','p').replace('-','m') for coeff in model.pdf.variables if coeff in model.pdf.variables ])
        tex_name = pdf_plot_point['tex'] 

        if i_pdf==0: name='SM'

        h[name]     = {}
        h_obs[name] = {}

        pdf['name'] = name
        
        for i_feature, feature in enumerate(data_model.feature_names):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *plot_options[feature]['binning'] )
        for i_observer, observer in enumerate(model.observers):
            h_obs[name][observer]    = ROOT.TH1F(name+'_'+observer+'_nom_obs',name+'_'+observer+'_obs', *plot_options[observer]['binning'] )

        # make reweights for x-check
        reweight     = copy.deepcopy(training_weights[()])
        # linear term
        for param1 in model.pdf.variables:
            reweight += (pdf[param1]-pdf_sm[param1])*training_weights[(param1,)] 
        reweight_lin  = copy.deepcopy( reweight )
        # quadratic term
        for param1 in model.pdf.variables:
            if pdf[param1]-pdf_sm[param1] ==0: continue
            for param2 in model.pdf.variables:
                if pdf[param2]-pdf_sm[param2] ==0: continue
                reweight += (.5 if param1!=param2 else 1)*(pdf[param1]-pdf_sm[param1])*(pdf[param2]-pdf_sm[param2])*training_weights[tuple(sorted((param1,param2)))]

        for i_feature, feature in enumerate(data_model.feature_names):
            binning = plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            #h_lin[name][feature] = helpers.make_TH1F( np.histogram(training_features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight_lin) )

            h[name][feature].SetLineWidth(2)
            h[name][feature].SetLineColor( pdf_plot_point['color'] )
            h[name][feature].SetMarkerStyle(0)
            h[name][feature].SetMarkerColor(pdf_plot_point['color'])
            h[name][feature].legendText = tex_name

        for i_observer, observer in enumerate(model.observers):
            binning = plot_options[observer]['binning']

            h_obs[name][observer] = helpers.make_TH1F( np.histogram(training_observers[:,i_observer], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            #h_lin[name][observer] = helpers.make_TH1F( np.histogram(training_observers[:,i_observer], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight_lin) )

            h_obs[name][observer].SetLineWidth(2)
            h_obs[name][observer].SetLineColor( pdf_plot_point['color'] )
            h_obs[name][observer].SetMarkerStyle(0)
            h_obs[name][observer].SetMarkerColor(pdf_plot_point['color'])
            h_obs[name][observer].legendText = tex_name

    for _h, feature_names, ratio_y, in [  [h_obs, model.observers, (0.8, 1.3)], [h, data_model.feature_names, (0.94, 1.1)] ]:
   
        ratio_y_low, ratio_y_high = ratio_y 
        for i_feature, feature in enumerate(feature_names):

            norm = _h[model.pdf_plot_points[0]['pdf']['name']][feature].Integral()
            if norm>0:
                for pdf_plot_point in model.pdf_plot_points:
                    _h[pdf_plot_point['pdf']['name']][feature].Scale(1./norm) 

            histos = [_h[pdf_plot_point['pdf']['name']][feature] for pdf_plot_point in model.pdf_plot_points]
            max_   = max( map( lambda h__:h__.GetMaximum(), histos ))

            for logY in [True, False]:

                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.68,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%plot_options[feature]['tex'])
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "feature_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+'.png' ))
                c1.Print( os.path.join( plot_directory_, feature+'.pdf' ))

            # Norm all shapes to 1
            for i_histo, histo in enumerate(histos):
                norm = histo.Integral()
                if norm>0:
                    histo.Scale(1./histo.Integral())

            # Divide all shapes by the SM
            ref = histos[0].Clone()
            for i_histo, histo in enumerate(histos):
                histo.Divide(ref)

            # Now plot shape differences
            for logY in [True, False]:
                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.78,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                c1.SetLogy(logY)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("shape wrt. SM")
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.01 if logY else ratio_y_low), (10 if logY else ratio_y_high))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "shape_plots", "nTraining_%i"%args.nTraining, "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

postfix = ""
if args.nJobs>0:
    postfix += "_resample%05i"%args.job
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
    time1 = time.time()
    bit = MultiBoostedInformationTree(
            training_features     = training_features,
            training_weights      = training_weights,
            base_points           = base_points,
            feature_names         = data_model.feature_names,
            **model.multi_bit_cfg
                )
    bit.boost()
    bit.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

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

#if args.bias is not None:
#    bias_weights = np.array(list(map( function, test_features[:, data_model.feature_names.index(args.bias[0])] )))
#    bias_weights /= np.mean(bias_weights)
#    test_weights = {k:v*bias_weights for k,v in test_weights.items()} 

# delete coefficients we don't need
if model.pdf.variables is not None:
    for key in list(test_weights.keys()):
        if not all( [k in model.pdf.variables for k in key]):
            del test_weights[key]

if args.training_plots:
    import gc

    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.06)

    # colors per derivative
    color = {}
    i_lin, i_diag, i_mixed = 0, 0, 0
    for der in bit.derivatives:
        if len(der) == 1:
            color[der] = ROOT.kAzure + i_lin;  i_lin   += 1
        elif len(der) == 2 and len(set(der)) == 1:
            color[der] = ROOT.kRed   + i_diag; i_diag  += 1
        else:
            color[der] = ROOT.kGreen + i_mixed; i_mixed += 1

    # Which iterations to plot
    plot_iterations = list(range(1, 10)) + list(range(10, bit.n_trees + 1, 10))
    #if isinstance(args.plot_iterations, list):
    #    plot_iterations = (plot_iterations + args.plot_iterations[1:]) if args.plot_iterations[0] < 0 else args.plot_iterations
    #    plot_iterations.sort()

    def _safe_hist(arr, bins, weights=None):
        h = helpers.make_TH1F(np.histogram(arr, bins=bins, weights=weights))
        h.SetDirectory(0)          # no TFile/directories
        # Let ROOT own it while it is drawn on a pad to avoid double delete:
        ROOT.SetOwnership(h, False)
        return h

    def _safe_div(numer, denom):
        denom2d = denom.reshape(-1, 1)
        out = np.zeros_like(numer, dtype=float)
        np.divide(numer, denom2d, out=out, where=(denom2d != 0))
        return out

    for max_n_tree in plot_iterations:
        if max_n_tree == 0:
            max_n_tree = 1

        # Keep Python refs for the duration of this iteration to avoid premature GC
        keep_alive = []

        test_predictions = bit.vectorized_predict(test_features, max_n_tree=max_n_tree)
        w0 = test_weights[()]

        # --- global truth/pred distributions per derivative ---
        th1d_pred, th1d_truth = {}, {}
        for i_der, der in enumerate(bit.derivatives):
            truth_ratio = (test_weights.get(der, test_weights.get(tuple(reversed(der))))) / w0
            qlo, qhi = np.quantile(truth_ratio, q=(0.01, 0.99))
            bins = np.linspace(min(0, qlo), qhi, 21) if len(der) == 2 else np.linspace(qlo, qhi, 21)

            h_truth = _safe_hist(truth_ratio, bins, weights=w0)
            h_pred  = _safe_hist(test_predictions[:, i_der], bins, weights=w0)

            for h in (h_truth, h_pred):
                h.SetLineColor(color[der]); h.SetMarkerColor(color[der]); h.SetMarkerStyle(0); h.SetLineWidth(2)
            h_truth.SetLineStyle(ROOT.kDashed)

            th1d_truth[der] = h_truth
            th1d_pred[der]  = h_pred
            keep_alive += [h_truth, h_pred]

        # --- feature-binned ratios ---
        for observables, features, postfix in [(data_model.feature_names, test_features, "")]:
            lin_binning = {
                f: np.linspace(plot_options[f]['binning'][1], plot_options[f]['binning'][2], plot_options[f]['binning'][0] + 1)
                for f in observables
            }

            yield_by_feature = {}
            ratio_pred_by_feature = {}
            ratio_truth_by_feature = {}

            for feature in observables:
                bins_lin = lin_binning[feature]
                binned = np.digitize(features[:, data_model.feature_names.index(feature)], bins_lin)
                mask = (binned.reshape(-1, 1) == np.arange(1, len(bins_lin))).T  # (B, N)

                h_w0 = np.array([w0[m].sum() for m in mask])  # (B,)
                h_der_pred = np.array([(w0.reshape(-1, 1) * test_predictions)[m].sum(axis=0) for m in mask])  # (B,M)
                h_der_truth = np.array([
                    (np.vstack([test_weights.get(der, test_weights.get(tuple(reversed(der)))) for der in bit.derivatives]).T)[m].sum(axis=0)
                    for m in mask
                ])  # (B, M)

                ratio_pred  = _safe_div(h_der_pred,  h_w0)
                ratio_truth = _safe_div(h_der_truth, h_w0)

                h_yield = _safe_hist(h_w0, bins_lin)
                h_yield.SetLineColor(ROOT.kGray + 2); h_yield.SetMarkerColor(ROOT.kGray + 2); h_yield.SetMarkerStyle(0)
                h_yield.GetXaxis().SetTitle(plot_options[feature]['tex']); h_yield.SetTitle("")
                yield_by_feature[feature] = h_yield
                keep_alive.append(h_yield)

                th_pred  = {der: _safe_hist(ratio_pred[:, i_der],  bins_lin) for i_der, der in enumerate(bit.derivatives)}
                th_truth = {der: _safe_hist(ratio_truth[:, i_der], bins_lin) for i_der, der in enumerate(bit.derivatives)}

                for der in bit.derivatives:
                    th_truth[der].SetLineColor(color[der]); th_truth[der].SetMarkerColor(color[der]); th_truth[der].SetMarkerStyle(0)
                    th_truth[der].SetLineWidth(2);          th_truth[der].SetLineStyle(ROOT.kDashed)
                    th_truth[der].GetXaxis().SetTitle(plot_options[feature]['tex'])
                    th_pred[der].SetLineColor(color[der]);  th_pred[der].SetMarkerColor(color[der]);  th_pred[der].SetMarkerStyle(0)
                    th_pred[der].SetLineWidth(2);           th_pred[der].GetXaxis().SetTitle(plot_options[feature]['tex'])

                ratio_pred_by_feature[feature]  = th_pred
                ratio_truth_by_feature[feature] = th_truth
                keep_alive += list(th_pred.values()) + list(th_truth.values())

            n_pads = len(observables) + 1
            n_col = int(sqrt(n_pads))
            n_rows = n_pads // n_col + (1 if n_pads % n_col else 0)

            for logY in (False, True):
                # Unique name for canvas to avoid ROOT reusing it internally
                c1 = ROOT.TCanvas(f"c1_{max_n_tree}_{int(logY)}", "multipads", 500 * n_col, 500 * n_rows)
                # Let ROOT own canvas/pads/primitives (avoid double delete from Python):
                ROOT.SetOwnership(c1, False)
                c1.Divide(n_col, n_rows)

                l = ROOT.TLegend(0.2, 0.1, 0.9, 0.85)
                ROOT.SetOwnership(l, False)
                l.SetNColumns(2); l.SetFillStyle(0); l.SetShadowColor(ROOT.kWhite); l.SetBorderSize(0)

                for i_feature, feature in enumerate(observables):
                    c1.cd(i_feature + 1); ROOT.gStyle.SetOptStat(0)

                    th1d_yield = yield_by_feature[feature]
                    th1d_ratio_pred  = ratio_pred_by_feature[feature]
                    th1d_ratio_truth = ratio_truth_by_feature[feature]

                    th1d_yield.Draw("hist")

                    for i_der, der in enumerate(bit.derivatives):
                        if i_feature == 0:
                            tex_name = ",".join(der)
                            l.AddEntry(th1d_ratio_truth[der], f"R({tex_name})")
                            l.AddEntry(th1d_ratio_pred[der],  f"#hat{{R}}({tex_name})")

                    if i_feature == 0:
                        l.AddEntry(th1d_yield, "yield (SM)")

                    max_ = max(h.GetMaximum() for h in th1d_ratio_truth.values())
                    max_ = (10 ** 1.5) * max_ if logY else 1.5 * max_
                    min_ = min(h.GetMinimum() for h in th1d_ratio_truth.values())
                    min_ = 0.1 if logY else (1.5 * min_ if min_ < 0 else 0.75 * min_)
                    if min_ < -0.1: min_ = -0.1

                    y_min, y_max = th1d_yield.GetMinimum(), th1d_yield.GetMaximum()
                    if y_max > 0:
                        for b in range(1, th1d_yield.GetNbinsX() + 1):
                            v = th1d_yield.GetBinContent(b)
                            th1d_yield.SetBinContent(b, (v - y_min) / max(1e-12, y_max) * (max_ - min_) * 0.95 + min_)

                    th1d_yield.Draw("hist")
                    ROOT.gPad.SetLogy(logY)
                    th1d_yield.GetYaxis().SetRangeUser(min_, max_)
                    th1d_yield.Draw("hist")

                    for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                        h.Draw("hsame")

                c1.cd(len(observables) + 1)
                l.Draw()

                # Add TLatex annotation (don’t keep ownership)
                t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.06)
                ROOT.SetOwnership(t, False)
                t.DrawLatex(0.29, 0.9, f'N_{{B}} = {max_n_tree:5d}')

                # Write file
                plot_directory_ = os.path.join(plot_directory, "training_plots", bit_name, "log" if logY else "lin")
                os.makedirs(plot_directory_, exist_ok=True)
                helpers.copyIndexPHP(plot_directory_)
                c1.Print(os.path.join(plot_directory_, f"epoch{postfix}_{max_n_tree:05d}.png"))

                # Close & detach the canvas from ROOT lists; do NOT call Delete()
                c1.Close()
                try:
                    ROOT.gROOT.GetListOfCanvases().Remove(c1)
                except Exception:
                    pass
                # Drop our last Python ref (ROOT owns all drawn primitives)
                del c1, l, t

        # Drop all Python refs we kept alive during this iteration
        keep_alive.clear()
        gc.collect()

