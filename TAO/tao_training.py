#!/usr/bin/env python
import sys, os
sys.path.insert(0, '..')

import yaml
import numpy as np

import importlib
from TAO import Tree
from Forest import Forest

import common.user as user
import common.syncer
import common.helpers as helpers

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
#argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--every",         action="store",      default=1, type=int,              help="Update plot at every 'every' iteration.")
argParser.add_argument('--data',          choices=['toy', 'challenge'], default='challenge', help="Which dataset to use")
argParser.add_argument("--forest_config",   action="store",       default="configs/tree_tao_v1.yaml", help="Which tree config?")
argParser.add_argument("--training",      action="store",       default="", help="A postfix?")
argParser.add_argument("--train_config",  action="store",       default="configs/training_tao_v1.yaml", help="Which training config?")
argParser.add_argument("--quantization",  nargs="*", type=int,  choices=[-1, 2, 3],  default=[], help="Quantization bit-widths per tree (each must be 2 or 3); empty list means no quantization.")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
argParser.add_argument('--logLevel',      action='store',       nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
args = argParser.parse_args()

from common.logger import get_logger
logger  = get_logger(args.logLevel, logFile = None)

# Make the tree
with open(args.forest_config, 'r') as f:
    forest_config = yaml.safe_load(f)
    #Use rng also in the data generation
    forest_config['rng'] = np.random.default_rng(forest_config.get('rng_seed', None))

# Check that the quantisation is consistent with the tree depth
for i_q,q in enumerate(args.quantization):
    if q<0:
        args.quantization[i_q] = None
if len(args.quantization)==1:
    # A single value threads over all dephts
    args.quantization = [args.quantization[0]]*(forest_config['max_depth']+1)
elif len(args.quantization)==forest_config['max_depth']+1 or len(args.quantization)==0:
    pass
else:
    raise RuntimeError( "Don't know what to do with quantization %r" % args.quantization )

forest_config['quantization'] = args.quantization

postfix = []
if len(args.training)>0:
    postfix.append( args.training )
if len(args.quantization)>0:
    postfix.append("quant_"+"_".join(["None" if i is None else str(i) for i in args.quantization]) )

postfix = "_"+"_".join(postfix) if len(postfix)>0 else ""
# Load training data
import importlib
training_module = importlib.import_module(f"data.{args.data}")
training_data = training_module.load_training_data(small=args.small, rng=forest_config['rng'])

# Where to store the training
model_directory = os.path.join(user.model_directory, "TAO", args.data, 
    os.path.splitext(os.path.basename(args.forest_config))[0]+postfix, 
    os.path.splitext(os.path.basename(args.train_config))[0]+("_small" if args.small else ""))

os.makedirs(model_directory, exist_ok=True)

if args.overwrite:
    start_epoch = 0
    logger.info("→ Overwrite specified, starting from scratch.")
    forest = Forest([Tree(config=forest_config) for _ in range(forest_config.get("ntrees", 1))], config=forest_config)
else:
    try:
        forest = Forest.load(model_directory)
        start_epoch = int(sorted([
            int(f.split('_')[-1].split('.')[0])
            for f in os.listdir(model_directory)
            if f.startswith('forest_epoch_') and f.endswith('.pkl')
        ])[-1]) + 1
        logger.info(f"→ Resuming from epoch {start_epoch}.")
    except FileNotFoundError:
        logger.info("→ No saved model found, starting from scratch.")
        forest = Forest([Tree(config=forest_config) for _ in range(forest_config.get("ntrees", 1))], config=forest_config)
        start_epoch = 0

# where to store the plots
plot_directory = os.path.join(user.plot_directory, "TAO", args.data, 
    os.path.splitext(os.path.basename(args.forest_config))[0]+postfix, 
    os.path.splitext(os.path.basename(args.train_config))[0]+("_small" if args.small else ""))
helpers.copyIndexPHP( os.path.join( plot_directory, "1D") )

# Load training config
import yaml
with open(args.train_config, 'r') as f:
    train_config = yaml.safe_load(f)

def plot1D(filename, X, y, y_pred, bins=50, weight=None, title="1D response", truth_2d=False, text = ""):
    import ROOT
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.07)
    tex.SetTextAlign(11)  # Align right

    if weight is None:
        weight = np.ones_like(y)

    d = X.shape[1]
    cols = min(3, d)
    rows = (d + cols - 1) // cols
    canvas = ROOT.TCanvas("c1d_combined", title, 300 * cols, 300 * rows)
    canvas.Divide(cols, rows)
    stuff = []
    for i_dim in range(d):
        x = X[:, i_dim]
        xmin, xmax = x.min(), x.max()

        h2 = ROOT.TH2F(f"h2_1d_{i_dim}", "", bins, xmin, xmax, bins, y.min(), y.max())
        hprof = ROOT.TProfile(f"hprof_1d_{i_dim}", "", bins, xmin, xmax)
        htruth = ROOT.TProfile(f"htruth_1d_{i_dim}", "", bins, xmin, xmax)
        stuff.append( h2)
        stuff.append( hprof)
        stuff.append( htruth)

        for xi, yi, y_pred_i, wi in zip(x, y, y_pred, weight):
            h2.Fill(xi, y_pred_i if not truth_2d else yi, wi)
            hprof.Fill(xi, y_pred_i, wi)
            htruth.Fill(xi, yi, wi)

        canvas.cd(i_dim + 1)
        h2.SetStats(False)
        h2.SetTitle(f";x_{i_dim};prediction")
        h2.Draw("COLZ")
        htruth.SetLineColor(ROOT.kBlack)
        htruth.SetLineWidth(2)
        htruth.Draw("SAME")
        hprof.SetLineColor(ROOT.kRed)
        hprof.SetMarkerColor(ROOT.kRed)
        hprof.SetLineWidth(2)
        hprof.Draw("SAME")
    if len(text)>0:
        lines = [(0.3, 0.95, text)]
        drawObjects = [tex.DrawLatex(*line) for line in lines]
        for o in drawObjects:
            o.Draw()
    canvas.Update()

    canvas.Print(filename)

# Training loop
forest.set_standardization(X_mean = training_data["X_mean"], X_std = training_data["X_std"]) 
for epoch in range(start_epoch, train_config['n_epochs']):
    logger.info(f"=== Epoch {epoch} ===")

    for i_batch, batch in enumerate(training_data['loader']):
        data, weights, raw_labels = training_data['loader'].split(batch)

        # truth
        y = raw_labels

        # reweight to equal class probability
        bkg_norm = training_data['weight_sums'][1] + training_data['weight_sums'][2] + training_data['weight_sums'][3]
        sig_norm = training_data['weight_sums'][0]
        weights[raw_labels > 0] *= (sig_norm / bkg_norm)
        forest.train_step(data, y, weights, train_config=train_config)

    y_pred = forest.predict(data)
    plot1D( os.path.join( plot_directory, "1D", f"epoch_{epoch:04d}.png" ), data, y, y_pred, weight = weights, text = f"Epoch = {epoch:04d}     ")

    if len(forest.trees)>1:
        #forest.print()
        forest.global_leaf_refit(data, y, weights, train_config=train_config)
        #print("After forest.global_leaf_refit")
        #forest.print()
        y_pred = forest.predict(data)
        plot1D( os.path.join( plot_directory, "1D", f"epoch_{epoch:04d}_gf.png" ), data, y, y_pred, weight = weights, text = f"Epoch = {epoch:04d} (GF)")

    common.syncer.makeRemoteGif(os.path.join( plot_directory, "1D" ), pattern="epoch_*.png", name="epoch" )
    common.syncer.sync()
    forest.save(model_directory, epoch=epoch)

#y_pred = forest.predict(data)
# 
#plot1D( os.path.join( plot_directory, "1D", f"epoch_gf.png" ), data, y, y_pred, weight = weights)
#    common.syncer.makeRemoteGif(os.path.join( plot_directory, "1D" ), pattern="epoch_*.png", name="epoch" )
#    #common.syncer.makeRemoteGif(os.path.join( plot_directory, "1D" ), pattern="epoch_truth_*.png", name="epoch" )
#    common.syncer.sync()
#    forest.save(model_directory, epoch=epoch)
#
#for i_batch, batch in enumerate(training_data['loader']):
#    data, weights, raw_labels = training_data['loader'].split(batch)
#    X = forest.standardize_input(data)
#    y = raw_labels
#    
#    # reweight to equal class probability
#    bkg_norm = training_data['weight_sums'][1] + training_data['weight_sums'][2] + training_data['weight_sums'][3]
#    sig_norm = training_data['weight_sums'][0]
#    weights[raw_labels > 0] *= (sig_norm / bkg_norm)
#
#    break
#
#print("Before forest.global_leaf_refit")
#forest.print()
#forest.global_leaf_refit(data, y, weights, train_config=train_config)
#print("After forest.global_leaf_refit")
#forest.print()
#
#y_pred = forest.predict(data)
#
# 
#plot1D( os.path.join( plot_directory, "1D", f"epoch_gf.png" ), data, y, y_pred, weight = weights)


#routing_masks = []
#leaf_specs    = []
#for t_idx, tree in enumerate(forest.trees):
#    routing, ordered_nodes = tree.route_all(X)
#    for n_idx, node in enumerate(ordered_nodes):
#        if node.is_leaf:
#            leaf_specs.append((t_idx, node))
#            routing_masks.append(routing[:, n_idx])

#K = len(leaf_specs)
#R = np.zeros((N, K + X.shape[1]))
#for j, mask in enumerate(routing_masks):


