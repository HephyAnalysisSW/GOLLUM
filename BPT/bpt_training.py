#!/usr/bin/env python

import numpy as np
import analytic_2D as model 
import os,sys
import os, sys
sys.path.insert(0, '..')
import time

import common.user
import common.syncer
from common.helpers import copyIndexPHP

from BoostedParametricTree import BoostedParametricTree

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--directory",     action="store",      default="v1", help="Subdirectory for output")
argParser.add_argument("--nTraining",     action="store",      default=100000, type=int,  help="Number of training events")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")

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


cfg = model.bpt_cfg
cfg.update( extra_args )


training_data = model.getEvents(args.nTraining)
total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])

bpt_name = "BPT_analytic_2D_nTraining_%i_nTrees_%i"%( args.nTraining, cfg["n_trees"])

model_directory = os.path.join( common.user.model_directory, args.directory )
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, bpt_name)+'.pkl'
try:
    print ("Trying to load %s from %s"%(bpt_name, filename))
    bpt = BoostedParametricTree.load(filename)
except (IOError, EOFError, ValueError):
    bpt = None

if bpt is None or args.overwrite:
    print ("Not found. Training.")
    time1 = time.time()
    bpt = BoostedParametricTree(
            training_data      = training_data,
            nominal_base_point = model.nominal_base_point,
            base_points        = model.base_points,
            parameters         = model.parameters,
            combinations       = model.combinations,
            feature_names      = model.feature_names,
            **cfg,
                )
    bpt.boost()

    bpt.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

predicted_reweights = np.exp( np.dot( bpt.vectorized_predict(training_data[model.nominal_base_point]['features'],  max_n_tree = None), bpt.VkA.transpose() ) )

plot_directory = os.path.join( common.user.plot_directory, args.directory )
os.makedirs(plot_directory, exist_ok=True)
copyIndexPHP( plot_directory )

import matplotlib.pyplot as plt
import numpy as np

hist_configs = []
colors = ['black', 'blue', 'green', 'red', 'orange', 'magenta', 'cyan']
for i_point, point in enumerate(model.base_points):
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'], 'name':'%s'%str(point) +" (truth.)", 'weights':training_data[point]['weights'], 'color':colors[i_point], 'linestyle':'--'} )
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'], 'name':'%s'%str(point) +" (pred.)", 'weights':predicted_reweights[:,i_point]*training_data[model.nominal_base_point]['weights'], 'color':colors[i_point], 'linestyle':'-'} )

def plot_weighted_histograms(hist_configs, bins=20, title='Overlayed Weighted Histograms'):
    """
    Plot and save overlayed 1D histograms with different weight vectors.

    Parameters:
    - hist_configs: list of dictionaries, each dictionary contains:
        - 'features': list or numpy array of features.
        - 'weights': weight vector (same length as features).
        - 'color': color for the histogram.
        - 'linestyle': line style for the histogram ('-' for continuous, '--' for dashed).
    - bins: number of bins for the histogram (default is 10).
    - title: title of the plot.
    """
    
    plt.figure(figsize=(10, 6))

    for config in hist_configs:
        features = config['features']
        weights = config['weights']
        color = config['color']
        linestyle = config['linestyle']
        name = config['name']
 
        plt.hist(features, bins=bins, weights=weights, color=color, linestyle=linestyle, histtype='step', linewidth=1.5, label=name)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Weighted Count')
    plt.legend([f'Histogram {i+1}' for i in range(len(hist_configs))])
    plt.grid(True)
    plt.legend(ncol=2)
 
    # Save the figure in PNG and PDF formats
    plt.savefig(os.path.join( plot_directory, 'weighted_histograms.png'))
    print ("Written ",os.path.join( plot_directory, 'weighted_histograms.png'))
    plt.savefig(os.path.join( plot_directory, 'weighted_histograms.pdf'))
    print ("Written ",os.path.join( plot_directory, 'weighted_histograms.pdf'))
    plt.show()

plot_weighted_histograms(hist_configs, bins=20)
common.syncer.sync()

