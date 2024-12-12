#!/usr/bin/env python
import copy
import numpy as np
import os,sys
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import time
import functools
import operator

import common.user
import common.syncer
from   common.helpers import copyIndexPHP
import models.analytic_2D as model 

import tensorflow as tf

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--directory",     action="store",      default="pnn", help="Subdirectory for output")
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

cfg = model.pnn_cfg
cfg.update( extra_args )

training_data = model.getEvents(args.nTraining)
total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])

# Base point matrix
base_points = np.array( model.base_points )
VkA  = np.zeros( [len(base_points), len(model.combinations) ], dtype='float32')
for i_base_point, base_point in enumerate(base_points):
    for i_comb1, comb1 in enumerate(model.combinations):
        VkA[i_base_point][i_comb1] += functools.reduce(operator.mul, [base_point[model.parameters.index(c)] for c in list(comb1)], 1)
VkA = tf.convert_to_tensor( VkA )

# Dissect inputs into nominal sample and variied
nominal_base_point_index = np.where(np.all(base_points==model.nominal_base_point,axis=1))[0]
assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( model.nominal_base_point, base_points)
nominal_base_point_index = nominal_base_point_index[0]
nominal_base_point_key   = tuple(model.nominal_base_point)

n_features = len(training_data[nominal_base_point_key]['features'][0])

# Complement training data
if 'weights' not in training_data[nominal_base_point_key]:
    training_data[nominal_base_point_key]['weights'] = np.ones(training_data[nominal_base_point_key]['features'].shape[0])

for k, v in training_data.items():
    if "features" not in v and "weights" not in v:
        raise RuntimeError( "Key %r has neither features nor weights" %k  )
    if k == nominal_base_point_key:
        if 'features' not in v:
            raise RuntimeError( "Nominal base point does not have features!" )
    else:
        if not 'features' in v:
            # we must have weights
            #v['features'] = training_data[nominal_base_point_key]['features']
            if len( training_data[nominal_base_point_key]['features'])!=len(v['weights']):
                raise runtimeerror("key %r has inconsistent length in weights"%v)
        if (not 'weights' in training_data[nominal_base_point_key].keys()) and 'weights' in v:
            raise RuntimeError( "Found no weights for nominal base point, but for a variation. This is not allowed" )
        if ('weights' in training_data[nominal_base_point_key].keys()) and 'weights' not in v:
            raise RuntimeError( "Found weights for nominal base point, but not for a variation. This is not allowed" )

    if 'weights' in v and 'features' in v:
        if len(v['weights'])!=len(v['features']):
            raise RuntimeError("Key %r has unequal length of weights and features: %i != %i" % (k, len(v['weights']), len(v['features'])) )

    if 'weights' in v:
        v['weights'] = tf.convert_to_tensor(v['weights'], dtype=tf.float32)
    if 'features' in v:
        v['features'] = tf.convert_to_tensor(v['features'], dtype=tf.float32)

model_directory = os.path.join( common.user.model_directory, args.directory )
os.makedirs(model_directory, exist_ok=True)

from PNN_tf import PNN  # Importing TensorFlow-based PNN

pnn = PNN(
        in_features  = n_features,
        out_features = len(model.combinations),
        VkA          = VkA,
        **cfg,
            )
pnn_name = "PNN_tf_analytic_2D_nTraining_%i_nLayers_%s"%( args.nTraining, "_".join(map( str, pnn.layer_size())))

optimizer = tf.keras.optimizers.Adam(learning_rate=model.pnn_cfg['learning_rate'])


for epoch in range(model.pnn_cfg['n_epochs']):
    total_loss = 0.0

    with tf.GradientTape() as tape:

        # Compute nominal values
        DeltaA_nominal = pnn(training_data[nominal_base_point_key]['features'])
        weights_nominal = training_data[nominal_base_point_key]['weights'] if 'weights' in training_data[nominal_base_point_key] else tf.ones(len(DeltaA_nominal))

        for i_base_point, base_point in enumerate(base_points):
            if i_base_point == nominal_base_point_index:
                continue

            # Retrieve features and weights for current ν
            if 'features' in training_data[tuple(base_point)]:
                DeltaA = pnn(training_data[tuple(base_point)]['features'])
            else:
                DeltaA = DeltaA_nominal

            if 'weights' in training_data[tuple(base_point)]:
                weights = training_data[tuple(base_point)]['weights']
            else:
                weights = weights_nominal

            # Compute weighted losses
            loss_0 = tf.reduce_sum(
                weights_nominal * tf.math.softplus(tf.linalg.matvec(DeltaA_nominal, VkA[i_base_point]))
            )
            loss_nu = tf.reduce_sum(
                weights * tf.math.softplus(-tf.linalg.matvec(DeltaA, VkA[i_base_point]))
            )

            # Total loss for the current ν
            loss = loss_0 + loss_nu
            loss -= (tf.reduce_sum(weights_nominal) + tf.reduce_sum(weights)) * np.log(2.0)
            total_loss += loss

       # Backpropagation
        gradients = tape.gradient(total_loss, pnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pnn.trainable_variables))

    print(f"Epoch {epoch + 1}/{model.pnn_cfg['n_epochs']}, Loss: {total_loss.numpy():.4f}")


# Final Predictions
predicted_reweights = np.exp(
    tf.matmul(pnn(training_data[nominal_base_point_key]['features']), tf.transpose(VkA)).numpy()
)

plot_directory = os.path.join( common.user.plot_directory, args.directory )
os.makedirs(plot_directory, exist_ok=True)
copyIndexPHP( plot_directory )

import matplotlib.pyplot as plt
import numpy as np

hist_configs = []
colors = ['black', 'blue', 'green', 'red', 'orange', 'magenta', 'cyan']
for i_point, point in enumerate(base_points):
    # truth
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'][:,0].numpy(), 
                          'name':'%s'%str(point) +" (truth.)", 
                          'weights':training_data[tuple(point)]['weights'].numpy(), 
                          'color':colors[i_point], 'linestyle':'--'} )
    # prediction
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'][:,0].numpy(), 
                          'name':'%s'%str(point) +" (pred.)", 
                          'weights':predicted_reweights[:,i_point]*training_data[model.nominal_base_point]['weights'].numpy(), 
                          'color':colors[i_point], 'linestyle':'-'} )

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

