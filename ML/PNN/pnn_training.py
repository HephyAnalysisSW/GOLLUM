#!/usr/bin/env python

import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import importlib
import tensorflow as tf
from ML.PNN.PNN import PNN

import common.user as user
import common.syncer
import common.helpers as helpers

import common.data_structure as data_structure

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--training",      action="store",      default="v1",                     help="Training version")
argParser.add_argument("--config",        action="store",      default="pnn_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the data
import common.datasets as datasets

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

## Do we use IC?
#if config.scale_with_ic:
#    from ML.IC.IC import InclusiveCrosssection
#    ic = InclusiveCrosssection.load(os.path.join(user.model_directory, "IC", "IC_"+args.selection+'.pkl'))
#    config.weight_sums = ic.weight_sums
#    print("We use this IC:")
#    print(ic)

# Do we use a Scaler?
if config.use_scaler:
    from ML.Scaler.Scaler import Scaler
    scaler = Scaler.load(os.path.join(user.model_directory, "Scaler", "Scaler_"+args.selection+'.pkl'))
    config.feature_means     = scaler.feature_means
    config.feature_variances = scaler.feature_variances

    print("We use this scaler:")
    print(scaler)

# Where to store the training
model_directory = os.path.join(user.model_directory, "PNN", args.selection, args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "PNN", args.selection, args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load PNN from {model_directory}")
        pnn = PNN.load(model_directory)
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        pnn    = PNN(config)
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    pnn    = PNN(config)

# Initialize for training
pnn.load_training_data(datasets, args.selection, n_split=(args.n_split if not args.small else 100))

max_batch = 1 if args.small else -1

# Determine the starting epoch
starting_epoch = 0
if not args.overwrite:
    latest_checkpoint = tf.train.latest_checkpoint(model_directory)
    if latest_checkpoint:
        try:
            starting_epoch = int(os.path.basename(latest_checkpoint))
        except ValueError:
            pass

# Training Loop
for epoch in range(starting_epoch, config.n_epochs):
    print(f"Epoch {epoch}/{config.n_epochs}")

    ## for debugging
    #for batch in pnn.data_loader:
    #    import numpy as np
    #    data, weights, raw_labels = pnn.data_loader.split(batch)
    #    data = (data - pnn.feature_means) / np.sqrt(pnn.feature_variances)
    #    break
    #assert False, ""

    true_histograms, pred_histograms = pnn.train_one_epoch(max_batch=max_batch, accumulate_histograms=(epoch%5==0))
    pnn.save(model_directory, epoch)  # Save model and config after each epoch

    if true_histograms is not None and pred_histograms is not None:
        # Plot convergence
        pnn.plot_convergence_root(
            true_histograms,
            pred_histograms,
            epoch,
            plot_directory,
            data_structure.feature_names,
        )
        common.syncer.makeRemoteGif(plot_directory, pattern="epoch_*.png", name="epoch" )
        common.syncer.makeRemoteGif(plot_directory, pattern="norm_epoch_*.png", name="norm_epoch" )

    if epoch%5==0 or not args.small:
        common.syncer.sync()

common.syncer.sync()

assert False, ""

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

