#!/usr/bin/env python

import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import importlib
import tensorflow as tf
from TFMC import TFMC

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
argParser.add_argument("--training",      action="store",      default="v6",                     help="Training version")
argParser.add_argument("--config",        action="store",      default="tfmc",                   help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the data
import common.datasets as datasets

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

# Do we use IC?
if config.use_ic:
    from ML.IC.IC import InclusiveCrosssection
    ic = InclusiveCrosssection.load(os.path.join(user.model_directory, "IC", "IC_"+args.selection+'.pkl'))
    config.weight_sums = ic.weight_sums
    print("We use this IC:")
    print(ic)
    
# Do we use a Scaler?
if config.use_scaler:
    from ML.Scaler.Scaler import Scaler 
    scaler = Scaler.load(os.path.join(user.model_directory, "Scaler", "Scaler_"+args.selection+'.pkl'))
    config.feature_means     = scaler.feature_means
    config.feature_variances = scaler.feature_variances

    print("We use this scaler:")
    print(scaler)

# Where to store the training
model_directory = os.path.join(user.model_directory, "TFMC", args.selection, args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "TFMC", args.selection, args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load TFMC from {model_directory}")
        tfmc = TFMC.load(model_directory)
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        tfmc = TFMC(config)
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    tfmc = TFMC(config)

# Initialize for training
tfmc.load_training_data(datasets, args.selection, n_split=(args.n_split if not args.small else 100))

max_batch = 1 if args.small else -1

# Determine the starting epoch
starting_epoch = 0
if not args.overwrite:
    latest_checkpoint = tf.train.latest_checkpoint(model_directory)
    if latest_checkpoint:
        try:
            starting_epoch = int(os.path.basename(latest_checkpoint)) + 1
        except ValueError:
            pass

# Training Loop
for epoch in range(starting_epoch, config.n_epochs):
    # Get the current learning rate
    current_lr = tf.keras.backend.get_value(tfmc.optimizer.learning_rate)  # Direct access
    print(f"Epoch {epoch}/{config.n_epochs} - Learning rate: {current_lr:.6f}")

    ## for debugging
    #for batch in tfmc.data_loader:
    #    import numpy as np
    #    data, weights, raw_labels = tfmc.data_loader.split(batch)
    #    data = (data - tfmc.feature_means) / np.sqrt(tfmc.feature_variances)
    #    break
    #assert False, ""

    #data, data_norm, weights, raw_labels, raw_labels_, remapped_labels, labels_one_hot, predictions, loss = tfmc.train_one_epoch(max_batch=max_batch, accumulate_histograms=(epoch%5==0))
    #assert False, ""

    true_histograms, pred_histograms = tfmc.train_one_epoch(max_batch=max_batch, accumulate_histograms=(epoch%5==0))
    tfmc.save(model_directory, epoch)  # Save model and config after each epoch

    if true_histograms is not None and pred_histograms is not None:
        # Plot convergence
        tfmc.plot_convergence_root(
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
