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
argParser.add_argument("--process",       action="store",      default=None,                     help="Which process?")
argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--every",         action="store",      default=5, type=int,              help="Make plot every this number of epochs.")
argParser.add_argument("--rebin",         action="store",      default=1, type=int,              help="Factor to rebin the plots.")
argParser.add_argument("--training",      action="store",      default="v1",                     help="Training version")
argParser.add_argument("--config",        action="store",      default="pnn_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the data
import common.datasets_hephy as datasets

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

subdirs= [arg for arg in [args.process, args.selection] if arg is not None]

# Do we use ICP?
if config.icp is not None:
    from ML.ICP.ICP import InclusiveCrosssectionParametrization
    icp_name = "ICP_"+"_".join(subdirs)+"_"+config.icp+".pkl"
    icp = InclusiveCrosssectionParametrization.load(os.path.join(user.model_directory, "ICP", icp_name))
    config.icp_predictor = icp.get_predictor()
    print("We use this ICP:",icp_name)
    print(icp)

# Do we use a Scaler?
if config.use_scaler:
    from ML.Scaler.Scaler import Scaler
    scaler_name = "Scaler_"+"_".join(subdirs)+'.pkl'
    scaler = Scaler.load(os.path.join(user.model_directory, "Scaler", scaler_name))
    config.feature_means     = scaler.feature_means
    config.feature_variances = scaler.feature_variances

    print("We use this scaler:", scaler_name)
    print(scaler)

# Where to store the training
model_directory = os.path.join(user.model_directory, "PNN", *subdirs,  args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "PNN", *subdirs,  args.config, args.training+("_small" if args.small else ""))
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
pnn.load_training_data(datasets=datasets, process=args.process, selection=args.selection, n_split=(args.n_split if not args.small else 100))

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

    # Manually evaluate and update the learning rate
    if hasattr(pnn, 'lr_schedule'):  # Ensure the schedule exists
        new_lr = pnn.lr_schedule(epoch)
        pnn.optimizer.learning_rate.assign(new_lr)  # Update the optimizer's learning rate
  
    # Print the current learning rate
    current_lr = tf.keras.backend.get_value(pnn.optimizer.learning_rate)  # Direct access
    print(f"Epoch {epoch}/{config.n_epochs} - Learning rate: {current_lr:.6f}")

    ## for debugging
    #for batch in pnn.data_loader:
    #    import numpy as np
    #    data, weights, raw_labels = pnn.data_loader.split(batch)
    #    data = (data - pnn.feature_means) / np.sqrt(pnn.feature_variances)
    #    break
    #assert False, ""

    true_histograms, pred_histograms = pnn.train_one_epoch(max_batch=max_batch, accumulate_histograms=(epoch%args.every==0), rebin=args.rebin)
    pnn.save(model_directory, epoch)  # Save model and config after each epoch

    if true_histograms is not None and pred_histograms is not None:
        # Plot convergence
        pnn.plot_convergence_root(
            true_histograms,
            pred_histograms,
            epoch,
            plot_directory,
            data_structure.feature_names,
            rebin=args.rebin,
        )
        common.syncer.makeRemoteGif(plot_directory, pattern="epoch_*.png", name="epoch" )
        common.syncer.makeRemoteGif(plot_directory, pattern="norm_epoch_*.png", name="norm_epoch" )

    if epoch%args.every==0 or not args.small:
        common.syncer.sync()

common.syncer.sync()
