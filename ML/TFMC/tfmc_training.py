#!/usr/bin/env python

import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import importlib

from TFMC import TFMC

import common.user as user
import common.syncer

import common.data_structure as data_structure

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument("--training",      action="store",      default="v2",                     help="Training version")
argParser.add_argument("--config",        action="store",      default="tfmc",                   help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

# import the data
import common.datasets as datasets

# Where to store the training
model_directory = os.path.join( common.user.model_directory, "TFMC", args.selection, args.config, args.training)
os.makedirs(model_directory, exist_ok=True)

# Initialize model
tfmc = TFMC(len(data_structure.feature_names), len(config.class_labels))

if not args.overwrite:
    try:
        print ("Trying to load TFMC from %s"%model_directory)
        tfmc.load(model_directory)
    except (IOError, EOFError, ValueError):
        pass

# Initialize for training
tfmc.load_training_data( datasets, args.selection)

# Training Loop
for epoch in range(config.n_epochs):
    print(f"Epoch {epoch + 1}/{config.n_epochs}")
    model.train_one_epoch(data_loader, config.class_labels, max_batch=max_batch)
    model.save(save_path, epoch)  # Save model after each epoch

    # Accumulate histograms
    true_histograms, pred_histograms, bin_edges = model.accumulate_histograms(
        data_loader, config.class_labels, max_batch=max_batch
    )

    # Plot convergence
    model.plot_convergence(
        true_histograms,
        pred_histograms,
        bin_edges,
        epoch,
        output_path,
        config.class_labels,
        data_structure.feature_names,  # Pass feature names
    )

    # Evaluate on the same data for simplicity (use a validation set in practice)
    model.evaluate(data_loader, class_labels, max_batch=max_batch)

# Load the saved model for further use
model.load(save_path)
common.syncer.sync()
