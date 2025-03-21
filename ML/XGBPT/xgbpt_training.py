#!/usr/bin/env python

import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import importlib
import tensorflow as tf
from ML.XGBPT.XGBPT import XGBPT

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
argParser.add_argument("--config",        action="store",      default="xgbpt_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the data
import common.datasets as datasets

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

# Where to store the training
model_directory = os.path.join(user.model_directory, "XGBPT", *subdirs,  args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)
config.model_dir = model_directory

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "XGBPT", *subdirs,  args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load XGBPT from {model_directory}")
        xgbpt = XGBPT.load(model_directory)
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        xgbpt  = XGBPT(config)
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    xgbpt  = XGBPT(config)

max_batch = 1 if args.small else -1

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load XGBPT from {model_directory}")
        xgbpt, last_epoch = XGBPT.load(model_directory, return_epoch=True)
        if xgbpt is None:
            print("No checkpoint found. Starting from scratch.")
            config = importlib.import_module(f"{args.configDir}.{args.config}")
            xgbpt = XGBPT(config)
            last_epoch = 0
        else:
            print(f"Resuming training at epoch {last_epoch + 1}")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        xgbpt = XGBPT(config)
        last_epoch = 0
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    xgbpt = XGBPT(config)
    last_epoch = 0

# Set the starting epoch for resuming training
xgbpt.start_epoch = last_epoch

# Initialize for training
xgbpt.load_training_data(datasets, args.selection, n_split=(args.n_split if not args.small else 100))

# Train the model
xgbpt.train(max_batch = max_batch, every=args.every, plot_directory=plot_directory)
