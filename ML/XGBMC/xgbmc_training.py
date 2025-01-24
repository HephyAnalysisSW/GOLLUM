import xgboost as xgb
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure
import common.user as user

from XGBMC import XGBMC

import importlib
import common.helpers as helpers
import common.datasets as datasets

# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite training?")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--n_split", action="store", default=10, type=int, help="How many batches?")
argParser.add_argument("--training", action="store", default="v1", help="Training version")
argParser.add_argument("--config", action="store", default="xgb_v1", help="Which config?")
argParser.add_argument("--every", action="store", default=5, type=int, help="Update plot at every 'every' iteration.")
argParser.add_argument("--configDir", action="store", default="configs", help="Where is the config?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
args = argParser.parse_args()

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
model_directory = os.path.join(user.model_directory, "XGBMC", args.selection, args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)
config.model_dir = model_directory

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "XGBMC", args.selection, args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

max_batch = 1 if args.small else -1

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load XGBMC from {model_directory}")
        xgbmc, last_epoch = XGBMC.load(model_directory)
        if xgbmc is None:
            print("No checkpoint found. Starting from scratch.")
            config = importlib.import_module(f"{args.configDir}.{args.config}")
            xgbmc = XGBMC(config)
            last_epoch = 0
        else:
            print(f"Resuming training at epoch {last_epoch + 1}")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        xgbmc = XGBMC(config)
        last_epoch = 0
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    xgbmc = XGBMC(config)
    last_epoch = 0

# Set the starting epoch for resuming training
xgbmc.start_epoch = last_epoch

# Initialize for training
xgbmc.load_training_data(datasets, args.selection, n_split=(args.n_split if not args.small else 100))

# Train the model
xgbmc.train(max_batch = max_batch, every=args.every, plot_directory=plot_directory)
