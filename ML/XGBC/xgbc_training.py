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
import common.syncer as syncer

from XGBC import XGBC

import importlib
import common.helpers as helpers
import common.datasets as datasets

# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite training?")
argParser.add_argument("--training", action="store", default="v1", help="Training version")
argParser.add_argument("--input_directory", action="store", default="/scratch-cbe/users/robert.schoefbeck/Challenge/output/toyFits/v5_train/", help="Which input directory")
argParser.add_argument("--config", action="store", default="xgbc_v1", help="Which config?")
argParser.add_argument("--configDir", action="store", default="configs", help="Where is the config?")
args = argParser.parse_args()

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

# Where to store the training
model_directory = os.path.join(user.model_directory, "XGBC", args.config, args.training)
os.makedirs(model_directory, exist_ok=True)
config.model_dir = model_directory

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "XGBC", args.config, args.training)
helpers.copyIndexPHP(plot_directory)

# Initialize model
if not args.overwrite:
    try:
        print(f"Trying to load XGBC from {model_directory}")
        xgbc, last_epoch = XGBC.load(model_directory, return_epoch=True)
        if xgbc is None:
            print("No checkpoint found. Starting from scratch.")
            config = importlib.import_module(f"{args.configDir}.{args.config}")
            xgbc = XGBC(config)
            last_epoch = 0
        else:
            print(f"Resuming training at epoch {last_epoch + 1}")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        config = importlib.import_module(f"{args.configDir}.{args.config}")
        xgbc = XGBC(config)
        last_epoch = 0
else:
    config = importlib.import_module(f"{args.configDir}.{args.config}")
    xgbc = XGBC(config)
    last_epoch = 0

# Set the starting epoch for resuming training
xgbc.start_epoch = last_epoch

# Initialize for training
xgbc.load_training_data(args.input_directory)

# Train the model
xgbc.train()

xgbc.plot_results( plot_directory )

syncer.sync()
