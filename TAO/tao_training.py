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

import common.data_structure as data_structure

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
#argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--every",         action="store",      default=1, type=int,              help="Update plot at every 'every' iteration.")
argParser.add_argument("--training",      action="store",      default="v1",                     help="Training version")
argParser.add_argument("--tree_config",        action="store",      default="configs/tree_tao_v1.yaml", help="Which tree config?")
argParser.add_argument("--train_config",    action="store",      default="configs/training_tao_v1.yaml", help="Which training config?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
argParser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
args = argParser.parse_args()

from common.logger import get_logger
logger  = get_logger(args.logLevel, logFile = None)

# Make the tree
with open(args.tree_config, 'r') as f:
    tree_config = yaml.safe_load(f)
    # global rng
    tree_config['rng'] = np.random.default_rng(tree_config.get('rng', None))

# Load training data
import TrainingData
training_data = TrainingData.load_training_data( selection = args.selection, 
    use_ic = True, 
    use_scaler = True, 
    n_split = 10000 if not args.small else 100 )

max_batch = 1 if args.small else -1

# Where to store the training
model_directory = os.path.join(user.model_directory, "TAO", args.selection, 
    os.path.splitext(os.path.basename(args.tree_config))[0], 
    os.path.splitext(os.path.basename(args.train_config))[0]+("_small" if args.small else ""))

os.makedirs(model_directory, exist_ok=True)

if args.overwrite:
    start_epoch = 0
    logger.info("→ Overwrite specified, starting from scratch.")
    forest = Forest([Tree(config=tree_config) for _ in range(tree_config.get("ntrees", 1))])
else:
    try:
        forest = Forest.load(model_directory)
        start_epoch = int(sorted([
            int(f.split('_')[-1].split('.')[0])
            for f in os.listdir(model_directory)
            if f.startswith('meta_epoch_') and f.endswith('.yaml')
        ])[-1]) + 1
        logger.info(f"→ Resuming from epoch {start_epoch}.")
    except FileNotFoundError:
        logger.info("→ No saved model found, starting from scratch.")
        forest = Forest([Tree(config=tree_config) for _ in range(tree_config.get("ntrees", 1))])
        start_epoch = 0

# where to store the plots
plot_directory = os.path.join(user.plot_directory, "TAO", args.selection, 
    os.path.splitext(os.path.basename(args.tree_config))[0], 
    os.path.splitext(os.path.basename(args.train_config))[0]+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

# Load training config
import yaml
with open(args.train_config, 'r') as f:
    train_config = yaml.safe_load(f)

##Test: one train step for first tree
#
#tree = forest.trees[0]
#tree.set_standardization(X_mean = training_data["X_mean"], X_std = training_data["X_std"]) 
#
#loader = training_data['loader']
#for i_batch, batch in enumerate(loader):
#    data, weights, raw_labels = loader.split(batch)
#
#    # truth
#    y = (raw_labels==0)
#
#    # reweight to equal class probability
#    bkg_norm = training_data['weight_sums'][1]+training_data['weight_sums'][2]+training_data['weight_sums'][3]
#    sig_norm = training_data['weight_sums'][0]
#    weights[raw_labels>0]*=(sig_norm/bkg_norm)
#
#    tree.train_step( data, y, weights, train_config=train_config)
#
#    break

# Training loop
for epoch in range(start_epoch, train_config['n_epochs']):
    logger.info(f"=== Epoch {epoch} ===")

    for i_batch, batch in enumerate(training_data['loader']):
        data, weights, raw_labels = training_data['loader'].split(batch)

        # truth
        y = (raw_labels == 0)

        # reweight to equal class probability
        bkg_norm = training_data['weight_sums'][1] + training_data['weight_sums'][2] + training_data['weight_sums'][3]
        sig_norm = training_data['weight_sums'][0]
        weights[raw_labels > 0] *= (sig_norm / bkg_norm)

        forest.train_step(data, y, weights, train_config=train_config)
        break

    forest.save(model_directory, epoch=epoch)
