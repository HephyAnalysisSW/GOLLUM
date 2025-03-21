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
argParser.add_argument("--selection",     action="store",      default="highMT_VBFJet",          help="Which selection?")
argParser.add_argument("--process",       action="store",      default=None,                     help="Which process?")
argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--rebin",         action="store",      default=1, type=int,              help="Factor to rebin the plots.")
argParser.add_argument("--training",      action="store",      default="v1",                     help="Training version")
argParser.add_argument("--config",        action="store",      default="pnn_quad_tes_jes_met",   help="Which config?")
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

model_directory = os.path.join(user.model_directory, "PNN", *subdirs,  args.config, args.training+("_small" if args.small else ""))

# where to store the plots
plot_directory  = os.path.join(user.plot_directory,  "PNN_validation", *subdirs,  args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

pnn = PNN.load(model_directory)
print(f"Loaded the PNN from {model_directory}")

# Initialize for training
pnn.load_training_data(datasets=datasets, process=args.process, selection=args.selection, n_split=(args.n_split if not args.small else 100))

max_batch = 1 if args.small else -1

# for debugging
loader = pnn.training_data[pnn.nominal_base_point_key]
for batch in loader:
    import numpy as np
    data, weights, raw_labels = loader.split(batch)
    break

assert False, ""

#if true_histograms is not None and pred_histograms is not None:
#    # Plot convergence
#    pnn.plot_convergence_root(
#        true_histograms,
#        pred_histograms,
#        epoch,
#        plot_directory,
#        data_structure.feature_names,
#        rebin=args.rebin,
#    )
#    common.syncer.makeRemoteGif(plot_directory, pattern="epoch_*.png", name="epoch" )
#    common.syncer.makeRemoteGif(plot_directory, pattern="norm_epoch_*.png", name="norm_epoch" )
#
#if epoch%args.every==0 or not args.small:
#    common.syncer.sync()
#
#common.syncer.sync()
