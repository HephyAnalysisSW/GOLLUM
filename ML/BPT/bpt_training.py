#!/usr/bin/env python

import numpy as np
import os,sys,time
import importlib
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
import common.syncer
import common.helpers as helpers
from   common.helpers import copyIndexPHP

from BoostedParametricTree import BoostedParametricTree

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument("--config",        action="store",      default="bpt_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument("--n_split",       action="store",      default=10, type=int,             help="How many batches?")
argParser.add_argument("--training",      action="store",      default="v3",                     help="Training version")
argParser.add_argument('--small',        action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

# import the data
import common.datasets as datasets

# Where to store the training
model_directory = os.path.join( user.model_directory, "BPT", args.selection, args.config, args.training+("_small" if args.small else ""))
os.makedirs(model_directory, exist_ok=True)

# where to store the plots
plot_directory = os.path.join(user.plot_directory, "BPT", args.selection, args.config, args.training+("_small" if args.small else ""))
helpers.copyIndexPHP(plot_directory)

bpt_name = f"BPT_{args.selection}_{args.config}"
filename = os.path.join(model_directory, ('small_' if args.small else '')+bpt_name)+'.pkl'

bpt = None
if not args.overwrite:
    try:
        print ("Trying to load %s from %s"%(bpt_name, filename))
        bpt = BoostedParametricTree.load(filename)
    except (IOError, EOFError, ValueError):
        pass

max_batch = 1     if args.small else -1
n_split   = 10000 if args.small else args.n_split
if bpt is None or args.overwrite:
    print ("Training.")
    time1 = time.time()
    bpt = BoostedParametricTree( config = config )

    bpt.load_training_data(datasets, args.selection, n_split=n_split, max_batch=max_batch)
    bpt.train             ()

    bpt.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

