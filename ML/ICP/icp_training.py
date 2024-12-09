#!/usr/bin/env python

import numpy as np
import os,sys,time
import importlib
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user
from ML.ICP.ICP import InclusiveCrosssectionParametrization

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true',                                   help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument("--config",        action="store",      default="icp_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',         action='store_true',                                   help="Only one batch, for debugging")
args = argParser.parse_args()

# import the config
config = importlib.import_module("%s.%s"%( args.configDir, args.config))

# import the data
import common.datasets as datasets

icp_name = f"ICP_{args.selection}_{args.config}"

model_directory = os.path.join( common.user.model_directory, "ICP" )
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, ('small_' if args.small else '')+icp_name)+'.pkl'

icp = None
if not args.overwrite:
    try:
        print ("Trying to load %s from %s"%(icp_name, filename))
        icp = InclusiveCrosssectionParametrization.load(filename)
    except (IOError, EOFError, ValueError):
        pass 

if icp is None or args.overwrite:
    print ("Training.")
    time1 = time.time()
    icp = InclusiveCrosssectionParametrization( config = config )

    icp.load_training_data(datasets, args.selection) 
    icp.train             (datasets, args.selection, small=args.small)

    icp.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Training time: %.2f seconds" % boosting_time)

print (f"Trained ICP with config {args.config} in selection {args.selection}")
prefix = "ICP: "+'\033[1m'+args.selection+'\033[0m'
print (prefix.ljust(50)+icp.__str__())
