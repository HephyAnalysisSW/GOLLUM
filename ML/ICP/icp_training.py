#!/usr/bin/env python

import numpy as np
import os,sys,time
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user
from ICP import ICP 

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument("--config",        action="store",      default="icp_quad_jes",           help="Which config?")
argParser.add_argument("--configDir",     action="store",      default="configs",                help="Where is the config?")
argParser.add_argument('--small',        action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the config
exec('import %s.%s as config'%( args.configDir, args.config))

# import the data
import common.datasets as datasets

print("Find training data for selection "+'\033[1m'+f"{args.selection} and config "+'\033[0m'+f"{args.config}")
training_data = {}
for base_point in config.base_points:
    values = config.get_alpha(base_point)
    data_loader = datasets.get_data_loader( selection=args.selection, values=values, selection_function=None, n_split=10)
    print ("ICP training data: Base point nu = %r, alpha = %r, file = %s"%( base_point, values, data_loader.file_path)) 
    training_data[base_point] = data_loader

icp_name = f"ICP_{args.selection}_{args.config}"

model_directory = os.path.join( common.user.model_directory, "ICP" )
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, icp_name)+'.pkl'

icp = None
if not args.overwrite:
    try:
        print ("Trying to load %s from %s"%(icp_name, filename))
        icp = ICP.load(filename)
    except (IOError, EOFError, ValueError):
        pass 

if icp is None or args.overwrite:
    print ("Training.")
    time1 = time.time()
    icp = ICP(
            nominal_base_point = config.nominal_base_point,
            base_points        = config.base_points,
            parameters         = config.parameters,
            combinations       = config.combinations,
                )

    icp.train(training_data, small=args.small)

    icp.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

