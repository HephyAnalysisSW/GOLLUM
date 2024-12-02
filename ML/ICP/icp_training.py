#!/usr/bin/env python

import numpy as np
import os,sys
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import models.analytic_2D as model 
import time

import common.user
import common.syncer
from common.helpers import copyIndexPHP

from ICP import ICP 

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--directory",     action="store",      default="v1", help="Subdirectory for output")
#argParser.add_argument("--nTraining",     action="store",      default=100000, type=int,  help="Number of training events")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--model",         action="store",      default="analytic_2D",                 help="Which model?")
argParser.add_argument("--modelDir",      action="store",      default="models",                 help="Which model directory?")

args, extra = argParser.parse_known_args(sys.argv[1:])

def parse_value( s ):
    try:
        r = int( s )
    except ValueError:
        try:
            r = float(s)
        except ValueError:
            r = s
    return r

extra_args = {}
key        = None
for arg in extra:
    if arg.startswith('--'):
        # previous no value? -> Interpret as flag
        #if key is not None and extra_args[key] is None:
        #    extra_args[key]=True
        key = arg.lstrip('-')
        extra_args[key] = True # without values, interpret as flag
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.items():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

cfg = model.pnn_cfg
cfg.update( extra_args )

# import the model
exec('import %s.%s as model'%( args.modelDir, args.model))

training_data = model.getEvents(args.nTraining)
total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])

icp_name = "ICP_%s_nTraining_%i"%( args.model, args.nTraining)

model_directory = os.path.join( common.user.model_directory, args.directory )
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, icp_name)+'.pkl'
try:
    print ("Trying to load %s from %s"%(icp_name, filename))
    icp = ICP.load(filename)
except (IOError, EOFError, ValueError):
    icp = None

if icp is None or args.overwrite:
    print ("Not found. Training.")
    time1 = time.time()
    icp = ICP(
            training_data      = training_data,
            nominal_base_point = model.nominal_base_point,
            base_points        = model.base_points,
            parameters         = model.parameters,
            combinations       = model.combinations,
                )

    icp.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    boosting_time = time2 - time1
    print ("Boosting time: %.2f seconds" % boosting_time)

