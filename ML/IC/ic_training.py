#!/usr/bin/env python

import numpy as np
import os,sys,time
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user
from ML.IC.IC import InclusiveCrosssection

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")
argParser.add_argument("--selection",     action="store",      default="lowMT_VBFJet",           help="Which selection?")
argParser.add_argument('--small',         action='store_true',  help="Only one batch, for debugging")
args = argParser.parse_args()

# import the data
import common.datasets_hephy as datasets_hephy

print("IC training for selection "+'\033[1m'+f"{args.selection}"+'\033[0m')

ic_name = f"IC_{args.selection}"

model_directory = os.path.join( common.user.model_directory, "IC" )
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, ('small_' if args.small else '')+ic_name)+'.pkl'

ic = None
if not args.overwrite:
    try:
        print ("Trying to load %s from %s"%(ic_name, filename))
        ic = InclusiveCrosssection.load(filename)
    except (IOError, EOFError, ValueError):
        pass 

if ic is None or args.overwrite:
    print ("Training.")
    time1 = time.time()
    ic = InclusiveCrosssection()

    ic.load_training_data(datasets_hephy, args.selection)
    ic.train             (datasets_hephy, args.selection, small=args.small)

    ic.save(filename)
    print ("Written %s"%( filename ))

    time2 = time.time()
    training_time = time2 - time1
    print ("Training time: %.2f seconds" % training_time)

print(f"Trained IC for this selection: {args.selection}")
print(ic)
