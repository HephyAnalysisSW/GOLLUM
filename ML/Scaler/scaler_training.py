#!/usr/bin/env python

import os
import sys
import time
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user
from ML.Scaler.Scaler import Scaler

# Parser
import argparse
argParser = argparse.ArgumentParser(description="Argument parser")
argParser.add_argument('--overwrite', action='store_true', help="Overwrite training?")
argParser.add_argument("--selection", action="store", default="lowMT_VBFJet", help="Which selection?")
argParser.add_argument("--process", action="store", default=None, help="Which proecss?")
argParser.add_argument('--small', action='store_true', help="Only one batch, for debugging")
args = argParser.parse_args()

# Import the data
import common.datasets_hephy as datasets

print("Scaler training for selection " + '\033[1m' + f"{args.selection}" + '\033[0m')

subdirs = [arg for arg in [args.process, args.selection] if arg is not None]
scaler_name = "Scaler_"+"_".join(subdirs)

model_directory = os.path.join(common.user.model_directory, "Scaler")
os.makedirs(model_directory, exist_ok=True)

filename = os.path.join(model_directory, ('small_' if args.small else '') + scaler_name) + '.pkl'

scaler = None
if not args.overwrite:
    try:
        print(f"Trying to load {scaler_name} from {filename}")
        scaler = Scaler.load(filename)
    except (IOError, EOFError, ValueError):
        pass

if scaler is None or args.overwrite:
    print("Training.")
    time1 = time.time()
    scaler = Scaler()

    scaler.load_training_data(datasets=datasets, selection=args.selection, process=args.process)
    scaler.train(small=args.small)

    scaler.save(filename)
    print(f"Written {filename}")

    time2 = time.time()
    training_time = time2 - time1
    print(f"Training time: {training_time:.2f} seconds")

print(scaler)
