#!/usr/bin/env python

import sys
sys.path.insert(0, "..")

import argparse
import common.user
from Workflow.Inference import Inference

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="ML inference.")
  parser.add_argument("-c","--config", help="Path to the config file.")
  parser.add_argument("-s","--save", action="store_true", help="Whether to save the ML predictions for the simulation.")
  parser.add_argument("-p","--predict", action="store_true", help="Whether to predict.")
  args = parser.parse_args()

  infer = Inference(args.config)
  if args.save:
    infer.save(filename="test",isData=False)
  if args.predict:
    r = infer.predict("lowMT_VBFJet",2.2,0,0,0,1.5,0,0,False)
    print(r)
    r = infer.predict("lowMT_VBFJet",2.2,0,0,0,1.5,0,0,True)
    print(r)
    infer.clossMLresults()
  # Below is the deprecated feature that calculates the ML prediction on the fly  
  #r = infer.testStat(1,0)
  #print(r)
