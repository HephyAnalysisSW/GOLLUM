#!/usr/bin/env python

import sys
sys.path.insert(0, "..")

import argparse
import common.user
from Workflow.Inference import Inference

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="ML inference.")
  parser.add_argument("-c","--config", help="Path to the config file.")
  args = parser.parse_args()

  infer = Inference(args.config)
  ts = infer.testStat(1,0)
  print(ts)
