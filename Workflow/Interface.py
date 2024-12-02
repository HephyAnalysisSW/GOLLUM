#!/usr/bin/env python

import yaml
import argparse
import networks.Models as ms

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="ML interface.")
  parser.add_argument("--train", action="store_true", help="Whether to train the network.")
  parser.add_argument("--predict", action="store_true", help="Whether to predict using the network.")
  args = parser.parse_args()
  assert args.train or args.predict, "Please specify which mode to use for the interface: --train or --predict."

  # First load the config file
  config_path = 'config.yaml'
  with open(config_path) as f:
    cfg = yaml.safe_load(f)
  
  for t in cfg['Tasks']:
    assert t in cfg, "{t} not defined in config!"
    m = ms.getModule(cfg[t]["module"])
    mm = m()
    print("Task {}: Module {} loaded.".format(t,cfg[t]["module"]))
    # Fit the model
    if args.train:
      print("Training...")
      #mm.fit()
    elif args.predict:
      print("Predicting...")
      #mm.predict()

