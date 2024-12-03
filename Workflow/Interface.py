#!/usr/bin/env python

import os
import yaml
import argparse
import importlib
import networks.Models as ms
import common.user


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="ML interface.")
  parser.add_argument("-t","--train", action="store_true", help="Whether to train the network.")
  parser.add_argument("-o","--overwrite", action="store_true", help="Whether to overwrite the network if already exists.")
  parser.add_argument("-p","--predict", action="store_true", help="Whether to predict using the network.")
  args = parser.parse_args()
  assert args.train or args.predict, "Please specify which mode to use for the interface: --train or --predict."

  # First load the config file
  config_path = 'config.yaml'
  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  # Import the data
  import common.datasets as datasets
  
  for t in cfg['Tasks']:
    assert t in cfg, "{t} not defined in config!"
    m = ms.getModule(cfg[t]["module"])
    if "config" in cfg[t]:
      config_path = cfg[t]["config"]
      #exec('import %s as config_%s'%( config_path, t))
      config_ = importlib.import_module(config_path)
      mm = m(config=config_)
    elif "hyper_param" in cfg[t]: # FIXME: this needs to be checked when dealing with algorithms without a config as the input
      mm = m(cfg[t]["hyper_param"])
    else:
      mm = m()
    print("Task {}: Module {} loaded.".format(t,cfg[t]["module"]))
    if len(cfg[t]["model_name_abs"])==0:
      model_directory = os.path.join( common.user.model_directory, t )
      model_path = os.path.join(model_directory, cfg[t]["model_name_rel"])
    else:
      model_directory = os.path.dirname( cfg[t]["model_name_abs"] )
      model_path = cfg[t]["model_name_abs"]
    # Fit the model
    if args.train:
      print("Training...")
      print("Will save model at {}".format(model_path))
      if not os.path.exists(model_directory):
        os.makedirs(model_directory)
      if os.path.exists(model_path) and not args.overwrite:
        raise Exception("Model path {} already exists!".format(model_path))
      mm.load_training_data(datasets, cfg[t]["selection"])
      mm.train(datasets, cfg[t]["selection"], small=True)
      mm.save(model_path)
      print("Model saved in {}".format(model_path))
    elif args.predict:
      print("Predicting...")
      mm = m.load(model_path)
      print("Model loaded from {}".format(model_path))
      print(mm)
      print(mm.predict((1,)))
      print(mm.predict((2,)))
      #mm.predict()

