#!/usr/bin/env python

from Workflow.networks.StartingKit import Model
from ML.BPT.BoostedParametricTree import BoostedParametricTree

def getModule(name):
  return eval(name)
