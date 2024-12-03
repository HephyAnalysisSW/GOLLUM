#!/usr/bin/env python

from Workflow.networks.StartingKit import Model
from ML.BPT.BoostedParametricTree import BoostedParametricTree
from ML.ICP.ICP import ICP

def getModule(name):
  return eval(name)
