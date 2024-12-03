#!/usr/bin/env python

from Workflow.networks.StartingKit import Model
from ML.BPT.BoostedParametricTree import BoostedParametricTree
from ML.ICP.ICP import ICP
from ML.IC.IC import IC

def getModule(name):
  return eval(name)
