#!/usr/bin/env python

from ML.BPT.BoostedParametricTree import BoostedParametricTree
from ML.ICP.ICP import InclusiveCrosssectionParametrization
from ML.IC.IC import InclusiveCrosssection
from ML.PNN.PNN import PNN
from ML.TFMC.TFMC import TFMC
from ML.XGBMC.XGBMC import XGBMC

def getModule(name):
  return eval(name)
