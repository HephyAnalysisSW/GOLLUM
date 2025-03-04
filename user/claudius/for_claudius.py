import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
    
import numpy as np

import common.user as user
import common.data_structure as data_structure
import common.selections as selections
import common.datasets_hephy as datasets_hephy

from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6")

from ML.XGBMC.XGBMC import XGBMC
xgbmc = XGBMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/XGBMC/lowMT_VBFJet/xgb_v1/v1")


# Iterate through the dataset
loader = datasets_hephy.get_data_loader(selection="lowMT_VBFJet", n_split=1)
for batch in loader:
    data, weights, labels = loader.split(batch)
    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )

    print(" class probabilities from TFMC")
    prob = tfmc.predict(data, ic_scaling = False)
    print(prob)
    #print(" class probabilities from XGBMC")
    #prob=xgbmc.predict(data, ic_scaling = False)
    #print(prob)

    break
