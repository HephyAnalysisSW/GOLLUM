import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.data_structure as data_structure

#MT_ind  = data_structure.feature_names.index("DER_mass_transverse_met_lep")
#LJ_ind  = data_structure.feature_names.index("PRI_jet_leading_pt")

from ML.TFMC.TFMC import TFMC
## https://schoef.web.cern.ch/schoef/Challenge/TFMC/lowMT_noVBFJet_ptH0to100/tfmc_2_reg/predicted_probabilities/predicted_class_htautau.png

tfmc = {'highMT_noVBFJet': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/highMT_noVBFJet/tfmc_2_reg/v6"),
        #'lowMT_noVBFJet_ptH0to100': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_noVBFJet_ptH0to100/tfmc_2_reg/v6"),
        }

ttbar_index = data_structure.labels.index('ttbar')
diboson_index = data_structure.labels.index('diboson')

# Put the elements of your selections here, i.e., cuts you want to apply
selections = {
        "MVAHighMTnoVBFJetTtbar"    : lambda data: data[tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, ttbar_index]>0.4],
        "MVAHighMTnoVBFJetDiboson"  : lambda data: data[(tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, ttbar_index]<=0.4) & (tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, diboson_index]>0.5)],
    }

all_selections = sorted(list(selections.keys()))

def print_all():
    print("All MVA selections: "+", ".join(all_selections)) 

# from selections import lowMT_VBFJet
globals().update(selections)
