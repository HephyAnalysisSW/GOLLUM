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
        'lowMT_noVBFJet_ptH0to100': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_noVBFJet_ptH0to100/tfmc_2_reg/v6"),
        'lowMT_noVBFJet_ptH100': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_noVBFJet_ptH100/tfmc_2_reg/v6"),
        'lowMT_VBFJet': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_2_reg/v6"),
        'highMT_VBFJet': TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/highMT_VBFJet/tfmc_2_reg/v6"),
        }

htautau_index = data_structure.labels.index('htautau')
ttbar_index = data_structure.labels.index('ttbar')
diboson_index = data_structure.labels.index('diboson')

# Put the elements of your selections here, i.e., cuts you want to apply
selections = {
        "MVAHighMTnoVBFJetTtbar"    : lambda data: data[tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, ttbar_index]>0.4],
        "MVAHighMTnoVBFJetDiboson"  : lambda data: data[(tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, ttbar_index]<=0.4) & (tfmc["highMT_noVBFJet"].predict(data[:,:28], ic_scaling=False)[:, diboson_index]>0.5)],
    }

def makeMVASelector( mva_key, output, lower, upper ):
    def sel( data, mva_key=mva_key, output=output, lower=lower, upper=upper):
        ind = data_structure.labels.index(output)
        pred = tfmc[mva_key].predict(data[:,:28], ic_scaling=False)[:, ind]
        return data[ (pred>lower) & (pred<=upper) ]
    return sel

selections.update( {
    "MVALowMTVBFJet_bin0":  makeMVASelector( 'lowMT_VBFJet', 'htautau', -0.005, 0.025 ),
    "MVALowMTVBFJet_bin1":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.025, 0.095 ),
    "MVALowMTVBFJet_bin2":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.095, 0.295 ), 
    "MVALowMTVBFJet_bin3":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.295, 0.625 ), 
    "MVALowMTVBFJet_bin4":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.625, 0.89  ), 
    "MVALowMTVBFJet_bin5":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.89 , 0.965 ), 
    "MVALowMTVBFJet_bin6":  makeMVASelector( 'lowMT_VBFJet', 'htautau',  0.965, 1.005 ),  

    "MVALowMTNoVBFJetPtH100_bin0":  makeMVASelector( 'lowMT_noVBFJet_ptH100', 'htautau', -0.005, 0.06 ),
    "MVALowMTNoVBFJetPtH100_bin1":  makeMVASelector( 'lowMT_noVBFJet_ptH100', 'htautau',  0.06, 0.17 ),
    "MVALowMTNoVBFJetPtH100_bin2":  makeMVASelector( 'lowMT_noVBFJet_ptH100', 'htautau',  0.17, 0.405 ), 
    "MVALowMTNoVBFJetPtH100_bin3":  makeMVASelector( 'lowMT_noVBFJet_ptH100', 'htautau',  0.405, 0.72 ), 
    "MVALowMTNoVBFJetPtH100_bin4":  makeMVASelector( 'lowMT_noVBFJet_ptH100', 'htautau',  0.72, 1.05  ), 

    "MVAHighMTVBJet_bin0":  makeMVASelector( 'highMT_VBFJet', 'htautau', -0.005, 0.045 ),
    "MVAHighMTVBJet_bin1":  makeMVASelector( 'highMT_VBFJet', 'htautau',  0.045, 0.32 ),
    "MVAHighMTVBJet_bin2":  makeMVASelector( 'highMT_VBFJet', 'htautau',  0.32, 0.665 ), 
    "MVAHighMTVBJet_bin3":  makeMVASelector( 'highMT_VBFJet', 'htautau',  0.665, 0.875 ), 
    "MVAHighMTVBJet_bin4":  makeMVASelector( 'highMT_VBFJet', 'htautau',  0.875, 1.05  ), 
    })



all_selections = sorted(list(selections.keys()))

def print_all():
    print("All MVA selections: "+", ".join(all_selections)) 

# from selections import lowMT_VBFJet
globals().update(selections)
