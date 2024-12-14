''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''
import numpy as np

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

n_epochs         = 200
#n_epoch_phaseout = 50  # This number of epochs  at the end for phaseout 

# Initial learning rate
learning_rate = 0.002

had_pt  = data_structure.feature_names.index("PRI_had_pt")
had_eta = data_structure.feature_names.index("PRI_had_eta")
had_phi = data_structure.feature_names.index("PRI_had_phi")
lep_pt  = data_structure.feature_names.index("PRI_lep_pt")
lep_eta = data_structure.feature_names.index("PRI_lep_eta")
lep_phi = data_structure.feature_names.index("PRI_lep_phi")
met     = data_structure.feature_names.index("PRI_met")
met_phi = data_structure.feature_names.index("PRI_met_phi")
mass_transverse_met_lep = data_structure.feature_names.index("DER_mass_transverse_met_lep")
mass_vis                = data_structure.feature_names.index("DER_mass_vis")

def preprocessor( features, features_norm ):

    return np.column_stack((
        features_norm[:, had_pt],
        features_norm[:, met],
        features[:, had_eta] - features[:, lep_eta],
        features[:, had_phi] - features[:, lep_phi],
        features[:, met_phi] - features[:, lep_phi],
        features[:, met_phi] - features[:, had_phi],
        features_norm[:, mass_transverse_met_lep],
        features_norm[:, mass_vis],
    ))

classes       = ['htautau', 'ztautau'] 
input_dim     = 8 
hidden_layers = [64,64]
activation    = 'relu'

#l1_reg        = 0.1
#l2_reg        = 0.05
#dropout_rate  = 0.2

use_ic        = True
use_scaler    = True
