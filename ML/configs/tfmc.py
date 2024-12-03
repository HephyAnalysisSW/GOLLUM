''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

n_epochs = 100

classes       = data_structure.labels
input_dim     = len(data_structure.feature_names)
hidden_layers = [64,64]

scale_with_ic = True
