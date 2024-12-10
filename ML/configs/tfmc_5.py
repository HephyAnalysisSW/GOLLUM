''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

n_epochs      = 200
#learning_rate = 0.00015
learning_rate = 0.01

classes       = data_structure.labels
input_dim     = len(data_structure.feature_names)
hidden_layers = [128,128]
activation    = 'LeakyReLU'

use_ic        = True
use_scaler    = True
