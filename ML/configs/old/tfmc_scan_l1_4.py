''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

n_epochs           = 300
n_epochs_phaseout  = 100
learning_rate = 0.01

classes       = data_structure.labels
input_dim     = len(data_structure.feature_names)
hidden_layers = [64,64,64]
activation    = 'relu'

l1_reg        = 0.01
l2_reg        = 0.05
dropout_rate  = 0.2

use_ic        = True
use_scaler    = True
