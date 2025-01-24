''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

classes       = data_structure.labels
input_dim     = len(data_structure.feature_names)

use_ic        = True
use_scaler    = True
        
learning_rate = 0.01  # Learning rate (eta)
max_depth = 6  # Maximum tree depth
subsample = 0.8  # Fraction of samples for training each tree
colsample_bytree = 0.8  # Fraction of features for training each tree
l1_reg = 0.1  # L1 regularization (alpha)
l2_reg = 0.05  # L2 regularization (lambda)
#num_boost_round = 300  # Number of boosting rounds
num_boost_round = 5  # Number of boosting rounds
seed = 42  # Random seed for reproducibility
verbose_eval = 10  # Frequency of logging during training

