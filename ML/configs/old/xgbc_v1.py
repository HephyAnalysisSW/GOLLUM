''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

input_dim     = 7 

truth_key = "mu_true"

feature_keys = [ 'mu_measured',
 'mu_measured_up',
 'mu_measured_down',
 'nu_tes',
 'nu_jes',
 'nu_met',
 'nu_bkg',
 'nu_tt',
 'nu_diboson']

learning_rate = 0.1  # Reduced learning rate for more gradual learning on a small dataset
max_depth = 4        # Lowered maximum tree depth to help prevent overfitting
l2_reg = 0.1         # Increased L2 regularization for better generalization
num_boost_round = 200  # Increased boosting rounds to compensate for the lower learning rate

subsample = 0.8  # Fraction of samples for training each tree
colsample_bytree = 1.0  # Fraction of features for training each tree
l1_reg = 0.1  # L1 regularization (alpha)
#num_boost_round = 300  # Number of boosting rounds
seed = 42  # Random seed for reproducibility
verbose_eval = 10  # Frequency of logging during training
