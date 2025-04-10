''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *

import common.data_structure as data_structure

parameters         = ['nu_jes']

# We learn a quadratic model for the nu_jes dependence in this config
combinations       = [('nu_jes',),  ('nu_jes', 'nu_jes'), ] 

# Base point coordinates in tes/jes/met; Example: jes, where we have +/- 3 sigma in steps
base_point_index = {
    0 : ( -3., ),
    1 : ( -2., ),
    2 : ( -1., ),
    3 : (  0., ),
    4 : (  1., ),
    5 : (  2., ),
    6 : (  3., ),
}

# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1, 1+base_point[0]*0.01, 0 )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

# Make a matrix
base_points        = [ base_point_index[i] for i in [0,1,2,3,4,5,6] ] 

# Pick out the "SM" base point
nominal_base_point = base_point_index[3]

# input dimensions
input_dim     = len(data_structure.feature_names)

# Scale external scaler (features)?
use_scaler    = True
# Use external inclusive xsec dependence? (Default: None)
icp           = "icp_quad_jes"

# hidden layers
hidden_layers = [128, 128]

# activation function
activation    = 'relu'

# Number of epochs
n_epochs           = 200
n_epochs_phaseout  = 50

# Learning rate 
learning_rate = 0.001

# Regularization
l1_reg          = 1
#l2_reg          = 0.01
#dropout         = 0.2
initialize_zero = True
                                 
