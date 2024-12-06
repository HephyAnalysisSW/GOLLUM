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
#base_point_index = {
#    0 : ( -3., ),
#    1 : ( -2., ),
#    2 : ( -1., ),
#    3 : (  0., ),
#    4 : (  1., ),
#    5 : (  2., ),
#    6 : (  3., ),
#}

base_point_index = {
    0 : ( -1., ),
    1 : (  0., ),
    2 : (  1., ),
}

# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1, 1+base_point[0]*0.01, 0 )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

# Make a matrix
#base_points        = [ base_point_index[i] for i in [0,1,2,3,4,5,6] ] 
base_points        = [ base_point_index[i] for i in [0,1,2] ] 

# Pick out the "SM" base point
#nominal_base_point = base_point_index[3]
nominal_base_point = base_point_index[1]

# Number of epochs
n_epochs = 100

# Learning rate 
learning_rate = 0.001

# input dimensions
input_dim     = len(data_structure.feature_names)

# First learn the global parametrization?
use_scaler    = True
use_icp       = False

# hidden layers
hidden_layers = [64, 32] 
