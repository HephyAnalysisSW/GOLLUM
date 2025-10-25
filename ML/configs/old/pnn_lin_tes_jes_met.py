''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

# Nuisance parameters
parameters         = ['nu_tes', 'nu_jes', 'nu_met']
# We learn a quadratic model for the nu_jes dependence in this config
combinations       = [('nu_tes',),  ('nu_jes',), ('nu_met',),] 

# Base point coordinates in tes/jes/met; Example: jes, where we have +/- 3 sigma in steps
base_point_index = {
 0:  (0, 0, 0.0),  
 1:  (-3, 0, 0.0), 
 2:  (-2, 0, 0.0), 
 3:  (-1, -1, 0.0),
 4:  (-1, -1, 1.0),
 5:  (-1, -1, 2.0),
 6:  (-1, 0, 0.0), 
 7:  (-1, 0, 1.0), 
 8:  (-1, 0, 2.0), 
 9:  (-1, 1, 0.0), 
 10: (-1, 1, 1.0), 
 11: (-1, 1, 2.0), 
 12: (0, -3, 0.0), 
 13: (0, -2, 0.0), 
 14: (0, -1, 0.0), 
 15: (0, -1, 1.0), 
 16: (0, -1, 2.0), 
 17: (0, 0, 1.0),  
 18: (0, 0, 2.0),  
 19: (0, 1, 0.0),  
 20: (0, 1, 1.0),  
 21: (0, 1, 2.0),  
 22: (0, 2, 0.0),  
 23: (0, 3, 0.0),  
 24: (1, -1, 0.0), 
 25: (1, -1, 1.0), 
 26: (1, -1, 2.0), 
 27: (1, 0, 0.0),  
 28: (1, 0, 1.0),  
 29: (1, 0, 2.0),  
 30: (1, 1, 0.0),  
 31: (1, 1, 1.0),  
 32: (1, 1, 2.0),  
 33: (2, 0, 0.0),  
 34: (3, 0, 0.0)   
}
 
# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1+base_point[0]*0.01, 1+base_point[1]*0.01, base_point[2] )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

# Make a matrix
base_points        = [ base_point_index[i] for i in range(35) ] 

# Pick out the "SM" base point
nominal_base_point = base_point_index[0]

# input dimensions
input_dim     = len(data_structure.feature_names)

# Scale external scaler (features)?
use_scaler    = True

# Use external inclusive xsec dependence? (Default: None)
icp           = "icp_quad_tes_jes_met"

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
#l1_reg          = 0.01
#l2_reg          = 0.01
#dropout         = 0.2
initialize_zero = True
