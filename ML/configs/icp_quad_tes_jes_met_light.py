''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *

parameters         = ['nu_tes', 'nu_jes', 'nu_met']
# We learn a quadratic model for the nu_jes dependence in this config
combinations       = [('nu_tes',),  ('nu_jes',), ('nu_met',), ('nu_tes', 'nu_tes'), ('nu_jes', 'nu_jes'), ('nu_met', 'nu_met'), ('nu_tes', 'nu_jes'), ('nu_tes', 'nu_met'), ('nu_jes', 'nu_met')]

# Base point coordinates in tes/jes/met; Example: jes, where we have +/- 2 sigma in steps
base_point_index = {
 0:  (0, 0, 0.0),  
 1:  (-2, 0, 0.0), 
 2:  (-1, -1, 0.0),
 3:  (-1, 0, 0.0), 
 4:  (-1, 0, 1.0), 
 5:  (-1, 1, 0.0), 
 6: (0, -2, 0.0), 
 7: (0, -1, 0.0), 
 8: (0, -1, 1.0), 
 9: (0, 0, 1.0),  
 10: (0, 0, 2.0),  
 11: (0, 1, 0.0),  
 12: (0, 1, 1.0),  
 13: (0, 2, 0.0),  
 14: (1, -1, 0.0), 
 15: (1, 0, 0.0),  
 16: (1, 0, 1.0),  
 17: (1, 1, 0.0),  
 18: (2, 0, 0.0)
}
 
# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1+base_point[0]*0.01, 1+base_point[1]*0.01, base_point[2] )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

# Make a matrix
base_points        = [ base_point_index[i] for i in range(19) ] 

# Pick out the "SM" base point
nominal_base_point = base_point_index[0]
