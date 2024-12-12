''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *

parameters         = ['nu_met']
# We learn a quadratic model for the nu_met dependence in this config
combinations       = [('nu_met',),  ('nu_met', 'nu_met'), ] 
# Base point coordinates in tes/jes/met; Example: jes, where we have +/- 3 sigma in steps
base_point_index = {
    0 : (  0., ),
    1 : (  .5, ),
    2 : (  1., ),
    3 : (  1.5, ),
    3 : (  2., ),
}

# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1,1,3*base_point[0] )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

# Make a matrix
base_points        = [ base_point_index[i] for i in [0,1,2,3] ] 
# Pick out the "SM" base point
nominal_base_point = base_point_index[0]
