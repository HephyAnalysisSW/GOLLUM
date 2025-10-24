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
combinations       = [('nu_tes',),  ('nu_jes',), ('nu_met',), 
                      ('nu_tes', 'nu_tes'), ('nu_jes', 'nu_jes'), ('nu_met', 'nu_met'), 
                      ('nu_tes', 'nu_jes'), ('nu_tes', 'nu_met'), ('nu_jes', 'nu_met')]

# Base point coordinates in tes/jes/met; Example: jes, where we have +/- 3 sigma in steps
base_point_index = {
 0:  (0, 0, 0.0),  
# 1:  (-3, 0, 0.0), 
# 2:  (-2, 0, 0.0), 
# 3:  (-1, -1, 0.0),
# 4:  (-1, -1, 1.0),
# 5:  (-1, -1, 2.0),
# 6:  (-1, 0, 0.0), 
# 7:  (-1, 0, 1.0), 
# 8:  (-1, 0, 2.0), 
# 9:  (-1, 1, 0.0), 
# 10: (-1, 1, 1.0), 
# 11: (-1, 1, 2.0), 
# 12: (0, -3, 0.0), 
# 13: (0, -2, 0.0), 
# 14: (0, -1, 0.0), 
# 15: (0, -1, 1.0), 
# 16: (0, -1, 2.0), 
# 17: (0, 0, 1.0),  
# 18: (0, 0, 2.0),  
# 19: (0, 1, 0.0),  
# 20: (0, 1, 1.0),  
# 21: (0, 1, 2.0),  
# 22: (0, 2, 0.0),  
# 23: (0, 3, 0.0),  
# 24: (1, -1, 0.0), 
# 25: (1, -1, 1.0), 
# 26: (1, -1, 2.0), 
# 27: (1, 0, 0.0),  
# 28: (1, 0, 1.0),  
# 29: (1, 0, 2.0),  
# 30: (1, 1, 0.0),  
# 31: (1, 1, 1.0),  
# 32: (1, 1, 2.0),  
# 33: (2, 0, 0.0),  
# 34: (3, 0, 0.0)   
}

# Make a matrix
base_points        = [ base_point_index[i] for i in range(len(base_point_index)) ] 
 
# translate nuisances to alpha values
def get_alpha( base_point ):
    return ( 1+base_point[0]*0.01, 1+base_point[1]*0.01, base_point[2] )

# For convenience, base_point_index should also know about the inverse dictionary
base_point_index.update ({val:key for key, val in base_point_index.items()})

input_dim = len(data_structure.feature_names)

# Pick out the "SM" base point
nominal_base_point = base_point_index[0]
use_scaler = True

# -----------------------------------------------------------------------------
# Network hyperparameters for conditional inflow
# -----------------------------------------------------------------------------

# Embedding network (maps nu -> embedding vector)
embed_hidden_layers = [64, 64]
embed_dim = 64           # size of nu embedding
activation = 'nn.LeakyReLU(0.1)'

# Flow (bijector) architecture
n_flow_layers = 3        # number of coupling/autoregressive layers
flow_hidden_layers = [128, 128]  # hidden dims in each coupling-net
coupling_mask = 'checkerboard'   # type of masking: 'checkerboard' or 'channel_wise'
use_batch_norm = False   # whether to insert batch-norm between flow blocks
# Spline settings (if using rational-quadratic flows)
use_spline = True
num_bins = 4            # number of bins for spline
bin_range = 3.0         # range for spline endpoints
min_bin_width  = 1e-2
min_bin_height = 1e-2
min_derivative = 1e-2
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

n_epochs = 200
batch_size = 512
learning_rate = 0.001
weight_decay = 1e-4

# Learning rate scheduler (optional)
lr_scheduler = {
    'type': 'CosineAnnealingLR',
    'T_max': n_epochs,
    'eta_min': 1e-6
}

# Gradient clipping (optional)
grad_norm_clip = 5.0

# Device
device = 'cuda'  # or 'cpu'

# Checkpointing and logging
save_interval = 10  # epochs
log_interval = 50   # iterations

# -----------------------------------------------------------------------------
# End of config
# -----------------------------------------------------------------------------
nu_plot_list = [
    (0,0,0),
#    (1,0,0),(-1,0,0),
#    (0,1,0),(0,-1,0),
#    (0,0,1),(0,0,2),
]
