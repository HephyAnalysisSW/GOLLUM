#!/usr/bin/env python
import torch
import copy
import numpy as np
import os,sys
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import time
import functools
import operator

import common.user
import common.syncer
from   common.helpers import copyIndexPHP
from   BPT import analytic_2D as model 

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--directory",     action="store",      default="bpn", help="Subdirectory for output")
argParser.add_argument("--nTraining",     action="store",      default=100000, type=int,  help="Number of training events")
argParser.add_argument('--overwrite',     action='store_true', help="Overwrite training?")

args, extra = argParser.parse_known_args(sys.argv[1:])

def parse_value( s ):
    try:
        r = int( s )
    except ValueError:
        try:
            r = float(s)
        except ValueError:
            r = s
    return r

extra_args = {}
key        = None
for arg in extra:
    if arg.startswith('--'):
        # previous no value? -> Interpret as flag
        #if key is not None and extra_args[key] is None:
        #    extra_args[key]=True
        key = arg.lstrip('-')
        extra_args[key] = True # without values, interpret as flag
        continue
    else:
        if type(extra_args[key])==type([]):
            extra_args[key].append( parse_value(arg) )
        else:
            extra_args[key] = [parse_value(arg)]
for key, val in extra_args.items():
    if type(val)==type([]) and len(val)==1:
        extra_args[key]=val[0]

cfg = model.bpn_cfg
cfg.update( extra_args )

training_data = model.getEvents(args.nTraining)
total_size    =  sum([len(s['features']) for s in training_data.values() if 'features' in s ])


# Base point matrix
base_points = np.array( model.base_points )
VkA  = np.zeros( [len(base_points), len(model.combinations) ], dtype='float32')
for i_base_point, base_point in enumerate(base_points):
    for i_comb1, comb1 in enumerate(model.combinations):
        VkA[i_base_point][i_comb1] += functools.reduce(operator.mul, [base_point[model.parameters.index(c)] for c in list(comb1)], 1)
VkA = torch.tensor( VkA )

# Dissect inputs into nominal sample and variied
nominal_base_point_index = np.where(np.all(base_points==model.nominal_base_point,axis=1))[0]
assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( model.nominal_base_point, base_points)
nominal_base_point_index = nominal_base_point_index[0]
nominal_base_point_key   = tuple(model.nominal_base_point)

n_features = len(training_data[nominal_base_point_key]['features'][0])

# Complement training data
if 'weights' not in training_data[nominal_base_point_key]:
    training_data[nominal_base_point_key]['weights'] = np.ones(training_data[nominal_base_point_key]['features'].shape[0])

for k, v in training_data.items():
    if "features" not in v and "weights" not in v:
        raise RuntimeError( "Key %r has neither features nor weights" %k  )
    if k == nominal_base_point_key:
        if 'features' not in v:
            raise RuntimeError( "Nominal base point does not have features!" )
    else:
        if not 'features' in v:
            # we must have weights
            #v['features'] = training_data[nominal_base_point_key]['features']
            if len( training_data[nominal_base_point_key]['features'])!=len(v['weights']):
                raise runtimeerror("key %r has inconsistent length in weights"%v)
        if (not 'weights' in training_data[nominal_base_point_key].keys()) and 'weights' in v:
            raise RuntimeError( "Found no weights for nominal base point, but for a variation. This is not allowed" )
        if ('weights' in training_data[nominal_base_point_key].keys()) and 'weights' not in v:
            raise RuntimeError( "Found weights for nominal base point, but not for a variation. This is not allowed" )

    if 'weights' in v and 'features' in v:
        if len(v['weights'])!=len(v['features']):
            raise RuntimeError("Key %r has unequal length of weights and features: %i != %i" % (k, len(v['weights']), len(v['features'])) )

    if 'weights' in v:
        v['weights'] = torch.tensor(v['weights'], dtype=torch.float32)
    if 'features' in v:
        v['features'] = torch.tensor(v['features'], dtype=torch.float32)

model_directory = os.path.join( common.user.model_directory, args.directory )
os.makedirs(model_directory, exist_ok=True)

from BPN import BPN
bpn = BPN.BPN(
        in_features  = n_features,
        out_features = len(model.combinations),
        VkA          = VkA,
        **cfg,
            )
bpn_name = "BPN_analytic_2D_nTraining_%i_nLayers_%s"%( args.nTraining, "_".join(map( str, bpn.layer_size())))

optimizer = torch.optim.Adam(bpn.parameters(), lr=model.bpn_cfg['learning_rate'])

# Training loop
ema_decay = 0.99  # Exponential moving average decay factos
ema_model = copy.deepcopy(bpn)  # Create a copy of the model for EMA

for epoch in range(model.bpn_cfg['n_epochs']):
    total_loss = 0.0

    # Compute nominal values
    DeltaA_nominal = bpn(training_data[nominal_base_point_key]['features'])
    weights_nominal = training_data[nominal_base_point_key]['weights'] if 'weights' in training_data[nominal_base_point_key] else torch.ones(len(DeltaA_nominal))

    for i_base_point, base_point in enumerate(base_points):
        if i_base_point == nominal_base_point_index:
            continue

        # Retrieve features and weights for current \nu
        if 'features' in training_data[tuple(base_point)]:
            DeltaA = bpn(training_data[tuple(base_point)]['features'])
        else:
            DeltaA = DeltaA_nominal
        if 'weights' in training_data[tuple(base_point)]:
            weights = training_data[tuple(base_point)]['weights']
        else:
            weights = weights_nominal

        # Compute weighted losses
        loss_0  = torch.sum(weights_nominal * torch.log1p(torch.exp(DeltaA_nominal @ VkA[i_base_point])))  # Soft⁺(ν_A Δ_A(x_0))
        loss_nu = torch.sum(weights * torch.log1p(torch.exp(-(DeltaA @ VkA[i_base_point]))))  # Soft⁺(-ν_A Δ_A(x_nu))

        # Total loss for the current \nu
        loss = loss_0 + loss_nu
        loss -= (weights_nominal.sum()+weights.sum())*np.log(2.)
        total_loss += loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(bpn.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()

    # Apply exponential moving average to stabilize model
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), bpn.parameters()):
            ema_param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data

    # Use the EMA model parameters for prediction
    bpn.load_state_dict(ema_model.state_dict())

    print(f"Epoch {epoch + 1}/{model.bpn_cfg['n_epochs']}, Loss: {total_loss.item():.4f}")

torch.set_grad_enabled(False)

#predicted_reweights = np.exp( np.dot( bpn.forward(features), bpt.VkA.transpose() ) )
predicted_reweights = np.exp( np.dot( bpn.forward(training_data[model.nominal_base_point]['features']).detach().numpy(), bpn.VkA.transpose(0,1) ) )

plot_directory = os.path.join( common.user.plot_directory, args.directory )
os.makedirs(plot_directory, exist_ok=True)
copyIndexPHP( plot_directory )

import matplotlib.pyplot as plt
import numpy as np

hist_configs = []
colors = ['black', 'blue', 'green', 'red', 'orange', 'magenta', 'cyan']
for i_point, point in enumerate(base_points):
    # truth
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'][:,0].detach().numpy(), 
                          'name':'%s'%str(point) +" (truth.)", 
                          'weights':training_data[tuple(point)]['weights'].detach().numpy(), 
                          'color':colors[i_point], 'linestyle':'--'} )
    # prediction
    hist_configs.append( {'features':training_data[model.nominal_base_point]['features'][:,0].detach().numpy(), 
                          'name':'%s'%str(point) +" (pred.)", 
                          'weights':predicted_reweights[:,i_point]*training_data[model.nominal_base_point]['weights'].detach().numpy(), 
                          'color':colors[i_point], 'linestyle':'-'} )

def plot_weighted_histograms(hist_configs, bins=20, title='Overlayed Weighted Histograms'):
    """
    Plot and save overlayed 1D histograms with different weight vectors.

    Parameters:
    - hist_configs: list of dictionaries, each dictionary contains:
        - 'features': list or numpy array of features.
        - 'weights': weight vector (same length as features).
        - 'color': color for the histogram.
        - 'linestyle': line style for the histogram ('-' for continuous, '--' for dashed).
    - bins: number of bins for the histogram (default is 10).
    - title: title of the plot.
    """
    
    plt.figure(figsize=(10, 6))

    for config in hist_configs:
        features = config['features']
        weights = config['weights']
        color = config['color']
        linestyle = config['linestyle']
        name = config['name']
 
        plt.hist(features, bins=bins, weights=weights, color=color, linestyle=linestyle, histtype='step', linewidth=1.5, label=name)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Weighted Count')
    plt.legend([f'Histogram {i+1}' for i in range(len(hist_configs))])
    plt.grid(True)
    plt.legend(ncol=2)
 
    # Save the figure in PNG and PDF formats
    plt.savefig(os.path.join( plot_directory, 'weighted_histograms.png'))
    print ("Written ",os.path.join( plot_directory, 'weighted_histograms.png'))
    plt.savefig(os.path.join( plot_directory, 'weighted_histograms.pdf'))
    print ("Written ",os.path.join( plot_directory, 'weighted_histograms.pdf'))
    plt.show()

plot_weighted_histograms(hist_configs, bins=20)
common.syncer.sync()

