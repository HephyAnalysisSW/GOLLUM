# 
# goal: make plots of pre-fit inputs directly from variations, instead of accessing the config and without using the learned ICPH surrogates
# - similar to what is in pdf/plot.py

# inputs: feature, binning, selection, sample (inc. era in the names), variation
# """

"""
Generate plots of pre-fit input variations directly from raw ±1 sigma input files.
This module creates histogram plots comparing nominal, down-varied, and up-varied weight distributions for different systematic uncertainties
without using learned ICPH surrogates.

Usage:
    python plot_from_inputs.py --feature <feature_name> [--binning <bin_edges>] [--selection <selection_string>]

Arguments:
    --feature, -f (str): Feature to plot (required)
    --binning, -b (list): Custom bin edges. If not provided, uses defaults from plot_options
    --selection, -s (str): String-based event selection

Outputs:
    PNG plots saved to plot_directory/binned_templates_from_inputs/<feature_name>/<era>/<sample>/

Supported systematics:
    - MODELING: Renormalization/factorization scales, shower ISR/FSR, alpha_s
    - EXPERIMENTAL: Pileup and other experimental uncertainties
"""


import os
import argparse as ap
from typing import Sequence

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)

import data.samples_RunII as samples
from data.plot_options import plot_options

import common.user as user
import common.syncer as syncer
import common.helpers as helpers

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
mh.style.use('CMS')

# TODO: finish this function
def get_branches_for_selection(selection: str) -> Sequence[str]:
    return []

parser = ap.ArgumentParser(description='make plots of pre-fit variations directly from raw +-1 sigma input files, instead of using ICPH surrogates')
parser.add_argument('--feature', '-f', required=True)
parser.add_argument('--binning', '-b', nargs="+", help='binning of plot. if not given, gets it from data.plot_options')
parser.add_argument('--selection', '-s', type=str, help='string-based selection')


args = parser.parse_args()

samples_to_use = ['TTLep_pow', 'TTSemi_pow', 'SingleTop', 'DrellYan']
eras = samples.ERAS

feature_name = args.feature
selection = args.selection

# Decide on bin edges
edges = None
if args.binning:
    # explicit list of thresholds
    edges = [float(x) for x in args.binning]
    print(f"[info] Using explicit bin edges from --binning ({len(edges)-1} bins).")
elif feature_name is not None and feature_name in plot_options and 'binning' in plot_options[feature_name]:
    # build thresholds from plot_options: [nBins, low, high]
    nBins, low, high = plot_options[feature_name]['binning']
    edges = [low + i*(high - low) / nBins for i in range(nBins + 1)]
    print(f"[info] Using binning from plot_options for '{feature_name}': "
          f"nBins={nBins}, low={low}, high={high} -> {len(edges)-1} bins.")
else:
    raise ValueError(f'No binning for feature {feature_name} given nor found in defaults.')

plot_directory = os.path.join(user.plot_directory, 'binned_templates_from_inputs', feature_name)
print(f"[info] Plots will be written under: {plot_directory}")
os.makedirs(plot_directory, exist_ok=True)

x_title = plot_options.get(feature_name, {}).get('tex', feature_name)
# simple color palette
colors = [
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 1,
    ROOT.kCyan + 1,
]

# legend columns (configurable)
legend_columns = 3

"""
systematics are divided into three categories:
- additional weights, to be multiplied by the overall weight
- variations in weights which are part of overall weight - should replace nominal weight
- kinematic variations - come from different files

the RDataLoader access will be different for each
"""

"""
additional weight to be multiplied by the overall weight. structure:

{group name: 
    variation name: [down variation branch name, up variation branch name]
}
"""
add_weight_groups = {
    'MODELING': { 
        'mu_ren': ['scale_ren0p5_fac1p0','scale_ren2p0_fac1p0'],
        'mu_fac': ['scale_ren1p0_fac0p5','scale_ren1p0_fac2p0'],
        'ShowerISR': ['shower_isr0p5_fsr1p0', 'shower_isr2p0_fsr1p0'], 
        'ShowerFSR': ['shower_isr1p0_fsr0p5', 'shower_isr1p0_fsr2p0'],
        'AlphaS': ['pdf_alphas_dn','pdf_alphas_up']
    },
}

"""
variations in weights which are part of overall weight. structure:

{group name: 
    variation name: [nominal branch name, down variation branch name, up variation branch name]
}
"""
replace_weight_groups = {
    'EXPERIMENTAL': {
        # 'L1Prefire': ['L1PreFiringWeight_Nom','L1PreFiringWeight_Dn','L1PreFiringWeight_Up'],
        'PU' : ['Pileup_SF','Pileup_SFDn','Pileup_SFUp'],
        # 'MuSF': ['lepMu_SF','lepMu_SFDn','lepMu_SFUp'],
        # 'EleSF': ['lepEle_SF','lepEle_SFDn','lepEle_SFUp'],
        # 'BTag_b_correlated': ['btagSF_fixedWP_SF','btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFDn',
        #                       'btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFUp'],
        # 'BTag_b_uncorrelated_<ERA>': ['btagSF_fixedWP_SF', 'btagSF_fixedWP_SF__CMS_eff_b_uncorrelated_<ERA>_heavy_SFDn',
        #                               'btagSF_fixedWP_SF__CMS_eff_b_uncorrelated_<ERA>_heavy_SFUp'],
        # 'BTag_l': ['btagSF_fixedWP_SF', 'btagSF_fixedWP_SF__CMS_eff_b_light_SFDn', 'btagSF_fixedWP_SF__CMS_eff_b_light_SFUp'],
    }
}

# TODO: add kinematic variation histograms from files, including era-decorrelated

# later will be converted into a loop
era = '2018'
sample = 'TTLep_pow'

nominal_sample = getattr(samples, f'{sample}_{era}_nominal')

# if selection:
#     requested_branches_for_selection = get_branches_for_selection(selection)
#     nominal_sample.addSelection(selection, requested_branches_for_selection)
     
if feature_name not in nominal_sample.feature_names:
    print(f'[warning] feature {feature_name} not in base RDataLoader, attempting to load with setFeatures')
    nominal_sample.setFeatures([feature_name])

nominal_features, nominal_weights = nominal_sample.materialize(0, what='fw', feature_names = [feature_name])

# nominal_features is an array of shape (n_events, n_features)
# even when just one feature is requested
nominal_feature_values = nominal_features[:,0]

nominal_hist = np.histogram(a=nominal_feature_values, bins=edges, weights=nominal_weights)

for group, uncertainty_names in replace_weight_groups.items():
    print(f"[info] group: {group}, uncertainties: {uncertainty_names.keys()}")
    for uncertainty_name, branch_names in uncertainty_names.items():

        print(f"{uncertainty_name}, {branch_names}")

        # using structure defined in replace_weight_groups
        nominal_weight_name = branch_names[0]
        down_var_weight_name = branch_names[1]
        up_var_weight_name = branch_names[2]

        if '<ERA>' in uncertainty_name:
            uncertainty_name = uncertainty_name.replace('<ERA>',str(era))
            nominal_weight_name = nominal_weight_name.replace('<ERA>',str(era))
            down_var_weight_name = down_var_weight_name.replace('<ERA>',str(era))
            up_var_weight_name = up_var_weight_name.replace('<ERA>',str(era))

        list_weights = nominal_sample.weight_branches
        print(list_weights)
        weight_to_remove_idx = list_weights.index(nominal_weight_name)
        
        list_weights[weight_to_remove_idx] = down_var_weight_name
        print(list_weights)
        down_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights)
        down_var_weights = down_var_sample.materialize(0, what='w')[0]

        down_var_hist = np.histogram(a=nominal_feature_values, bins=edges, weights=down_var_weights)
        
        list_weights[weight_to_remove_idx] = up_var_weight_name
        print(list_weights)
        up_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights)
        up_var_weights = up_var_sample.materialize(0, what='w')[0]
        
        up_var_hist = np.histogram(a=nominal_feature_values, bins=edges, weights=up_var_weights)

        # TODO: add the actual plotting in ROOT
        individual_plot_dir = os.path.join(plot_directory,era,sample)
        fig, axes = plt.subplots(nrows=2)
        mh.histplot(
            [nominal_hist, down_var_hist, up_var_hist],
            histtype='step',
            color = ['C0','C1','C2'],
            alpha=0.7,
            label=['Nominal', 'Down', 'Up'],
            ax=axes[0]
        )
        axes[0].legend(loc='upper right')
        mh.yscale_legend(soft_fail=True)
        axes[0].set_yscale('log')
        plt.savefig(f'{individual_plot_dir}/{uncertainty_name}.png')

syncer.sync()