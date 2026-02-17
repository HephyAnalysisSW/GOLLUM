"""
Script to create pre-fit variation plots directly from raw ±1σ input files.

Generates two types of plots:
1. Binned template comparisons: nominal vs up/down variations for each uncertainty
2. Pre-fit stacks: combined samples with uncertainty bands, summed in quadrature

Arguments:
    --selection, -s: String-based selection in Python/Awkward format (e.g. use & instead of &&)
    --version, -v: Version tag to append to output directories
    --debug, -d: Enable debug mode (uses only 2016/2016APV era, one uncertainty per type, two features)
"""

import os, sys
import gc # explicit garbage collection to avoid ridiculous (+100GB) memory usage
import itertools
import re
import argparse as ap
from typing import Sequence, Optional
import time
import logging

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)

import data.samples_RunII as samples
from data.plot_options import plot_options

import common.user as user
import common.syncer as syncer
import common.helpers as helpers

import numpy as np

import copy

# NB: this is a first prototype and hasn't been fully battle tested
def get_branches_for_selection(selection: str) -> Sequence[str]:
    
    # NB: the order matters, operators with more chars should be removed first
    comparison_operators = ["==","!=",">=","<=",">","<","&","|","(",")"]

    string_no_comparisons = selection
    for operator in comparison_operators:
        
        # adding space to avoid variables sticking together in situations like "x>y" -> "xy"
        string_no_comparisons = string_no_comparisons.replace(operator," ")
    
    branches_for_selection = []
    for string in string_no_comparisons.split():

        if string.isdigit() or (string in branches_for_selection):
            continue
        
        # to take into account cuts with abs()
        # should only require branches inside abs()
        if 'abs' in string:
            continue

        branches_for_selection.append(string)

    return branches_for_selection

"""
systematics are divided into three categories:
- variations in weights which are part of overall weight - should replace nominal weight
- additional weights, to be multiplied by the overall weight
- kinematic variations - come from different files

the RDataLoader access will be different for each
"""

"""
variations in weights which are part of overall weight in nominal sample.
should replace nominal weight by down or up variations

structure:

{group name: 
    variation name: [nominal branch name, down variation branch name, up variation branch name]
}
"""
replace_weight_groups = {
    'EXPERIMENTAL': {
        'L1Prefire': ['L1PreFiringWeight_Nom','L1PreFiringWeight_Dn','L1PreFiringWeight_Up'],
        'PU' : ['Pileup_SF','Pileup_SFDn','Pileup_SFUp'],
        'MuSF': ['lepMu_SF','lepMu_SFDn','lepMu_SFUp'],
        'EleSF': ['lepEle_SF','lepEle_SFDn','lepEle_SFUp'],
        'BTag_b_correlated': ['btagSF_fixedWP_SF','btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFDn',
                              'btagSF_fixedWP_SF__CMS_eff_b_correlated_heavy_SFUp'],
        # TODO: add this line when the quantities are added back into make_ntuple
        # 'BTag_b_uncorrelated_<ERA>': ['btagSF_fixedWP_SF', 'btagSF_fixedWP_SF__CMS_eff_b_uncorrelated_<ERA>_heavy_SFDn',
        #                               'btagSF_fixedWP_SF__CMS_eff_b_uncorrelated_<ERA>_heavy_SFUp'],
        'BTag_l': ['btagSF_fixedWP_SF', 'btagSF_fixedWP_SF__CMS_eff_b_light_SFDn', 'btagSF_fixedWP_SF__CMS_eff_b_light_SFUp'],
    }
}


"""
additional weight to be multiplied by the overall weight of nominal sample. structure:

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
file directories and names follow the structure {era}/{sample}_{tag}
different sample with kinematic variations. structure:

{group name: 
    variation name: [down variation file tag, up variation file tag]
}

"""
kinematic_variation_groups = {
        'JER': {
            'CMS_res_j_0_<ERA>': ['CMS_res_j_0_<ERA>_down', 'CMS_res_j_0_<ERA>_up'],
            'CMS_res_j_1_<ERA>': ['CMS_res_j_1_<ERA>_down', 'CMS_res_j_1_<ERA>_up'],
            'CMS_res_j_2_<ERA>': ['CMS_res_j_2_<ERA>_down', 'CMS_res_j_2_<ERA>_up'],
            'CMS_res_j_3_<ERA>': ['CMS_res_j_3_<ERA>_down', 'CMS_res_j_3_<ERA>_up'],
            'CMS_res_j_4_<ERA>': ['CMS_res_j_4_<ERA>_down', 'CMS_res_j_4_<ERA>_up'],
            'CMS_res_j_5_<ERA>': ['CMS_res_j_5_<ERA>_down', 'CMS_res_j_5_<ERA>_up'],
        },
        'JES1': {
            'CMS_scale_j_FlavorPureBottom': ['CMS_scale_j_FlavorPureBottom_down', 'CMS_scale_j_FlavorPureBottom_up'],
            'CMS_scale_j_FlavorPureCharm': ['CMS_scale_j_FlavorPureCharm_down', 'CMS_scale_j_FlavorPureCharm_up'],
            'CMS_scale_j_FlavorPureGluon': ['CMS_scale_j_FlavorPureGluon_down', 'CMS_scale_j_FlavorPureGluon_up'],
            'CMS_scale_j_FlavorPureQuark': ['CMS_scale_j_FlavorPureQuark_down', 'CMS_scale_j_FlavorPureQuark_up'],
        },
        'JES2': {
            'CMS_scale_j_Regrouped_Absolute': ['CMS_scale_j_Regrouped_Absolute_down', 'CMS_scale_j_Regrouped_Absolute_up'],
            'CMS_scale_j_Regrouped_Absolute_<ERA>': ['CMS_scale_j_Regrouped_Absolute_<ERA>_down', 'CMS_scale_j_Regrouped_Absolute_<ERA>_up'],
            'CMS_scale_j_Regrouped_BBEC1': ['CMS_scale_j_Regrouped_BBEC1_down', 'CMS_scale_j_Regrouped_BBEC1_up'],
            'CMS_scale_j_Regrouped_BBEC1_<ERA>': ['CMS_scale_j_Regrouped_BBEC1_<ERA>_down', 'CMS_scale_j_Regrouped_BBEC1_<ERA>_up'],
            'CMS_scale_j_Regrouped_EC2': ['CMS_scale_j_Regrouped_EC2_down', 'CMS_scale_j_Regrouped_EC2_up'],
            'CMS_scale_j_Regrouped_EC2_<ERA>': ['CMS_scale_j_Regrouped_EC2_<ERA>_down', 'CMS_scale_j_Regrouped_EC2_<ERA>_up'],
        },
        'JES3': {
            'CMS_scale_j_Regrouped_HF': ['CMS_scale_j_Regrouped_HF_down','CMS_scale_j_Regrouped_HF_up'],
            'CMS_scale_j_Regrouped_RelativeBal': ['CMS_scale_j_Regrouped_RelativeBal_down','CMS_scale_j_Regrouped_RelativeBal_up'],
            'CMS_scale_j_Regrouped_RelativeSample_<ERA>': ['CMS_scale_j_Regrouped_RelativeSample_<ERA>_down','CMS_scale_j_Regrouped_RelativeSample_<ERA>_up'],
            'Uncl': ['Uncl_down', 'Uncl_up'],
        },
}

# returns bin edges in Numpy format (low edges + upper edge of last bin)
def get_bin_edges(feature_name: str, print_add_info = False) -> Optional[Sequence[float]]:
    
    if feature_name in plot_options:
        
        # prioritize set of bin edges for pre-fit plots
        # to have plots similar to TOP-20-006
        if 'bin_edges' in plot_options[feature_name]:
            # gets thresholds directly from plot_options
            edges = plot_options[feature_name]['bin_edges']
            if print_add_info:
                logger.info(f"Using bin edges from plot_options for '{feature_name}': {edges}")
        
        elif 'binning' in plot_options[feature_name]:
            # build thresholds from plot_options: [nBins, low, high]
            nBins, low, high = plot_options[feature_name]['binning']
            edges = [low + i*(high - low) / nBins for i in range(nBins + 1)]
            if print_add_info:
                logger.info(f"Using equidistant binning from plot_options for '{feature_name}': "
                f"nBins={nBins}, low={low}, high={high} -> {len(edges)-1} bins.")
        
        return edges
    
    else:
        return None
    
if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = ap.ArgumentParser(description='make plots of pre-fit variations directly from raw +-1 sigma input files, instead of using ICPH surrogates')
    parser.add_argument('--selection', '-s', type=str, help='string-based selection in Python/Awkward format, e.g. use & instead of && for chaining selections')
    parser.add_argument('--version', '-v', help = 'version, to be added to the relevant directories')
    parser.add_argument('--debug', '-d', help = 'enables debug mode, keeping only era and one uncertainty of each type', action="store_true")

    args = parser.parse_args()

    samples_to_use = ['TTLep_pow', 'TTSemi_pow', 'SingleTop', 'DrellYan']
    eras = samples.ERAS

    selection = args.selection
    debug = args.debug
    version = args.version

    if debug:
        logger.info("Debug mode: printing additional information, plotting 2016 and 2016APV only, only using one of each type of uncertainty and only plotting two features.")
        logger.setLevel(logging.DEBUG)
        eras = ["2016","2016APV"]
        replace_weight_groups = {'EXPERIMENTAL': {'L1Prefire': ['L1PreFiringWeight_Nom','L1PreFiringWeight_Dn','L1PreFiringWeight_Up']}}
        add_weight_groups = {'MODELING': {'mu_ren': ['scale_ren0p5_fac1p0','scale_ren2p0_fac1p0']}}
        kinematic_variation_groups = {'JER': {'CMS_res_j_0_<ERA>': ['CMS_res_j_0_<ERA>_down', 'CMS_res_j_0_<ERA>_up']},
                                      'JES2': {'CMS_scale_j_Regrouped_Absolute_<ERA>': ['CMS_scale_j_Regrouped_Absolute_<ERA>_down', 'CMS_scale_j_Regrouped_Absolute_<ERA>_up']}}

    if selection:
        if "abs" in selection and not (("np.abs" in selection) or ("ak.abs" not in selection)):
            logger.warning("selection requires absolute values using abs, replacing by np.abs")
            selection.replace("abs","np.abs")
            
        requested_branches_for_selection = get_branches_for_selection(selection)
        logger.info(f"{selection=}, {requested_branches_for_selection=}")

    if version:
        logger.info(f"Writing plots under sub-directories with name {version}")

    """
    stores histograms of nominal, up and down variations
    for each era, sample, feature, uncertainty set
    
    used to separate creation of RDataLoaders,
    fetching of templates for each uncertainty
    from creation of pre-fit binned template comparisons and stacks
    
    structure of dictionary:
    {
        era: {
            sample: {
                feature: {
                    uncertainty: [nominal, down variation template, up variation templates]
                }
            }
        }
    }
    """

    # initializing here to avoid adding another layer of for loop separating era and sample
    histogram_dict = {}
    for era in eras:
        histogram_dict[era] = {}
        for sample in samples_to_use:
            histogram_dict[era][sample] = {}

    # used to have a simple way to iterate over all the features
    # once the nominal sample object is deleted
    list_features = []

    for i_era_sample, (era, sample) in enumerate(itertools.product(eras, samples_to_use)):
        print(f"era: {era}, sample: {sample}")
        
        nominal_sample = getattr(samples, f'{sample}_{era}_nominal')
        
        # initializating dict here to avoid bugs
        for i_feature, feature_name in enumerate(nominal_sample.feature_names):

            if debug and i_feature > 1:
                continue
            
            histogram_dict[era][sample][feature_name] = {}

            # simple way to let us know the binning only once and not pollute the terminal
            if i_era_sample==0:
                get_bin_edges(feature_name, print_add_info=True)
                list_features.append(feature_name)

        t0 = time.perf_counter()

        if selection:
            nominal_sample.addSelection(selection, requested_branches_for_selection)

        dt0 = time.perf_counter() - t0
        logger.debug(f"time to add selections on nominal sample: {dt0:.6f} s")

        t1 = time.perf_counter()
        # this could be replaced by opening the files, and issuing a TTree->Draw command (optimized)
        nominal_feature_values, nominal_weights = nominal_sample.materialize(0, what='fw')

        dt1 = time.perf_counter() - t1
        logger.debug(f"time to materialize nominal sample: {dt1:.6f} s")

        t2 = time.perf_counter()

        # list of weights in nominal sample
        weight_names = nominal_sample.weight_branches

        """
        To take into account the three types of handling systematic variations
        at the RDataLoader level
        """
        for mode in ["replace", "add", "kinematics"]:

            if mode == "replace":
                syst_groups = replace_weight_groups
                if i_era_sample == 0:
                    logger.info("plotting variations for scale factors which are part of the overall weight, " \
                "e.g. pileup reweighting scale factor uncertainty")
            elif mode == "add":
                syst_groups = add_weight_groups
                if i_era_sample == 0:
                    logger.info("plotting variations implemented as scale factors which multiply the overall event weight, " \
                "e.g. QCD scale variations")
            elif mode == "kinematics":
                syst_groups = kinematic_variation_groups
                if i_era_sample == 0:
                    logger.info("plotting variations from varied kinematics (JME)")

            for group, uncertainty_names in syst_groups.items():
                if i_era_sample == 0:
                    logger.info(f"group: {group}, uncertainties: {uncertainty_names.keys()}")

                for uncertainty_name, branch_names in uncertainty_names.items():

                    if i_era_sample == 0:
                        logger.info(f"uncertainty name: {uncertainty_name}, branch names: {branch_names}")

                    # logic different for each type of systematic variations
                    if mode == "replace":
                
                        """
                        For variations, cloning sample replacing nominal weight branch with varied one,
                        since one can only change the weight branches at RDataLoader creation.
                        """
                        treplace = time.perf_counter()
                        # using structure defined in replace_weight_groups
                        nominal_weight_name = branch_names[0]
                        down_var_weight_name = branch_names[1]
                        up_var_weight_name = branch_names[2]

                        if '<ERA>' in uncertainty_name:
                            uncertainty_name = uncertainty_name.replace('<ERA>',str(era))
                            down_var_weight_name = down_var_weight_name.replace('<ERA>',str(era))
                            up_var_weight_name = up_var_weight_name.replace('<ERA>',str(era))

                        weight_names_varied = weight_names
                        weight_to_remove_idx = weight_names_varied.index(nominal_weight_name)
                        
                        weight_names_varied[weight_to_remove_idx] = down_var_weight_name
                        down_var_sample = nominal_sample.clone_from_files(nominal_sample.files, weight_names_varied)

                        weight_names_varied[weight_to_remove_idx] = up_var_weight_name
                        up_var_sample = nominal_sample.clone_from_files(nominal_sample.files, weight_names_varied)

                        # fixes cases where there are two uncertainties with the same
                        # nominal weight name (e.g. b-tag SF)
                        weight_names_varied[weight_to_remove_idx] = nominal_weight_name

                        dtreplace_clone_files = time.perf_counter() - treplace
                        logger.debug(f"time to edit era strings and clone files for replace-type alternative sample: {dtreplace_clone_files:.6f} s")

                        down_var_weights = down_var_sample.materialize(0, what='w')[0]
                        up_var_weights = up_var_sample.materialize(0, what='w')[0]

                        # doing this to have a generic assignment outside of "if" condition
                        down_var_feature_values = nominal_feature_values
                        up_var_feature_values = nominal_feature_values
                        
                        dtreplace = time.perf_counter() - treplace - dtreplace_clone_files
                        logger.debug(f"time to materialize weights from replace-type alternative samples: {dtreplace:.6f} s")
                    
                    elif mode == "add":
                
                        tadd = time.perf_counter()
                        """
                        For variations, cloning nominal sample add branch for variation varied one,
                        since one can only change the weight branches at RDataLoader creation.
                        """

                        # using structure defined in add_weight_groups
                        down_var_weight_name = branch_names[0]
                        up_var_weight_name = branch_names[1]

                        if '<ERA>' in uncertainty_name:
                            uncertainty_name = uncertainty_name.replace('<ERA>',str(era))
                            down_var_weight_name = down_var_weight_name.replace('<ERA>',str(era))
                            up_var_weight_name = up_var_weight_name.replace('<ERA>',str(era))                        
                        
                        down_var_sample = nominal_sample.clone_from_files(nominal_sample.files, weight_names+[down_var_weight_name])
                        up_var_sample = nominal_sample.clone_from_files(nominal_sample.files, weight_names+[up_var_weight_name])

                        dtadd_clone_files = time.perf_counter() - tadd
                        logger.debug(f"time to edit era strings and clone files for add-type alternative sample: {dtadd_clone_files:.6f} s")

                        down_var_weights = down_var_sample.materialize(0, what='w')[0]
                        down_var_feature_values = nominal_feature_values
                        
                        up_var_weights = up_var_sample.materialize(0, what='w')[0]
                        up_var_feature_values = nominal_feature_values

                        dtadd = time.perf_counter() - tadd - dtadd_clone_files
                        logger.debug(f"time to materialize weights from add-type alternative samples: {dtadd:.6f} s")

                    elif mode == "kinematics":
                        
                        tkin = time.perf_counter()
                        """
                        For kinematic variations, cloning from different files, keeping the same branch names,
                        since one can only change the weight branches at RDataLoader creation.
                        """

                        # using structure defined in add_weight_groups
                        down_var_file_tag = branch_names[0]
                        up_var_file_tag = branch_names[1]

                        if '<ERA>' in uncertainty_name:
                            uncertainty_name = uncertainty_name.replace('<ERA>',str(era))
                            down_var_file_tag = down_var_file_tag.replace('<ERA>',str(era))
                            up_var_file_tag = up_var_file_tag.replace('<ERA>',str(era))

                        # all era-decorrelated JES variations consider 2016 and 2016APV as single 2016 era
                        # the opposite is true for JER variations
                        if 'CMS_res_j' not in uncertainty_name:
                            # uncertainty_name = uncertainty_name.replace("APV","")
                            down_var_file_tag = down_var_file_tag.replace("APV","")
                            up_var_file_tag = up_var_file_tag.replace("APV","")

                        down_var_sample = getattr(samples, f'{sample}_{era}_{down_var_file_tag}')
                        up_var_sample = getattr(samples, f'{sample}_{era}_{up_var_file_tag}')

                        dtkin_get_files = time.perf_counter() - tkin
                        logger.debug(f"time to edit era strings and fetch files for samples with kinematic variations: {dtkin_get_files:.6f} s")

                        down_var_feature_values, down_var_weights = down_var_sample.materialize(0, what='fw')
                        # parsing output of materialize into a simple array
                        # also done when loading the nominal sample                        
                        up_var_feature_values, up_var_weights = up_var_sample.materialize(0, what='fw')                        

                        dtkin = time.perf_counter() - tkin - dtkin_get_files
                        logger.debug(f"time to materialize kinematic-type alternative samples: {dtkin:.6f} s")

                        # tkin_clean = time.perf_counter()
                        
                        # del down_var_feature_values, up_var_feature_values
                        # gc.collect()
                        
                        # dtkin_clean = time.perf_counter() - tkin_clean
                        # logger.debug(f"time to clean feature values from kinematic-type alternative sample: {dtkin_clean:.6f} s")


                    for i_feature, feature_name in enumerate(nominal_sample.feature_names):

                        if debug and i_feature > 1:
                            continue

                        edges = get_bin_edges(feature_name)
                        
                        if not edges:
                            if i_era_sample == 0:
                                logger.warning(f"feature {feature_name} does not have default binning in plot_options, skipping histogram filling")
                            continue

                        if i_era_sample == 0:
                            logger.debug(f"{feature_name=}, {edges=}")
                        
                        nominal_hist_entries = np.histogram(a=nominal_feature_values[:,i_feature], bins=edges, weights=nominal_weights)[0]
                        
                        down_var_hist_entries = np.histogram(a=down_var_feature_values[:,i_feature], bins=edges, weights=down_var_weights)[0]
                        up_var_hist_entries = np.histogram(a=up_var_feature_values[:,i_feature], bins=edges, weights=up_var_weights)[0]

                        # converting Numpy histogram edges (low edges + upper edge of last bin)
                        # to ROOT binning (lower edges only), bin 0 is underflow bin, bin nbins+1 is overflow
                        n_bins = len(edges)-1

                        # # central histogram with variable binning, created only once
                        h_central_name = f"h_central_{era}_{sample}_{uncertainty_name}_{feature_name}"
                        h_central = ROOT.TH1F(h_central_name, "", n_bins, np.array(edges, dtype=float))
                        for i in range(n_bins):
                            h_central.SetBinContent(i + 1, nominal_hist_entries[i])

                        h_down_name = f"h_{era}_{sample}_{uncertainty_name}_{feature_name}_Down"
                        h_down = ROOT.TH1F(h_down_name, "", n_bins, np.array(edges, dtype=float))
                        for i in range(n_bins):
                            h_down.SetBinContent(i + 1, down_var_hist_entries[i])

                        h_up_name = f"h_{era}_{sample}_{uncertainty_name}_{feature_name}_Up"
                        h_up = ROOT.TH1F(h_up_name, "", n_bins, np.array(edges, dtype=float))
                        for i in range(n_bins):
                            h_up.SetBinContent(i + 1, up_var_hist_entries[i])

                        histogram_dict[era][sample][feature_name][uncertainty_name] = []

                        histogram_dict[era][sample][feature_name][uncertainty_name].append(h_central)
                        histogram_dict[era][sample][feature_name][uncertainty_name].append(h_down)
                        histogram_dict[era][sample][feature_name][uncertainty_name].append(h_up)
                        

                    # t_clean = time.perf_counter()

                    # del down_var_feature_values, up_var_feature_values
                    # del down_var_weights, up_var_weights
                    down_var_sample._arr_cache.clear()
                    up_var_sample._arr_cache.clear()
                    del down_var_sample, up_var_sample

                    gc.collect()
                    
                    # dt_clean = time.perf_counter() - t_clean
                    # logger.debug(f"time to do general memory clean from any alternative sample: {dt_clean:.6f} s")


                # t_root_compare = time.perf_counter()

                # dt_root_compare = time.perf_counter() - t_root_compare
                # logger.debug(f"time to make the ROOT plot comparison: {dt_root_compare:.6f} s")

        nominal_sample._arr_cache.clear()
        del nominal_sample, nominal_feature_values, nominal_weights
        gc.collect()

        if i_era_sample == 0:
            logger.info("after first sample/era combination, no longer printing variable/branch information")


    logger.debug(histogram_dict)




    """
    ### Binned template comparisons ###

    making binned template input comparisons from nominal, down and up variation templates from previous step

    remember the structure of histogram_dict:
        histogram_dict[era][sample][feature_name][uncertainty_name] = [h_nominal, h_down, h_up]
    
    """
    # legend columns (configurable) - for binned template plots
    legend_columns_binned_templates = 3

    plot_directory_binned_templates = os.path.join(user.plot_directory, 'binned_templates_from_inputs')
    if debug:
        plot_directory_binned_templates = os.path.join(plot_directory_binned_templates,"debug")
    elif version:
        plot_directory_binned_templates = os.path.join(plot_directory_binned_templates,version)

    logger.info(f"Creating binned templates from sample inputs, will be written under {plot_directory_binned_templates}")
    os.makedirs(plot_directory_binned_templates, exist_ok=True)

    # simple color palette
    binned_template_colors = [
        ROOT.kRed + 1,
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 1,
        ROOT.kCyan + 1,
    ]

    # creating a single dictionary with all the uncertainty groups
    uncertainty_groups = replace_weight_groups
    uncertainty_groups.update(add_weight_groups)
    uncertainty_groups.update(kinematic_variation_groups)

    for i_era_sample, (era, sample) in enumerate(itertools.product(eras, samples_to_use)):

        individual_plot_dir = os.path.join(plot_directory_binned_templates,era,sample)
        os.makedirs(individual_plot_dir, exist_ok=True)
        helpers.copyIndexPHP(individual_plot_dir)
            
        if i_era_sample == 0:
            logger.info(f"{era=}, {sample=}")

        for feature_name in list_features:

            if i_era_sample == 0:
                logger.info(f"{feature_name=}")

            if feature_name not in plot_options:
                if i_era_sample == 0:
                    logger.warning(f"feature {feature_name} not in plot_options, skipping histogram")
                continue

            logY = plot_options.get(feature_name, {}).get('logY', False)
            x_title = plot_options.get(feature_name, {}).get('tex', feature_name)
            
            for group, uncertainties in uncertainty_groups.items():
                
                canvas_name = f"{group}_{feature_name}_{era}_{sample}"
                # stretch in y
                c = ROOT.TCanvas(canvas_name, canvas_name, 800, 900)

                # three pads: legend (top), yields (middle), ratios (bottom)
                padLegend = ROOT.TPad(canvas_name + "_legend", canvas_name + "_legend", 0.0, 0.80, 1.0, 1.0)
                padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 0.80)
                padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

                padLegend.SetBottomMargin(0.05)
                padLegend.SetTopMargin(0.10)
                padLegend.SetLeftMargin(0.10)
                padLegend.SetRightMargin(0.10)
                padLegend.SetFillStyle(0)

                padTop.SetBottomMargin(0.)
                padTop.SetTopMargin(0.08)
                padTop.SetLeftMargin(0.10)
                padTop.SetRightMargin(0.05)

                padBottom.SetTopMargin(0.0)
                padBottom.SetBottomMargin(0.30)
                padBottom.SetLeftMargin(0.10)
                padBottom.SetRightMargin(0.05)

                padLegend.Draw()
                padTop.Draw()
                padBottom.Draw()

                # legend (created here, drawn later in padLegend)
                legend = ROOT.TLegend(0.02, 0.10, 0.98, 0.90)
                legend.SetBorderSize(0)
                legend.SetFillStyle(0)
                legend.SetNColumns(legend_columns_binned_templates)

                # ------------- TOP PAD: absolute yields -------------
                padTop.cd()
                padTop.SetTicks(1, 1)
                if logY:
                    padTop.SetLogy(True)

                # collect all histograms to draw in the top pad, format them and draw them

                first_uncertainty_in_group = list(uncertainties.keys())[0]
                if '<ERA>' in first_uncertainty_in_group:
                    first_uncertainty_in_group = first_uncertainty_in_group.replace('<ERA>',str(era))
                
                # collect and format central histogram
                h_central = histogram_dict[era][sample][feature_name][first_uncertainty_in_group][0]
                h_central.SetLineColor(ROOT.kBlack)
                h_central.SetLineWidth(2)
                # # no title on the top pad
                h_central.SetTitle("")
                h_central.GetXaxis().SetTitle(x_title)
                h_central.GetYaxis().SetTitle("Events")
                h_central.GetYaxis().SetTitleSize(0.06)
                h_central.GetYaxis().SetLabelSize(0.045)
                # # x label on bottom pad only
                h_central.GetXaxis().SetLabelSize(0)
                h_central.GetXaxis().SetTitleSize(0)

                legend.AddEntry(h_central, "Nominal", "l")

                # storing nominal and variation histograms to draw in same canvas
                # since these will be looped over several times
                h_variations = [h_central]

                color_index = 0
                for uncertainty_name in uncertainties:

                    if '<ERA>' in uncertainty_name:
                        uncertainty_name = uncertainty_name.replace('<ERA>',str(era))

                    color = binned_template_colors[color_index % len(binned_template_colors)]

                    h_down = histogram_dict[era][sample][feature_name][uncertainty_name][1]
                    h_down.SetLineColor(color)
                    h_down.SetLineStyle(ROOT.kDashed)
                    h_down.SetLineWidth(1)
                    
                    h_up = histogram_dict[era][sample][feature_name][uncertainty_name][2]
                    h_up.SetLineColor(color)
                    h_up.SetLineStyle(ROOT.kSolid)
                    h_up.SetLineWidth(1)
                    
                    # remove unnecessary "CMS_" tag from name
                    uncertainty_name_for_plot = uncertainty_name.replace("CMS_","") 
                    legend.AddEntry(h_down, f"{uncertainty_name_for_plot} -1#sigma","l")
                    legend.AddEntry(h_up, f"{uncertainty_name_for_plot} +1#sigma","l")

                    h_variations.append(h_down)
                    h_variations.append(h_up)

                    color_index += 1

                # getting reasonable y range (absolute yields)
                max_y = max(h.GetMaximum() for h in h_variations)
                if logY:
                    h_central.SetMinimum(0.8)
                    h_central.SetMaximum(1.2 * max_y if max_y > 0 else 1.0)
                else:
                    h_central.SetMinimum(0.0)
                    h_central.SetMaximum(1.2 * max_y if max_y > 0 else 1.0)                    

                h_central.Draw("HIST")

                for h in h_variations[1:]:
                    h.Draw("HIST SAME")

                # ------------- BOTTOM PAD: ratios -------------
                padBottom.cd()
                padBottom.SetTicks(1, 1)

                # ratio central
                ratio_central_name = h_central.GetName() + "_ratio"
                h_ratio_central = h_central.Clone(ratio_central_name)
                h_ratio_central.SetDirectory(0)
                h_ratio_central.Divide(h_central)
                h_ratio_central.SetLineColor(ROOT.kBlack)
                h_ratio_central.SetLineWidth(2)
                h_ratio_central.SetTitle("")

                h_ratio_central.GetYaxis().SetTitle("var / nominal")
                h_ratio_central.GetYaxis().SetNdivisions(505)
                h_ratio_central.GetYaxis().SetTitleSize(0.09)
                h_ratio_central.GetYaxis().SetTitleOffset(0.5)
                h_ratio_central.GetYaxis().SetLabelSize(0.08)

                h_ratio_central.GetXaxis().SetTitle(x_title)
                h_ratio_central.GetXaxis().SetTitleSize(0.1)
                h_ratio_central.GetXaxis().SetLabelSize(0.08)

                # build ratio histos for variations
                h_ratio_vars = [h_ratio_central]
                for h in h_variations[1:]:
                    r_name = h.GetName() + "_ratio"
                    h_r = h.Clone(r_name)
                    h_r.SetDirectory(0)
                    h_r.Divide(h_central)
                    h_ratio_vars.append(h_r)

                # ratio y-range based on max relative deviation from 1
                max_dev = 0.0
                for h in h_ratio_vars:
                    for i in range(1, n_bins+1):
                        val = h.GetBinContent(i)
                        if val != 0:
                            dev = abs(val - 1.0)
                            if dev > max_dev:
                                max_dev = dev

                if max_dev <= 0.0:
                    r_min, r_max = 0.9, 1.1
                else:
                    # 30% larger than max deviation, symmetric around 1
                    half_range = 1.3 * max_dev
                    r_min = 1.0 - half_range
                    r_max = 1.0 + half_range

                h_ratio_central.SetMinimum(max(r_min,-2.0))
                h_ratio_central.SetMaximum(min(r_max,2.0))

                h_ratio_central.Draw("HIST")
                for h_r in h_ratio_vars[1:]:
                    h_r.Draw("HIST SAME")

                # line at 1
                line = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
                line.SetLineStyle(ROOT.kDashed)
                line.SetLineColor(ROOT.kBlack)
                line.Draw("SAME")

                # ------------- LEGEND PAD -------------
                padLegend.cd()
                # no frame, no axes, just the legend
                legend.Draw()

                ## Saving histograms
                c.cd()
                c.Update()

                out_png = os.path.join(individual_plot_dir, canvas_name + ".png")
                out_pdf = os.path.join(individual_plot_dir, canvas_name + ".pdf")
                c.SaveAs(out_png)
                c.SaveAs(out_pdf)

    
    """
    making pre-fit stacks from nominal, up and down variation templates from previous step

    sums variations for each sample in quadrature separately for up and down variations
    sets symmetric uncertainty in each bin of combined histogram as max(up, down) for that bins

    follows very closely the structure in plot.py

    remember the structure of histogram_dict:
        histogram_dict[era][sample][feature_name][uncertainty_name] = [h_nominal, h_down, h_up]
    """

    plot_directory_stacks = os.path.join(user.plot_directory, 'prefit_stacks_from_inputs')
    if debug:
        plot_directory_stacks = os.path.join(plot_directory_stacks,"debug")
    elif version:
        plot_directory_stacks = os.path.join(plot_directory_stacks,version)        

    # logger.info(f"Pre-fit stacks saved to folder {plot_directory_stacks}")
    from data.colors import get_color

    legend_columns_prefit_stacks = 2

    for era in eras:
        
        for feature_name in list_features:

            edges = get_bin_edges(feature_name)
            
            # this situation will likely never happen
            # but keeping the safety here
            if not edges:
                continue

            logY = plot_options.get(feature_name, {}).get('logY', False)
            x_title = plot_options.get(feature_name, {}).get('tex', feature_name)        

            # converting Numpy histogram edges (low edges + upper edge of last bin)
            # to ROOT binning (lower edges only), bin 0 is underflow bin, bin nbins+1 is overflow
            n_bins = len(edges)-1

            # sample_histogram_dict = histogram_dict[era]

            # build total histogram and each of the variations
            h_total = ROOT.TH1F(f"h_prefit_total_{era}","", n_bins, np.array(edges, dtype=float))
            h_unc_down = ROOT.TH1F(f"h_prefit_unc_down_{era}","", n_bins, np.array(edges, dtype=float))
            h_unc_up = ROOT.TH1F(f"h_prefit_unc_up_{era}","", n_bins, np.array(edges, dtype=float))
            
            # nominal histogram is the same for all the different uncertainties
            # convoluted way to fetch the nominal histogram but it works
            first_uncertainty = list(histogram_dict[era][samples_to_use[0]][feature_name].keys())[0]
            if '<ERA>' in first_uncertainty:
                first_uncertainty = first_uncertainty.replace('<ERA>',str(era))
            
            for sample in samples_to_use:
                
                h_nominal = histogram_dict[era][sample][feature_name][first_uncertainty][0]
                h_total.Add(h_nominal)
                
                uncertainty_histogram_dict = histogram_dict[era][sample][feature_name]

                for i_uncertainty, (uncertainty, histograms) in enumerate(uncertainty_histogram_dict.items()):
                    
                    # creating histograms with uncertainties
                    # summing in quadrature each of the variations
                    down_variation_histo = histograms[1]
                        
                    h_var_down = down_variation_histo - h_nominal
                    h_var_down.Multiply(h_var_down)
                    h_unc_down.Add(h_var_down)

                    up_variation_histo = histograms[2]

                    h_var_up = up_variation_histo - h_nominal
                    h_var_up.Multiply(h_var_up)
                    h_unc_up.Add(h_var_up)

            # uncertainty band via TBoxes + line-only histos
            uncertainty_boxes = []

            h_line_up   = h_total.Clone(f"h_prefit_up_{era}")
            h_line_down = h_total.Clone(f"h_prefit_down_{era}")

            # lines only, no fill
            h_line_up.SetFillStyle(0)
            h_line_up.SetFillColor(0)
            h_line_down.SetFillStyle(0)
            h_line_down.SetFillColor(0)
            h_line_up.SetLineColor(ROOT.kGray + 2)
            h_line_down.SetLineColor(ROOT.kGray + 2)
            h_line_up.SetLineWidth(1)
            h_line_down.SetLineWidth(1)

            logger.debug(f"Creating uncertainty boxes.")

            for ib in range(1, n_bins+1):
                x1 = edges[ib-1]
                x2 = edges[ib]
                nom = h_total.GetBinContent(ib)
                err_up = np.sqrt(h_unc_up.GetBinContent(ib))
                err_down = np.sqrt(h_unc_down.GetBinContent(ib))

                if nom > 0.0:
                    y_low  = nom - err_down
                    y_high = nom + err_up
                    logger.debug(f"{ib=}, {nom=}, {y_low=}, {y_high=}, {(y_low/nom)=}, {(y_high/nom)=}")

                    box = ROOT.TBox(x1, y_low, x2, y_high)
                    box.SetFillColor(ROOT.kGray + 1)
                    box.SetFillStyle(3345)
                    box.SetLineWidth(0)
                    uncertainty_boxes.append(box)

                    h_line_up.SetBinContent(ib, y_high)
                    h_line_down.SetBinContent(ib, y_low)
                else:
                    # no prediction in this bin -> keep lines at 0
                    h_line_up.SetBinContent(ib, 0.0)
                    h_line_down.SetBinContent(ib, 0.0)

            # "data" histogram: copy of total, errors = sqrt(yield)
            h_data = h_total.Clone(f"h_prefit_data_{era}")
            h_data.SetDirectory(0)
            for ib in range(1, n_bins+1):
                y = h_data.GetBinContent(ib)
                if y<0:
                    y=0
                h_data.SetBinError(ib, np.sqrt(y))
            h_data.SetMarkerStyle(ROOT.kFullCircle)
            h_data.SetMarkerSize(1.0)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetFillStyle(0)
            
            nominal_histograms = [histogram_dict[era][sample][feature_name][first_uncertainty][0] for sample in samples_to_use]
            nominal_histogram_integrals = [histogram.Integral() for histogram in nominal_histograms]

            # adding plots to stack in ascending order of yields
            yield_order = sorted(range(len(samples_to_use)), key=lambda i: nominal_histogram_integrals[i])
            sample_labels_sorted = [samples_to_use[i] for i in yield_order]

            stack_name = f"stack_prefit_{era}"
            hs = ROOT.THStack(stack_name, "")
            for i_sample, sample in enumerate(sample_labels_sorted):

                h_nominal = histogram_dict[era][sample][feature_name][first_uncertainty][0]

                sample_color = get_color(sample) if callable(get_color) else ROOT.kGray + 1

                h_nominal.SetLineColor(ROOT.kBlack)
                h_nominal.SetFillColor(sample_color)
                h_nominal.SetLineWidth(1)

                hs.Add(h_nominal, "hist")

            canvas_name = f"c_prefit_{era}"
            c_stack = ROOT.TCanvas(canvas_name, canvas_name, 800, 800)

            padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 1.0)
            padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

            padTop.SetBottomMargin(0.0)
            padTop.SetTopMargin(0.08)
            padTop.SetLeftMargin(0.10)
            padTop.SetRightMargin(0.05)
            padTop.SetTicks(1, 1)

            padBottom.SetTopMargin(0.0)
            padBottom.SetBottomMargin(0.30)
            padBottom.SetLeftMargin(0.10)
            padBottom.SetRightMargin(0.05)
            padBottom.SetTicks(1, 1)

            padTop.Draw()
            padBottom.Draw()

            # ---- TOP PAD: absolute yields ----
            padTop.cd()
            if logY:
                padTop.SetLogy(True)

            hs.Draw("HIST")
            hs.GetXaxis().SetTitle(x_title)
            hs.GetYaxis().SetTitle("Events")

            # font sizes / alignment (top pad)
            hs.GetYaxis().SetTitleSize(0.05)     # a bit smaller
            hs.GetYaxis().SetTitleOffset(1.1)    # helps align with bottom pad title
            hs.GetYaxis().SetLabelSize(0.045)
            hs.GetXaxis().SetLabelSize(0)
            hs.GetXaxis().SetTitleSize(0)

            # y-range
            max_y = max(hs.GetMaximum(), h_data.GetMaximum())
            if logY:
                hs.SetMinimum(0.5)
                hs.SetMaximum(10.0 * max_y if max_y > 0 else 1.0)
            else:
                hs.SetMinimum(0.0)
                hs.SetMaximum(1.5 * max_y if max_y > 0 else 1.0)

            # draw uncertainty band, lines, and data
            for box in uncertainty_boxes:
                box.Draw("SAME")
            h_data.Draw("E SAME")

            # legend
            leg = ROOT.TLegend(0.50, 0.60, 0.88, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetNColumns(legend_columns_prefit_stacks)

            leg.AddEntry(h_data, "Data (Asimov)", "lep")
            for sample in sample_labels_sorted:
                h_nominal = histogram_dict[era][sample][feature_name][first_uncertainty][0]
                leg.AddEntry(h_nominal, sample, "f")
                logging.debug(f"{sample=}")
            leg.AddEntry(uncertainty_boxes[0], "Uncertainty", "f")
            leg.Draw()

            # ---- BOTTOM PAD: ratios ----
            padBottom.cd()

            # ratio central
            h_ratio_central = h_total.Clone(f"h_prefit_ratio_{era}")
            h_ratio_central.SetDirectory(0)
            h_ratio_central.Divide(h_total)  # becomes 1 where non-zero
            h_ratio_central.SetLineColor(ROOT.kBlack)
            h_ratio_central.SetLineWidth(2)
            h_ratio_central.SetTitle("")

            h_ratio_central.GetYaxis().SetTitle("var / nominal")
            h_ratio_central.GetYaxis().SetNdivisions(505)
            h_ratio_central.GetYaxis().SetTitleSize(0.09)
            h_ratio_central.GetYaxis().SetTitleOffset(0.5)
            h_ratio_central.GetYaxis().SetLabelSize(0.08)

            h_ratio_central.GetXaxis().SetTitle(x_title)
            h_ratio_central.GetXaxis().SetTitleSize(0.10)
            h_ratio_central.GetXaxis().SetLabelSize(0.08)

            # ratio uncertainty band via TBoxes + line-only histos
            ratio_boxes = []

            h_ratio_line_up   = h_ratio_central.Clone(f"h_prefit_ratio_up_{era}")
            h_ratio_line_down = h_ratio_central.Clone(f"h_prefit_ratio_down_{era}")
            h_ratio_line_up.SetDirectory(0)
            h_ratio_line_down.SetDirectory(0)

            # lines only, no fill
            h_ratio_line_up.SetFillStyle(0)
            h_ratio_line_up.SetFillColor(0)
            h_ratio_line_down.SetFillStyle(0)
            h_ratio_line_down.SetFillColor(0)
            h_ratio_line_up.SetLineColor(ROOT.kGray + 2)
            h_ratio_line_down.SetLineColor(ROOT.kGray + 2)
            h_ratio_line_up.SetLineWidth(1)
            h_ratio_line_down.SetLineWidth(1)

            for ib in range(1, n_bins+1):
                x1 = edges[ib-1]
                x2 = edges[ib]
                nom = h_total.GetBinContent(ib)
                err_up = np.sqrt(h_unc_up.GetBinContent(ib))
                err_down = np.sqrt(h_unc_down.GetBinContent(ib))   

                if nom > 0.0:
                    rel_up = err_up / nom
                    rel_down = err_down / nom
                    
                    y_low  = 1.0 - rel_down
                    y_high = 1.0 + rel_up

                    box = ROOT.TBox(x1, y_low, x2, y_high)
                    box.SetFillColor(ROOT.kGray + 1)
                    box.SetFillStyle(3345)
                    box.SetLineWidth(0)
                    ratio_boxes.append(box)

                    h_ratio_line_up.SetBinContent(ib, y_high)
                    h_ratio_line_down.SetBinContent(ib, y_low)
                else:
                    # no prediction in this bin -> keep lines at 1
                    h_ratio_line_up.SetBinContent(ib, 1.0)
                    h_ratio_line_down.SetBinContent(ib, 1.0)

            # ratio y-range from max relative deviation
            max_dev = 0.0
            for ib in range(1, n_bins+1):
                nom = h_total.GetBinContent(ib)

                err_up = np.sqrt(h_unc_up.GetBinContent(ib))
                err_down = np.sqrt(h_unc_down.GetBinContent(ib))

                if nom > 0:
                    dev_up = err_up / nom
                    dev_down = err_down / nom
                    if max(dev_up,dev_down) > max_dev:
                        max_dev = max(dev_up,dev_down)

            if max_dev <= 0.0:
                r_min, r_max = 0.9, 1.1
            else:
                half_range = 1.3 * max_dev
                r_min = 1.0 - half_range
                r_max = 1.0 + half_range

            h_ratio_central.SetMinimum(max(r_min,-2.0))
            h_ratio_central.SetMaximum(min(r_max,2.0))

            # draw ratio
            h_ratio_central.Draw("HIST")
            for box in ratio_boxes:
                box.Draw("SAME")
            h_ratio_line_up.Draw("HIST SAME")
            h_ratio_line_down.Draw("HIST SAME")

            # line at 1
            line = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
            line.SetLineStyle(ROOT.kDashed)
            line.SetLineColor(ROOT.kBlack)
            line.Draw("SAME")

            c_stack.cd()
            c_stack.Update()

            helpers.copyIndexPHP(plot_directory_stacks)
            out_png = os.path.join(plot_directory_stacks, f"{era}_{feature_name}_prefit.png")
            out_pdf = os.path.join(plot_directory_stacks, f"{era}_{feature_name}_prefit.pdf")
            c_stack.SaveAs(out_png)
            c_stack.SaveAs(out_pdf)        


    syncer.sync()
