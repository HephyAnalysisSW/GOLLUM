"""
plot_from_inputs.py

This module creates histogram plots comparing nominal, down-varied, and up-varied weight distributions for different systematic uncertainties
without using learned ICPH surrogates, i.e. directly from input variations.

Usage:
    python plot_from_inputs.py --feature <feature_name> [--binning <bin_edges>] [--selection <selection_string>]

Arguments:
    --feature, -f (str): Feature to plot (required)
    --binning, -b (list): Custom bin edges. If not provided, uses defaults from plot_options
    --selection, -s (str): String-based event selection

Outputs:
    PNG and PDF plots saved to plot_directory/binned_templates_from_inputs/<feature_name>/<era>/<sample>/

Supported systematics:
    - MODELING: Renormalization/factorization scales, shower ISR/FSR, alpha_s
    - EXPERIMENTAL: Pileup and other experimental uncertainties
"""


import os
import itertools
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

# TODO: finish this function
def get_branches_for_selection(selection: str) -> Sequence[str]:
    return []

"""
systematics are divided into three categories:
- additional weights, to be multiplied by the overall weight
- variations in weights which are part of overall weight - should replace nominal weight
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
    

# simple color palette
colors = [
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 1,
    ROOT.kCyan + 1,
]

if __name__ == "__main__":

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

    logY = plot_options.get(feature_name, {}).get('logY', False)

    plot_directory = os.path.join(user.plot_directory, 'binned_templates_from_inputs', feature_name)
    print(f"[info] Plots will be written under: {plot_directory}")
    os.makedirs(plot_directory, exist_ok=True)

    x_title = plot_options.get(feature_name, {}).get('tex', feature_name)

    # legend columns (configurable)
    legend_columns = 3


    i_sample_era = 0
    for era, sample in itertools.product(eras, samples_to_use):
        print(f"era: {era}, sample: {sample}")
        

        nominal_sample = getattr(samples, f'{sample}_{era}_nominal')

        # if selection:
        #     requested_branches_for_selection = get_branches_for_selection(selection)
        #     nominal_sample.addSelection(selection, requested_branches_for_selection)
            
        if feature_name not in nominal_sample.feature_names:
            print(f'[warning] feature {feature_name} not in base RDataLoader, attempting to load with setFeatures')
            nominal_sample.setFeatures([feature_name])

        nominal_feature_values, nominal_weights = nominal_sample.materialize(0, what='fw', feature_names = [feature_name])

        # nominal_features is an array of shape (n_events, n_features)
        # even when just one feature is requested
        nominal_feature_values = nominal_feature_values[:,0]

        nominal_hist = np.histogram(a=nominal_feature_values, bins=edges, weights=nominal_weights)
        nominal_hist_entries = nominal_hist[0]

        # list of weights in nominal sample
        list_weights = nominal_sample.weight_branches

        """
        To take into account the three types of handling systematic variations
        at the RDataLoader level
        """
        # for mode in ["replace", "add", "kinematics"]:
        for mode in ["kinematics"]:

            if mode == "replace":
                syst_groups = replace_weight_groups
                print("[info] plotting variations for scale factors which are part of the overall weight" \
                "e.g. pileup reweighting scale factor uncertainty")
            elif mode == "add":
                syst_groups = add_weight_groups
                print("[info] plotting variations implemented as scale factors which multiply the overall event weight, " \
                "e.g. QCD scale variations")
            elif mode == "kinematics":
                print("[info] plotting variations from varied kinematics (JME)")
                syst_groups = kinematic_variation_groups

            for group, uncertainty_names in syst_groups.items():
                if i_sample_era == 0:
                    print(f"[info] group: {group}, uncertainties: {uncertainty_names.keys()}")

                individual_plot_dir = os.path.join(plot_directory,era,sample)
                os.makedirs(individual_plot_dir, exist_ok=True)
                helpers.copyIndexPHP(individual_plot_dir)

                canvas_name = f"{era}_{sample}_{group}"
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
                legend.SetNColumns(legend_columns)

                # ------------- TOP PAD: absolute yields -------------
                padTop.cd()
                padTop.SetTicks(1, 1)
                if logY:
                    padTop.SetLogy(True)

                # central histogram with variable binning
                h_central_name = f"h_central_{era}_{sample}_{group}"
                h_central = ROOT.TH1F(h_central_name, "", len(edges) - 1, np.array(edges))
                for i in range(len(edges)-1):
                    h_central.SetBinContent(i + 1, nominal_hist_entries[i])

                h_central.SetLineColor(ROOT.kBlack)
                h_central.SetLineWidth(2)
                # no title on the top pad
                h_central.SetTitle("")
                h_central.GetXaxis().SetTitle(x_title)

                h_central.GetYaxis().SetTitle("Events")
                h_central.GetYaxis().SetTitleSize(0.06)
                h_central.GetYaxis().SetLabelSize(0.045)
                # x label on bottom pad only
                h_central.GetXaxis().SetLabelSize(0)
                h_central.GetXaxis().SetTitleSize(0)

                legend.AddEntry(h_central, "nominal", "l")

                # keep references alive
                h_variations = [h_central]

                color_index = 0

                for uncertainty_name, branch_names in uncertainty_names.items():
                    color = colors[color_index % len(colors)]
                    color_index += 1

                    # logic different for each type of systematic variations
                    if mode == "replace":
                
                        """
                        For variations, cloning sample replacing nominal weight branch with varied one,
                        since one can only change the weight branches at RDataLoader creation.
                        """
                
                        # using structure defined in replace_weight_groups
                        nominal_weight_name = branch_names[0]
                        down_var_weight_name = branch_names[1]
                        up_var_weight_name = branch_names[2]

                        if '<ERA>' in uncertainty_name:
                            uncertainty_name = uncertainty_name.replace('<ERA>',str(era))
                            down_var_weight_name = down_var_weight_name.replace('<ERA>',str(era))
                            up_var_weight_name = up_var_weight_name.replace('<ERA>',str(era))

                        list_weights_varied = list_weights
                        weight_to_remove_idx = list_weights_varied.index(nominal_weight_name)
                        
                        list_weights_varied[weight_to_remove_idx] = down_var_weight_name
                        down_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights_varied)

                        list_weights_varied[weight_to_remove_idx] = up_var_weight_name
                        up_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights_varied)

                        # fixes cases where there are two uncertainties with the same
                        # nominal weight name (e.g. b-tag SF)
                        # list_weights_varied[weight_to_remove_idx] = nominal_weight_name

                        down_var_weights = down_var_sample.materialize(0, what='w')[0]
                        down_var_hist_entries = np.histogram(a=nominal_feature_values, bins=edges, weights=down_var_weights)[0]
                        
                        up_var_weights = up_var_sample.materialize(0, what='w')[0]
                        up_var_hist_entries = np.histogram(a=nominal_feature_values, bins=edges, weights=up_var_weights)[0]
                    
                    elif mode == "add":
                
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
                        
                        # list_weights = nominal_sample.weight_branches

                        down_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights+[down_var_weight_name])
                        up_var_sample = nominal_sample.clone_from_files(nominal_sample.files, list_weights+[up_var_weight_name])

                        down_var_weights = down_var_sample.materialize(0, what='w')[0]
                        down_var_hist_entries = np.histogram(a=nominal_feature_values, bins=edges, weights=down_var_weights)[0]
                        
                        up_var_weights = up_var_sample.materialize(0, what='w')[0]
                        up_var_hist_entries = np.histogram(a=nominal_feature_values, bins=edges, weights=up_var_weights)[0]

                    elif mode == "kinematics":
                        
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
                            uncertainty_name = uncertainty_name.replace("APV","")
                            down_var_file_tag = down_var_file_tag.replace("APV","")
                            up_var_file_tag = up_var_file_tag.replace("APV","")

                        down_var_sample = getattr(samples, f'{sample}_{era}_{down_var_file_tag}')
                        up_var_sample = getattr(samples, f'{sample}_{era}_{up_var_file_tag}')

                        down_var_feature_values, down_var_weights = down_var_sample.materialize(0, what='fw', feature_names=[feature_name])
                        # parsing output of materialize into a simple array
                        # also done when loading the nominal sample
                        down_var_feature_values = down_var_feature_values[:,0]
                        down_var_hist_entries = np.histogram(a=down_var_feature_values, bins=edges, weights=down_var_weights)[0]
                        
                        up_var_feature_values, up_var_weights = up_var_sample.materialize(0, what='fw', feature_names=[feature_name])
                        up_var_feature_values = up_var_feature_values[:,0]
                        up_var_hist_entries = np.histogram(a=up_var_feature_values, bins=edges, weights=up_var_weights)[0]
                    
                    # to avoid ballooning memory usage
                    del down_var_sample, up_var_sample
                    
                    h_down_name = f"h_{group}_{uncertainty_name}_Down"
                    h_down = ROOT.TH1F(h_down_name, "", len(edges) - 1, np.array(edges))
                    for i in range(len(edges)-1):
                        h_down.SetBinContent(i + 1, down_var_hist_entries[i])
                    
                    h_down.SetLineColor(color)
                    h_down.SetLineStyle(ROOT.kDashed)
                    h_down.SetLineWidth(1)
                    legend.AddEntry(h_down, f"{uncertainty_name} -1#sigma", "l")
                    h_variations.append(h_down)


                    h_up_name = f"h_{group}_{uncertainty_name}_Up"
                    h_up = ROOT.TH1F(h_up_name, "", len(edges) - 1, np.array(edges))
                    for i in range(len(edges)-1):
                        h_up.SetBinContent(i + 1, up_var_hist_entries[i])
                    
                    h_up.SetLineColor(color)
                    h_up.SetLineStyle(ROOT.kSolid)
                    h_up.SetLineWidth(1)
                    legend.AddEntry(h_up, f"{uncertainty_name} +1#sigma", "l")
                    h_variations.append(h_up)
                

                # y range (absolute yields)
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
                ratio_central_name = h_central_name + "_ratio"
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
                    for i in range(1, len(edges) + 1):
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

                h_ratio_central.SetMinimum(r_min)
                h_ratio_central.SetMaximum(r_max)

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

                c.cd()
                c.Update()

                out_png = os.path.join(individual_plot_dir, canvas_name + ".png")
                out_pdf = os.path.join(individual_plot_dir, canvas_name + ".pdf")
                c.SaveAs(out_png)
                c.SaveAs(out_pdf)
        
        i_sample_era += 1
        print("[info]: after first sample/era combination, no longer printing debug information")

    syncer.sync()