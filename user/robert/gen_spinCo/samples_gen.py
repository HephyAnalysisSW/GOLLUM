import glob
import os
import sys
from dataclasses import dataclass

import ROOT
import uproot

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory
from data.RDataLoader import RDataLoader

parton_features = [
    "parton_Top_pt", 
    "parton_Top_eta", 
    "parton_Top_phi", 
    "parton_Top_mass", 
    "parton_Top_y", 
    "parton_Top_pdgId", 
    "parton_AntiTop_pt", 
    "parton_AntiTop_eta", 
    "parton_AntiTop_phi", 
    "parton_AntiTop_mass", 
    "parton_AntiTop_y", 
    "parton_AntiTop_pdgId", 
    "parton_ttbar_pt", 
    "parton_ttbar_mass", 
    "parton_ttbar_eta", 
    "parton_ttbar_y", 
    "parton_ttbar_dEta", 
    "parton_ttbar_dAbsEta", 
    "parton_Mtt", 
    "parton_cosTheta_t", 
    "parton_cosThetaPlus_n", 
    "parton_cosThetaMinus_n", 
    "parton_cosThetaPlus_r", 
    "parton_cosThetaMinus_r", 
    "parton_cosThetaPlus_k", 
    "parton_cosThetaMinus_k", 
    "parton_cosThetaPlus_r_star", 
    "parton_cosThetaMinus_r_star", 
    "parton_cosThetaPlus_k_star", 
    "parton_cosThetaMinus_k_star", 
    "parton_xi_nn", 
    "parton_xi_rr", 
    "parton_xi_kk", 
    "parton_xi_nr_plus", 
    "parton_xi_nr_minus", 
    "parton_xi_rk_plus", 
    "parton_xi_rk_minus", 
    "parton_xi_nk_plus", 
    "parton_xi_nk_minus", 
    "parton_xi_r_star_k", 
    "parton_xi_k_r_star", 
    "parton_xi_kk_star", 
    "parton_cos_phi", 
    "parton_cos_phi_lab", 
    "parton_abs_delta_phi_ll_lab", 
    "parton_c_hel", 
    "parton_c_han", 
    "parton_hasGenTops", 
    "parton_hasGenSpin", 
    "parton_genSpinCat", 
]
observers = [
    "run", 
    "luminosityBlock", 
    "event", 
    "genWeight",
] 

tt2l = RDataLoader(
    input_paths=["/groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples/RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v1/"],
    tree_name="Events",
    branches=[],
    selection=None,
    n_split=100,
    splitting_strategy="files",
    strict_branches=False,
    weight_branches=[],
    feature_names=parton_features,
    observer_names=observers,
)

tt2l_noSpinCo = RDataLoader(
    input_paths=["/groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples/RunIISummer20UL17NanoAODv9__TTTo2L2Nu-noSC_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v2/"],
    tree_name="Events",
    branches=[],
    selection=None,
    n_split=100,
    splitting_strategy="files",
    strict_branches=False,
    weight_branches=[],
    feature_names=parton_features,
    observer_names=observers,
)

#if __name__=="__main__":
#    for shard in range(100):
#        X, O, w = tt2l.materialize(shard=shard, what="fow", n=None)
#        break
#
