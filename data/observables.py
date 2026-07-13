"""
Group-aware observable schema used across loaders and models.

- Define logical groups (top_kinematics, lepton_kinematics, asymmetry, spin_correlation).
- Provide utilities to resolve groups to a flat branch list.
- Carry along observer branches (non-features) when needed.
"""
from typing import Dict, List, Iterable

OLD_TOP_KINEMATICS = [
    "tr_ttbar_pt",
    "tr_ttbar_mass",
    "tr_top_pt",
    "tr_topBar_pt",
    "tr_top_eta",
    "tr_topBar_eta",
]

OLD_LEPTON_KINEMATICS = [
    "recoLep0_pt",
    "recoLep1_pt",
    "recoLepPos_pt",
    "recoLepNeg_pt",
    "recoLep01_pt",
    "recoLep01_mass",
]

OLD_ASYMMETRY = [
    "tr_ttbar_dEta",
    "tr_ttbar_dAbsEta",
    "recoLep_dEta",
    "recoLep_dAbsEta",
]

OLD_SPIN_CORRELATION = [
    "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r",
    "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star",
    "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star",
    "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus",
    "tr_xi_rk_plus", "tr_xi_rk_minus", "tr_xi_nk_plus", "tr_xi_nk_minus",
    "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
    "tr_cos_phi", "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",
]

OLD_BASIC_EVENT = [
    "nBTag", "nrecoJet", "jet0_pt", "jet1_pt", "ht", "recoLep0_pt", "recoLep1_pt",
]

TOP_KINEMATICS = [
    "tr_ttbar_pt",
    "tr_ttbar_mass",
    "tr_ttbar_y",
    "tr_Top_pt",
    "tr_AntiTop_pt",
    "tr_Top_y",
    "tr_AntiTop_y",
]

TOP_KINEMATICS_NO_PT_TTBAR = [
    "tr_ttbar_mass",
    "tr_ttbar_y",
    "tr_Top_pt",
    "tr_AntiTop_pt",
    "tr_Top_y",
    "tr_AntiTop_y",
]

LEPTON_KINEMATICS = [
    "lep0_pt",
    "lep1_pt",
    "dilep_pt",
    "dilep_eta",
    "dilep_mass",
]

ASYMMETRY = [
    "tr_ttbar_dEta",
    "tr_ttbar_dAbsEta",
    "dilep_dEta",
    "dilep_dAbsEta",
]

SPIN_CORRELATION = [
    "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r",
    "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star",
    "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star",
    "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus",
    "tr_xi_rk_plus", "tr_xi_rk_minus", "tr_xi_nk_plus", "tr_xi_nk_minus",
    "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
    "tr_cos_phi", "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",
]

THRESHOLD = ["tr_ttbar_mass", "tr_ttbar_y", "tr_ttbar_pt", "tr_ttbar_beta_plus", "tr_c_hel", "tr_c_han", 
       "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus",
       "tr_xi_rk_plus", "tr_xi_rk_minus", "tr_xi_nk_plus", "tr_xi_nk_minus",
       "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star"]

BASIC_EVENT = [
    "nBJet", "nSelJet", "jet0_pt", "jet1_pt", "lep0_pt", "lep1_pt", "ht",
]

# Generator-level observers (not directly features)
OBSERVERS = ["weight", "Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF", "run", "luminosityBlock", "event"]

# kinematics added for the EFT analysis
ADDITIONAL_KINEMATICS = [
    "lepMinus_pt", "lepMinus_phi", "lepMinus_eta",
    "lepPlus_pt", "lepPlus_phi", "lepPlus_eta",
    "dilep_dPhi", "dilep_absDPhi",
    "jet0_phi", "jet0_mass",
    "jet1_phi", "jet1_mass",
    "dijet_dPhi", "dijet_dEta",
    "dijet_pt", "dijet_mass",
    "bjet0_pt", "bjet1_pt",
    "bjet0_eta", "bjet1_eta",
    "dibjet_pt", "dibjet_mass",
    "lb0_pt", "max_obj_pair_pt",
    "pseudo_mtt" # m(llbb)
]

ALL_FEATURES = TOP_KINEMATICS + LEPTON_KINEMATICS + ASYMMETRY + SPIN_CORRELATION + BASIC_EVENT + ADDITIONAL_KINEMATICS
