"""
Group-aware observable schema used across loaders and models.

- Define logical groups (top_kinematics, lepton_kinematics, asymmetry, spin_correlation).
- Provide utilities to resolve groups to a flat branch list.
- Carry along observer branches (non-features) when needed.
"""
from typing import Dict, List, Iterable

TOP_KINEMATICS = [
    "tr_ttbar_pt",
    "tr_ttbar_mass",
    "tr_Top_pt",
    "tr_AntiTop_pt",
    "tr_Top_eta",
    "tr_AntiTop_eta",
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

BASIC_EVENT = [
    "nBJet", "inSelJet", "jet0_pt", "jet1_pt", "lep0_pt", "lep1_pt", "ht",
]

# Generator-level observers (not directly features)
OBSERVERS = ["weight", "Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]

GROUPS: Dict[str, List[str]] = {
    "top_kinematics": TOP_KINEMATICS,
    "lepton_kinematics": LEPTON_KINEMATICS,
    "asymmetry": ASYMMETRY,
    "spin_correlation": SPIN_CORRELATION,
    "basic_event": BASIC_EVENT,
}

ALL_FEATURES = TOP_KINEMATICS + LEPTON_KINEMATICS + ASYMMETRY + SPIN_CORRELATION + BASIC_EVENT
