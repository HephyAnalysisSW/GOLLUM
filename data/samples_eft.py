from __future__ import annotations

import os
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from data.RDataLoader import RDataLoader
import common.user as user
from data.plot_options_eft import plot_options
from typing import Optional


BASE_DIRECTORY = "/groups/hephy/cms/ricardo.barrue/CMGRDF_ntuples_ttbar_EFT/v3-2_nJ2p_nB2p_2l"

# no longer needed since lumi normalization is done at cmgrdf level
# LUMI = {
#     "2016APV": 19.52,
#     "2016": 16.81,
#     "2017": 41.48,
#     "2018": 59.83,
# }

# kinematics not in original ttbar samples
# will need to run make_ntuple on those again
additional_kinematics = [
    "jet0_phi", "jet0_mass",
    "jet1_phi", "jet1_mass",
    "dijet_dPhi", "dijet_dEta",
    "dijet_pt", "dijet_mass",
    "bjet0_pt", "bjet1_pt",
    "bjet0_eta", "bjet1_eta",
    "dibjet_pt", "dibjet_mass",
    "lepMinus_pt", "lepMinus_phi", "lepMinus_eta",
    "lepPlus_pt", "lepPlus_phi", "lepPlus_eta",
    "dilep_dPhi", "dilep_absDPhi",
    "lj0_pt", "max_obj_pair_pt",
    "pseudo_mtt"
]

from data.observables import TOP_KINEMATICS, LEPTON_KINEMATICS, ASYMMETRY

ALL_FEATURES = TOP_KINEMATICS + LEPTON_KINEMATICS + ASYMMETRY

wc_names = [
    "cQd1",
    "ctj1",
    "cQj31",
    "ctj8",
    "ctd1",
    "ctd8",
    "ctGRe",
    "ctGIm",
    "cQj11",
    "cQj18",
    "ctu8",
    "cQd8",
    "ctu1",
    "cQu1",
    "cQj38",
    "cQu8",
]


def _derivative_branches(wcs):
    out = ["EFTWeight_SM"]
    for j, wc_j in enumerate(wcs):
        out.append(f"der_{wc_j}")
        for k in range(j + 1):
            out.append(f"der_{wc_j}_{wcs[k]}")
    return out


eft_derivatives = _derivative_branches(wc_names)

observers = [
    "weight", # lumi*xs/sumw factor (from CMGRDF) 
    "Generator_weight", # 1.0 for all samples
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "LHEWeight_originalXWGTUP",
    "nEFTfitCoefficients",
] + eft_derivatives



def _eft_loader(*relpaths: str) -> RDataLoader:

    loader = RDataLoader(
        input_paths=[os.path.join(BASE_DIRECTORY, relpath) for relpath in relpaths],
        tree_name="Events",
        branches=ALL_FEATURES,
        selection=None,
        n_split=1,
        splitting_strategy="files",
        strict_branches=False,
        weight_branches=[
            "weight", # weight already contains lumi normalization from CMGRDF
            "EFTWeight_SM", # SM weight, since Generator_weight is one for all the generated samples
            "L1PreFiringWeight_Nom",
            "JetPUID_SF",
            "Pileup_SF",
            "btagSF_fixedWP_SF",
            "lepEle_SF",
            "lepMu_SF",
        ],
        feature_names=ALL_FEATURES,
        observer_names=observers,
    )

    # matching the selection in samples_RunII.py
    loader.addSelection(
            "(dilep_mass > 20) & (lep1_pt>20) & (tr_isvalid>0) & (isOS>0) & (offZ>0)",
            required_branches=["dilep_mass", "lep1_pt", "isOS", "offZ", "tr_isvalid"],
        )
    # selection to truncate outliers (see samples_eft_gen) can be added in the config
    return loader

# configs
TT01j2l_EFT_2016APV_mtt_0to700 = _eft_loader(
    "2016APV/TT01j2l_UL16APV_mtt_0to700_nominal.root",
)
TT01j2l_EFT_2016APV_mtt_700to900 = _eft_loader(
    "2016APV/TT01j2l_UL16APV_mtt_700to900_nominal.root",
)
TT01j2l_EFT_2016APV_mtt_900toInf = _eft_loader(
    "2016APV/TT01j2l_UL16APV_mtt_900toInf_nominal.root",
)

TT01j2l_EFT_2016_mtt_0to700 = _eft_loader(
    "2016/TT01j2l_UL16_mtt_0to700_nominal.root",
)
TT01j2l_EFT_2016_mtt_700to900 = _eft_loader(
    "2016/TT01j2l_UL16_mtt_700to900_nominal.root",
)
TT01j2l_EFT_2016_mtt_900toInf = _eft_loader(
    "2016/TT01j2l_UL16_mtt_900toInf_nominal.root",
)


# 2017 files cannot be generated atm, as they're
# missing some of the branches to make jet corrections
# TT01j2l_EFT_2017_mtt_0to700 = _eft_loader(
#     "2017/TT01j2l_UL17_mtt_0to700_nominal.root",
# )
# TT01j2l_EFT_2017_mtt_700to900 = _eft_loader(
#     "2017/TT01j2l_UL17_mtt_700to900_nominal.root",
# )
# TT01j2l_EFT_2017_mtt_900toInf = _eft_loader(
#     "2017/TT01j2l_UL17_mtt_900toInf_nominal.root",
# )

TT01j2l_EFT_2018_mtt_0to700 = _eft_loader(
    "2018/TT01j2l_UL18_mtt_0to700_nominal.root",
)
TT01j2l_EFT_2018_mtt_700to900 = _eft_loader(
    "2018/TT01j2l_UL18_mtt_700to900_nominal.root",
)
TT01j2l_EFT_2018_mtt_900toInf = _eft_loader(
    "2018/TT01j2l_UL18_mtt_900toInf_nominal.root",
)


TT01j2l_EFT_2016APV = _eft_loader(
    "2016APV/TT01j2l_UL16APV_mtt_0to700_nominal.root",
    "2016APV/TT01j2l_UL16APV_mtt_700to900_nominal.root",
    "2016APV/TT01j2l_UL16APV_mtt_900toInf_nominal.root",
)
TT01j2l_EFT_2016 = _eft_loader(
    "2016/TT01j2l_UL16_mtt_0to700_nominal.root",
    "2016/TT01j2l_UL16_mtt_700to900_nominal.root",
    "2016/TT01j2l_UL16_mtt_900toInf_nominal.root",
)
# TT01j2l_EFT_2017 = _eft_loader(
#     "2017/TT01j2l_UL17_mtt_0to700_nominal.root",
#     "2017/TT01j2l_UL17_mtt_700to900_nominal.root",
#     "2017/TT01j2l_UL17_mtt_900toInf_nominal.root",
# )

TT01j2l_EFT_2018 = _eft_loader(
    "2018/TT01j2l_UL18_mtt_0to700_nominal.root",
    "2018/TT01j2l_UL18_mtt_700to900_nominal.root",
    "2018/TT01j2l_UL18_mtt_900toInf_nominal.root",
)

TT01j2l_EFT_RunII = _eft_loader(
    "2016APV/TT01j2l_UL16APV_mtt_0to700_nominal.root",
    "2016APV/TT01j2l_UL16APV_mtt_700to900_nominal.root",
    "2016APV/TT01j2l_UL16APV_mtt_900toInf_nominal.root",
    "2016/TT01j2l_UL16_mtt_0to700_nominal.root",
    "2016/TT01j2l_UL16_mtt_700to900_nominal.root",
    "2016/TT01j2l_UL16_mtt_900toInf_nominal.root",
    # "2017/TT01j2l_UL17_mtt_0to700_nominal.root",
    # "2017/TT01j2l_UL17_mtt_700to900_nominal.root",
    # "2017/TT01j2l_UL17_mtt_900toInf_nominal.root",
    "2018/TT01j2l_UL18_mtt_0to700_nominal.root",
    "2018/TT01j2l_UL18_mtt_700to900_nominal.root",
    "2018/TT01j2l_UL18_mtt_900toInf_nominal.root",
)

if __name__ == "__main__":
    print("Base:_nominal.root", TT01j2l_EFT_2018_mtt_0to700)
    F, O, W = TT01j2l_EFT_2018_mtt_0to700.materialize(0, "fow")
    print("Shapes:_nominal.root", F.shape, O.shape, W.shape)
