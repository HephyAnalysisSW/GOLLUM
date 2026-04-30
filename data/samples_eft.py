from __future__ import annotations

import os
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from data.RDataLoader import RDataLoader
import common.user as user
from data.plot_options_eft import plot_options


BASE_DIRECTORY = "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/TTbarEFT-centralGen-ntuples"
LUMI = {
    "2016APV": 19.52,
    "2016": 16.81,
    "2017": 41.48,
    "2018": 59.83,
}

kinematics = [
    "l0_pt",
    "l0_eta",
    "l0_phi",
    "l1_pt",
    "l1_eta",
    "l1_phi",
    "lminus_pt",
    "lminus_eta",
    "lminus_phi",
    "lplus_pt",
    "lplus_eta",
    "lplus_phi",
    "dilep_pt",
    "dilep_mass",
    "dilep_dphi",
    "dilep_deta",
    "j0_pt",
    "j0_eta",
    "j0_phi",
    "j1_pt",
    "j1_eta",
    "j1_phi",
    "dijet_pt",
    "dijet_mass",
    "dijet_dphi",
    "dijet_deta",
    "pseudo_mtt",
    "nJets",
    "max_obj_pair_pt",
]
KINEMATICS = list(kinematics)

TOP_KINEMATICS = [
    "j0_pt",
    "j0_eta",
    "j0_phi",
    "j1_pt",
    "j1_eta",
    "j1_phi",
    "dijet_pt",
    "dijet_mass",
    "pseudo_mtt",
]

LEPTON_KINEMATICS = [
    "l0_pt",
    "l0_eta",
    "l0_phi",
    "l1_pt",
    "l1_eta",
    "l1_phi",
    "lminus_pt",
    "lminus_eta",
    "lminus_phi",
    "lplus_pt",
    "lplus_eta",
    "lplus_phi",
    "dilep_pt",
    "dilep_mass",
]

ASYMMETRY = [
    "dilep_dphi",
    "dilep_deta",
    "dijet_dphi",
    "dijet_deta",
    "nJets",
    "max_obj_pair_pt",
]

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
    "weight1fb",
    "Generator_weight",
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "LHEWeight_originalXWGTUP",
    "nEFTfitCoefficients",
] + eft_derivatives


def _eft_loader(*relpaths: str, lumi: float) -> RDataLoader:
    return RDataLoader(
        input_paths=[os.path.join(BASE_DIRECTORY, relpath) for relpath in relpaths],
        tree_name="Events",
        branches=observers + kinematics,
        selection=None,
        n_split=1,
        splitting_strategy="events",
        strict_branches=False,
        weight_branches=[
            "weight1fb",
            "EFTWeight_SM",
        ],
        feature_names=kinematics,
        observer_names=observers,
        weight_rescale=lumi,
    )


TT01j2l_EFT_2016APV_mtt_0to700 = _eft_loader(
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_0to700",
    lumi=LUMI["2016APV"],
)
TT01j2l_EFT_2016APV_mtt_700to900 = _eft_loader(
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_700to900",
    lumi=LUMI["2016APV"],
)
TT01j2l_EFT_2016APV_mtt_900toInf = _eft_loader(
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_900toInf",
    lumi=LUMI["2016APV"],
)

TT01j2l_EFT_2016_mtt_0to700 = _eft_loader(
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_0to700",
    lumi=LUMI["2016"],
)
TT01j2l_EFT_2016_mtt_700to900 = _eft_loader(
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_700to900",
    lumi=LUMI["2016"],
)
TT01j2l_EFT_2016_mtt_900toInf = _eft_loader(
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_900toInf",
    lumi=LUMI["2016"],
)

TT01j2l_EFT_2017_mtt_0to700 = _eft_loader(
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_0to700",
    lumi=LUMI["2017"],
)
TT01j2l_EFT_2017_mtt_700to900 = _eft_loader(
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_700to900",
    lumi=LUMI["2017"],
)
TT01j2l_EFT_2017_mtt_900toInf = _eft_loader(
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_900toInf",
    lumi=LUMI["2017"],
)

TT01j2l_EFT_2018_mtt_0to700 = _eft_loader(
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_0to700",
    lumi=LUMI["2018"],
)
TT01j2l_EFT_2018_mtt_700to900 = _eft_loader(
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_700to900",
    lumi=LUMI["2018"],
)
TT01j2l_EFT_2018_mtt_900toInf = _eft_loader(
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_900toInf",
    lumi=LUMI["2018"],
)


TT01j2l_EFT_2016APV = _eft_loader(
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_0to700",
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_700to900",
    "hnelson2__mc__UL16APV/UL16APV_TT01j2l_mtt_900toInf",
    lumi=LUMI["2016APV"],
)
TT01j2l_EFT_2016 = _eft_loader(
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_0to700",
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_700to900",
    "hnelson2__mc__UL16/UL16_TT01j2l_mtt_900toInf",
    lumi=LUMI["2016"],
)
TT01j2l_EFT_2017 = _eft_loader(
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_0to700",
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_700to900",
    "hnelson2__mc__UL17/UL17_TT01j2l_mtt_900toInf",
    lumi=LUMI["2017"],
)
TT01j2l_EFT_2018 = _eft_loader(
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_0to700",
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_700to900",
    "hnelson2__mc__UL18/UL18_TT01j2l_mtt_900toInf",
    lumi=LUMI["2018"],
)


if __name__ == "__main__":
    print("Base:", TT01j2l_EFT_2018_mtt_0to700)
    F, O, W = TT01j2l_EFT_2018_mtt_0to700.materialize(0, "fow")
    print("Shapes:", F.shape, O.shape, W.shape)
