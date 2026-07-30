from __future__ import annotations

import os
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from data.RDataLoader import RDataLoader
import common.user as user
import observables
from data.plot_options import plot_options
from typing import Optional, List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# used in EFT sample factory class
from data.samples_RunII import Factory as Factory_RunII
from data.samples_RunII import BASE_DIRECTORY as BASE_DIRECTORY_RUNII

BASE_DIRECTORY_EFT = "/groups/hephy/cms/ricardo.barrue/CMGRDF_ntuples_ttbar_EFT/v3-2_nJ2p_nB2p_2l"

# no longer needed since lumi normalization is done at cmgrdf level
# LUMI = {
#     "2016APV": 19.52,
#     "2016": 16.81,
#     "2017": 41.48,
#     "2018": 59.83,
# }

wc_names = [
    "cQd1",
    "ctj1",
    "cQj31",
    "ctj8", # ML4EFT
    "ctd1",
    "ctd8", # ML4EFT
    "ctGRe", # ML4EFT
    "ctGIm", # ML4EFT
    "cQj11",
    "cQj18", # ML4EFT
    "ctu8", # ML4EFT 
    "cQd8", # ML4EFT
    "ctu1",
    "cQu1",
    "cQj38", # ML4EFT
    "cQu8", # ML4EFT
]


# lower triangular matrix in coefficients
# same as in postprocessing script (make_ntuple)
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
    "LHEWeight_originalXWGTUP",
    "nEFTfitCoefficients",
    "run",
    "luminosityBlock",
    "event",
] + eft_derivatives



def _eft_loader(*relpaths: str) -> RDataLoader:

    loader = RDataLoader(
        input_paths=[os.path.join(BASE_DIRECTORY_EFT, relpath) for relpath in relpaths],
        tree_name="Events",
        branches=observables.ALL_FEATURES + observers,
        selection=None,
        n_split=1,
        splitting_strategy="files",
        strict_branches=True,
        weight_branches=[
            "weight", # weight contains lumi*xs/sumw normalization from CMGRDF
            "EFTWeight_SM", # SM weight, since Generator_weight is one for all the generated samples
            "L1PreFiringWeight_Nom",
            "JetPUID_SF",
            "Pileup_SF",
            "btagSF_fixedWP_SF",
            "lepEle_SF",
            "lepMu_SF",
        ],
        feature_names=observables.ALL_FEATURES,
        observer_names=observers,
        weight_rescale=1000.0 # forgot to convert the cross-section to fb when processing the samples
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

# Factory class to allow running fits with EFT samples (closure studies) and mixing EFT samples with Run 2 samples
class Factory:

    def __init__( self, 
            BASE_DIRECTORY: str = BASE_DIRECTORY_EFT,
            features: Optional[List[str]] = None,
            selection: Optional[str] = None,
            selection_features: Optional[List[str]] = None,
            ):
        if type(BASE_DIRECTORY) == str:
            self.BASE_DIRECTORY = Path(BASE_DIRECTORY)
        else:
            self.BASE_DIRECTORY = BASE_DIRECTORY

        self.features = features
        self.selection = selection
        self.selection_features = selection_features

        self.Factory_RunII = Factory_RunII(BASE_DIRECTORY_RUNII,
                                           self.features,
                                           self.selection,
                                           self.selection_features)
    
    def get(self, process: str, era: Optional[str] = None, tag: Optional[str] = None) -> RDataLoader:

        loader = None
        if era is None and tag is None:
            try:
                loader = globals()[process]
                if self.features:
                    loader.setFeatures( self.features )
                if self.selection:
                    loader.addSelection( self.selection, required_branches = self.selection_features )
                return loader             
            except KeyError:
                logger.debug(f"EFT loader with name {process} not found! Reverting to Run 2 sample module.")
                loader = self.Factory_RunII.get(process)
                return loader
        else:
            logger.debug("Asking for era and tag, only available for Run 2 sample module.")
            loader = self.Factory_RunII.get(process, era, tag)
            return loader


if __name__ == "__main__":

    # lower triangular matrix of derivatives
    print(eft_derivatives)

    base_loader = TT01j2l_EFT_2016_mtt_0to700
    print("Base:_nominal.root", base_loader)
    F, O, W = base_loader.materialize(0, "fow")
    print("Shapes:_nominal.root", F.shape, O.shape, W.shape)

    factory = Factory(
        features=base_loader.feature_names,
        selection=base_loader.selection,
        selection_features=base_loader.feature_names
    )

    L_EFT = factory.get("TT01j2l_EFT_2016")
    F, O, W = L_EFT.materialize(0, "fow")
    print("Shapes:L_EFT.root", F.shape, O.shape, W.shape)
    print("w[:5]",W[:5])

    L_from_samples_RunII = factory.get("TTLep_pow_2016")
    F, O, W = L_from_samples_RunII.materialize(0, "fow")
    print(L_from_samples_RunII)
    print("Shapes:L_from_samples_RunII.root", F.shape, O.shape, W.shape)
    print("w[:5]",W[:5])