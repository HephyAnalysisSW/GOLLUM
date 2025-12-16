"""
Global selection definitions for RDataLoader.

Each selection is a callable: (ak.Array) -> np.ndarray[bool]
"""
import numpy as np
import awkward as ak
from typing import Dict, Callable, Union

from RDataLoader import SelectionFn, RDataLoader
import data.observables as observables

# Pre-defined global selections - from TOP-20-006
# Based on v2.2+ CMGRDF ntuple names
SELECTIONS: Dict[str, SelectionFn] = {
    "nL2": lambda ar: ar["nLepton_good"] == 2,  # type:ignore
    "nJ2p": lambda ar: ar["nSelJet"] >= 2,  # type:ignore
    "nB2p": lambda ar: ar["nBJet"] >= 2,  # type:ignore
    "goodTop": lambda ar: ar["tr_isvalid"] == 1,  # type:ignore
    "minMll": lambda ar: ar["dilep_mass"] > 20,  # type:ignore
    "SameFlavorMET": lambda ar: (ak.abs(ar["lep0_pdgId"]) != ak.abs(ar["lep1_pdgId"])) | (ar["MET_pt"] > 40),  # type:ignore
    "SameFlavorMll": lambda ar: (ak.abs(ar["lep0_pdgId"]) != ak.abs(ar["lep1_pdgId"])) | (ar["dilep_mass"] < 76) | (ar["dilep_mass"] > 106)  # type:ignore
}

# Reconstruction-level variables used for event selection that are not input features
SELECTION_OBSERVERS = ["nLepton_good", "tr_isvalid", "lep0_pdgId", "lep1_pdgId", "MET_pt"]

# testing with a single file
if __name__ == "__main__":

    # full loader
    tt2l = RDataLoader(
        input_paths=["/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-2_nJ2p_nB2p_trvalid/2016/TTLep_pow_nominal.root"],
        tree_name="Events",
        branches=SELECTION_OBSERVERS + observables.ALL_FEATURES,
        # selection=[SELECTIONS["minMll"], SELECTIONS["goodTop"]],
        n_split=1,
        splitting_strategy="events",
        strict_branches=False,
        weight_branches=[
            "weight",
        ],
        feature_names=observables.ALL_FEATURES,
        observer_names=SELECTION_OBSERVERS,
    )

    # loader with selection
    tt2l_sel = RDataLoader(
        input_paths=["/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-2_nJ2p_nB2p_trvalid/2016/TTLep_pow_nominal.root"],
        tree_name="Events",
        branches=SELECTION_OBSERVERS + observables.ALL_FEATURES,
        selection=[SELECTIONS["minMll"]],
        n_split=1,
        splitting_strategy="events",
        strict_branches=False,
        weight_branches=[
            "weight",
        ],
        feature_names=observables.ALL_FEATURES,  # includes dilep_mass
        observer_names=SELECTION_OBSERVERS,
    )

    dilep_mass = tt2l.features(feature_names=["dilep_mass"])
    dilep_mass_with_selection = tt2l_sel.features(feature_names=["dilep_mass"])
    print(f"minimum dilepton mass: without cut: {ak.min(dilep_mass)}; with cut: {ak.min(dilep_mass_with_selection)}")