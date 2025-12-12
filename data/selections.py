"""
Global selection definitions for RDataLoader.

Each selection is a callable: (ak.Array) -> np.ndarray[bool]
"""
import numpy as np
import awkward as ak
from typing import Dict, Callable, Union

from RDataLoader import SelectionFn

# Pre-defined global selections - from TOP-20-006
# Based on v2.2+ CMGRDF ntuple names
SELECTIONS: Dict[str, SelectionFn] = {
    "nL2": lambda ar: ar["nLepton_good"] == 2,
    "nJ2p": lambda ar: ar["nSelJet"] >= 2,
    "nB2p": lambda ar: ar["nBJet"] >= 2,
    "goodTop": lambda ar: ar["tr_isvalid"] == 1,
    "minMll": lambda ar: ar["dilep_mass"] > 20,
    "SameFlavorMET": lambda ar: (np.abs(ar["lep0_pdgId"]) - np.abs(ar["lep1_pdgId"]) != 0) + (ar["MET_pt"] > 40)
    "SameFlavorMll": lambda ar: (np.abs(ar["lep0_pdgId"]) - np.abs(ar["lep1_pdgId"]) != 0) + (ar["dilep_mass"] < 76) + (ar["dilep_mass"] > 106)
}


def get_selection(name_or_expr: Union[str, SelectionFn]) -> SelectionFn:
    """
    Resolve a selection by name or expression string.
    
    Args:
        name_or_expr: Either a key in SELECTIONS, or a callable
        
    Returns:
        A selection function
    """
    if callable(name_or_expr):
        return name_or_expr
    
    if name_or_expr in SELECTIONS:
        return SELECTIONS[name_or_expr]
    
    # not implementing a fancier/automated method, since it will
    # open up much more complexity which then becomes hard to manage