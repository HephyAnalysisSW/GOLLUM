# File: data/samples.py
from __future__ import annotations
from typing import Dict, List, Optional

import os
import sys
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from RDataLoader import RDataLoader
from SelectionView import SelectionView
import observables
import common.user as user

# Global, always loaded branches 
_EXTRA_REWEIGHTS = [
    "reweightTopPt",
    "reweightTrigger", "reweightTriggerUp", "reweightTriggerDown",
    "reweightL1Prefire", "reweightL1PrefireUp", "reweightL1PrefireDown",
    "reweightPU", "reweightPUUp", "reweightPUDown",
    "reweightLeptonSF", "reweightLeptonSFUp", "reweightLeptonSFDown",
    "reweightBTagSF1a_SF",
    "reweightBTagSF1a_SF_b_Up", "reweightBTagSF1a_SF_b_Down",
    "reweightBTagSF1a_SF_l_Up", "reweightBTagSF1a_SF_l_Down",
]

# -----------------------------
# Base sample (nominal)
# -----------------------------
tt2l = RDataLoader(
    input_paths=[os.path.join(
        user.training_data_dir,
        "training-ntuples-v7/MVA-training/"
        "PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/"
        "TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root",
    )],
    tree_name="Events",
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY + _EXTRA_REWEIGHTS,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    # Base product (explicit):
    weight_branches=[
        "weight",
        "reweightTopPt",
        "reweightTrigger",
        "reweightL1Prefire",
        "reweightPU",
        "reweightLeptonSF",
        "reweightBTagSF1a_SF",
    ],
    feature_names=observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names=observables.OBSERVERS + _EXTRA_REWEIGHTS,
)

def _replace(items: List[str], nominal: str, new_value: str) -> List[str]:
    """Return a copy of items with 'nominal' replaced by new_value (if present)."""
    out = list(items)
    try:
        i = out.index(nominal)
        out[i] = new_value
    except ValueError:
        # nominal not present -> append the replacement
        out.append(new_value)
    return out

def make_pdf_weight_views(
    base_loader,
    name_prefix: str,
    *,
    base_weights: Optional[List[str]] = None,
    selection_fn=None,
    selection_feature_names: Optional[List[str]] = None,
) -> Dict[str, SelectionView]:
    """
    Build a dict of SelectionView objects for the requested weight variations.
    - All views REPLACE the varied group's nominal with Up/Down, keeping all other base factors.
    - If selection_fn is given, it is applied to every created view (broadcasted selection).
    """
    if base_weights is None:
        base_weights = list(getattr(base_loader, "weight_branches", []) or [])
    if not base_weights:
        base_weights = []

    groups = {
        "Trigger":   ("reweightTrigger",   "reweightTriggerUp",   "reweightTriggerDown"),
        "L1Prefire": ("reweightL1Prefire", "reweightL1PrefireUp", "reweightL1PrefireDown"),
        "PU":        ("reweightPU",        "reweightPUUp",        "reweightPUDown"),
        "LeptonSF":  ("reweightLeptonSF",  "reweightLeptonSFUp",  "reweightLeptonSFDown"),
    }

    btag_nom = "reweightBTagSF1a_SF"
    btag_b_up, btag_b_dn = "reweightBTagSF1a_SF_b_Up", "reweightBTagSF1a_SF_b_Down"
    btag_l_up, btag_l_dn = "reweightBTagSF1a_SF_l_Up", "reweightBTagSF1a_SF_l_Down"

    views: Dict[str, SelectionView] = {}

    # One-group Up/Down variations
    for tag, (nom, up, down) in groups.items():
        w_up   = _replace(base_weights, nom,   up)
        w_down = _replace(base_weights, nom, down)
        views[f"{name_prefix}_{tag}Up"] = SelectionView(
            base_loader, name=f"{tag}Up", weight=w_up,
            selection_fn=selection_fn, selection_feature_names=selection_feature_names
        )
        views[f"{name_prefix}_{tag}Down"] = SelectionView(
            base_loader, name=f"{tag}Down", weight=w_down,
            selection_fn=selection_fn, selection_feature_names=selection_feature_names
        )

    # BTag “b” and “l” variations (replace scalar factor, no squaring)
    w_b_up   = _replace(base_weights, btag_nom, btag_b_up)
    w_b_down = _replace(base_weights, btag_nom, btag_b_dn)
    w_l_up   = _replace(base_weights, btag_nom, btag_l_up)
    w_l_down = _replace(base_weights, btag_nom, btag_l_dn)

    views[f"{name_prefix}_BTag_b_Up"] = SelectionView(
        base_loader, name="BTag_b_Up", weight=w_b_up,
        selection_fn=selection_fn, selection_feature_names=selection_feature_names
    )
    views[f"{name_prefix}_BTag_b_Down"] = SelectionView(
        base_loader, name="BTag_b_Down", weight=w_b_down,
        selection_fn=selection_fn, selection_feature_names=selection_feature_names
    )
    views[f"{name_prefix}_BTag_l_Up"] = SelectionView(
        base_loader, name="BTag_l_Up", weight=w_l_up,
        selection_fn=selection_fn, selection_feature_names=selection_feature_names
    )
    views[f"{name_prefix}_BTag_l_Down"] = SelectionView(
        base_loader, name="BTag_l_Down", weight=w_l_down,
        selection_fn=selection_fn, selection_feature_names=selection_feature_names
    )

    return views

# -----------------------------
# Parton-channel selections (nominal, inherit base weights)
# -----------------------------
def _id_columns(obs_matrix: np.ndarray, obs_names=None):
    names = observables.OBSERVERS if obs_names is None else obs_names
    ix1, ix2 = names.index("Generator_id1"), names.index("Generator_id2")
    id1 = obs_matrix[:, ix1].astype(int, copy=False)
    id2 = obs_matrix[:, ix2].astype(int, copy=False)
    return id1, id2

def sel_GG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names); return (id1 == 21) & (id2 == 21)

def sel_QG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1,2,3,4,5,6], dtype=int)
    return ((id1 == 21) & np.isin(np.abs(id2), abs_ids)) | ((id2 == 21) & np.isin(np.abs(id1), abs_ids))

def sel_QQ(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1,2,3,4,5,6], dtype=int)
    return (np.isin(np.abs(id1), abs_ids) & np.isin(np.abs(id2), abs_ids) & (id1 * id2 < 0))

tt2l_GG = SelectionView(tt2l, name="tt2l_GG",
                        selection_fn=lambda G,n: sel_GG(G,n),
                        selection_feature_names=["Generator_id1","Generator_id2"])

tt2l_QG = SelectionView(tt2l, name="tt2l_QG",
                        selection_fn=lambda G,n: sel_QG(G,n),
                        selection_feature_names=["Generator_id1","Generator_id2"])

tt2l_QQ = SelectionView(tt2l, name="tt2l_QQ",
                        selection_fn=lambda G,n: sel_QQ(G,n),
                        selection_feature_names=["Generator_id1","Generator_id2"])

# -----------------------------
# Systematic variations:
#   1) no selection (global)
#   2) with GG/QG/QQ selections (broadcasted)
# -----------------------------
_views_global = make_pdf_weight_views(tt2l, name_prefix="tt2l")
globals().update(_views_global)

_views_GG = make_pdf_weight_views(
    tt2l, name_prefix="tt2l_GG",
    selection_fn=lambda G,n: sel_GG(G,n),
    selection_feature_names=["Generator_id1","Generator_id2"],
)
globals().update(_views_GG)

_views_QG = make_pdf_weight_views(
    tt2l, name_prefix="tt2l_QG",
    selection_fn=lambda G,n: sel_QG(G,n),
    selection_feature_names=["Generator_id1","Generator_id2"],
)
globals().update(_views_QG)

_views_QQ = make_pdf_weight_views(
    tt2l, name_prefix="tt2l_QQ",
    selection_fn=lambda G,n: sel_QQ(G,n),
    selection_feature_names=["Generator_id1","Generator_id2"],
)
globals().update(_views_QQ)

__all__ = [
    "tt2l",
    # global variations:
    *_views_global.keys(),
    # nominal selections:
    "tt2l_GG", "tt2l_QG", "tt2l_QQ",
    # selection-broadcasted variations:
    *_views_GG.keys(),
    *_views_QG.keys(),
    *_views_QQ.keys(),
    # selection helpers:
    "sel_GG", "sel_QG", "sel_QQ",
]

if __name__ == "__main__":
    print("Base:", tt2l)
    print("Nominal GG:", tt2l_GG)
    print("Example global view:", _views_global.get("tt2l_TriggerUp"))
    F,O,W = tt2l.materialize(0,"fow")
    print("Shapes:", F.shape, O.shape, W.shape)

