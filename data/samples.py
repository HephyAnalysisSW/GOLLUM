# File: data/samples.py
from __future__ import annotations
import os
import sys
import numpy as np

# project roots (keep as in your current layout)
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from RDataLoader import RDataLoader
from SelectionView import SelectionView
import observables
import common.user as user

# Add these once so they're loaded from ROOT (include central SF!)
_EXTRA_REWEIGHTS = ["reweightLeptonSF", "reweightLeptonSFUp", "reweightLeptonSFDown"]

# -----------------------------
# Base sample (no selection)
# -----------------------------
tt2l = RDataLoader(
    input_paths=[os.path.join(
        user.training_data_dir,
        "training-ntuples-v7/MVA-training/"
        "PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/"
        "TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root",
    )],
    tree_name="Events",
    # Load the union needed for training + observers (+ reweights)
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY + _EXTRA_REWEIGHTS,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    # NEW: explicit product for base weights; if you remove this, weights default to ones.
    weight_branches=["weight", "reweightLeptonSF"],
    feature_names=observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names=observables.OBSERVERS + _EXTRA_REWEIGHTS,  # includes "weight", Generator_id*, and SF branches
)

# -----------------------------
# Selections (GG, QG, QQ)
# -----------------------------
def _id_columns(obs_matrix: np.ndarray, obs_names=None):
    names = observables.OBSERVERS if obs_names is None else obs_names
    ix_id1 = names.index("Generator_id1")
    ix_id2 = names.index("Generator_id2")
    id1 = obs_matrix[:, ix_id1].astype(int, copy=False)
    id2 = obs_matrix[:, ix_id2].astype(int, copy=False)
    return id1, id2

def sel_GG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names)
    return (id1 == 21) & (id2 == 21)

def sel_QG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return ((id1 == 21) & np.isin(np.abs(id2), abs_ids)) | ((id2 == 21) & np.isin(np.abs(id1), abs_ids))

def sel_QQ(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return (np.isin(np.abs(id1), abs_ids) & np.isin(np.abs(id2), abs_ids) & (id1 * id2 < 0))

# -----------------------------
# Views (inherit features/observers; optional selection and weight override)
# NOTE: weight override REPLACES the base weight when provided (no multiplication with base).
#       Pass None to inherit the base product ["weight","reweightLeptonSF"].
# -----------------------------

# Global Up/Down (no further selection); replace with weight * reweightLeptonSFUp/Down
tt2l_LeptonSFUp = SelectionView(
    base=tt2l,
    name="LeptonSFUp",
    selection_fn=None,
    weight=["weight", "reweightLeptonSFUp"],
)

tt2l_LeptonSFDown = SelectionView(
    base=tt2l,
    name="LeptonSFDown",
    selection_fn=None,
    weight=["weight", "reweightLeptonSFDown"],
)

# Nominal GG (inherits base weight: ["weight","reweightLeptonSF"])
tt2l_GG = SelectionView(
    base=tt2l,
    name="GG",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    # weight=None  -> inherit base weight
)

# GG with LeptonSF Up/Down (explicit replacement with product)
tt2l_GG_LeptonSFUp = SelectionView(
    base=tt2l,
    name="GG_LeptonSFUp",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    weight=["weight", "reweightLeptonSFUp"],
)

tt2l_GG_LeptonSFDown = SelectionView(
    base=tt2l,
    name="GG_LeptonSFDown",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    weight=["weight", "reweightLeptonSFDown"],
)

tt2l_QG = SelectionView(
    base=tt2l,
    name="QG",
    selection_fn=lambda obs_mat, names: sel_QG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    # weight=None -> inherit base weight
)

tt2l_QQ = SelectionView(
    base=tt2l,
    name="QQ",
    selection_fn=lambda obs_mat, names: sel_QQ(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    # weight=None -> inherit base weight
)

__all__ = [
    "tt2l",
    "tt2l_LeptonSFUp", "tt2l_LeptonSFDown",
    "tt2l_GG", "tt2l_GG_LeptonSFUp", "tt2l_GG_LeptonSFDown",
    "tt2l_QG", "tt2l_QQ",
    "sel_GG", "sel_QG", "sel_QQ",
]

# -----------------------------
# Minimal demo using the materialize API
# -----------------------------
if __name__ == "__main__":
    print("\n=== Demo: base.materialize ===")
    try:
        F, O, W = tt2l.materialize(shard=0, what="fow")
        print(" base shapes:", F.shape, O.shape, W.shape, "| W[:3] =", W[:3])
    except Exception as e:
        print(" base.materialize failed (likely no files available here):", e)

    print("\n=== Demo: view.materialize (GG) ===")
    try:
        Fo = tt2l_GG.materialize(shard=0, what="wof")
        print(" GG shapes (w,o,f):", [a.shape for a in Fo])
    except Exception as e:
        print(" view.materialize failed:", e)

