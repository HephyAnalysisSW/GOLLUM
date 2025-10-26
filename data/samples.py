# File: data/samples.py
from __future__ import annotations
import os
import sys
import numpy as np

# project roots (keep as in your current layout)
sys.path.insert(0, '..')

from RDataLoader import RDataLoader
from SelectionView import SelectionView
import observables
import common.user as user


# add these two names once
_EXTRA_REWEIGHTS = ["reweightLeptonSFUp", "reweightLeptonSFDown"]

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
    # Load the union needed for training + observers
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY + _EXTRA_REWEIGHTS,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    feature_names=observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names=observables.OBSERVERS+ _EXTRA_REWEIGHTS,  # must include "weight", id1, id2, etc.
    # weight defaults to "weight"; override here if needed:
    # weight="weight",
    # weight_branches=[],  # if using a callable, list required branches
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

# Names of observer columns needed to evaluate the mask fast
_SELECTION_OBS = ["Generator_id1", "Generator_id2"]

# -----------------------------
# First-class views (behave like loaders)
# -----------------------------

# Nominal GG (uses base weight)
tt2l_GG = SelectionView(
    base=tt2l,
    name="GG",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    # weight=None  # default: use base weight
)

# Up variation: weight * reweightLeptonSFUp
tt2l_GG_LeptonSFUp = SelectionView(
    base=tt2l,
    name="GG_LeptonSFUp",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    weight="reweightLeptonSFUp",
)

# Down variation: weight * reweightLeptonSFDown
tt2l_GG_LeptonSFDown = SelectionView(
    base=tt2l,
    name="GG_LeptonSFDown",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
    weight="reweightLeptonSFDown",
)

tt2l_QG = SelectionView(
    base=tt2l,
    name="QG",
    selection_fn=lambda obs_mat, names: sel_QG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
)

tt2l_QQ = SelectionView(
    base=tt2l,
    name="QQ",
    selection_fn=lambda obs_mat, names: sel_QQ(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
)

__all__ = [
    "tt2l",
    "tt2l_GG", "tt2l_QG", "tt2l_QQ",
    "sel_GG", "sel_QG", "sel_QQ",
]

# -----------------------------
# Minimal demo using the new materialize API
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

