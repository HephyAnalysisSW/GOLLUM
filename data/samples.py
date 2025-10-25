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
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    feature_names=observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names=observables.OBSERVERS,  # must include "weight", id1, id2, etc.
)

# -----------------------------
# Selections (GG, QG, QQ)
#   We build masks from OBSERVERS (only id1/id2 are actually needed).
#   The SelectionView will request selection_feature_names just for mask eval.
# -----------------------------
def _id_columns(obs_matrix: np.ndarray, obs_names=None):
    names = observables.OBSERVERS if obs_names is None else obs_names
    ix_id1 = names.index("Generator_id1")
    ix_id2 = names.index("Generator_id2")
    id1 = obs_matrix[:, ix_id1].astype(int, copy=False)
    id2 = obs_matrix[:, ix_id2].astype(int, copy=False)
    return id1, id2

def sel_GG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    """gluon–gluon fusion: (id1==21) & (id2==21)"""
    id1, id2 = _id_columns(obs_matrix, obs_names)
    return (id1 == 21) & (id2 == 21)

def sel_QG(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    """quark–gluon mixed (either leg gluon, the other (anti)quark)"""
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return ((id1 == 21) & np.isin(np.abs(id2), abs_ids)) | ((id2 == 21) & np.isin(np.abs(id1), abs_ids))

def sel_QQ(obs_matrix: np.ndarray, obs_names=None) -> np.ndarray:
    """quark–antiquark annihilation: both legs are (anti)quarks with opposite sign"""
    id1, id2 = _id_columns(obs_matrix, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return (np.isin(np.abs(id1), abs_ids) & np.isin(np.abs(id2), abs_ids) & (id1 * id2 < 0))

# Names of observer columns needed to evaluate the mask fast
_SELECTION_OBS = ["Generator_id1", "Generator_id2"]

# -----------------------------
# First-class views (behave like loaders)
#   - Use observers to compute masks (selection_feature_names = _SELECTION_OBS)
#   - Inherit feature_names / observer_names from base
# -----------------------------
tt2l_GG = SelectionView(
    base=tt2l,
    name="GG",
    selection_fn=lambda obs_mat, names: sel_GG(obs_mat, names),
    selection_feature_names=["Generator_id1", "Generator_id2"],
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
# Minimal demo using the new loader helpers & view masks
# -----------------------------
if __name__ == "__main__":
    print("\n=== Demo: base sample (single read) ===")
    # Use high-level convenience: features_and_observers (internally reads shard 0 once and caches)
    X_base, G_base = tt2l.features_and_observers(shard=0, n=None)
    if len(X_base) > 0:
        print("[base] first event — features:", X_base[0])
        print("[base] first event — observers:", G_base[0])
    else:
        print("[base] shard is empty")

    # --------------------------------------------------------------

    print("\n=== Demo: three views over the same base (no additional read) ===")
    # Ensure shard 0 is resident in cache (no-op if already loaded above)
    _ = tt2l.load_selection_shard(0)

    views = [("GG", tt2l_GG), ("QG", tt2l_QG), ("QQ", tt2l_QQ)]
    for name, view in views:
        # Ask loader to compute/cache the mask for this view on shard 0
        mask = tt2l.compute_mask(selection_name=view.name,
                                 selection_fn=view.selection_fn,
                                 shard=0,
                                 observer_names=view.observer_names)

        # Materialize masked features/observers using the cached shard
        X_sel = tt2l.features_from_mask(shard=0, mask=mask, feature_names=view.feature_names)
        G_sel = tt2l.observers_from_mask(shard=0, mask=mask, observer_names=view.observer_names)

        if len(X_sel) == 0:
            print(f"[{name}] selection empty on this shard")
            continue

        print(f"[{name}] first selected event — features:", X_sel[0])
        print(f"[{name}] first selected event — observers:", G_sel[0])

        # Only demonstrate a single logical iteration per view

