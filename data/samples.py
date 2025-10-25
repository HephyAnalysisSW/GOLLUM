# File: data/samples.py
from __future__ import annotations
import os
import sys
import numpy as np

# project roots (keep as in your current layout)
sys.path.insert(0, '..')

from RDataLoader import RDataLoader
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
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    feature_names=observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names=observables.OBSERVERS,
)


# -----------------------------
# Selections (GG, QG, QQ)
# -----------------------------
def _id_columns(training_observers: np.ndarray, obs_names=None):
    """
    Resolve id1/id2 columns from observers using the canonical order in observables.OBSERVERS.
    """
    names = observables.OBSERVERS if obs_names is None else obs_names
    ix_id1 = names.index("Generator_id1")
    ix_id2 = names.index("Generator_id2")
    id1 = training_observers[:, ix_id1].astype(int, copy=False)
    id2 = training_observers[:, ix_id2].astype(int, copy=False)
    return id1, id2


def sel_GG(training_observers: np.ndarray, obs_names=None) -> np.ndarray:
    """gluon–gluon fusion: (id1==21) & (id2==21)"""
    id1, id2 = _id_columns(training_observers, obs_names)
    return (id1 == 21) & (id2 == 21)


def sel_QG(training_observers: np.ndarray, obs_names=None) -> np.ndarray:
    """quark–gluon mixed (either leg gluon, the other (anti)quark)"""
    id1, id2 = _id_columns(training_observers, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return ((id1 == 21) & np.isin(np.abs(id2), abs_ids)) | ((id2 == 21) & np.isin(np.abs(id1), abs_ids))


def sel_QQ(training_observers: np.ndarray, obs_names=None) -> np.ndarray:
    """quark–antiquark annihilation: both legs are quarks with opposite sign"""
    id1, id2 = _id_columns(training_observers, obs_names)
    abs_ids = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    return (np.isin(np.abs(id1), abs_ids) & np.isin(np.abs(id2), abs_ids) & (id1 * id2 < 0))


# -----------------------------
# Lightweight “views” over tt2l
# (the loader changes will wire these up to reuse I/O)
# -----------------------------
class SelectionView:
    """
    Minimal placeholder for a selection-based view over a base loader.
    The base loader handles I/O; this view only carries the selection metadata.
    """
    def __init__(self, base: RDataLoader, name: str, selection_fn):
        self.base = base
        self.name = name
        self.selection_fn = selection_fn
        # Optionally override feature/observer sets per view later:
        self.feature_names = getattr(base, "feature_names", None)
        self.observer_names = getattr(base, "observer_names", None)


# Three selected samples/views based on tt2l
tt2l_GG = SelectionView(base=tt2l, name="GG", selection_fn=sel_GG)
tt2l_QG = SelectionView(base=tt2l, name="QG", selection_fn=sel_QG)
tt2l_QQ = SelectionView(base=tt2l, name="QQ", selection_fn=sel_QQ)


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

