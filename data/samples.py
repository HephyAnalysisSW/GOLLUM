# File: data/samples.py
from __future__ import annotations
from typing import Dict, List, Optional

import os
import sys
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

import observables
import common.user as user

# -----------------------------
# Base sample (nominal)
# -----------------------------
tt2l = RDataLoader(
    input_paths=[os.path.join(
        user.training_data_dir,
        "training-ntuples-v8/MVA-training/PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root",
    )],
    tree_name="Events",
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
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
    observer_names=observables.OBSERVERS,
)

jes_dir = "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/training-ntuples-v8-jec/MVA-training/"

# Forgot to add the weights in the make_ntuple step. But we don't need them.
tt2l_0p5_jesTotalDown = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_0p5_jesTotalDown/TTLep_0p5_jesTotalDown.root"))
tt2l_0p5_jesTotalUp   = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_0p5_jesTotalUp/TTLep_0p5_jesTotalUp.root"))
tt2l_1p0_jesTotalDown = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_1p0_jesTotalDown/TTLep_1p0_jesTotalDown.root"))
tt2l_1p0_jesTotalUp   = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_1p0_jesTotalUp/TTLep_1p0_jesTotalUp.root"))
tt2l_1p5_jesTotalDown = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_1p5_jesTotalDown/TTLep_1p5_jesTotalDown.root"))
tt2l_1p5_jesTotalUp   = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_1p5_jesTotalUp/TTLep_1p5_jesTotalUp.root"))
tt2l_2p0_jesTotalDown = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_2p0_jesTotalDown/TTLep_2p0_jesTotalDown.root"))
tt2l_2p0_jesTotalUp   = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_2p0_jesTotalUp/TTLep_2p0_jesTotalUp.root"))
tt2l_jesNominal       = tt2l.clone_from_files(os.path.join( jes_dir, "JEC_for_PDF_minDLmass20-dilepM-offZ1/TTLep_nominal/TTLep_nominal.root"))

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

tt2l_GG_0p5_jesTotalDown = SelectionView(tt2l_0p5_jesTotalDown , name = "tt2l_GG_0p5_jesTotalDown", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_GG_0p5_jesTotalUp   = SelectionView(tt2l_0p5_jesTotalUp   , name = "tt2l_GG_0p5_jesTotalUp", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_GG_1p0_jesTotalDown = SelectionView(tt2l_1p0_jesTotalDown , name = "tt2l_GG_1p0_jesTotalDown", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_GG_1p0_jesTotalUp   = SelectionView(tt2l_1p0_jesTotalUp   , name = "tt2l_GG_1p0_jesTotalUp", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_GG_1p5_jesTotalDown = SelectionView(tt2l_1p5_jesTotalDown , name = "tt2l_GG_1p5_jesTotalDown", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_GG_1p5_jesTotalUp   = SelectionView(tt2l_1p5_jesTotalUp   , name = "tt2l_GG_1p5_jesTotalUp", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_GG_2p0_jesTotalDown = SelectionView(tt2l_2p0_jesTotalDown , name = "tt2l_GG_2p0_jesTotalDown", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_GG_2p0_jesTotalUp   = SelectionView(tt2l_2p0_jesTotalUp   , name = "tt2l_GG_2p0_jesTotalUp", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_GG_jesNominal       = SelectionView(tt2l_jesNominal       , name = "tt2l_GG_jesNominal", selection_fn=lambda G,n: sel_GG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])

tt2l_QG_0p5_jesTotalDown = SelectionView(tt2l_0p5_jesTotalDown , name = "tt2l_QG_0p5_jesTotalDown", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QG_0p5_jesTotalUp   = SelectionView(tt2l_0p5_jesTotalUp   , name = "tt2l_QG_0p5_jesTotalUp", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QG_1p0_jesTotalDown = SelectionView(tt2l_1p0_jesTotalDown , name = "tt2l_QG_1p0_jesTotalDown", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QG_1p0_jesTotalUp   = SelectionView(tt2l_1p0_jesTotalUp   , name = "tt2l_QG_1p0_jesTotalUp", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QG_1p5_jesTotalDown = SelectionView(tt2l_1p5_jesTotalDown , name = "tt2l_QG_1p5_jesTotalDown", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QG_1p5_jesTotalUp   = SelectionView(tt2l_1p5_jesTotalUp   , name = "tt2l_QG_1p5_jesTotalUp", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QG_2p0_jesTotalDown = SelectionView(tt2l_2p0_jesTotalDown , name = "tt2l_QG_2p0_jesTotalDown", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QG_2p0_jesTotalUp   = SelectionView(tt2l_2p0_jesTotalUp   , name = "tt2l_QG_2p0_jesTotalUp", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QG_jesNominal       = SelectionView(tt2l_jesNominal       , name = "tt2l_QG_jesNominal", selection_fn=lambda G,n: sel_QG(G,n), selection_feature_names=["Generator_id1","Generator_id2"])

tt2l_QQ_0p5_jesTotalDown = SelectionView(tt2l_0p5_jesTotalDown , name = "tt2l_QQ_0p5_jesTotalDown", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QQ_0p5_jesTotalUp   = SelectionView(tt2l_0p5_jesTotalUp   , name = "tt2l_QQ_0p5_jesTotalUp", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QQ_1p0_jesTotalDown = SelectionView(tt2l_1p0_jesTotalDown , name = "tt2l_QQ_1p0_jesTotalDown", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"]) 
tt2l_QQ_1p0_jesTotalUp   = SelectionView(tt2l_1p0_jesTotalUp   , name = "tt2l_QQ_1p0_jesTotalUp", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QQ_1p5_jesTotalDown = SelectionView(tt2l_1p5_jesTotalDown , name = "tt2l_QQ_1p5_jesTotalDown", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QQ_1p5_jesTotalUp   = SelectionView(tt2l_1p5_jesTotalUp   , name = "tt2l_QQ_1p5_jesTotalUp", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QQ_2p0_jesTotalDown = SelectionView(tt2l_2p0_jesTotalDown , name = "tt2l_QQ_2p0_jesTotalDown", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QQ_2p0_jesTotalUp   = SelectionView(tt2l_2p0_jesTotalUp   , name = "tt2l_QQ_2p0_jesTotalUp", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])
tt2l_QQ_jesNominal       = SelectionView(tt2l_jesNominal       , name = "tt2l_QQ_jesNominal", selection_fn=lambda G,n: sel_QQ(G,n), selection_feature_names=["Generator_id1","Generator_id2"])

#__all__ = [
#    "tt2l",
#    # nominal selections:
#    "tt2l_GG", "tt2l_QG", "tt2l_QQ",
#    # selection helpers:
#    "sel_GG", "sel_QG", "sel_QQ",
#]

if __name__ == "__main__":
    print("Base:", tt2l)
    print("Nominal GG:", tt2l_GG)
    F,O,W = tt2l.materialize(0,"fow")
    print("Shapes:", F.shape, O.shape, W.shape)
