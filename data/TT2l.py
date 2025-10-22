# File: data/datasets_min.py
from __future__ import annotations
import os

from RDataLoader import RDataLoader
import observables 
import sys
sys.path.insert(0, '..')
import common.user as user

# --- input file (same construction as before) ---
_PATH = os.path.join(
    user.training_data_dir,
    "training-ntuples-v7/MVA-training/"
    "PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/"
    "TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root",
)


#loader_top_kinematics = RDataLoader(
#    input_paths=[_PATH],
#    tree_name="Events",
#    branches=OBS + TOP_KINEMATICS,
#    selection=None,
#    n_split=1,
#    splitting_strategy="events",
#    strict_branches=False,
#)

loader = RDataLoader(
    input_paths=[_PATH],
    tree_name="Events",
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
)
