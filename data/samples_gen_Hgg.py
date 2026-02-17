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

# OBJ: TBranch   H_pt    H_pt/F : 0 at: 0x55a227027f10
# OBJ: TBranch   H_y H_y/F : 0 at: 0x55a227030f10
# OBJ: TBranch   mgg mgg/F : 0 at: 0x55a227031470
# OBJ: TBranch   ptgg    ptgg/F : 0 at: 0x55a2270319d0
# OBJ: TBranch   ygg ygg/F : 0 at: 0x55a227031f30
# OBJ: TBranch   Generator_weight    Generator_weight/F : 0 at: 0x55a227032490
# OBJ: TBranch   Generator_scalePDF  Generator_scalePDF/F : 0 at: 0x55a227032ab0
# OBJ: TBranch   Generator_x1    Generator_x1/F : 0 at: 0x55a2270330d0
# OBJ: TBranch   Generator_x2    Generator_x2/F : 0 at: 0x55a227033630
# OBJ: TBranch   Generator_id1   Generator_id1/I : 0 at: 0x55a227033b90
# OBJ: TBranch   Generator_id2   Generator_id2/I : 0 at: 0x55a227036070
# OBJ: TBranch   LHEWeight_originalXWGTUP    LHEWeight_originalXWGTUP/F : 0 at: 0x55a2270365a0
# OBJ: TBranch   nLHEPdfWeight   nLHEPdfWeight/I : 0 at: 0x55a227036bc0
# OBJ: TBranch   LHEPdfWeight    LHEPdfWeight[nLHEPdfWeight]/F : 0 at: 0x55a227037120

kinematics = ["H_pt", "H_y", "mgg", "ptgg", "ygg"]
observers  = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF"]

H_gg_2016 = RDataLoader(
    input_paths=[os.path.join(
        "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/Hgg-gen-ntuples/",
        "RunIISummer20UL16NanoAODAPVv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_preVFP_v11-v2/",
    )],
    tree_name="Events",
    branches = kinematics + kinematics,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,
    weight_branches=[],
    feature_names=kinematics,
    observer_names=observers,
)


if __name__ == "__main__":
    print("Base:", H_gg_2016)
    F,O,W = H_gg_2016.materialize(0,"fow")
    print("Shapes:", F.shape, O.shape, W.shape)
