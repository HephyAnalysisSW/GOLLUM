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

def _hgg_loader(dirname: str) -> RDataLoader:
    return RDataLoader(
        input_paths=[os.path.join(
            "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/Hgg-gen-ntuples/",
            dirname,
        )],
        tree_name="Events",
        branches=kinematics + kinematics,
        selection=None,
        n_split=1,
        splitting_strategy="events",
        strict_branches=False,
        weight_branches=[],
        feature_names=kinematics,
        observer_names=observers,
    )


H_gg_2016APV = _hgg_loader(
    "RunIISummer20UL16NanoAODAPVv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_preVFP_v11-v2"
)
H_gg_2016APV_ext1 = _hgg_loader(
    "RunIISummer20UL16NanoAODAPVv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_preVFP_v11_ext1-v4"
)
H_gg_2016 = _hgg_loader(
    "RunIISummer20UL16NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_v17-v2"
)
H_gg_2016_ext1 = _hgg_loader(
    "RunIISummer20UL16NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mcRun2_asymptotic_v17_ext1-v4"
)
H_gg_2017 = _hgg_loader(
    "RunIISummer20UL17NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mc2017_realistic_v9-v1"
)
H_gg_2017_ext1 = _hgg_loader(
    "RunIISummer20UL17NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_mc2017_realistic_v9_ext1-v4"
)
H_gg_2018 = _hgg_loader(
    "RunIISummer20UL18NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_upgrade2018_realistic_v16_L1v1-v1"
)
H_gg_2018_ext1 = _hgg_loader(
    "RunIISummer20UL18NanoAODv9__GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8__106X_upgrade2018_realistic_v16_L1v1_ext1-v4"
)

ttbar_kinematics = [
    "ttbar_pt",
    "ttbar_mass",
    "ttbar_y",
    "t_pt",
    "tbar_pt",
    "t_y",
    "tbar_y",
]

ttbar_observers = [
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "Generator_scalePDF",
    "ttbar_decay_class",
    "ttbar_ne",
    "ttbar_nmu",
    "ttbar_ntau",
    "t_decay_flavor",
    "tbar_decay_flavor",
]


def _ttbar_loader(dirname: str) -> RDataLoader:
    return RDataLoader(
        input_paths=[os.path.join(
            "/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/TTbar-gen-ntuples/",
            dirname,
        )],
        tree_name="Events",
        branches=ttbar_observers + ttbar_kinematics,
        selection=None,
        n_split=1,
        splitting_strategy="events",
        strict_branches=False,
        weight_branches=[],
        feature_names=ttbar_kinematics,
        observer_names=ttbar_observers,
    )


TTbar_2L2Nu_2016APV = _ttbar_loader(
    "RunIISummer20UL16NanoAODAPVv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mcRun2_asymptotic_preVFP_v11-v1"
)
TTbar_SemiLeptonic_2016APV = _ttbar_loader(
    "RunIISummer20UL16NanoAODAPVv9__TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8__106X_mcRun2_asymptotic_preVFP_v11-v1"
)
TTbar_2L2Nu_2016 = _ttbar_loader(
    "RunIISummer20UL16NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mcRun2_asymptotic_v17-v1"
)
TTbar_SemiLeptonic_2016 = _ttbar_loader(
    "RunIISummer20UL16NanoAODv9__TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8__106X_mcRun2_asymptotic_v17-v1"
)
TTbar_2L2Nu_2017 = _ttbar_loader(
    "RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v1"
)
TTbar_SemiLeptonic_2017 = _ttbar_loader(
    "RunIISummer20UL17NanoAODv9__TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v1"
)
TTbar_2L2Nu_2018 = _ttbar_loader(
    "RunIISummer20UL18NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_upgrade2018_realistic_v16_L1v1-v1"
)
TTbar_SemiLeptonic_2018 = _ttbar_loader(
    "RunIISummer20UL18NanoAODv9__TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8__106X_upgrade2018_realistic_v16_L1v1-v1"
)


if __name__ == "__main__":
    print("Base:", H_gg_2016)
    F,O,W = H_gg_2016.materialize(0,"fow")
    print("Shapes:", F.shape, O.shape, W.shape)
