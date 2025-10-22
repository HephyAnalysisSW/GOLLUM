# File: data/datasets_min.py
from __future__ import annotations
import os

from RDataLoader import RDataLoader
import observables
import sys
sys.path.insert(0, '..')
import common.user as user

tt2l = RDataLoader(
    input_paths=[ os.path.join(
            user.training_data_dir, "training-ntuples-v7/MVA-training/PDF_tr-minDLmass20-dilepM-offZ1-njet3p-btagM2p/TTLep_Summer16_preVFP/TTLep_Summer16_preVFP.root",
        )],
    tree_name="Events",
    branches=observables.OBSERVERS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=False,

    feature_names  = observables.TOP_KINEMATICS + observables.LEPTON_KINEMATICS + observables.ASYMMETRY,
    observer_names = observables.OBSERVERS,
)

if __name__ == "__main__":
    # how many rows to show
    N_SHOW = 5

    # 1) Traditional: load awkward array, then pull scalars
    arr = tt2l[0]
    X_trad = tt2l.scalar_branches(arr, observables.LEPTON_KINEMATICS + observables.ASYMMETRY)[:N_SHOW]
    G_trad = tt2l.scalar_branches(arr, observables.OBSERVERS)[:N_SHOW]

    print("\n[traditional] iterate over awkward array + scalar_branches")
    for i in range(len(X_trad)):
        print(f"  i={i}  feats[0:3]={X_trad[i][:3]}  gen={G_trad[i]}")

    # 2) New: features() — no awkward array in user code
    X = tt2l.features(shard=0, n=N_SHOW)
    print("\n[features()] iterate over NumPy features")
    for i, row in enumerate(X):
        print(f"  i={i}  feats[0:3]={row[:3]}")

    # 3) New: features_and_observers() — both matrices at once
    X2, G2 = tt2l.features_and_observers(shard=0, n=N_SHOW)
    print("\n[features_and_observers()] iterate over NumPy features + observers")
    for i in range(len(X2)):
        print(f"  i={i}  feats[0:3]={X2[i][:3]}  gen={G2[i]}")

