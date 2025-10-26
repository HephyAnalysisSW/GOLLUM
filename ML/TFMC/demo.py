import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# Load trained TFMC
from ML.TFMC.TFMC import TFMC
model_dir = "/groups/hephy/cms/robert.schoefbeck/SBIPDF/models/TFMC/tfmc_split_processes/"
tfmc = TFMC.load(model_dir)

# Import the new datasets directly (RDataLoader / SelectionView objects)
import data.samples as samples

# choose which of the new datasets to test on
dataset_names = ["tt2l"]  # adjust as you like

def load_features_and_weights(loader, shard=0):
    """
    Works for both base RDataLoader and SelectionView.
    Returns (X, w) on the requested shard.
    """
    # If it's a SelectionView, build mask from the base and materialize masked arrays
    if hasattr(loader, "base") and hasattr(loader, "selection_fn"):
        base = loader.base
        # observer/feature names
        obs_names  = getattr(loader, "observer_names", None) or getattr(base, "observer_names", None)
        feat_names = getattr(loader, "feature_names",  None) or getattr(base, "feature_names",  None)
        # compute mask and materialize
        mask = base.compute_mask(
            selection_name=getattr(loader, "name", "view"),
            selection_fn=loader.selection_fn,
            shard=shard,
            observer_names=obs_names,
        )
        X = base.features_from_mask(shard=shard, mask=mask, feature_names=feat_names)
        G = base.observers_from_mask(shard=shard, mask=mask, observer_names=obs_names)
    else:
        # Base sample
        X, G = loader.features_and_observers(shard=shard, n=None)

    # extract weights (defaults to column named 'weight')
    obs_names = getattr(loader, "observer_names", None) or getattr(getattr(loader, "base", loader), "observer_names", [])
    try:
        w_idx = obs_names.index("weight")
    except ValueError:
        raise RuntimeError("Observer 'weight' not found in observer_names.")
    w = G[:, w_idx]
    return X, w

# loop chosen datasets
for name in dataset_names:
    if not hasattr(samples, name):
        print(f"[warn] dataset '{name}' not found in data.samples")
        continue
    loader = getattr(samples, name)

    # just first shard for a quick demo
    if len(getattr(loader, "base", loader)) == 0:
        print(f"[warn] dataset '{name}' has zero shards")
        continue

    X, w = load_features_and_weights(loader, shard=0)

    # probabilities and IC-scaled posteriors
    probs = tfmc.predict(X, probability=True)
    dcr   = tfmc.predict(X, probability=False)

    print(f"\nDataset: {name}")
    print("  X shape:", X.shape)
    print("  first 5 probs row-sum:", probs[:5].sum(axis=1))
    print("  first 5 dcr   row-sum:", dcr[:5].sum(axis=1))
    print("  first 5 probs:\n", probs[:5])
    print("  first 5 dcr:\n", dcr[:5])

