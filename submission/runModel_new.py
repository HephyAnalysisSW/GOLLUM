from Model import Model
import h5py
import pandas as pd
import numpy as np
from toy_generator import run_pseudo_experiments 

test_toy, pkl_toy = run_pseudo_experiments(
    tes=None,
    jes=None,
    soft_met=None,
    ttbar_scale=None,
    diboson_scale=None,
    bkg_scale=None,
    num_pseudo_experiments=100,
    num_of_sets=1,
    ground_truth_mus=[2.0],
)


# run fit
m = Model(get_train_set=None, systematics=None)
results = m.predict(test_toy)
print(results)
