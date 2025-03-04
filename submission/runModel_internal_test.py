import sys
import os
#sys.path.insert( 0, '..')
from model import Model

import h5py
import pandas as pd
import numpy as np


def load_h5_to_test_set(h5_file_path):
    with h5py.File(h5_file_path, "r") as hf:
        full_data = hf["data"][:]  # (N, 30)
    test_data = pd.DataFrame(full_data[:, :28])  # (N, 28)
    test_weights = full_data[:, 28]  # (N,)
    test_set = {
        "data": test_data,
        "weights": test_weights
    }
    return test_set

# define toy
h5_file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_2/set_2.0_pseudo_exp_10.h5"

# convert to "official" format
test_toy = load_h5_to_test_set(h5_file_path)

# run fit
m = Model(get_train_set=None, systematics=None)
results = m.predict(test_toy)
print(results)
