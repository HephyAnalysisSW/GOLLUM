import sys
import os
import h5py
import pandas as pd
import numpy as np
import argparse
sys.path.insert( 0, '..')
import common.user as user

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

parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument("-c", "--config", help="Path to the config file.")
args = parser.parse_args()

config_name = os.path.basename(args.config).replace(".yaml", "")
output_directory = os.path.join ( user.output_directory, config_name)

# define toy
h5_file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_2/set_2.0_pseudo_exp_10.h5"

# convert to "official" format
test_toy = load_h5_to_test_set(h5_file_path)

# run fit
from Model import Model
m = Model(get_train_set=None, systematics=None, config_path=args.config)
m.cfg["tmp_path"] = os.path.join( output_directory, f"tmp_data" )
results = m.predict(test_toy)
print(results)
