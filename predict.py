import sys
import os
import json
from model import Model

import h5py
import pandas as pd
import numpy as np

import argparse

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

parser = argparse.ArgumentParser(description="Prediction.")
parser.add_argument("--TEST_DATA_CLEAN_PATH", help="Path to the test data.")
#parser.add_argument("--nevent", type=int, help="Path to the test data.")
parser.add_argument("--SUBMISSION_DIR", help="Path to save the results.")
args = parser.parse_args()

# convert to "official" format
test_toy = load_h5_to_test_set(args.TEST_DATA_CLEAN_PATH)
#test_toy_reduced = {}
#nevent = min(args.nevent,len(test_toy['weights']))
#for k in test_toy.keys():
#  test_toy_reduced[k] = test_toy[k][0:nevent]

# run fit
m = Model(get_train_set=None, systematics=None)
results = m.predict(test_toy)

os.makedirs(args.SUBMISSION_DIR, exist_ok=True)
with open(os.path.join(args.SUBMISSION_DIR,'results.json'),'w') as f:
  json.dump(results,f)

print("Results saved in {}".format(os.path.join(args.SUBMISSION_DIR,'results.json')))
