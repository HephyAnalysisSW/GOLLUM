import numpy as np
import io
import h5py
from Inference import Inference
from test_set import load_h5_to_test_set


h5_file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_2/set_2.0_pseudo_exp_0.h5"

toy_from_memory = load_h5_to_test_set(h5_file_path)

inference = Inference(cfg={}, toy_from_memory=toy_from_memory)

print("Saving toy dataset to memory...")
inference.saveToyToMemory()

assert isinstance(inference.toy_from_memory, io.BytesIO), "Error: self.toy_from_memory is not a BytesIO object"

print("Loading toy dataset from memory...")
inference.toy_from_memory.seek(0) 
with h5py.File(inference.toy_from_memory, "r") as hf:
    loaded_data = hf["data"][:] 

assert loaded_data.shape == (961611, 30), f"Error: Loaded data has incorrect shape {loaded_data.shape}, expected (100, 30)"

assert np.all(loaded_data[:, -2] == 1), "Error: Weights column is not all 1"

assert np.all(loaded_data[:, -1] == -1), "Error: Labels column is not all -1"


print("All tests passed! saveToyToMemory() and loadToyFromMemory() are working correctly.")
