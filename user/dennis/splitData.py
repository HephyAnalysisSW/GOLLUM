import sys, os
import math
import numpy as np
import ROOT

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from common.FeatureSelector import FeatureSelector
from common.features import feature_names, class_labels
import common.features as features
from data_loader.data_loader_2 import H5DataLoader

import h5py

output_dir = "/scratch-cbe/users/dennis.schwarz/HiggsChallenge"
print(f"Write split files to {output_dir}")

# Create new HDF5 files for each label
output_files = {
    label: h5py.File(f'{output_dir}/{label}_nominal.h5', 'w') for label in class_labels
}


# Initialize datasets for each label
datasets = {}
for label, h5file in output_files.items():
    datasets[label] = {
        'data': h5file.create_dataset('data', shape=(0,30), maxshape=(None,30), dtype='float32'),
        'weights': h5file.create_dataset('weights', shape=(0,), maxshape=(None,), dtype='float32'),
        'detailed_labels': h5file.create_dataset('detailed_labels', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
    }

# Function to append data to HDF5 datasets
def append_to_dataset(dataset, new_data):
    current_size = dataset.shape[0]
    new_size = current_size + new_data.shape[0]
    dataset.resize(new_size, axis=0)
    dataset[current_size:] = new_data

# Process batches
# Load data using H5DataLoader
NBatches = 100
dataLoader = H5DataLoader(
    os.path.join(user.data_directory, "nominal.h5"),
    ['data', 'weights', 'detailed_labels'],
    batch_size=None,
    n_split=NBatches,
    selection_function=None
)

# Iterate through the batches
NBatchesDone = 0
for batch in dataLoader:
    data = batch['data']
    weights = batch['weights']
    rawLabels = batch['detailed_labels']
    stringLabels = np.array([label.decode('utf-8') for label in rawLabels])

    combined_data = np.concatenate([data, weights, rawLabels], axis=1)

    # Split data based on labels
    for label in class_labels:
        mask = stringLabels == label
        if np.any(mask):
            append_to_dataset(datasets[label]['data'], combined_data[mask])

    NBatchesDone += 1
    print("%i/%i batches processed"%(NBatchesDone,NBatches))

# Close all output files
for h5file in output_files.values():
    h5file.close()

print("Data successfully split into separate HDF5 files.")
