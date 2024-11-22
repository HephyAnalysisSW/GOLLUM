import h5py
import numpy as np

import h5py
import numpy as np

class H5DataLoader:
    def __init__(self, file_path, datasets, batch_size):
        """
        Initialize the data loader.
        
        Parameters:
        - file_path: str, path to the HDF5 file.
        - datasets: list of str, names of the datasets to load (e.g., ['data', 'weights', 'detailed_labels']).
        - batch_size: int, number of samples per batch.
        """
        self.file_path = file_path
        self.datasets = datasets
        self.batch_size = batch_size
        self._init_dataset()
    
    def _init_dataset(self):
        """Load dataset metadata and initialize indices."""
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_sizes = [f[ds].shape[0] for ds in self.datasets]
            if len(set(self.dataset_sizes)) > 1:
                raise ValueError("All datasets must have the same first dimension size.")
            self.dataset_size = self.dataset_sizes[0]
            self.indices = np.arange(self.dataset_size)  # Sequential indices
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(self.dataset_size / self.batch_size))
    
    def __iter__(self):
        """Create an iterator for the dataset."""
        self.current_index = 0
        return self
    
    def __next__(self):
        """Load and return the next batch."""
        if self.current_index >= self.dataset_size:
            raise StopIteration
        
        # Compute batch indices
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        
        # Load data for the current batch
        with h5py.File(self.file_path, 'r') as f:
            batch = {ds: f[ds][batch_indices] for ds in self.datasets}
        
        self.current_index += self.batch_size
        return batch

file_path = "/eos/vbc/group/mlearning/data/PUdata/syst_train_set_test.h5"
# Assume 'data.h5' contains datasets: 'data', 'weights', 'detailed_labels'
datasets = ['data', 'weights', 'detailed_labels']
batch_size = 64

# Initialize the data loader
data_loader = H5DataLoader(file_path, datasets, batch_size)

# Iterate through the dataset
for batch in data_loader:
    data = batch['data']
    weights = batch['weights']
    labels = batch['detailed_labels']
    print(data, weights, labels)

