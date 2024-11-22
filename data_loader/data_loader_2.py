import h5py
import numpy as np

class H5DataLoader:
    def __init__(self, file_path, datasets, batch_size=None, n_split=None):
        """
        Initialize the data loader.
        
        Parameters:
        - file_path: str, path to the HDF5 file.
        - datasets: list of str, names of the datasets to load (e.g., ['data', 'weights', 'detailed_labels']).
        - batch_size: int, number of samples per batch (takes precedence over n_split).
        - n_split: int, number of splits to divide the dataset (ignored if batch_size is provided).
        """
        self.file_path = file_path
        self.datasets = datasets
        self.batch_size = batch_size
        self.n_split = n_split
        self._init_dataset()
    
    def _init_dataset(self):
        """Load dataset metadata and initialize batch size or splits."""
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_sizes = [f[ds].shape[0] for ds in self.datasets]
            if len(set(self.dataset_sizes)) > 1:
                raise ValueError("All datasets must have the same first dimension size.")
            self.dataset_size = self.dataset_sizes[0]
        
        # Determine batch size if not provided
        if self.batch_size is None:
            if self.n_split is None:
                raise ValueError("Either batch_size or n_split must be provided.")
            self.batch_size = int(np.ceil(self.dataset_size / self.n_split))
        
        self.indices = np.arange(self.dataset_size)
    
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
batch_size = None #64**2
n_split    = 1000

# Initialize the data loader
data_loader = H5DataLoader(file_path, datasets, batch_size=batch_size, n_split=n_split)

# Iterate through the dataset
for batch in data_loader:
    data = batch['data']
    weights = batch['weights']
    labels = batch['detailed_labels']
    print(data.shape, weights.shape, labels.shape)

