import h5py
import numpy as np

import sys
sys.path.insert(0, '..')

import common.data_structure as data_structure

class H5DataLoader:

    def __init__(self, file_path, batch_size=None, n_split=None, selection_function=None):
        """
        Initialize the data loader.

        Parameters:
        - file_path: str, path to the HDF5 file.
        - batch_size: int, number of samples per batch (takes precedence over n_split).
        - n_split: int, number of splits to divide the dataset (ignored if batch_size is provided).
        - selection_function: callable, a function that takes a numpy array and returns a reduced dataset.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.n_split = n_split
        self.selection_function = selection_function  # Optional selection function
        self._init_dataset()

    def _init_dataset(self):
        """Load dataset metadata and determine batch size or splits."""
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_size = f['data'].shape[0]  # Assume all datasets have the same size

        # Determine batch size if not provided
        if self.batch_size is None:
            if self.n_split is None:
                raise ValueError("Either batch_size or n_split must be provided.")
            self.batch_size = int(np.ceil(self.dataset_size / self.n_split))
        print(f"data_loader_2: Initialize reading from {self.file_path}")

    def set_selection(self, selection_function):
        """
        Update the selection function.

        Parameters:
        - selection_function: callable, a function that takes a numpy array and returns a reduced dataset.
        """
        self.selection_function = selection_function

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

        # Load the batch
        with h5py.File(self.file_path, 'r') as f:
            start = self.current_index
            end = min(self.current_index + self.batch_size, self.dataset_size)
            batch_data = f['data'][start:end] 

        # Apply the selection function if provided
        if self.selection_function:
            batch_data = self.selection_function(batch_data)
            
        self.current_index += self.batch_size
        return batch_data

    @staticmethod
    def features( arr ):
        return( arr[:, :len(data_structure.feature_names)] )

    @staticmethod
    def weights( arr ):
        return( arr[:, data_structure.weight_index] )

    @staticmethod
    def labels( arr ):
        return( arr[:, data_structure.label_index] )

    @staticmethod
    def split( arr ):
        return H5DataLoader.features(arr), H5DataLoader.weights(arr), H5DataLoader.labels(arr)

    @staticmethod
    def get_weight_sum( data_loader, small=False):
        from tqdm import tqdm
        sum_ = 0.
        for batch in tqdm(data_loader, desc="Computing weight sum", unit="batch"):
            sum_ += data_loader.weights(batch).sum()
            if small: break
        return sum_

if __name__=="__main__":

    import selections

    batch_size = None #64**2
    n_split    = 10000

    file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/jes_0p99_met_1.h5"

    # Example1: load all the data, then select
    data_loader_1 = H5DataLoader(
        file_path = file_path, 
        batch_size= batch_size, 
        n_split   = n_split,
        )

    # select only the "inclusive" selection
    for batch in data_loader_1:
        features1, weights1, labels1 = data_loader_1.split(selections.inclusive(batch))
        break

    # Example2 (equivalent): loop over selected data 
    data_loader_2 = H5DataLoader(
        file_path = file_path, 
        batch_size= batch_size, 
        n_split   = n_split,
        selection_function=selections.inclusive,
        )

    for batch in data_loader_2:
        features2, weights2, labels2 = data_loader_2.split(batch)
        break
