import h5py
import numpy as np

import sys
sys.path.insert(0, '..')
    
import logging
logger = logging.getLogger('UNC')

import common.data_structure as data_structure

class H5DataLoader:

    def __init__(self, file_path, batch_size=None, n_split=None, process=None, selection_function=None):
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
        self.process = process # Optional process selection
        self.selection_function = selection_function  # Optional selection function
        try:
            self._init_dataset()
        except Exception as e:
            logger.error(f"Problem opening {self.file_path}")
            raise e

    def _init_dataset(self):
        """Load dataset metadata and determine batch size or splits."""
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_size = f['data'].shape[0]  # Assume all datasets_hephy have the same size

        # Determine batch size if not provided
        if self.batch_size is None:
            if self.n_split is None:
                raise ValueError("Either batch_size or n_split must be provided.")
            self.batch_size = int(np.ceil(self.dataset_size / self.n_split))
        logger.info(f"data_loader_2: Initialize reading from {self.file_path}")

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
        try:
            with h5py.File(self.file_path, 'r') as f:
                start = self.current_index
                end = min(self.current_index + self.batch_size, self.dataset_size)
                batch_data = f['data'][start:end]
        except Exception as e:
            logger.error(f"Problem in {self.file_path}")
            raise e

        # Apply the selection function if provided
        if self.selection_function:
            batch_data = self.selection_function(batch_data)

        # Apply the process selection 
        if self.process:
            batch_data = batch_data[batch_data[:,data_structure.label_index]==data_structure.label_encoding[self.process]]
            
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
    def get_weight_sum( data_loader, small=False, selection=None):
        from tqdm import tqdm
        sum_ = 0.
        for batch in tqdm(data_loader, desc="Computing weight sum", unit="batch"):
            if selection is not None:
                sum_ += data_loader.weights(selection(batch)).sum()
            else:
                sum_ += data_loader.weights(batch).sum()
            if small: break
        return sum_

if __name__=="__main__":

    import selections

    batch_size = None #64**2
    n_split    = 10000

    file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/tes_0p99_jes_0p99_met_2.h5"
    file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/tes_1p01_jes_1p01_met_6.h5"

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
