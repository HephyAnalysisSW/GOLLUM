import h5py
import numpy as np

datasets = ['data', 'weights', 'detailed_labels']

class H5DataLoader:
    def __init__(self, file_path, datasets=datasets, batch_size=None, n_split=None, selection_function=None):
        """
        Initialize the data loader.
        
        Parameters:
        - file_path: str, path to the HDF5 file.
        - datasets: list of str, names of the datasets to load (e.g., ['data', 'weights', 'detailed_labels']).
        - batch_size: int, number of samples per batch (takes precedence over n_split).
        - n_split: int, number of splits to divide the dataset (ignored if batch_size is provided).
        - selection_function: callable, a function that takes a numpy array and returns a boolean mask for selection.
        """
        self.file_path = file_path
        self.datasets = datasets
        self.batch_size = batch_size
        self.n_split = n_split
        self.selection_function = selection_function  # Optional selection function
        self._init_dataset()
    
    def _init_dataset(self):
        """Load dataset metadata and initialize batch size or splits."""
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_sizes = [f[ds].shape[0] for ds in self.datasets]
            if len(set(self.dataset_sizes)) > 1:
                raise ValueError("All datasets must have the same first dimension size.")
            self.dataset_size = self.dataset_sizes[0]
        
        # Initialize indices with all data
        self.indices = np.arange(self.dataset_size)
        
        # Determine batch size if not provided
        if self.batch_size is None:
            if self.n_split is None:
                raise ValueError("Either batch_size or n_split must be provided.")
            self.batch_size = int(np.ceil(self.dataset_size / self.n_split))
    
    def set_selection(self, selection_function):
        """
        Update the selection function.

        Parameters:
        - selection_function: callable, a function that takes a numpy array and returns a boolean mask for selection.
        """
        self.selection_function = selection_function

    def _apply_selection_to_indices(self, batch_indices):
        """
        Apply the selection function to filter the given indices.

        Parameters:
        - batch_indices: array-like, indices of the current batch.

        Returns:
        - Filtered indices based on the selection function.
        """
        if not self.selection_function:
            return batch_indices

        with h5py.File(self.file_path, 'r') as f:
            data = f["data"][batch_indices]  # Load only the relevant data for filtering
        mask = self.selection_function(data)  # Apply the selection function
        return batch_indices[mask]  # Return filtered indices

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
        batch_indices = self._apply_selection_to_indices(batch_indices)  # Filter indices

        # If no data remains after selection, skip this batch
        if len(batch_indices) == 0:
            self.current_index += self.batch_size
            return self.__next__()

        # Load data for the current batch
        with h5py.File(self.file_path, 'r') as f:
            batch = {ds: f[ds][batch_indices] for ds in self.datasets}
        
        self.current_index += self.batch_size
        return batch

if __name__=="__main__":

    import sys
    sys.path.insert( 0, "..")

    batch_size = None #64**2
    n_split    = 1000

    from common.FeatureSelector import FeatureSelector
    selector = FeatureSelector().build_selector([(300, "PRI_had_pt"), ("PRI_lep_eta", 0)])

    data_loader = H5DataLoader(
        file_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/nominal.h5" , 
        batch_size= batch_size, 
        n_split   = n_split,
        selection_function = selector 
        )

    for batch in data_loader:
        data = batch['data']
        weights = batch['weights']
        labels = batch['detailed_labels']
        print(data.shape, weights.shape, labels.shape)

        break
