import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# columns = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']

class HiggsDataset(Dataset):
    def __init__(self, file_path, dataset_names, indices=None, load_mode='multiple', single_dataset_name=None):
        """
        Initialize the dataset.
        
        Parameters:
        - file_path: str, path to the HDF5 file.
        - dataset_names: list of str, names of the datasets to load (e.g., ['data', 'labels', 'detailed_labels', 'weights']).
        - indices: list or numpy array, indices to use for the dataset.
        - load_mode: str, 'single' or 'multiple', specifies whether to load a single dataset or multiple datasets.
        - single_dataset_name: str, the name of the dataset to load when load_mode is 'single'.
        """
        self.file_path = file_path
        self.dataset_names = dataset_names
        self.indices = indices
        self.load_mode = load_mode
        self.single_dataset_name = single_dataset_name

        # Only store dataset size, do not load data into memory
        with h5py.File(self.file_path, 'r') as f:
            if load_mode == 'single':
                self.dataset_size = f[self.single_dataset_name].shape[0]
            elif load_mode == 'multiple':
                self.dataset_size = f[dataset_names[0]].shape[0]

        if self.indices is None:
            self.indices = np.arange(self.dataset_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        index = self.indices[idx]

        # Load data on demand to reduce memory usage
        with h5py.File(self.file_path, 'r') as f:
            if self.load_mode == 'single':
                data = f[self.single_dataset_name][index]
            elif self.load_mode == 'multiple':
                data = {name: f[name][index] for name in self.dataset_names}

        return data

# Function to create DataLoader
def create_dataloader(file_path, dataset_names, batch_size, split_ratio=0.7, load_mode='multiple', single_dataset_name=None, light=False):
    with h5py.File(file_path, 'r') as f:
        dataset_size = f[dataset_names[0]].shape[0] if load_mode == 'multiple' else f[single_dataset_name].shape[0]

    indices = np.arange(dataset_size)
    if light:
        indices = indices[:int(dataset_size * 0.01)]

    split_point = int(len(indices) * split_ratio)
    train_indices, test_indices = indices[:split_point], indices[split_point:]

    train_dataset = HiggsDataset(file_path, dataset_names, train_indices, load_mode, single_dataset_name)
    test_dataset = HiggsDataset(file_path, dataset_names, test_indices, load_mode, single_dataset_name)

    # Reduce num_workers to 1 or 0 to avoid memory overhead from multiple workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
