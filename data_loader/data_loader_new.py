import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# The order of the 30 features in 'data' is as follows: PRI_had_pt, PRI_had_eta, PRI_had_phi, PRI_lep_pt, PRI_lep_eta, PRI_lep_phi, 
# PRI_met, PRI_met_phi, PRI_jet_num, PRI_jet_leading_pt, PRI_jet_leading_eta, PRI_jet_leading_phi, PRI_jet_subleading_pt, PRI_jet_subleading_eta, PRI_jet_subleading_phi, 
# PRI_jet_all_pt, DER_mass_transverse_met_lep, DER_mass_vis, DER_pt_h, DER_deltaeta_jet_jet, DER_mass_jet_jet, DER_prodeta_jet_jet, DER_deltar_had_lep, 
# DER_pt_tot, DER_sum_pt, DER_pt_ratio_lep_tau, DER_met_phi_centrality, DER_lep_eta_centrality, event_weight, labels

class HiggsDataset(Dataset):
    """
    Dataset for loading Higgs signal and background data from memory.
    """

    def __init__(self, data):
        """
        Args:
            data (numpy array): The full dataset array containing 30 features.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        return torch.tensor(sample, dtype=torch.float32)

def split_dataset(data, train_ratio=0.7):
    """
    Split dataset into training and testing datasets.
    """
    num_train = int(len(data) * train_ratio)

    train_data = data[:num_train]
    test_data = data[num_train:]

    return train_data, test_data

def create_higgs_dataloaders(path_to_file, batch_size=32):
    """
    Create DataLoaders for Higgs dataset.

    Args:
        path_to_file (str): Path to the .h5 file containing the datasets.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: DataLoaders for training and testing datasets of htautau, ztautau, diboson, and ttbar.
    """
    dataloaders = {}

    with h5py.File(path_to_file, 'r') as f:
        for label in ['htautau_data', 'ztautau_data', 'diboson_data', 'ttbar_data']:
            data = f[label][:]

            # Split dataset into training and testing
            train_data, test_data = split_dataset(data)

            # Create training and testing datasets
            train_dataset = HiggsDataset(train_data)
            test_dataset = HiggsDataset(test_data)

            # Create DataLoaders for training and testing
            dataloaders[f'train_{label}'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            dataloaders[f'test_{label}'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return (
        dataloaders['train_htautau_data'], dataloaders['test_htautau_data'],
        dataloaders['train_ztautau_data'], dataloaders['test_ztautau_data'],
        dataloaders['train_diboson_data'], dataloaders['test_diboson_data'],
        dataloaders['train_ttbar_data'], dataloaders['test_ttbar_data']
    )

def create_lightweight_dataloaders(path_to_file, batch_size=32):
    """
    Create lightweight DataLoaders containing 1% of each dataset.

    Args:
        path_to_file (str): Path to the .h5 file containing the datasets.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: DataLoaders for lightweight training and testing datasets.
    """
    dataloaders = {}

    with h5py.File(path_to_file, 'r') as f:
        for label in ['htautau_data', 'ztautau_data', 'diboson_data', 'ttbar_data']:
            data = f[label][:]

            # Select 1% of the data randomly
            num_samples = int(len(data) * 0.01)
            indices = np.random.choice(len(data), num_samples, replace=False)

            data_subset = data[indices]

            # Split dataset into training and testing
            train_data, test_data = split_dataset(data_subset)

            # Create training and testing datasets
            train_dataset = HiggsDataset(train_data)
            test_dataset = HiggsDataset(test_data)

            # Create DataLoaders for training and testing
            dataloaders[f'train_{label}'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            dataloaders[f'test_{label}'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return (
        dataloaders['train_htautau_data'], dataloaders['test_htautau_data'],
        dataloaders['train_ztautau_data'], dataloaders['test_ztautau_data'],
        dataloaders['train_diboson_data'], dataloaders['test_diboson_data'],
        dataloaders['train_ttbar_data'], dataloaders['test_ttbar_data']
    )
