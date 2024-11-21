import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# The order of the 28 features in 'data' is as follows: PRI_had_pt, PRI_had_eta, PRI_had_phi, PRI_lep_pt, PRI_lep_eta, PRI_lep_phi, PRI_met, PRI_met_phi, PRI_jet_num, PRI_jet_leading_pt, PRI_jet_leading_eta, PRI_jet_leading_phi, PRI_jet_subleading_pt, PRI_jet_subleading_eta, PRI_jet_subleading_phi, PRI_jet_all_pt, DER_mass_transverse_met_lep, DER_mass_vis, DER_pt_h, DER_deltaeta_jet_jet, DER_mass_jet_jet, DER_prodeta_jet_jet, DER_deltar_had_lep, DER_pt_tot, DER_sum_pt, DER_pt_ratio_lep_tau, DER_met_phi_centrality, DER_lep_eta_centrality

class HiggsDataset(Dataset):
    """
    Dataset for loading Higgs signal and background data from HDF5 file.
    """

    def __init__(self, path_to_file, return_label=True):
        """
        Args:
            path_to_file (str): Path to the .h5 file containing the data.
            return_label (bool): Whether to return the label of each sample.
        """
        self.path_to_file = path_to_file
        self.return_label = return_label

        # Load data from HDF5 file
        with h5py.File(self.path_to_file, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['labels'][:]
            self.weights = f['weights'][:]
            self.detailed_labels = f['detailed_labels'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'data': self.data[idx],
            'weights': self.weights[idx],
            'detailed_labels': self.detailed_labels[idx]
        }

        if self.return_label:
            sample['label'] = self.labels[idx]

        return sample


def get_higgs_dataloader(path_to_file, batch_size=32, return_label=True, shuffle=True):
    """
    Create a DataLoader for Higgs dataset.

    Args:
        path_to_file (str): Path to the .h5 file containing the data.
        batch_size (int): Batch size for the DataLoader.
        return_label (bool): Whether to return the label of each sample.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = HiggsDataset(path_to_file, return_label=return_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
