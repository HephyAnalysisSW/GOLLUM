import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.data_structure as data_structure 

class Scaler:
    def __init__(self):
        self.feature_means = {}
        self.feature_variances = {}
        self.selection = None
        self.process   = None

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls()
            new_instance.feature_means = old_instance.feature_means
            new_instance.feature_variances = old_instance.feature_variances
            new_instance.selection   = old_instance.selection if hasattr(old_instance, "selection") else None
            new_instance.process     = old_instance.process if hasattr(old_instance, "process") else None

            return new_instance

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename, 'wb') as file_:
            pickle.dump(self, file_)

    def load_training_data(self, datasets_hephy, selection, process=None, n_split=10):
        self.data_loader = datasets_hephy.get_data_loader(selection=selection, process=process, selection_function=None, n_split=n_split)
        self.selection   = selection
        self.process     = process

    def train(self, small=False):
        num_features = None
        feature_sums = None
        feature_sq_sums = None
        total_samples = 0

        for batch in tqdm(self.data_loader, desc="Computing feature mean/variance", unit="batch"):
            data = batch[:, :-2]  # Exclude weights and labels
            if feature_sums is None:
                num_features = data.shape[1]
                feature_sums = np.zeros(num_features)
                feature_sq_sums = np.zeros(num_features)

            feature_sums += np.sum(data, axis=0)
            feature_sq_sums += np.sum(data**2, axis=0)
            total_samples += data.shape[0]

            if small:
                break

        self.feature_means = feature_sums / total_samples
        self.feature_variances = (feature_sq_sums / total_samples) - (self.feature_means**2)

    def __str__(self):
        if hasattr(self, "selection") and self.selection is not None:
            selection = '\033[1m'+self.selection+'\033[0m'
        else:
            selection = "(not set)"
        if not hasattr(self, "process"):
            process = "(not set)"
        elif self.process is None:
            process = '(\033[1m'+'not set'+'\033[0m)'
        else:
            process = '(\033[1m'+self.process+'\033[0m)'

        line = f"Scaler: selection {selection} process {process}"

        lines = [line]
        for i, feature_name in enumerate(data_structure.feature_names):
            mean = self.feature_means[i]
            variance = self.feature_variances[i]
            lines.append(f"{feature_name}: mean={mean:.3f}, variance={variance:.3f}")
        return "\n".join(lines)

    def normalize(self, data):
        """Normalize data using the computed mean and variance."""
        return (data - self.feature_means) / np.sqrt(self.feature_variances)

