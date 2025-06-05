from typing import List
import numpy as np
import glob
import os
import pickle
import yaml
import logging
logger = logging.getLogger('UNC')

class Forest:
    def __init__(self, trees: List):
        self.trees = trees

        self.X_mean = None  # Standardization
        self.X_std  = None

    def set_standardization( self, X_mean, X_std ):
        self.X_mean = X_mean  # Standardization
        self.X_std  = X_std

    def standardize_input(self, X):
        """
        Standardize input features: subtract mean and divide by std.
        Stores the mean and std in the tree instance for use during inference.
        """
        if self.X_mean is None or self.X_std is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1.0  # avoid division by zero

        return (X - self.X_mean) / self.X_std

    def predict(self, X, summed=True):
        """
        Predict by summing the outputs of all trees.

        Args:
            X (np.ndarray): input data of shape (N, d)

        Returns:
            predictions (np.ndarray): array of shape (N,) with predicted values
        """
        X = self.standardize_input(X)
        if summed:
            return np.sum(np.array([tree.predict(X) for tree in self.trees]), axis=0)
        else:
            return np.array([tree.predict(X) for tree in self.trees])

    def train_step(self, X, y, w, train_config):
        """
        Train all trees  
        """

        logger.debug("→ Training all trees in the forest.")
    
        predictions = self.predict( X )

        for i_tree, tree in enumerate(self.trees):
            logger.debug(f"→ Training tree {i_tree}/{len(self.trees)}")

            predictions -= tree.predict(X)
            tree.train_step( X, y-predictions, w, train_config=train_config)
            predictions += tree.predict(X)

        #print (predictions.shape, predictions)

    def save(self, path, epoch):
        os.makedirs(path, exist_ok=True)

        # Save everything in one binary pickle
        forest_data = {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'trees': self.trees,
        }

        forest_file = os.path.join(path, f"forest_epoch_{epoch}.pkl")
        with open(forest_file, 'wb') as f:
            pickle.dump(forest_data, f)

    @classmethod
    def load(cls, path, epoch=None):
        if epoch is None:
            pkl_files = sorted(glob.glob(os.path.join(path, "forest_epoch_*.pkl")))
            if not pkl_files:
                raise FileNotFoundError("No saved forests found in directory.")
            latest_file = pkl_files[-1]
            epoch = int(latest_file.split('_')[-1].split('.')[0])

        forest_file = os.path.join(path, f"forest_epoch_{epoch}.pkl")
        with open(forest_file, 'rb') as f:
            forest_data = pickle.load(f)

        forest = cls(forest_data['trees'])
        forest.set_standardization(forest_data['X_mean'], forest_data['X_std'])
        return forest
