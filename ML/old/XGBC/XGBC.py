import xgboost as xgb
import numpy as np
import os
import glob
import pickle
import matplotlib
matplotlib.use("Agg")  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure
import common.user as user
from tqdm import tqdm
from math import ceil, sqrt

class XGBC:
    def __init__(self, config=None, truth_key=None, feature_keys=None, input_dim=None, model_dir=None, num_boost_round=None):
        """
        Initialize the XGBoost regression model.

        Parameters:
        - config: configuration object with hyperparameters.
        - input_dim: int, number of features in the input data.
        - classes: list of class labels.
        """
        if config is not None:
            self.config = config
            self.input_dim = len(config.feature_keys)
            self.model_dir = config.model_dir
            self.num_boost_round = config.num_boost_round
            self.truth_key=config.truth_key
            self.feature_keys=config.feature_keys
        elif not ( input_dim is None or model_dir is None or num_boost_round is None):
            self.config=None
            self.input_dim=input_dim
            self.model_dir=model_dir
            self.num_boost_round=num_boost_round 
            self.truth_key=truth_key
            self.feature_keys=feature_keys
        else:
            raise Exception("Please provide a config.")

        self.model = None

    def load_training_data(self, data_dir):
        """
        Loads training data from a directory containing .npz files.

        Each .npz file is expected to have a key defined by self.truth_key (e.g. 'mu_true')
        and a key 'mu_measured'. This function computes the training target as the difference:
            target = mu_true - mu_measured
        and uses the features from self.config.feature_keys as input features.

        Parameters
        ----------
        data_dir : str
            The directory containing the .npz training files.
        """
        # Get list of all .npz files in the directory
        file_list = glob.glob(os.path.join(data_dir, '*.npz'))
        
        targets = []
        inputs = []
        
        for file in file_list:
            # Load the file. allow_pickle=True ensures we can load data that might be pickled.
            data = np.load(file, allow_pickle=True)
            
            # Compute the target as the difference between the truth and 'mu_measured'.
            target = data[self.truth_key] - data["mu_measured"]
            features = [data[key] for key in self.config.feature_keys]
            
            # Combine the features into a 2D array (one column per feature).
            features = np.column_stack(features)
            
            targets.append(target)
            inputs.append(features)
        
        # Concatenate all target vectors and feature matrices from the files along the first axis.
        self.truth = np.concatenate(targets, axis=0)
        self.data  = np.concatenate(inputs, axis=0)


    def train(self, plot_directory=None):
        """
        Train the XGBoost regressor on the entire training dataset.
        
        The model is trained to predict the difference:
             (truth key value - mu_measured)
        rather than the raw truth value.
        """
        # Ensure that training data has been loaded via load_training_data.
        if not hasattr(self, 'data') or not hasattr(self, 'truth'):
            raise ValueError("Training data not loaded. Call load_training_data first.")

        # Initialize training parameters if not already set, using a regression objective.
        if not hasattr(self, "params"):
            self.params = {
                'objective': 'reg:squarederror',  # Regression objective
                'eta': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'lambda': self.config.l2_reg,
                'alpha': self.config.l1_reg,
                'eval_metric': 'rmse',
                'seed': self.config.seed,
            }

        # Create a DMatrix from the entire dataset using the computed difference as label.
        dtrain = xgb.DMatrix(self.data, label=self.truth)

        # Determine remaining boosting rounds if training is resumed.
        current_round = getattr(self, 'start_epoch', 0)
        rounds_to_train = self.num_boost_round - current_round

        # Train the model (if self.model exists, resume training; otherwise, start from scratch).
        self.model = xgb.train(params=self.params,
                               dtrain=dtrain,
                               num_boost_round=rounds_to_train,
                               xgb_model=self.model)

        # Update the training counter and save the model.
        self.start_epoch = self.num_boost_round
        self.save(epoch=self.start_epoch)


    def save(self, epoch):
        """
        Save the current XGBoost model and its metadata.
        
        Parameters:
        - epoch: int, the current training epoch (or total number of boosting rounds).
        """
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.json")
        metadata_path = os.path.join(self.model_dir, f"model_metadata_{epoch:04d}.pkl")
        os.makedirs(self.model_dir, exist_ok=True)

        if self.model is not None:
            self.model.save_model(model_path)
            print( f"Model written to {model_path}.")
            metadata = {
                'epoch': epoch,
                'input_dim': self.input_dim,
                'model_dir': self.model_dir,
                'truth_key': self.truth_key,
                'feature_keys': self.feature_keys,
                'num_boost_round': self.num_boost_round,
                'params': self.params,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                print( f"Metadata written to {metadata_path}.")
        else:
            raise Exception("Model is not trained yet!")


    @classmethod
    def load(cls, model_dir, return_epoch=False):
        """
        Load the most recent saved model and its metadata from the given directory.

        Parameters:
        - model_dir: str, the directory containing the saved model files.
        - return_epoch: bool, if True, also return the epoch at which the model was saved.

        Returns:
        - instance: an instance of XGBC with the loaded model.
        - epoch (optional): int, the epoch number from the saved metadata.
        """
        model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".json")]
        if not model_files:
            return None, 0

        model_files.sort()
        last_model_file = model_files[-1]
        epoch = int(last_model_file.split('_')[1].split('.')[0])
        model_path = os.path.join(model_dir, last_model_file)
        metadata_path = os.path.join(model_dir, last_model_file.replace('model', 'model_metadata').replace('.json', '.pkl'))

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(config=None,
                       input_dim=metadata['input_dim'],
                       model_dir=metadata['model_dir'],
                       truth_key=metadata['truth_key'],
                       feature_keys=metadata['feature_keys'],
                       num_boost_round=metadata['num_boost_round'])
        instance.model = xgb.Booster()
        instance.model.load_model(model_path)
        instance.model_dir = model_dir
        instance.params = metadata['params']
        print(f"Model and metadata loaded from {model_path}, epoch {epoch}")
        
        if return_epoch:
            return instance, epoch
        else:
            return instance

    def predict(self, data):
        """
        Predict regression values for the provided input data.

        Parameters
        ----------
        data : np.ndarray
            A 2D array of input features.

        Returns
        -------
        predictions : np.ndarray
            A 1D array of predicted regression values.
        """
        dtest = xgb.DMatrix(data)
        return self.model.predict(dtest)

    def plot_results(self, plot_directory):
        """
        Generate two 2D histogram plots with the x-axis as mu_truth and 
        the y-axis as the difference (measured or predicted) minus mu_truth.
        
          1. For measured: y = mu_measured - mu_truth.
          2. For predicted: y = (mu_measured + predicted_diff) - mu_truth.
        
        The plots are saved in the provided plot_directory.
        """
        import matplotlib.pyplot as plt
        os.makedirs(plot_directory, exist_ok=True)

        # Retrieve the 'mu_measured' feature from the input data.
        try:
            mu_measured_index = self.feature_keys.index("mu_measured")
        except ValueError:
            raise ValueError("'mu_measured' key not found in feature_keys")

        from matplotlib.colors import LinearSegmentedColormap
        base_cmap = plt.cm.get_cmap('viridis', 256)
        newcolors = base_cmap(np.linspace(0, 1, 256))
        newcolors[0, :] = np.array([1, 1, 1, 1])
        newcmp = LinearSegmentedColormap.from_list('viridis_white0', newcolors)

        # Extract the measured values.
        mu_measured = self.data[:, mu_measured_index]
        # Compute mu_truth from the fact that self.truth = (mu_true - mu_measured)
        mu_truth = self.truth + mu_measured

        # For measured residual: y = mu_measured - mu_truth.
        measured_residual = mu_measured - mu_truth

        vmin, vmax = 0, 500  # Set your desired range for the color axis

        # Plot 1: 2D histogram of (mu_measured - mu_truth) vs mu_truth.
        plt.figure()
        h1 = plt.hist2d(mu_truth, measured_residual, bins=50, cmap=newcmp, vmin=vmin, vmax=vmax)
        plt.xlabel(self.truth_key)  # mu_truth on x-axis
        plt.ylabel("Measured Residual (mu_measured - " + self.truth_key + ")")
        plt.title("2D Histogram: Measured Residual vs " + self.truth_key)
        plt.colorbar(h1[3], label='Counts')
        plt.grid(True)
        plot_path1 = os.path.join(plot_directory, "hist2d_measured_residual_vs_truth.png")
        plt.savefig(plot_path1)
        plt.close()

        # Compute predictions for the training data.
        predicted_diff = self.predict(self.data)  # predicted_diff approximates (mu_truth - mu_measured)
        # Predicted mu_truth is given by: mu_measured + predicted_diff.
        # Thus, predicted residual = (mu_measured + predicted_diff) - mu_truth.
        predicted_residual = (mu_measured + predicted_diff) - mu_truth

        # Plot 2: 2D histogram of predicted residual vs mu_truth.
        plt.figure()
        h2 = plt.hist2d(mu_truth, predicted_residual, bins=50, cmap=newcmp, vmin=vmin, vmax=vmax)
        plt.xlabel(self.truth_key)  # mu_truth on x-axis
        plt.ylabel("Predicted Residual ((mu_measured + predicted_diff) - " + self.truth_key + ")")
        plt.title("2D Histogram: Predicted Residual vs " + self.truth_key)
        plt.colorbar(h2[3], label='Counts')
        plt.grid(True)
        plot_path2 = os.path.join(plot_directory, "hist2d_predicted_residual_vs_truth.png")
        plt.savefig(plot_path2)
        plt.close()

