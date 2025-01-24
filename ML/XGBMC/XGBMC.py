import xgboost as xgb
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure
import common.user as user
from tqdm import tqdm

class XGBMC:
    def __init__(self, config=None, input_dim=None, classes=None, model_dir=None, num_boost_round=None):
        """
        Initialize the XGBoost multiclass classifier model.

        Parameters:
        - config: configuration object with hyperparameters.
        - input_dim: int, number of features in the input data.
        - classes: list of class labels.
        """
        if config is not None:
            self.config = config
            self.input_dim = config.input_dim
            self.classes = config.classes
            self.model_dir = config.model_dir
            self.num_boost_round = config.num_boost_round
        elif not ( input_dim is None or classes is None or model_dir is None or num_boost_round is None):
            self.config=None
            self.input_dim=input_dim
            self.classes=classes
            self.model_dir=model_dir
            self.num_boost_round=num_boost_round 
        else:
            raise Exception("Please provide a config.")

        self.num_classes = len(self.classes)
        self.model = None
        self.feature_means = getattr(config, 'feature_means', None)
        self.feature_variances = getattr(config, 'feature_variances', None)
        self.weight_sums = getattr(config, 'weight_sums', None)

    def load_training_data(self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader(selection=selection, selection_function=None, n_split=n_split)

    def train(self):
        """
        Train the XGBoost classifier using the batched data loader.
        """
        if self.data_loader is None:
            raise ValueError("Data loader is not initialized. Call `load_training_data` first.")

        if not hasattr( self, "params" ):
            self.params = {
                'objective': 'multi:softprob',  # Multiclass classification with soft probabilities
                'num_class': self.num_classes,  # Number of classes
                'eta': self.config.learning_rate,  # Learning rate
                'max_depth': self.config.max_depth,  # Max depth of trees
                'subsample': self.config.subsample,  # Fraction of samples per tree
                'colsample_bytree': self.config.colsample_bytree,  # Fraction of features per tree
                'lambda': self.config.l2_reg,  # L2 regularization (alpha)
                'alpha': self.config.l1_reg,  # L1 regularization (lambda)
                'eval_metric': 'mlogloss',  # Cross-entropy loss
                'seed': self.config.seed,  # Random seed
            }

        for epoch in tqdm(range(self.start_epoch, self.num_boost_round), desc="Training epochs"):
            for i_batch, batch in enumerate(self.data_loader):
                data, weights, raw_labels = self.data_loader.split(batch)
                data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

                if self.weight_sums:
                    class_weights = np.array([
                        self.weight_sums[data_structure.label_encoding[label]] for label in self.classes
                    ])
                    total = sum(class_weights)
                    class_weights = np.array([total / class_weights[i] for i in range(len(class_weights))])
                    weights *= class_weights[raw_labels.astype('int')]

                dtrain = xgb.DMatrix(data_norm, label=raw_labels, weight=weights)

                if self.model is None:
                    self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=1, xgb_model=None)
                else:
                    self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=1, xgb_model=self.model)

            self.save(epoch=epoch + 1)

    def predict(self, data, ic_scaling=True):

        # apply scaler
        data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

        dtest = xgb.DMatrix(data_norm)
        class_probs = self.model.predict(dtest)  # Probabilities for each class

        # put back the inclusive xsec
        if ic_scaling:
            class_weights = np.array([self.weight_sums[data_structure.label_encoding[label]] for label in self.classes])
            total = sum(class_weights)
            class_weights = np.array([total / class_weights[i] for i in range(len(class_weights))])
            return class_probs/class_weights # DCR
        else:
            return class_probs

    def compute_likelihood_ratio(self, X_test):
        """
        Compute the likelihood ratio based on predicted probabilities.

        Parameters:
        - X_test: np.ndarray, test features.

        Returns:
        - np.ndarray, likelihood ratios.
        """
        probabilities = self.predict(X_test)
        return probabilities / (1 - probabilities)

    def save(self, epoch):
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.json")
        metadata_path = os.path.join(self.model_dir, f"model_metadata_{epoch:04d}.pkl")

        os.makedirs(self.model_dir, exist_ok=True)

        if self.model:
            self.model.save_model(model_path)
            metadata = {
                'feature_means': self.feature_means,
                'feature_variances': self.feature_variances,
                'weight_sums': self.weight_sums,
                'epoch': epoch,
                'input_dim': self.input_dim,
                'classes': self.classes,
                'model_dir': self.model_dir,
                'num_boost_round': self.num_boost_round,
                'params':self.params,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            #print(f"Model and metadata saved for epoch {epoch}")
        else:
            raise Exception("Model is not trained yet!")

    @classmethod
    def load(cls, model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".json")]
        if not model_files:
            return None, 0

        model_files.sort()
        last_model_file = model_files[-1]
        epoch = int(last_model_file.split('_')[1].split('.')[0])
        model_path = os.path.join(model_dir, last_model_file)
        metadata_path = os.path.join( model_dir, last_model_file.replace('model', 'model_metadata').replace('.json', '.pkl'))
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        instance = cls(config=None, input_dim=metadata['input_dim'], classes=metadata['classes'],model_dir=metadata['model_dir'],num_boost_round=metadata['num_boost_round'])
        instance.model = xgb.Booster()
        instance.model.load_model(model_path)
        instance.model_dir = model_dir
        instance.feature_means = metadata['feature_means']
        instance.feature_variances = metadata['feature_variances']
        instance.weight_sums = metadata['weight_sums']
        instance.params = metadata['params']
        print(f"Model and metadata loaded from {model_path}, epoch {epoch}")
        return instance, epoch
