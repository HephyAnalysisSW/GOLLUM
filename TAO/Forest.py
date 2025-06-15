from typing import List
import numpy as np
import glob
import os
import pickle
import logging
#from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from scipy import sparse

logger = logging.getLogger('UNC')

from quantize import quantize

class Forest:
    def __init__(self, trees: List, config: dict):
        self.trees = trees
        self.X_mean = None  # Standardization
        self.X_std  = None
        self.config = config

    def set_standardization(self, X_mean, X_std):
        self.X_mean = X_mean
        self.X_std  = X_std

    def standardize_input(self, X):
        if self.X_mean is None or self.X_std is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1.0
        return (X - self.X_mean) / self.X_std

    def predict(self, X, summed=True):
        X = self.standardize_input(X)
        if summed:
            return np.sum(np.array([tree.predict(X) for tree in self.trees]), axis=0)
        else:
            return np.array([tree.predict(X) for tree in self.trees])

    def train_step(self, X, y, w, train_config):
        logger.debug("→ Training all trees in the forest.")
        std_inputs = self.standardize_input(X)
        predictions = self.predict(X)
        for i_tree, tree in enumerate(self.trees):
            logger.debug(f"→ Training tree {i_tree}/{len(self.trees)}")
            predictions -= tree.predict(std_inputs)
            tree.train_step(X=std_inputs, y=y - predictions, w=w, train_config=train_config)
            predictions += tree.predict(std_inputs)

    def global_leaf_refit(self, X, y, w, train_config):
        """
        Global refit of all leaf node predictions via joint L1-regularized regression.

        Uses a sparse design matrix where each event contributes (1, x_i)
        only to the block corresponding to its leaf in each tree.
        """
        logger.info("→ Performing global leaf refit...")
        X = self.standardize_input(X)
        N, d = X.shape

        # Collect routing matrices and ordered nodes per tree
        routing_list = []        # routing_list[t] is N x M_t boolean
        ordered_nodes_list = []  # ordered_nodes_list[t] are nodes in tree t
        for tree in self.trees:
            routing, ordered_nodes = tree.route_all(X)
            routing_list.append(routing)
            ordered_nodes_list.append(ordered_nodes)

        # Build a flat list of all leaf specifications
        leaf_specs = []  # list of (t, leaf_idx, node, column_base)
        for t, ordered_nodes in enumerate(ordered_nodes_list):
            leaf_nodes = [n for n in ordered_nodes if n.is_leaf]
            for l, node in enumerate(leaf_nodes):
                # column block base index for this leaf: j*(d+1)
                j = len(leaf_specs)
                col_base = j * (d + 1)
                leaf_specs.append((t, node, ordered_nodes.index(node), col_base))

        K = len(leaf_specs)
        n_cols = K * (d + 1)

        # Build sparse matrix entries: each event i contributes to T leaves
        rows, cols, data = [], [], []
        for (t, node, node_idx, col_base) in leaf_specs:
            mask = routing_list[t][:, node_idx]  # which samples go to this leaf
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            # Intercept term (1.0)
            rows.extend(idxs.tolist())
            cols.extend([col_base] * idxs.size)
            data.extend([1.0] * idxs.size)
            # Feature terms
            for f in range(d):
                rows.extend(idxs.tolist())
                cols.extend([col_base + 1 + f] * idxs.size)
                # add x_{i,f} for each i in idxs
                data.extend(X[idxs, f].tolist())

        X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(N, n_cols))

        alpha_leaf = train_config.get("alpha_leaf", 0.0)
        logger.debug("→ Solving global Lasso regression with sparse design matrix...")
        clf = Lasso(alpha=alpha_leaf, fit_intercept=False, max_iter=1000)
        clf.fit(X_sparse, y, sample_weight=w)

        # Reshape coefficients into per-leaf weight vectors
        theta = clf.coef_.reshape(K, d + 1)
        #for j, (t, node, _, col_base) in enumerate(leaf_specs):
        #    b_j = theta[j, 0]
        #    W_j = theta[j, 1:].reshape(1, -1)
        #    node.b = b_j
        #    node.W = W_j

        for j, (t, node, _, col_base) in enumerate(leaf_specs):
            b_j = theta[j, 0]
            W_j = theta[j, 1:].reshape(1, -1)
            # apply quantization if requested
            if node.quantization is not None:
                W_j, b_j = quantize(W_j, b_j, node.quantization)
            node.b = b_j
            node.W = W_j

        if logger.level<=10:
            logger.debug("After fit:")
            self.print()

    def save(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        forest_data = {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'trees': self.trees,
            'config': self.config,
        }
        forest_file = os.path.join(path, f"forest_epoch_{epoch}.pkl")
        with open(forest_file, 'wb') as f:
            pickle.dump(forest_data, f)
    
    def print(self):
        for i_tree, tree in enumerate(self.trees):
            print(f"Forest tree {i_tree}/{len(self.trees)}")
            tree.print()

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

        forest = cls(forest_data['trees'], config=forest_data['config'])
        forest.set_standardization(forest_data['X_mean'], forest_data['X_std'])
        return forest

