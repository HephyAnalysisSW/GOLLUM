import numpy as np
import itertools
import os
import yaml
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')

from   common.helpers import copyIndexPHP
import common.user as user
import common.syncer
import logging
logger = logging.getLogger('UNC')

from quantize import quantize

class TreeNode:
    def __init__(self, node_id, depth, is_leaf=False, mode='regression'):
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = is_leaf

        self.W = None  # (1, d) for oblique splits
        self.b = None

        self.n_instances = None  # Number of training instances 

        self.left = None
        self.right = None

    def is_internal(self):
        return not self.is_leaf

    def set_split(self, W, b):
        self.W = W
        self.b = b

    def set_prediction(self, b, W=None):
        self.b = b
        self.W = W

    def prediction(self, X):
        if self.W is None:
            return np.full(X.shape[0], self.b)
        else:
            return (X @ self.W.T).flatten() + self.b

class Tree:
    def __init__(self, config):
        if type(config)==str:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif type(config)==dict:
            self.config = config

        self.max_depth    = self.config['max_depth']
        self.input_dim    = self.config['input_dim']
        self.order        = self.config.get('order',      'reverse_bfs')
        self.mode         = self.config.get('mode',       'regression')
        self.quantization = self.config.get('quantization', None)

        # Relevant for initializing the lef nodes with 1/Ntree
        self.ntrees = self.config.get('ntrees', 1)

        self.root = None
        self.nodes = {}

        # make an rng or take external one
        if "rng" in config:
            self.rng = config.get('rng')
        else:
            self.rng = np.random.default_rng(config.get('rng_seed', None))

        self.X_mean = None  # Standardization
        self.X_std  = None

        self._build_tree()

    def set_standardization( self, X_mean, X_std ):
        self.X_mean = X_mean  # Standardization
        self.X_std  = X_std

    def _build_tree(self):
        def add_node(current_depth, node_id):
            is_leaf = (current_depth == self.max_depth)
            node = TreeNode(node_id=node_id, depth=current_depth, is_leaf=is_leaf)

            if is_leaf:
                # If we have ntrees instances, we initialize with 1/ntrees to arrive at O(1)
                W = self.rng.normal(size=(1, self.input_dim))/self.ntrees
                b = self.rng.normal()/self.ntrees
                node.set_prediction(b=b, W=W if self.mode=="lasso" else None ) 
            else:
                W = self.rng.normal(size=(1, self.input_dim))
                b = self.rng.normal()
                node.set_split(W, b)

                node.left = add_node(current_depth + 1, 2 * node_id + 1)
                node.right = add_node(current_depth + 1, 2 * node_id + 2)

            if is_leaf:
                # initialize leaf to small random, then quantize
                W = self.rng.normal(size=(1, self.input_dim)) / self.ntrees
                b = self.rng.normal()           / self.ntrees
                if self.mode=="lasso":
                    W, b = quantize(W, b, self.quantization)
                    node.set_prediction(b=b, W=W)
                else:
                    node.set_prediction(b=b, W=None)
            else:
                # interior split → quantize too
                W = self.rng.normal(size=(1, self.input_dim))
                b = self.rng.normal()
                W, b = quantize(W, b, self.quantization)
                node.set_split(W, b)

            self.nodes[node_id] = node
            return node

        self.root = add_node(current_depth=0, node_id=0)

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

    def route_all(self, X):
        """
        Returns:
            result: boolean array of shape (N_samples, N_nodes),
                    where result[i, j] is True if sample i passes through the j-th node
                    as defined by self._get_nodes().
            ordered_nodes: list of nodes (internal or all), in traversal order.
        """
        N = X.shape[0]
        ordered_nodes = self._get_nodes(internal=False)
        result = np.zeros((N, len(ordered_nodes)), dtype=bool)

        def emit_mask(node, current_mask):
            idx = node_to_index[node]
            result[:, idx] = current_mask

            if node.is_leaf:
                return

            decision_values = (X @ node.W.T).flatten() + node.b

            go_right = decision_values > 0
            go_left = ~go_right

            emit_mask(node.left, current_mask & go_left)
            emit_mask(node.right, current_mask & go_right)

        node_to_index = {node: i for i, node in enumerate(ordered_nodes)}
        emit_mask(self.root, np.ones(N, dtype=bool))

        # Tell each node how many entries it has
        for i_node, node in enumerate(ordered_nodes):
            node.n_instances = np.count_nonzero( result[:, i_node] )
        return result, ordered_nodes

    #def prune_by_min_node_size(self, min_size):
    #    """
    #    Prune internal nodes based on cached leaf_indices.
    #    """
    #    print(f"→ Pruning with min node size: {min_size}")
    #    keep = set(self.leaf_indices.keys())

    #    for node in self._get_nodes():
    #        if node.is_leaf:
    #            continue
    #        if (node.left is None or node.left.node_id not in keep or
    #            node.right is None or node.right.node_id not in keep):
    #            continue
    #        left_count = len(self.leaf_indices.get(node.left.node_id, []))
    #        right_count = len(self.leaf_indices.get(node.right.node_id, []))

    #        if left_count < min_size or right_count < min_size:
    #            print(f"  ↳ Pruning node {node.node_id} (left: {left_count}, right: {right_count})")

    #            left_id = node.left.node_id
    #            right_id = node.right.node_id

    #            node.is_leaf = True
    #            node.left = None
    #            node.right = None
    #            node.W = None
    #            node.b = None

    #            keep.discard(left_id)
    #            keep.discard(right_id)
    #            self.leaf_indices[node.node_id] = (
    #                self.leaf_indices.get(left_id, []) + self.leaf_indices.get(right_id, [])
    #            )

    #    print("✓ Pruning complete.")

    def train_step(self, X, y, w, train_config):
        """
        Full TAO-style training step with single routing pass.
        """
        logger.debug("→ Standardizing input...")
        X = self.standardize_input(X)

        logger.debug("→ Routing all data...")
        routing, ordered_nodes = self.route_all(X)

        #print(routing.shape, routing)
        #assert False, ""

        if logger.level<=10:
            logger.debug("Before fit:")
            self.print()

        logger.debug("→ Computing new predictions")
        for i_node, node in enumerate(ordered_nodes):
            mask = routing[:, i_node]
            #print (i_node, node.is_leaf, np.count_nonzero(mask), mask)
            #assert False, ""
            if node.is_leaf:
                self._update_leaf_from_routing(
                    node, mask, X, y, w, alpha_leaf=train_config['alpha_leaf']
                )
            else:
                self._update_split_node_tao(
                    X, y, w,
                    node=node,
                    mask=mask,
                    alpha=train_config['alpha'],
                )
        if logger.level<=10:
            logger.debug("After fit:")
            self.print()

    def _update_leaf_from_routing(self, node, mask, X, y, w, alpha_leaf=0.01):
        """
        Update prediction for a single leaf node using the routing mask.
        
        Args:
            node (TreeNode): the leaf node
            mask (np.ndarray): boolean mask of shape (N,) indicating which samples reach this leaf
            y (np.ndarray): labels
            w (np.ndarray): weights
            alpha_leaf (float): L1 regularization strength for "lasso" mode
        """

        logger.debug(f"Fitting leaf node {node.node_id}")

        if not np.any(mask):
            node.set_prediction(0.0, W=np.zeros(self.input_dim) if self.mode=="lasso" else None)
            return

        y_sub = y[mask]
        w_sub = w[mask]
        if self.mode == "regression":
            weighted_mean = np.sum(w_sub * y_sub) / np.sum(w_sub)
            node.set_prediction(weighted_mean)

        elif self.mode == "classification":
            y_bar = np.sum(w_sub * y_sub) / np.sum(w_sub)
            eps = 1e-8
            y_bar = np.clip(y_bar, eps, 1 - eps)
            v = np.log(y_bar / (1 - y_bar))
            node.set_prediction(v)

        elif self.mode == "lasso":
            X_sub = X[mask]
            if X_sub.shape[0] < 2:
                node.set_prediction(0.0, W=np.zeros(self.input_dim) if self.mode=="lasso" else None)
                return

            clf = Lasso(alpha=alpha_leaf, fit_intercept=True, max_iter=1000)
            clf.fit(X_sub, y_sub, sample_weight=w_sub)
            W_new = clf.coef_.reshape(1, -1)
            b_new = clf.intercept_
            # quantize leaf weights & bias
            Wq, bq = quantize(W_new, b_new, self.quantization)
            node.set_prediction(b=bq, W=Wq)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _accumulate_leaf_losses(self, root, X, y, w):
        """
        Vectorized loss evaluation for a subtree rooted at `root`,
        for a subset of examples only (X, y, w all pre-masked).
        
        Returns:
            losses (np.ndarray): array of shape (len(X),)
        """
        N = X.shape[0]
        losses = np.zeros(N)

        def emit_loss(node, mask):
            if not np.any(mask):
                return

            if node.is_leaf:
                if self.mode == "regression":
                    residual = y[mask] - node.b
                    losses[mask] = w[mask] * residual**2
                elif self.mode == "classification":
                    v = node.b
                    pi = 1 / (1 + np.exp(-v))
                    eps = 1e-8
                    ce_loss = -y[mask] * np.log(pi + eps) - (1 - y[mask]) * np.log(1 - pi + eps)
                    losses[mask] = w[mask] * ce_loss
                elif self.mode == "lasso":
                    residual = y[mask] - (X[mask] @ node.W.T).flatten() - node.b
                    losses[mask] = w[mask] * residual**2
                return

            decision = (X @ node.W.T).flatten() + node.b
            go_right = decision > 0
            go_left = ~go_right

            emit_loss(node.left, mask & go_left)
            emit_loss(node.right, mask & go_right)

        emit_loss(root, np.ones(N, dtype=bool))
        return losses

    def _update_split_node_tao(self, X, y, w, node, mask, alpha=0.01):
        if node.is_leaf or not np.any(mask):
            return

        X_sub = X[mask]
        y_sub = y[mask]
        w_sub = w[mask]

        logger.debug(f"Fitting internal node {node.node_id} with {X_sub.shape[0]} events")

        left_losses  = self._accumulate_leaf_losses(node.left, X_sub, y_sub, w_sub)
        right_losses = self._accumulate_leaf_losses(node.right, X_sub, y_sub, w_sub)

        delta_loss = left_losses - right_losses
        y_target = np.sign(delta_loss).astype(int)
        sample_weight = np.abs(delta_loss)

        if np.sum(sample_weight) == 0 or len(np.unique(y_target)) < 2:
            return

        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1 / alpha)

        #clf.fit(X_sub, y_target, sample_weight=sample_weight)

        # Fit an L1-regularized linear model to residuals (MSE loss)
        #from sklearn.linear_model import SGDRegressor
        #clf = SGDRegressor(
        #    loss='squared_error',   # MSE
        #    penalty='l1',           # L1 regularization
        #    alpha=alpha,            # strength of penalty
        #    fit_intercept=True,
        #    max_iter=1000,
        #    tol=1e-3,
        #    random_state=self.config.get('rng_seed', None),
        #)
        #clf.fit(X_sub, y_target, sample_weight=sample_weight)

        #node.W = clf.coef_.reshape(1, -1)
        #node.b = clf.intercept_[0]

        clf.fit(X_sub, y_target, sample_weight=sample_weight)
        W_new = clf.coef_.reshape(1, -1)
        b_new = clf.intercept_[0]
        # quantize split
        Wq, bq = quantize(W_new, b_new, self.quantization)
        node.W, node.b = Wq, bq

    def print(self):
        threshold = 1e-3

        def _print(node, prefix=""):
            if node is None:
                return

            n_inst_str = f"(n_inst = {getattr(node, 'n_instances', None)})"

            if node.is_leaf:
                if node.W is None:
                    # Scalar leaf
                    print(f"{prefix}[Leaf] Node {node.node_id} at depth {node.depth} → prediction = {node.b:.4f} {n_inst_str}")
                else:
                    # Linear leaf
                    terms = [
                        f"{w:.3f}*x{j}"
                        for j, w in enumerate(node.W.flatten())
                        if abs(w) >= threshold
                    ]
                    w_str = " + ".join(terms) if terms else "0"
                    print(f"{prefix}[Leaf] Node {node.node_id} at depth {node.depth} → prediction = {w_str} + {node.b:.3f} {n_inst_str}")
            else:
                terms = [
                    f"{w:.3f}*x{j}"
                    for j, w in enumerate(node.W.flatten())
                    if abs(w) >= threshold
                ]
                w_str = " + ".join(terms) if terms else "0"
                decision_str = f"{w_str} + {node.b:.3f} >= 0"
                print(f"{prefix}[Split] Node {node.node_id} at depth {node.depth} → {decision_str} {n_inst_str}")
                _print(node.left, prefix + "  ")
                _print(node.right, prefix + "  ")

        _print(self.root)

    def _get_nodes(self, internal = True):
        if self.order == "reverse_bfs":
            return sorted(
                [n for n in self.nodes.values() if (n.is_internal() or not internal)],
                key=lambda n: -n.depth
            )
        elif self.order == "bfs":
            return sorted(
                [n for n in self.nodes.values() if (n.is_internal() or not internal)],
                key=lambda n: n.depth
            )
        elif self.order == "preorder":
            result = []
            def dfs(node):
                if node is None:
                    return
                if node.is_internal() or not internal:
                    result.append(node)
                dfs(node.left)
                dfs(node.right)
            dfs(self.root)
            return result
        else:
            raise ValueError(f"Unknown order: {self.order}")

    def predict(self, X):
        """
        Predict output for new data using emit-style vectorized tree traversal.

        Args:
            X (np.ndarray): input data of shape (N, d)

        Returns:
            predictions (np.ndarray): array of shape (N,) with predicted values
        """
        X = self.standardize_input(X)  # standardize
        N = X.shape[0]
        predictions = np.zeros(N)

        def emit_prediction(node, mask):
            if not np.any(mask):
                return
            if node.is_leaf:
                predictions[mask] = node.prediction(X[mask])
                return
            decision = (X @ node.W.T).flatten() + node.b
            go_right = decision > 0
            go_left = ~go_right
            emit_prediction(node.left, mask & go_left)
            emit_prediction(node.right, mask & go_right)

        emit_prediction(self.root, np.ones(N, dtype=bool))
        return predictions
