#!/usr/bin/env python
# Standard imports
import sys
import pickle
import numpy as np
import operator
import functools

sys.path.insert(0, '..'); sys.path.insert(0, '../..')
import ML.BIT.NumbaMultiNode as MultiNode

default_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2,
    "loss" : "MSE",  # or "CrossEntropy"
    "learn_global_score": False,
}

class MultiBoostedInformationTree:
    """
    Lightweight container for a boosted tree ensemble.

    - Holds: cfg, node_cfg, and the trained trees.
    - Does NOT hold training features, mutable training weights, base_points, or feature_names.
    - Training is performed externally (e.g. in pdf_bit_training.py).
    """

    @staticmethod
    def sort_comb(comb):
        return tuple(sorted(comb))

    def __init__(self, **kwargs):

        # cfg and node_cfg from kwargs keys known by the Node
        self.cfg = dict(default_cfg)  # avoid mutating the module-level default
        self.node_cfg = {}

        for (key, val) in kwargs.items():
            if key in MultiNode.default_cfg.keys():
                self.node_cfg[key] = val
            elif key in default_cfg.keys():
                self.cfg[key] = val
            else:
                raise RuntimeError("Got unexpected keyword arg: %s:%r" % (key, val))

        self.node_cfg['loss'] = self.cfg['loss']

        for (key, val) in self.cfg.items():
            setattr(self, key, val)

        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02
        if self.learning_rate == "auto":
            self.learning_rate = 1 - 0.02**(1. / self.n_trees)

        # Will hold the trees
        self.trees = []

        # Filled after first tree exists (or after load if missing)
        self.derivatives = None

    def _sanitize_after_load(self):
        # Drop any legacy training-state / metadata attributes if present
        for attr in ("training_features", "training_weights", "base_points", "feature_names"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

        if not hasattr(self, "trees") or self.trees is None:
            self.trees = []

        if not hasattr(self, "cfg") or self.cfg is None:
            self.cfg = dict(default_cfg)

        if not hasattr(self, "node_cfg") or self.node_cfg is None:
            self.node_cfg = {}

        # Backfill common config attributes if missing
        for k, v in default_cfg.items():
            if not hasattr(self, k):
                setattr(self, k, self.cfg.get(k, v))
            self.cfg[k] = getattr(self, k)

        # Backfill derivatives if missing and possible
        if (not hasattr(self, "derivatives")) or (self.derivatives is None):
            if len(self.trees) > 0:
                try:
                    self.derivatives = self.trees[0].derivatives[1:]
                except Exception:
                    self.derivatives = None

    def __setstate__(self, state):
        self.__dict__ = state
        self._sanitize_after_load()

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file_:
            obj = pickle.load(file_)
        if hasattr(obj, "_sanitize_after_load"):
            obj._sanitize_after_load()
        return obj

    def save(self, filename):
        with open(filename, 'wb') as file_:
            pickle.dump(self, file_, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_prediction_cache(self):
        split_feature_parts = []
        split_value_parts = []
        left_child_parts = []
        right_child_parts = []
        is_leaf_parts = []
        leaf_value_parts = []
        tree_offsets = [0]

        for tree in self.trees:
            if hasattr(tree, "_ensure_prediction_state"):
                tree._ensure_prediction_state()
                split_feature = tree._predict_split_feature
                split_value = tree._predict_split_value
                left_child = tree._predict_left_child
                right_child = tree._predict_right_child
                is_leaf = tree._predict_is_leaf
                leaf_value = tree._predict_leaf_value
            else:
                tmp = tree.vectorized_predict(np.zeros((1, 1), dtype=np.float64))
                raise RuntimeError(f"Tree {tree} does not expose prediction state; got tmp shape {tmp.shape}")

            offset = tree_offsets[-1]
            split_feature_parts.append(split_feature)
            split_value_parts.append(split_value)
            left_child_parts.append(np.where(left_child >= 0, left_child + offset, left_child))
            right_child_parts.append(np.where(right_child >= 0, right_child + offset, right_child))
            is_leaf_parts.append(is_leaf)
            leaf_value_parts.append(leaf_value)
            tree_offsets.append(offset + len(split_feature))

        self._predict_tree_offsets = np.asarray(tree_offsets, dtype=np.int32)
        if split_feature_parts:
            self._predict_split_feature = np.concatenate(split_feature_parts).astype(np.int32, copy=False)
            self._predict_split_value = np.concatenate(split_value_parts).astype(np.float64, copy=False)
            self._predict_left_child = np.concatenate(left_child_parts).astype(np.int32, copy=False)
            self._predict_right_child = np.concatenate(right_child_parts).astype(np.int32, copy=False)
            self._predict_is_leaf = np.concatenate(is_leaf_parts).astype(np.int8, copy=False)
            self._predict_leaf_value = np.concatenate(leaf_value_parts).astype(np.float64, copy=False)
        else:
            self._predict_split_feature = np.zeros(0, dtype=np.int32)
            self._predict_split_value = np.zeros(0, dtype=np.float64)
            self._predict_left_child = np.zeros(0, dtype=np.int32)
            self._predict_right_child = np.zeros(0, dtype=np.int32)
            self._predict_is_leaf = np.zeros(0, dtype=np.int8)
            self._predict_leaf_value = np.zeros((0, 0), dtype=np.float64)
        self._predict_cache_n_trees = len(self.trees)

    def _ensure_prediction_cache(self):
        if getattr(self, "_predict_cache_n_trees", None) != len(self.trees):
            self._build_prediction_cache()

    def predict(self, feature_array, max_n_tree=None, summed=True, last_tree_counts_full=False):
        """
        Parameters
        ----------
        feature_array : np.ndarray, shape (N, d)
            Input features.
        max_n_tree : int or None
            Use only the first max_n_tree trees if set.
        summed : bool
            If True, returns the learning-rate-weighted sum over trees (shape (N, K)).
            If False, returns per-tree predictions (shape (T, N, K)).
        last_tree_counts_full : bool
            If True and using all trees, set the last tree's learning rate to 1.
        """
        T = max_n_tree if max_n_tree is not None else self.n_trees
        T = min(T, len(self.trees))

        learning_rates = self.learning_rate * np.ones(T, dtype=np.float64)
        if T > 0 and last_tree_counts_full and (max_n_tree is None or max_n_tree == self.n_trees) and (T == len(self.trees)):
            learning_rates[-1] = 1.0
        if self.cfg.get("learn_global_score", False) and T > 0:
            learning_rates[0] = 1.0

        if T == 0:
            K = 0
            if len(self.trees) > 0:
                tmp = self.trees[0].vectorized_predict(feature_array[:1])
                K = max(0, tmp.shape[1] - 1)
            return np.zeros((feature_array.shape[0], K), dtype=np.float64) if summed else \
                   np.zeros((0, feature_array.shape[0], K), dtype=np.float64)

        first = self.trees[0].vectorized_predict(feature_array[:1])
        K = first.shape[1] - 1
        N = feature_array.shape[0]

        if summed:
            self._ensure_prediction_cache()
            acc = np.zeros((N, K), dtype=np.float64)
            kernel = MultiNode._predict_forest_ratio_numba_parallel if N >= 4096 else MultiNode._predict_forest_ratio_numba
            kernel(
                np.asarray(feature_array),
                self._predict_tree_offsets[:T],
                self._predict_split_feature,
                self._predict_split_value,
                self._predict_left_child,
                self._predict_right_child,
                self._predict_is_leaf,
                self._predict_leaf_value,
                learning_rates,
                acc,
            )
            return acc
        else:
            out = np.empty((T, N, K), dtype=np.float64)
            for t in range(T):
                raw = self.trees[t].vectorized_predict(feature_array)
                if raw.dtype != np.float64:
                    raw = raw.astype(np.float64, copy=False)
                denom = raw[:, :1]
                num   = raw[:, 1:]
                np.divide(num, denom, out=out[t], where=(denom != 0.0))
                out[t] *= learning_rates[t]
            return out

#    def losses(self, feature_array, weight_dict, max_n_tree=None, last_tree_counts_full=False):
#        """
#        LEGACY / UNUSED (per your note).
#
#        This computes a tree-by-tree loss proxy using base_points and derivatives.
#        It is kept only for backwards compatibility / occasional diagnostics.
#
#        IMPORTANT:
#        - This function reads base_points from self.trees[0].base_points (stored inside the trained tree).
#        - The BIT container itself does NOT store base_points.
#
#        If you truly never use it, you can delete it safely once you’re confident nothing calls it.
#        """
#        base_points      = self.trees[0].base_points
#        base_point_const = np.array([[functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der], 1)
#                                      for der in self.derivatives] for point in base_points]).astype('float')
#        for i_der, der in enumerate(self.derivatives):
#            if not (len(der) == 2 and der[0] == der[1]): continue
#            for i_point in range(len(base_points)):
#                base_point_const[i_point][i_der] /= 2.
#
#        predictions = np.array([tree.vectorized_predict(feature_array) for tree in self.trees[:max_n_tree]])
#        predictions = predictions[:, :, 1:] / np.expand_dims(predictions[:, :, 0], -1)
#
#        weight_ratio = np.array([(weight_dict[der] / weight_dict[()] if der in weight_dict else
#                                  weight_dict[tuple(reversed(der))] / weight_dict[()])
#                                 for der in self.derivatives]).transpose().astype('float')
#
#        return -(weight_dict[()][np.newaxis, ..., np.newaxis] * np.dot((predictions - (weight_ratio[np.newaxis, ...])), base_point_const) ** 2).sum(axis=(1, 2))
