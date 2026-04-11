#!/usr/bin/env python
# Standard imports
import sys
import pickle
import numpy as np
import operator
import functools

sys.path.insert(0, '..'); sys.path.insert(0, '../..')
import ML.BIT.GpuMultiNode as MultiNode

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
            acc = np.zeros((N, K), dtype=np.float64)
            for t in range(T):
                raw = self.trees[t].vectorized_predict(feature_array)  # (N, 1+K)
                if raw.dtype != np.float64:
                    raw = raw.astype(np.float64, copy=False)
                denom = raw[:, :1]
                num   = raw[:, 1:]
                ratio = np.empty_like(num)
                np.divide(num, denom, out=ratio, where=(denom != 0.0))
                acc += learning_rates[t] * ratio
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
