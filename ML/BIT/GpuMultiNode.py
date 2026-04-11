#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt
import itertools
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')

import functools

# --- Numba setup (graceful fallback) ---
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(n): return range(n)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

default_cfg = {
    "max_depth":        4,
    "min_size" :        50,
    "max_n_split":      -1,     # similar to TMVA
    "base_points":      None,
    "feature_names":    None,
    "positive":         False,
    "min_node_size_neg_adjust": False,
    "loss" : "MSE", # or "CrossEntropy"
    "_get_only_score":False,
    "split_mode": "exact",   # "exact" (current) or "binned"
    "n_bins": 256,           # used only if split_mode == "binned"
    "quantile_bins": False,  # if True: quantile edges; else uniform edges

}


def _compute_global_quantile_cuts(feature_matrix, n_bins, cut_sample_rows=None):
    if cut_sample_rows is None or cut_sample_rows <= 0 or cut_sample_rows >= feature_matrix.shape[0]:
        sample = feature_matrix
    else:
        sample = feature_matrix[:cut_sample_rows]
    quantile_levels = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)[1:-1]
    if quantile_levels.size == 0:
        return np.zeros((feature_matrix.shape[1], 0), dtype=np.float32)
    return np.quantile(sample, quantile_levels, axis=0).T.astype(np.float32)


def _quantize_feature_matrix(feature_matrix, cuts):
    bins = np.empty(feature_matrix.shape, dtype=np.uint16 if cuts.shape[1] + 1 > 256 else np.uint8)
    for feature in range(feature_matrix.shape[1]):
        bins[:, feature] = np.searchsorted(cuts[feature], feature_matrix[:, feature], side="left")
    return bins


def _child_cfg_without_cached_bins(cfg):
    child_cfg = dict(cfg)
    child_cfg.pop("binned_features", None)
    return child_cfg

# ------------------ Numba kernels for heavy numerics ------------------

@njit(cache=True)
def _rowdot(A, B_T):
    # A: (n, d), B_T: (d, k)  -> (n, k)
    return A @ B_T

@njit(cache=True, parallel=True)
def _mse_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, base_point_const_T):
    # Fused matmul + sum-of-squares: avoids allocating intermediate (n, k) matrices.
    # Parallelised over rows (independent per split candidate).
    n = sorted_weight_sums.shape[0]
    d = sorted_weight_sums.shape[1]
    k = base_point_const_T.shape[1]

    gains = np.empty(n, dtype=np.float64)

    for i in prange(n):
        sL = 0.0
        sR = 0.0
        for j in range(k):
            lij = 0.0
            rij = 0.0
            for m in range(d):
                lij += sorted_weight_sums[i, m] * base_point_const_T[m, j]
                rij += sorted_weight_sums_right[i, m] * base_point_const_T[m, j]
            sL += lij * lij
            sR += rij * rij
        gains[i] = sL / sorted_weight_sums[i, 0] + sR / sorted_weight_sums_right[i, 0]

    return gains

@njit(cache=True, parallel=True)
def _crossentropy_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, base_point_const_T):
    # Fused matmul + cross-entropy: avoids allocating intermediate (n, k) matrices.
    # Parallelised over rows (independent per split candidate).
    n = sorted_weight_sums.shape[0]
    d = sorted_weight_sums.shape[1]
    k = base_point_const_T.shape[1]

    gains = np.empty(n, dtype=np.float64)

    # avoid divide-by-zero by tiny eps (keeps parity with numpy nan_to_num behavior)
    eps = 1e-300

    for i in prange(n):
        nL = sorted_weight_sums[i, 0]
        nR = sorted_weight_sums_right[i, 0]
        if nL <= 0.0 or nR <= 0.0:
            gains[i] = -np.inf
            continue

        s = 0.0
        for j in range(k):
            # inline dot products
            lij = 0.0
            rij = 0.0
            for m in range(d):
                lij += sorted_weight_sums[i, m] * base_point_const_T[m, j]
                rij += sorted_weight_sums_right[i, m] * base_point_const_T[m, j]

            rL = lij / nL
            rR = rij / nR
            one_rL = 1.0 + rL
            one_rR = 1.0 + rR

            # clamp away from 0 to avoid log(0)
            if one_rL == 0.0: one_rL = eps
            if one_rR == 0.0: one_rR = eps
            if rL == 0.0: rL = eps
            if rR == 0.0: rR = eps

            s += 0.5 * ( -2.0 * np.log(one_rL) + rL * ( 2.0 * (np.log(abs(rL)) - np.log(abs(one_rL))) ) )
            s += 0.5 * ( -2.0 * np.log(one_rR) + rR * ( 2.0 * (np.log(abs(rR)) - np.log(abs(one_rR))) ) )

        gains[i] = sorted_weight_sums[i, 0] * s
    # shift by min to make positive (as in original)
    # (Do it in Python caller to keep parity with np.nan_to_num etc.)
    return gains

@njit(cache=True, parallel=True)
def _bin_sums_parallel(idx, TW, nbin):
    """Accumulate weighted sums per bin without sorting.

    Uses thread-local accumulators (no locks/atomics needed).
    ~150x faster than argsort + fancy-index + reduceat for large N.

    Parameters
    ----------
    idx : int64[N]   bin index (0 .. nbin-1) for each event
    TW  : float64[N, D]  per-event weight vector
    nbin : int        number of bins
    """
    N, D = TW.shape
    nthreads = numba.get_num_threads()
    chunk = (N + nthreads - 1) // nthreads
    # Thread-local storage avoids race conditions without atomics
    local_sums = np.zeros((nthreads, nbin, D), dtype=np.float64)
    for t in prange(nthreads):
        s = t * chunk
        e_end = min(s + chunk, N)
        for e in range(s, e_end):
            b = idx[e]
            for d in range(D):
                local_sums[t, b, d] += TW[e, d]
    # Reduce thread-local sums
    bin_sums = np.zeros((nbin, D), dtype=np.float64)
    for t in range(nthreads):
        for b in range(nbin):
            for d in range(D):
                bin_sums[b, d] += local_sums[t, b, d]
    return bin_sums


@njit(cache=True)
def _predict_tree_numba(feature_matrix, split_feature, split_value, left_child, right_child, is_leaf, leaf_value, out):
    n_rows = feature_matrix.shape[0]
    pred_dim = leaf_value.shape[1]
    for i in range(n_rows):
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            if feature_matrix[i, feature] <= split_value[node]:
                node = left_child[node]
            else:
                node = right_child[node]
        for c in range(pred_dim):
            out[i, c] = leaf_value[node, c]


@njit(cache=True, parallel=True)
def _predict_tree_numba_parallel(feature_matrix, split_feature, split_value, left_child, right_child, is_leaf, leaf_value, out):
    n_rows = feature_matrix.shape[0]
    pred_dim = leaf_value.shape[1]
    for i in prange(n_rows):
        node = 0
        while is_leaf[node] == 0:
            feature = split_feature[node]
            if feature_matrix[i, feature] <= split_value[node]:
                node = left_child[node]
            else:
                node = right_child[node]
        for c in range(pred_dim):
            out[i, c] = leaf_value[node, c]


@njit(cache=True)
def _predict_forest_ratio_numba(feature_matrix, tree_offsets, split_feature, split_value, left_child, right_child, is_leaf, leaf_value, learning_rates, out):
    n_rows = feature_matrix.shape[0]
    n_trees = learning_rates.shape[0]
    pred_dim = out.shape[1]
    for i in range(n_rows):
        for c in range(pred_dim):
            out[i, c] = 0.0
        for t in range(n_trees):
            node = tree_offsets[t]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                if feature_matrix[i, feature] <= split_value[node]:
                    node = left_child[node]
                else:
                    node = right_child[node]
            denom = leaf_value[node, 0]
            if denom != 0.0:
                lr = learning_rates[t]
                for c in range(pred_dim):
                    out[i, c] += lr * (leaf_value[node, c + 1] / denom)


@njit(cache=True, parallel=True)
def _predict_forest_ratio_numba_parallel(feature_matrix, tree_offsets, split_feature, split_value, left_child, right_child, is_leaf, leaf_value, learning_rates, out):
    n_rows = feature_matrix.shape[0]
    n_trees = learning_rates.shape[0]
    pred_dim = out.shape[1]
    for i in prange(n_rows):
        for c in range(pred_dim):
            out[i, c] = 0.0
        for t in range(n_trees):
            node = tree_offsets[t]
            while is_leaf[node] == 0:
                feature = split_feature[node]
                if feature_matrix[i, feature] <= split_value[node]:
                    node = left_child[node]
                else:
                    node = right_child[node]
            denom = leaf_value[node, 0]
            if denom != 0.0:
                lr = learning_rates[t]
                for c in range(pred_dim):
                    out[i, c] += lr * (leaf_value[node, c + 1] / denom)

# -------------------------------------------------------

class MultiNode:
    def __init__( self, features, training_weights, _depth=0, **kwargs):

        ## basic BDT configuration + kwargs
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for attr, val in self.cfg.items():
            setattr( self, attr, val )

        # data set
        self.features           = features
        self.split_bin          = -1
        self.binned_features    = kwargs.get('binned_features', None)
        self.precomputed_cuts   = kwargs.get('precomputed_cuts', None)
        if self.features is not None:
            self.size = len(self.features)
            self.n_features = self.features.shape[1]
        elif self.binned_features is not None:
            self.size = len(self.binned_features)
            self.n_features = self.binned_features.shape[1]
        else:
            raise RuntimeError("Need either features or binned_features for training.")

        if self.cfg['loss'] not in ["MSE", "CrossEntropy"]:
            raise RuntimeError( "Unknown loss. Should be 'MSE' or 'CrossEntropy'." ) 

        # Master node: We expect a dict  
        if type(training_weights)==dict:
            self.coefficients            = sorted(list(set(sum(map(list,training_weights.keys()),[]))))

            self.first_derivatives  = sorted(list(itertools.combinations_with_replacement(self.coefficients,1))) 
            self.second_derivatives = sorted(list(itertools.combinations_with_replacement(self.coefficients,2))) 
            self.derivatives        = [tuple()] + self.first_derivatives + self.second_derivatives

            self.training_weights   = {tuple(sorted(key)):val for key,val in training_weights.items()}

            assert ('base_points' in kwargs) and kwargs['base_points'] is not None, "Must provide base_points in cfg"
            assert all( [ key in self.training_weights for key in self.derivatives ]), "Incomplete list of keys in training_weights?"

            # precoumputed base_point_const
            self.base_points      = kwargs['base_points']
            self.base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in self.derivatives] for point in self.base_points]).astype('float')
            for i_der, der in enumerate(self.derivatives):
                if not (len(der)==2 and der[0]==der[1]): continue
                for i_point in range(len(self.base_points)):
                    self.base_point_const[i_point][i_der]/=2.

            assert np.linalg.matrix_rank(self.base_point_const) == self.base_point_const.shape[0], \
                   "Base points not linearly independent! Found rank %i for %i base_points" %( np.linalg.matrix_rank(self.base_point_const), self.base_point_const.shape[0])

            # make another version of base_point_const that contains the [1,0,0,...] vector -> used for testing positivity of the zeroth coefficient
            const = np.zeros((1,len(self.derivatives)))
            const[0,0]=1
            self.base_point_const_for_pos = np.concatenate((const, self.base_point_const))

            self.cfg['base_point_const']         = self.base_point_const
            self.cfg['base_point_const_for_pos'] = self.base_point_const_for_pos
            self.cfg['derivatives'] = self.derivatives 
            self.cfg['feature_names'] = None if not ('feature_names' in kwargs) else kwargs['feature_names'] 
            self.feature_names      = self.cfg['feature_names']
            self.training_weights   = np.array([training_weights[der] for der in self.derivatives]).transpose().astype('float')
        # inside tree -> we need not re-compute the base-point consts, we copy them
        else:
            self.training_weights           = training_weights
            self.base_point_const           = kwargs['base_point_const']
            self.base_point_const_for_pos   = kwargs['base_point_const_for_pos']
            self.derivatives                = kwargs['derivatives']
            self.feature_names              = kwargs['feature_names']

        self.cfg.pop('binned_features', None)

        # keep track of recursion depth
        self._depth             = _depth

        self.split(_depth=_depth)
        self.prune()

        # Let's not leak the dataset.
        del self.training_weights
        del self.features 
        del self.split_left_group 

    def _build_prediction_state(self):
        split_feature = []
        split_value = []
        left_child = []
        right_child = []
        is_leaf = []
        leaf_value = []

        def visit(node):
            node_id = len(split_feature)
            split_feature.append(-1)
            split_value.append(0.0)
            left_child.append(-1)
            right_child.append(-1)
            is_leaf.append(0)
            leaf_value.append(np.zeros(len(self.derivatives), dtype=np.float64))

            if isinstance(node, ResultNode):
                is_leaf[node_id] = 1
                leaf_value[node_id] = np.asarray(node.coefficient_sum, dtype=np.float64)
            else:
                split_feature[node_id] = int(node.split_i_feature)
                split_value[node_id] = float(node.split_value)
                left_child[node_id] = visit(node.left)
                right_child[node_id] = visit(node.right)
            return node_id

        visit(self)
        self._predict_split_feature = np.asarray(split_feature, dtype=np.int32)
        self._predict_split_value = np.asarray(split_value, dtype=np.float64)
        self._predict_left_child = np.asarray(left_child, dtype=np.int32)
        self._predict_right_child = np.asarray(right_child, dtype=np.int32)
        self._predict_is_leaf = np.asarray(is_leaf, dtype=np.int8)
        self._predict_leaf_value = np.asarray(leaf_value, dtype=np.float64)

    def _ensure_prediction_state(self):
        if not hasattr(self, "_predict_split_feature"):
            self._build_prediction_state()

#    def get_split_vectorized( self ):
#        ''' determine where to split the features, with numba-accelerated FI maximization
#        '''
#
#        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
#        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None
#
#        # for a valid binary split, we need at least twice the mean size
#        assert self.size >= 2*self.min_size
#
#        # precompute transposes used in kernels
#        B_T  = self.base_point_const.transpose()
#        loss_mode = self.cfg['loss']
#
#        for i_feature in range(len(self.features[0])):
#            feature_values = self.features[:,i_feature]
#
#            feature_sorted_indices = np.argsort(feature_values)
#            # cumulative sums along rows (sorted)
#            sorted_weight_sums     = np.cumsum(self.training_weights[feature_sorted_indices],axis=0) # 2D cumsum
#
#            # respect min size for split & optional max_n_split
#            if self.max_n_split<2:
#                plateau_and_split_range_mask = np.ones(self.size-1, dtype=np.dtype('bool'))
#            else:
#                min_, max_ = float(np.min(feature_values)), float(np.max(feature_values))
#                # create at most 'max_n_split' plateaus between min/max
#                bins = np.linspace(min_, max_, self.max_n_split+2, endpoint=True)
#                digit = np.digitize(feature_values[feature_sorted_indices], bins)
#                plateau_and_split_range_mask = (digit[1:] != digit[:-1])
#                plateau_and_split_range_mask = np.insert(plateau_and_split_range_mask, 0, False)[:-1]
#
#            if self.min_size > 1:
#                plateau_and_split_range_mask[0:self.min_size-1] = False
#                plateau_and_split_range_mask[-self.min_size+1:] = False
#            plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)
#
#            total_weight_sum         = sorted_weight_sums[-1]
#            sorted_weight_sums       = sorted_weight_sums[0:-1]
#            sorted_weight_sums_right = total_weight_sum-sorted_weight_sums
#
#            # mask negative definite splits
#            if self.cfg['positive']:
#                pos       = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums,self.base_point_const_for_pos.transpose())>=0)
#                pos_right = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums_right,self.base_point_const_for_pos.transpose())>=0)
#                plateau_and_split_range_mask &= pos
#                plateau_and_split_range_mask &= pos_right
#
#            # Never allow negative yields
#            plateau_and_split_range_mask &= (sorted_weight_sums[:,0]>0)
#            plateau_and_split_range_mask &= (sorted_weight_sums_right[:,0]>0)
#
#            # --------- compute neg_loss_gains (JIT kernels) ----------
#            if loss_mode == 'MSE':
#                if NUMBA_AVAILABLE:
#                    neg_loss_gains = _mse_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, B_T)
#                else:
#                    L = sorted_weight_sums @ B_T
#                    R = sorted_weight_sums_right @ B_T
#                    neg_loss_gains = (np.sum(L*L, axis=1)/sorted_weight_sums[:,0]
#                                      + np.sum(R*R, axis=1)/sorted_weight_sums_right[:,0])
#            elif loss_mode == 'CrossEntropy':
#                if NUMBA_AVAILABLE:
#                    neg_loss_gains = _crossentropy_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, B_T)
#                    # shift by min to make positive (parity with original)
#                    neg_loss_gains -= np.nanmin(neg_loss_gains)
#                else:
#                    with np.errstate(divide='ignore', invalid='ignore'):
#                        r       = (sorted_weight_sums @ B_T)/sorted_weight_sums[:,0].reshape(-1,1)
#                        r_right = (sorted_weight_sums_right @ B_T)/sorted_weight_sums_right[:,0].reshape(-1,1)
#                        neg_loss_gains  = sorted_weight_sums[:,0]*np.sum( ( 0.5*np.log((1./(1.+r))**2) + r*0.5*np.log((r/(1.+r))**2) ), axis=1)
#                        neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( 0.5*np.log((1./(1.+r_right))**2) + r_right*0.5*np.log((r_right/(1.+r_right))**2) ), axis=1)
#                        neg_loss_gains -= np.nanmin(neg_loss_gains)
#
#            # Optional: min_node_size_neg_adjust (unchanged logic)
#            if self.cfg['min_node_size_neg_adjust']:
#                with np.errstate(divide='ignore', invalid='ignore'):
#                    sorted_pos_sums       = np.cumsum( (self.training_weights[:,0]>0).astype('int')[feature_sorted_indices])
#                    total_pos_sum         = sorted_pos_sums[-1]
#                    sorted_pos_sums       = sorted_pos_sums[0:-1]
#                    sorted_pos_sums_right = total_pos_sum-sorted_pos_sums
#                    sorted_neg_sum       = np.arange(1, len(self.training_weights[:,0])) - sorted_pos_sums
#                    sorted_neg_sum_right = np.arange(len(self.training_weights[:,0])-1,0,-1) - sorted_pos_sums_right
#                    f      = sorted_neg_sum/np.maximum(sorted_pos_sums, 1).astype(float)
#                    f_right= sorted_neg_sum_right/np.maximum(sorted_pos_sums_right, 1).astype(float)
#
#                    plateau_and_split_range_mask &= (np.arange(1, len(self.training_weights[:,0]))>(1+f)/(1-f)*self.min_size)
#                    plateau_and_split_range_mask &= (np.arange(len(self.training_weights[:,0])-1,0,-1)>(1+f_right)/(1-f_right)*self.min_size)
#
#            gain_masked = np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask
#            argmax_fi   = np.argmax(gain_masked)
#            gain        = gain_masked[argmax_fi]
#            value       = feature_values[feature_sorted_indices[argmax_fi]]
#
#            debug_self_split_gain = self.split_gain
#            if gain > self.split_gain: 
#                self.split_i_feature = i_feature
#                self.split_value     = value
#                self.split_gain      = gain
#
#            if np.count_nonzero(self.features[:,self.split_i_feature]<=self.split_value) == 1:
#                print ("sorted_weight_sums[:,0]      ", sorted_weight_sums[:,0])
#                print ("sorted_weight_sums_right[:,0]", sorted_weight_sums_right[:,0])
#                print ("plateau_and_split_range_mask", plateau_and_split_range_mask)
#                if loss_mode == 'MSE':
#                    L = sorted_weight_sums @ B_T
#                    R = sorted_weight_sums_right @ B_T
#                    print ("neg_loss_gains left", np.sum(L*L,axis=1)/sorted_weight_sums[:,0] )
#                    print ("neg_loss_gains right", np.sum(R*R,axis=1)/sorted_weight_sums_right[:,0])
#                print ("neg_loss_gains", neg_loss_gains)
#                print ("np.nan_to_num(neg_loss_gains)", np.nan_to_num(neg_loss_gains))
#                print ("np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask", np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask)
#                print ("argmax_fi", np.argmax(np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask) )
#                print ("gain", gain )
#                print ("found split?", gain > debug_self_split_gain, "gain", gain, "self.split_gain (before)",debug_self_split_gain, "self.split_gain(after)",self.split_gain)
#                print ("self.split_left_group", self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool'))
#                print ("non_zero", np.count_nonzero(self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')))
#                print ()
#                assert False, "single-entry node!!"
#
#        assert not np.isnan(self.split_value)
#
#        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')

    def get_split_vectorized( self ):
        """Determine where to split the features.

        Modes:
          - exact  (default): current behavior (argsort + full cumsum over events)
          - binned: bin feature values into n_bins, accumulate *weighted sums* per bin,
                    do prefix sums over bins only, evaluate gains at bin boundaries.
        """

        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None

        # for a valid binary split, we need at least twice the mean size (in events)
        assert self.size >= 2*self.min_size

        # precompute transposes used in kernels
        B_T  = self.base_point_const.transpose()
        loss_mode = self.cfg['loss']

        mode = getattr(self, "split_mode", "exact")
        if mode not in ("exact", "binned"):
            mode = "exact"
        n_bins = int(getattr(self, "n_bins", 256))
        if n_bins < 2:
            mode = "exact"
        use_precomputed_bins = (
            mode == "binned"
            and getattr(self, "binned_features", None) is not None
            and getattr(self, "precomputed_cuts", None) is not None
        )

        # convenience (used in both modes)
        TW = self.training_weights  # shape (N, D)
        TW_gpu = None
        B_T_gpu = None
        Mpos_gpu = None

        if mode == "binned":
            if not CUPY_AVAILABLE:
                raise RuntimeError("Requested GPU backend but cupy is not available.")
            TW_gpu = cp.asarray(TW, dtype=cp.float64)
            B_T_gpu = cp.asarray(B_T, dtype=cp.float64)
            if self.cfg['positive']:
                Mpos_gpu = cp.asarray(self.base_point_const_for_pos.transpose(), dtype=cp.float64)

        bins_gpu = cp.asarray(self.binned_features, dtype=cp.int32) if use_precomputed_bins else None

        for i_feature in range(self.n_features):

            # ------------------------ BINNED MODE ------------------------
            if mode == "binned":
                if use_precomputed_bins:
                    cuts_i = self.precomputed_cuts[i_feature]
                    nbin = int(cuts_i.shape[0] + 1)
                    if nbin < 2:
                        continue
                    idx = bins_gpu[:, i_feature]
                else:
                    feature_values = self.features[:, i_feature]
                    min_, max_ = float(np.min(feature_values)), float(np.max(feature_values))
                    if not np.isfinite(min_) or not np.isfinite(max_) or max_ <= min_:
                        continue
                    quant = bool(getattr(self, "quantile_bins", False))

                    if quant:
                        qs = np.linspace(0.0, 1.0, n_bins + 1, endpoint=True)
                        edges = np.quantile(feature_values, qs)
                        edges = np.unique(edges)
                        if edges.size < 3:
                            continue
                        nbin = int(edges.size - 1)
                    else:
                        edges = np.linspace(min_, max_, n_bins + 1, endpoint=True)
                        nbin = int(n_bins)

                    feature_values_gpu = cp.asarray(feature_values, dtype=cp.float64)
                    edges_gpu = cp.asarray(edges, dtype=cp.float64)
                    idx = cp.searchsorted(edges_gpu, feature_values_gpu, side="right") - 1
                    idx = cp.clip(idx, 0, nbin - 1).astype(cp.int32, copy=False)

                # per-bin event counts (for min_size constraint)
                bin_counts = cp.bincount(idx, minlength=nbin).astype(cp.int64, copy=False)

                # accumulate weighted sums per bin on GPU
                bin_sums = cp.stack(
                    [cp.bincount(idx, weights=TW_gpu[:, _d], minlength=nbin) for _d in range(TW.shape[1])],
                    axis=1,
                ).astype(cp.float64, copy=False)

                # prefix sums over bins -> candidates are boundaries between bins
                prefix_sums = cp.cumsum(bin_sums, axis=0)         # (nbin, D)
                left_sums   = prefix_sums[:-1]                    # (nbin-1, D)
                total_sum   = prefix_sums[-1]                     # (D,)
                right_sums  = total_sum - left_sums               # (nbin-1, D)

                prefix_cnt  = cp.cumsum(bin_counts)               # (nbin,)
                left_cnt    = prefix_cnt[:-1]                     # (nbin-1,)
                total_cnt   = prefix_cnt[-1]
                right_cnt   = total_cnt - left_cnt                # (nbin-1,)

                # candidate mask
                mask = cp.ones(nbin - 1, dtype=cp.bool_)

                # respect min_size in EVENTS (parity with original)
                if self.min_size > 1:
                    mask &= (left_cnt  >= self.min_size)
                    mask &= (right_cnt >= self.min_size)

                # optional negative-adjust logic (approximate in binned space)
                if self.cfg['min_node_size_neg_adjust']:
                    # count positives in column 0 per bin
                    pos0 = (TW_gpu[:, 0] > 0).astype(cp.float64)
                    pos_per_bin = cp.bincount(idx, weights=pos0, minlength=nbin).astype(cp.float64, copy=False)
                    prefix_pos  = cp.cumsum(pos_per_bin)[:-1]
                    left_pos    = prefix_pos
                    right_pos   = cp.sum(pos_per_bin) - left_pos

                    left_neg  = left_cnt.astype(cp.float64)  - left_pos
                    right_neg = right_cnt.astype(cp.float64) - right_pos

                    left_pos_safe  = cp.maximum(left_pos,  1.0)
                    right_pos_safe = cp.maximum(right_pos, 1.0)
                    f      = left_neg  / left_pos_safe
                    f_right= right_neg / right_pos_safe

                    mask &= (left_cnt.astype(cp.float64)  > (1+f)/(1-f) * self.min_size)
                    mask &= (right_cnt.astype(cp.float64) > (1+f_right)/(1-f_right) * self.min_size)

                # positivity constraints (use weighted sums, as in original)
                if self.cfg['positive']:
                    posL = cp.all((left_sums  @ Mpos_gpu) >= 0, axis=1)
                    posR = cp.all((right_sums @ Mpos_gpu) >= 0, axis=1)
                    mask &= posL
                    mask &= posR

                # Never allow negative yields (weighted yield is column 0)
                mask &= (left_sums[:, 0]  > 0)
                mask &= (right_sums[:, 0] > 0)

                # --------- compute neg_loss_gains on GPU ----------
                if loss_mode == 'MSE':
                    L = left_sums @ B_T_gpu
                    R = right_sums @ B_T_gpu
                    neg_loss_gains = (cp.sum(L*L, axis=1)/left_sums[:,0]
                                      + cp.sum(R*R, axis=1)/right_sums[:,0])

                elif loss_mode == 'CrossEntropy':
                    with cp.errstate(divide='ignore', invalid='ignore'):
                        r       = (left_sums  @ B_T_gpu)/left_sums[:,0].reshape(-1,1)
                        r_right = (right_sums @ B_T_gpu)/right_sums[:,0].reshape(-1,1)
                        neg_loss_gains  = left_sums[:,0]*cp.sum((0.5*cp.log((1./(1.+r))**2) + r*0.5*cp.log((r/(1.+r))**2)), axis=1)
                        neg_loss_gains += right_sums[:,0]*cp.sum((0.5*cp.log((1./(1.+r_right))**2) + r_right*0.5*cp.log((r_right/(1.+r_right))**2)), axis=1)
                        neg_loss_gains -= cp.nanmin(neg_loss_gains)

                gain_masked = cp.nan_to_num(neg_loss_gains) * mask.astype(neg_loss_gains.dtype, copy=False)
                argmax_fi   = int(cp.asnumpy(cp.argmax(gain_masked)))
                gain        = float(cp.asnumpy(gain_masked[argmax_fi]))

                # split at boundary after bin argmax_fi
                if use_precomputed_bins:
                    value = float(cuts_i[argmax_fi])
                else:
                    value = edges[argmax_fi + 1]

                if gain > self.split_gain:
                    self.split_i_feature = i_feature
                    self.split_value     = value
                    self.split_gain      = gain
                    self.split_bin       = int(argmax_fi)

                continue  # next feature

            # ------------------------ EXACT MODE (CURRENT) ------------------------
            feature_values = self.features[:, i_feature]
            feature_sorted_indices = np.argsort(feature_values)

            sorted_weight_sums     = np.cumsum(TW[feature_sorted_indices], axis=0)  # (N, D)

            # respect min size for split & optional max_n_split
            if self.max_n_split < 2:
                plateau_and_split_range_mask = np.ones(self.size-1, dtype=np.dtype('bool'))
            else:
                min_, max_ = float(np.min(feature_values)), float(np.max(feature_values))
                bins = np.linspace(min_, max_, self.max_n_split+2, endpoint=True)
                digit = np.digitize(feature_values[feature_sorted_indices], bins)
                plateau_and_split_range_mask = (digit[1:] != digit[:-1])
                plateau_and_split_range_mask = np.insert(plateau_and_split_range_mask, 0, False)[:-1]

            if self.min_size > 1:
                plateau_and_split_range_mask[0:self.min_size-1] = False
                plateau_and_split_range_mask[-self.min_size+1:] = False
            plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)

            total_weight_sum         = sorted_weight_sums[-1]
            sorted_weight_sums       = sorted_weight_sums[0:-1]
            sorted_weight_sums_right = total_weight_sum - sorted_weight_sums

            # mask negative definite splits
            if self.cfg['positive']:
                Mpos = self.base_point_const_for_pos.transpose()
                pos       = np.all((sorted_weight_sums       @ Mpos) >= 0, axis=1)
                pos_right = np.all((sorted_weight_sums_right @ Mpos) >= 0, axis=1)
                plateau_and_split_range_mask &= pos
                plateau_and_split_range_mask &= pos_right

            # Never allow negative yields
            plateau_and_split_range_mask &= (sorted_weight_sums[:,0]       > 0)
            plateau_and_split_range_mask &= (sorted_weight_sums_right[:,0] > 0)

            # --------- compute neg_loss_gains (JIT kernels) ----------
            if loss_mode == 'MSE':
                if NUMBA_AVAILABLE:
                    neg_loss_gains = _mse_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, B_T)
                else:
                    L = sorted_weight_sums @ B_T
                    R = sorted_weight_sums_right @ B_T
                    neg_loss_gains = (np.sum(L*L, axis=1)/sorted_weight_sums[:,0]
                                      + np.sum(R*R, axis=1)/sorted_weight_sums_right[:,0])

            elif loss_mode == 'CrossEntropy':
                if NUMBA_AVAILABLE:
                    neg_loss_gains = _crossentropy_neg_loss_gains(sorted_weight_sums, sorted_weight_sums_right, B_T)
                    neg_loss_gains -= np.nanmin(neg_loss_gains)
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r       = (sorted_weight_sums @ B_T)/sorted_weight_sums[:,0].reshape(-1,1)
                        r_right = (sorted_weight_sums_right @ B_T)/sorted_weight_sums_right[:,0].reshape(-1,1)
                        neg_loss_gains  = sorted_weight_sums[:,0]*np.sum( ( 0.5*np.log((1./(1.+r))**2) + r*0.5*np.log((r/(1.+r))**2) ), axis=1)
                        neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( 0.5*np.log((1./(1.+r_right))**2) + r_right*0.5*np.log((r_right/(1.+r_right))**2) ), axis=1)
                        neg_loss_gains -= np.nanmin(neg_loss_gains)

            if self.cfg['min_node_size_neg_adjust']:
                with np.errstate(divide='ignore', invalid='ignore'):
                    sorted_pos_sums       = np.cumsum( (TW[:,0]>0).astype('int')[feature_sorted_indices])
                    total_pos_sum         = sorted_pos_sums[-1]
                    sorted_pos_sums       = sorted_pos_sums[0:-1]
                    sorted_pos_sums_right = total_pos_sum - sorted_pos_sums
                    sorted_neg_sum       = np.arange(1, len(TW[:,0])) - sorted_pos_sums
                    sorted_neg_sum_right = np.arange(len(TW[:,0])-1,0,-1) - sorted_pos_sums_right
                    f      = sorted_neg_sum/np.maximum(sorted_pos_sums, 1).astype(float)
                    f_right= sorted_neg_sum_right/np.maximum(sorted_pos_sums_right, 1).astype(float)

                    plateau_and_split_range_mask &= (np.arange(1, len(TW[:,0]))>(1+f)/(1-f)*self.min_size)
                    plateau_and_split_range_mask &= (np.arange(len(TW[:,0])-1,0,-1)>(1+f_right)/(1-f_right)*self.min_size)

            gain_masked = np.nan_to_num(neg_loss_gains) * plateau_and_split_range_mask
            argmax_fi   = np.argmax(gain_masked)
            gain        = gain_masked[argmax_fi]
            value       = feature_values[feature_sorted_indices[argmax_fi]]

            if gain > self.split_gain:
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain      = gain
                self.split_bin       = -1

        assert not np.isnan(self.split_value)

        if use_precomputed_bins and self.split_bin >= 0:
            self.split_left_group = self.binned_features[:, self.split_i_feature] <= self.split_bin
        else:
            self.split_left_group = self.features[:,self.split_i_feature] <= self.split_value if not np.isnan(self.split_value) else np.ones(self.size, dtype='bool')


    def coefficient_sum( self, group ):
        return np.sum(self.training_weights[group],axis=0)

    def negative_fraction( self, group ):
        ''' lambda ~ omega*n = omega*(n^+ - n^-) -> this function returns ~ n^+/n^-
        '''
        neg = float(np.count_nonzero(self.training_weights[group][:,0]<0))
        return neg/( len(group) - neg )

    # everything we want to store in the terminal nodes
    def __store( self, group ):
        return {
            'size': np.count_nonzero(group),
            'coefficient_sum': self.coefficient_sum(group),
            'f'   : self.negative_fraction(group), 
            }

    # Create child splits for a node or make terminal
    def split(self, _depth=0):

        # Find the best split
        if self.cfg["_get_only_score"]:
            # store derivatives in the left box and do not split further 
            self.split_i_feature, self.split_value, self.split_left_group = 0, +float('inf'), None
            self.split_bin    = -1
            self.left        = ResultNode(derivatives=self.derivatives, **self.__store(np.ones(self.size,dtype=bool)))
            self.right       = ResultNode(derivatives=self.derivatives, **self.__store(np.zeros(self.size,dtype=bool)))
            return

        self.get_split_vectorized()

        # check for max depth or a 'no' split
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group):
            # stop splitting further. Put everything in the left node
            self.split_value = float('inf')
            self.split_bin   = -1
            self.left        = ResultNode(derivatives=self.derivatives, **self.__store(np.ones(self.size,dtype=bool)))
            self.right       = ResultNode(derivatives=self.derivatives, **self.__store(np.zeros(self.size,dtype=bool)))
            return

        # process left child
        if np.count_nonzero(self.split_left_group) < 2*self.min_size:
            self.left             = ResultNode(derivatives=self.derivatives, **self.__store(self.split_left_group) )
        else:
            child_cfg = _child_cfg_without_cached_bins(self.cfg)
            self.left             = MultiNode(
                None if self.features is None else self.features[self.split_left_group],
                training_weights = self.training_weights[self.split_left_group],
                binned_features = None if self.binned_features is None else self.binned_features[self.split_left_group],
                _depth=self._depth+1,
                **child_cfg,
            )
        # process right child
        if np.count_nonzero(~self.split_left_group) < 2*self.min_size:
            self.right            = ResultNode(derivatives=self.derivatives, **self.__store(~self.split_left_group) )
        else:
            child_cfg = _child_cfg_without_cached_bins(self.cfg)
            self.right            = MultiNode(
                None if self.features is None else self.features[~self.split_left_group],
                training_weights = self.training_weights[~self.split_left_group],
                binned_features = None if self.binned_features is None else self.binned_features[~self.split_left_group],
                _depth=self._depth+1,
                **child_cfg,
            )

    # Prediction    
    def predict( self, features):
        ''' obtain the result by recursively descending down the tree
        '''
        node = self.left if features[self.split_i_feature]<=self.split_value else self.right
        if isinstance(node, ResultNode):
            return node.coefficient_sum 
        else:
            return node.predict(features)

    def vectorized_predict(self, feature_matrix):
        N = len(feature_matrix)
        D = len(self.derivatives)
        if N == 0:
            return np.zeros((0, D), dtype=np.float64)

        if not NUMBA_AVAILABLE:
            predictions = np.zeros((N, D), dtype=np.float64)
            stack = [(self, np.arange(N, dtype=np.intp))]
            while stack:
                node, indices = stack.pop()
                if isinstance(node, ResultNode):
                    predictions[indices] = node.coefficient_sum
                else:
                    go_left = feature_matrix[indices, node.split_i_feature] <= node.split_value
                    left_idx  = indices[ go_left]
                    right_idx = indices[~go_left]
                    if len(right_idx):
                        stack.append((node.right, right_idx))
                    if len(left_idx):
                        stack.append((node.left, left_idx))
            return predictions

        self._ensure_prediction_state()
        predictions = np.empty((N, D), dtype=np.float64)
        kernel = _predict_tree_numba_parallel if N >= 4096 else _predict_tree_numba
        kernel(
            np.asarray(feature_matrix),
            self._predict_split_feature,
            self._predict_split_value,
            self._predict_left_child,
            self._predict_right_child,
            self._predict_is_leaf,
            self._predict_leaf_value,
            predictions,
        )
        return predictions

    # remove the 'inf' splits
    def prune( self ):
        if not isinstance(self.left, ResultNode) and self.left.split_value==float('+inf'):
            self.left = self.left.left
        elif not isinstance(self.left, ResultNode):
            self.left.prune()
        if not isinstance(self.right, ResultNode) and self.right.split_value==float('+inf'):
            self.right = self.right.left
        elif not isinstance(self.right, ResultNode):
            self.right.prune()

    # Print a decision tree
    def print_tree(self, _depth=0):
        print('%s[%s <= %.3f]' % ((self._depth*' ', "X%d"%self.split_i_feature if self.feature_names is None else self.feature_names[self.split_i_feature], self.split_value)))
        for node in [self.left, self.right]:
            node.print_tree(_depth = _depth+1)

    def get_list(self):
        ''' recursively obtain all thresholds '''
        return [ (self.split_i_feature, self.split_value), self.left.get_list(), self.right.get_list() ] 

class ResultNode:
    ''' Simple helper class to store result value.
    '''
    def __init__( self, derivatives=None, **kwargs):
        for k, v in kwargs.items():
            setattr( self, k, v)
        self.derivatives     = derivatives

    @staticmethod
    def prefac(der):
        return (0.5 if (len(der)==2 and len(set(der))==1) else 1. )

    def print_tree(self, _depth=0):
        r_poly_str = "".join(["*".join(["{:+.2e}".format(self.prefac(der)*self.coefficient_sum[i_der]/self.coefficient_sum[0])] + list(self.derivatives[i_der]) ) for i_der, der in enumerate(self.derivatives)])
        c_poly_str = "".join(["*".join(["{:+.2e}".format(self.prefac(der)*self.coefficient_sum[i_der])] + list(self.derivatives[i_der]) ) for i_der, der in enumerate(self.derivatives)])
        try:
            unc = 1./sqrt(self.size)*sqrt((1+self.f)/(1-self.f))
        except ZeroDivisionError:
            unc = 0 
        print_string = '%s(%6i, unc=%1.3f) r = %s   c = %s' % ((_depth)*' ', self.size, 1./sqrt(self.size)*sqrt((1+self.f)/(1-self.f)), r_poly_str, c_poly_str)
        print(print_string)

    def get_list(self):
        ''' recursively obtain all thresholds (bottom of recursion)'''
        return self.coefficient_sum 
