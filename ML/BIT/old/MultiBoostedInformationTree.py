#!/usr/bin/env python
# Standard imports
import cProfile
import sys
#sys.path.insert( 0, '..')
#sys.path.insert( 0, '.')
import time
import pickle
import copy
import itertools
import numpy as np
import operator
import functools

sys.path.insert(0, '..'); sys.path.insert(0, '../..')
import ML.BIT.MultiNode as MultiNode

default_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "MSE", # or "CrossEntropy" 
#    "bagging_fraction": 1.,
    "learn_global_score": False,
}

class MultiBoostedInformationTree:

    def __init__( self, training_features, training_weights, 
                    **kwargs ):

        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        self.node_cfg = {}
        for (key, val) in kwargs.items():
            if key in MultiNode.default_cfg.keys():
                self.node_cfg[key] = val 
            elif key in default_cfg.keys():
                self.cfg[key]      = val
            else:
                raise RuntimeError( "Got unexpected keyword arg: %s:%r" %( key, val ) )
        self.node_cfg['loss'] = self.cfg['loss'] 

        for (key, val) in self.cfg.items():
                setattr( self, key, val )

        # Attempt to learn 98%. (1-learning_rate)^n_trees = 0.02 -> After the fit, the score is at least down to 2% 
        if self.learning_rate == "auto":
            self.learning_rate = 1-0.02**(1./self.n_trees)

        self.training_weights   = copy.deepcopy(training_weights)
        if training_weights is not None:
            self.training_weights = {self.sort_comb(key):val for key,val in self.training_weights.items()}

        self.training_features  = training_features

        # Will hold the trees
        self.trees              = []

    @staticmethod 
    def sort_comb( comb ):
        return tuple(sorted(comb))

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls( None, None, 
                    n_trees             = old_instance.n_trees, 
                    learning_rate       = old_instance.learning_rate,
                    learn_global_score  = old_instance.learn_global_score if hasattr( old_instance, "learn_global_score") else False,
                    feature_names       = old_instance.feature_names if hasattr( old_instance, "feature_names") else None,
                    )
            new_instance.trees = old_instance.trees

            new_instance.derivatives = old_instance.trees[0].derivatives[1:]

            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

    def boost( self ):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0
        for n_tree in range(self.n_trees):

            training_time = 0

            # store the score vector in the first tree:
            _get_only_score = ( (n_tree==0) and self.cfg["learn_global_score"] )
            self.node_cfg["_get_only_score"] = _get_only_score 

            # fit to data
            time1 = time.process_time()
            root = MultiNode.MultiNode(   
                            self.training_features, 
                            training_weights = self.training_weights,
                            **self.node_cfg 
                        )

            if n_tree==0:
                self.derivatives = root.derivatives[1:]

            time2 = time.process_time()
            weak_learner_time += time2 - time1
            training_time      = time2 - time1

            self.trees.append( root )

            # Recall current tree
            time1 = time.process_time()

            prediction   = root.vectorized_predict(self.training_features)
            len_         = len(prediction)
            delta_weight = self.training_weights[tuple()].reshape(len_,-1)*prediction[:,1:]/prediction[:,0].reshape(len_,-1)
            learning_rate = 1. if _get_only_score else self.learning_rate 
            for i_der, der in enumerate(root.derivatives[1:]):
                self.training_weights[der] += -learning_rate*delta_weight[:,i_der]

            time2 = time.process_time()
            update_time   += time2 - time1
            training_time += time2 - time1

            self.trees[-1].training_time = training_time 

            # update the bar
            if self.n_trees>=toolbar_width:
                try:
                    if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")
                    sys.stdout.flush()
                except OSError:
                    pass

        try:
            sys.stdout.write("]\n") # this ends the progress bar
        except OSError:
            pass

        print ("weak learner time: %.2f" % weak_learner_time)
        print ("update time: %.2f" % update_time)
       
        # purge training data
        del self.training_weights       
        del self.training_features      

    #def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
    #    # list learning rates
    #    learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
    #    # keep the last tree?
    #    if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
    #        learning_rates[-1] = 1
    #    # Does the first tree hold the global score?
    #    if self.cfg["learn_global_score"]:
    #         learning_rates[0] = 1
    #        
    #    predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])
    #    predictions = predictions[:,1:]/predictions[:,0].reshape(-1,1)
    #    if summed:
    #        return np.dot(learning_rates, predictions)
    #    else:
    #        return learning_rates.reshape(-1, 1)*predictions
    
    #def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
    #    # list learning rates
    #    learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
    #    # keep the last tree?
    #    if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
    #        learning_rates[-1] = 1
    #    # Does the first tree hold the global score?
    #    if self.cfg["learn_global_score"]:
    #         learning_rates[0] = 1
    #        
    #    predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
    #    predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)
    #    if summed:
    #        return np.sum(learning_rates.reshape(-1,1,1)*predictions, axis=0)
    #    else:
    #        return learning_rates.reshape(-1,1,1)*predictions 

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
            If False, returns per-tree predictions (shape (T, N, K)) like before.
        last_tree_counts_full : bool
            If True and using all trees, set the last tree's learning rate to 1.
        """
        # number of trees to use
        T = max_n_tree if max_n_tree is not None else self.n_trees

        # learning rates per tree
        learning_rates = self.learning_rate * np.ones(T, dtype=np.float64)
        if last_tree_counts_full and (max_n_tree is None or max_n_tree == self.n_trees):
            learning_rates[-1] = 1.0
        if self.cfg.get("learn_global_score", False):
            # first tree holds global score
            learning_rates[0] = 1.0

        # Fast exit if no trees
        if T == 0:
            # determine K from a single tree if available (fallback to 0)
            K = 0
            if self.n_trees > 0:
                tmp = self.trees[0].vectorized_predict(feature_array)
                K = max(0, tmp.shape[1] - 1)
            return np.zeros((feature_array.shape[0], K), dtype=np.float64) if summed else \
                   np.zeros((0, feature_array.shape[0], K), dtype=np.float64)

        # We need output dimension K; get it from the first tree cheaply.
        first = self.trees[0].vectorized_predict(feature_array[:1])
        # first has shape (1, 1+K): [denom, num1, num2, ...]
        K = first.shape[1] - 1

        N = feature_array.shape[0]

        if summed:
            # --- streaming accumulation: O(N*K) memory ---
            acc = np.zeros((N, K), dtype=np.float64)
            # work buffers reused to avoid reallocs
            # We’ll compute `ratio = num / denom` safely via np.divide
            for t in range(T):
                raw = self.trees[t].vectorized_predict(feature_array)  # shape (N, 1+K)
                # ensure float64 for stable accumulation
                if raw.dtype != np.float64:
                    raw = raw.astype(np.float64, copy=False)
                denom = raw[:, :1]              # (N,1)
                num   = raw[:, 1:]              # (N,K)
                ratio = np.empty_like(num)      # (N,K)
                # safe division: where denom==0 -> 0 (same effect as original would be NaN; 0 is safer)
                np.divide(num, denom, out=ratio, where=(denom != 0.0))
                acc += learning_rates[t] * ratio
            return acc
        else:
            # --- per-tree tensor (memory heavy, keep legacy behavior) ---
            # If you still hit memory issues here, consider adding a chunked writer or return a generator.
            out = np.empty((T, N, K), dtype=np.float64)
            for t in range(T):
                raw = self.trees[t].vectorized_predict(feature_array)  # (N, 1+K)
                if raw.dtype != np.float64:
                    raw = raw.astype(np.float64, copy=False)
                denom = raw[:, :1]
                num   = raw[:, 1:]
                np.divide(num, denom, out=out[t], where=(denom != 0.0))
                out[t] *= learning_rates[t]
            return out

    def losses( self, feature_array, weight_dict, max_n_tree = None, last_tree_counts_full = False):
        ## list learning rates
        #learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        ## keep the last tree?
        #if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
        #    learning_rates[-1] = 1
        ## Does the first tree hold the global score?
        #if self.cfg["learn_global_score"]:
        #     learning_rates[0] = 1

        # recover base points from tree
        base_points      = self.trees[0].base_points
        base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in self.derivatives] for point in base_points]).astype('float')
        for i_der, der in enumerate(self.derivatives):
            if not (len(der)==2 and der[0]==der[1]): continue
            for i_point in range(len(base_points)):
                base_point_const[i_point][i_der]/=2.
        
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)

        weight_ratio = np.array( [ (weight_dict[der]/weight_dict[()] if der in weight_dict else weight_dict[tuple(reversed(der))]/weight_dict[()]) for der in self.derivatives]).transpose().astype('float')
        # losses
        return -( weight_dict[()][np.newaxis,...,np.newaxis]*np.dot( (predictions - (weight_ratio[np.newaxis,...])), base_point_const )**2).sum(axis=(1,2))
