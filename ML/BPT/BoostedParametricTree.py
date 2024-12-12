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
import Node
from tqdm import tqdm

from data_loader.data_loader_2 import H5DataLoader

class BoostedParametricTree:
    def __init__( self, config ):

        self.config      = config
        self.config_name = config.__name__
        self.base_points   = np.array(config.base_points)
        self.n_base_points = len(self.base_points)

        self.nominal_base_point = np.array( config.nominal_base_point, dtype='float')
        self.combinations       = config.combinations
        self.parameters         = config.parameters

        # We should copy all the pieces needed at inference time to the config. Otherwise, a change to the config effects inference post-training.
        self.n_trees            = config.n_trees
        self.learning_rate      = config.learning_rate

        # Base point matrix
        self.VkA  = np.zeros( [len(self.base_points), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(self.base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                self.VkA[i_base_point][i_comb1] += functools.reduce(operator.mul, [base_point[self.parameters.index(c)] for c in list(comb1)], 1)

        # Dissect inputs into nominal sample and variied
        nominal_base_point_index = np.where(np.all(self.base_points==self.nominal_base_point,axis=1))[0]
        assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( self.nominal_base_point, self.base_points)
        self.nominal_base_point_index = nominal_base_point_index[0]
        self.nominal_base_point_key   = tuple(self.nominal_base_point)

        nu_mask = np.ones(len(self.base_points), bool)
        nu_mask[self.nominal_base_point_index] = 0

        # remove the nominal from the list of all the base_points
        self.masked_base_points = self.base_points[nu_mask]

        # computing base-point matrix
        C    = np.zeros( [len(self.combinations), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                for i_comb2, comb2 in enumerate(self.combinations):
                    C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[self.parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

        assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

        self.CInv = np.linalg.inv(C)

        self._VKA = np.zeros( (len(self.masked_base_points), len(self.combinations)) )
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_combination, combination in enumerate(self.combinations):
                res=1
                for var in combination:
                    res*=base_point[self.parameters.index(var)]

                self._VKA[i_base_point, i_combination ] = res

        # Compute matrix Mkk from non-nominal base_points
        self.MkA  = np.dot(self._VKA, self.CInv).transpose()
        self.Mkkp = np.dot(self._VKA, self.MkA )

        # Will hold the trees
        self.trees              = []

    def load_training_data(self, datasets, selection, n_split=10, max_batch=-1):
        all_features = []
        all_weights = []
        all_enumeration = []

        for i_base_point, base_point in enumerate(self.base_points):
            base_point = tuple(base_point)
            values = self.config.get_alpha(base_point)
            data_loader = datasets.get_data_loader(
                selection=selection,
                values=values,
                selection_function=None,
                n_split=n_split
            )
            print(
                f"ICP training data: Base point nu = {base_point}, alpha = {values}, file = {data_loader.file_path}"
            )

            # Loop through the dataloader and collect data with tqdm
            i_batch = 0
            for batch in tqdm(data_loader, desc=f"Processing base point {i_base_point}/{len(self.base_points)}", unit="batch"):
                features, weights, _ = data_loader.split(batch)
                all_features.append(features)
                all_weights.append(weights)
                all_enumeration.append(np.full(len(features), i_base_point))

                i_batch+=1
                if i_batch>=max_batch: break

        # Concatenate all collected data
        self.features = np.concatenate(all_features, axis=0)
        self.weights = np.concatenate(all_weights, axis=0)
        self.enumeration = np.concatenate(all_enumeration, axis=0)

    @staticmethod 
    def sort_comb( comb ):
        return tuple(sorted(comb))

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            import importlib
            old_instance = pickle.load(file_)
            config_ = importlib.import_module(old_instance.config_name)
            new_instance = cls(config = config_)

            new_instance.combinations = old_instance.combinations
            new_instance.parameters   = old_instance.parameters
            new_instance.n_trees      = old_instance.n_trees
            new_instance.learning_rate= old_instance.learning_rate
            new_instance.VkA          = old_instance.VkA
            new_instance.CInv         = old_instance.CInv
            new_instance.Mkkp         = old_instance.Mkkp
            new_instance.MkA          = old_instance.MkA
            new_instance.n_base_points=old_instance.n_base_points,
            new_instance.nominal_base_point_index=old_instance.nominal_base_point_index,
            new_instance.feature_names= old_instance.feature_names if hasattr( old_instance, "feature_names") else None,

            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        _config = self.config
        self.config=None
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )
        self.config = _config

    def train( self ):

        toolbar_width = min(20, self.n_trees)

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0

        # reweight only the non-base point events
        reweight_mask = self.enumeration!=self.nominal_base_point_index
        
        for n_tree in range(self.n_trees):

            training_time = 0

            # store the param vector in the first tree:
            _get_only_param = ( (n_tree==0) and self.config.learn_global_param )
            self.config._get_only_param = _get_only_param 

            # fit to data
            time1 = time.process_time()

            root = Node.Node( 
                 features     = self.features,
                 weights      = self.weights,
                 enumeration  = self.enumeration,
                 Mkkp         = self.Mkkp,
                 MkA          = self.MkA,
                 n_base_points=self.n_base_points,
                 nominal_base_point_index=self.nominal_base_point_index,
                 combinations = self.combinations,
                 feature_names= self.feature_names if hasattr( self, "feature_names") else None,
                 min_size     = self.config.min_size,  
                 max_depth    = self.config.max_depth,
                 loss         = self.config.loss )

            time2 = time.process_time()
            weak_learner_time += time2 - time1
            training_time      = time2 - time1

            self.trees.append( root )

            # Recall current tree
            time1 = time.process_time()

            # reweight the non-nominal data
            learning_rate = 1. if _get_only_param else self.learning_rate 
            self.weights[reweight_mask] *=\
                np.exp(-learning_rate*np.einsum('ij,ij->i',  
                    root.vectorized_predict( self.features[reweight_mask] ), 
                    self.VkA[self.enumeration[reweight_mask]])
                    )

            time2 = time.process_time()
            update_time   += time2 - time1
            training_time += time2 - time1

            self.trees[-1].training_time = training_time 

            # update the bar
            if self.n_trees>=toolbar_width:
                if n_tree % (self.n_trees/toolbar_width)==0:   sys.stdout.write("-")

            try:
                sys.stdout.flush()
            except OSError:
                pass

        sys.stdout.write("]\n") # this ends the progress bar
        print ("weak learner time: %.2f" % weak_learner_time)
        print ("update time: %.2f" % update_time)
       
        # purge training data
        del self.enumeration
        del self.features   
        del self.weights    

    def predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rtes
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
        # Does the first tree hold the global param?
        if self.cfg["learn_global_param"]:
             learning_rates[0] = 1
            
        predictions = np.array([ tree.predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        if summed:
            return np.dot(learning_rates, predictions)
        else:
            return learning_rates.reshape(-1, 1)*predictions
    
    def vectorized_predict( self, feature_array, max_n_tree = None, summed = True, last_tree_counts_full = False):
        # list learning rates
        learning_rates = self.learning_rate*np.ones(max_n_tree if max_n_tree is not None else self.n_trees)
        # keep the last tree?
        if last_tree_counts_full and (max_n_tree is None or max_n_tree==self.n_trees):
            learning_rates[-1] = 1
        # Does the first tree hold the global param?
        if self.cfg["learn_global_param"]:
             learning_rates[0] = 1
            
        predictions = np.array([ tree.vectorized_predict( feature_array ) for tree in self.trees[:max_n_tree] ])
        #predictions = predictions[:,:,1:]/np.expand_dims(predictions[:,:,0], -1)
        if summed:
            return np.sum(learning_rates.reshape(-1,1,1)*predictions, axis=0)
        else:
            return learning_rates.reshape(-1,1,1)*predictions 

