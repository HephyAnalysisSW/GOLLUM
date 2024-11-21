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

default_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "CrossEntropy", 
    "learn_global_param": False,
    "min_size": 50,
}

class BoostedParametricTree:

    def __init__( self, training_data, combinations, nominal_base_point, base_points, parameters, **kwargs ):

        # make cfg and node_cfg from the kwargs keys known by the Node
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        self.node_cfg = {}
        for (key, val) in kwargs.items():
            if key in Node.default_cfg.keys():
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

        # Make sure of the format
        #if "base_points" in kwargs:
        #    self.base_points = kwargs["base_points"]
        #elif training_data is not None: 
        #    self.base_points = np.array( sorted(list(training_data.keys())), dtype='float')
        #else:
        #    raise RuntimeError("Did not find base_points.")
        self.base_points = np.array(base_points)

        self.n_base_points = len(self.base_points)

        self.nominal_base_point = np.array( nominal_base_point, dtype='float')
        self.combinations       = combinations
        self.parameters         = parameters

        # Base point matrix
        self.VkA  = np.zeros( [len(self.base_points), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(self.base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                self.VkA[i_base_point][i_comb1] += functools.reduce(operator.mul, [base_point[parameters.index(c)] for c in list(comb1)], 1)
            
        # Dissect inputs into nominal sample and variied
        nominal_base_point_index = np.where(np.all(self.base_points==self.nominal_base_point,axis=1))[0]
        assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( self.nominal_base_point, self.base_points)
        self.nominal_base_point_index = nominal_base_point_index[0]
        self.nominal_base_point_key   = tuple(self.nominal_base_point)

        nu_mask = np.ones(len(self.base_points), bool)
        nu_mask[self.nominal_base_point_index] = 0

        # remove the nominal from the list of all the base_points
        masked_base_points = self.base_points[nu_mask]

        # computing base-point matrix
        C    = np.zeros( [len(self.combinations), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(masked_base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                for i_comb2, comb2 in enumerate(self.combinations):
                    C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

        assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

        self.CInv = np.linalg.inv(C)

        # Compute matrix Mkk from non-nominal base_points
        self._VKA = np.zeros( (len(masked_base_points), len(self.combinations)) )
        for i_base_point, base_point in enumerate(masked_base_points):
            for i_combination, combination in enumerate(self.combinations):
                res=1
                for var in combination:
                    res*=base_point[parameters.index(var)]

                self._VKA[i_base_point, i_combination ] = res

        self.MkA  = np.dot(self._VKA, self.CInv).transpose()
        self.Mkkp = np.dot(self._VKA, self.MkA )

        if training_data is not None:
            # Complement training data
            if 'weights' not in training_data[self.nominal_base_point_key]:
                training_data[self.nominal_base_point_key]['weights'] = np.ones(training_data[self.nominal_base_point_key]['features'].shape[0])

            for k, v in training_data.items():
                if "features" not in v and "weights" not in v:
                    raise RuntimeError( "Key %r has neither features nor weights" %k  )
                if k == self.nominal_base_point_key:
                    if 'features' not in v:
                        raise RuntimeError( "Nominal base point does not have features!" )
                else:
                    if not 'features' in v:
                        # we must have weights
                        v['features'] = training_data[self.nominal_base_point_key]['features']
                        if len(v['features'])!=len(v['weights']):
                            raise runtimeerror("key %r has inconsistent length in weights"%v) 
                    if (not 'weights' in training_data[self.nominal_base_point_key].keys()) and 'weights' in v:
                        raise RuntimeError( "Found no weights for nominal base point, but for a variation. This is not allowed" )

                    if not 'weights' in v:
                        v['weights'] = np.ones(v['features'].shape[0])

                if len(v['weights'])!=len(v['features']):
                    raise RuntimeError("Key %r has unequal length of weights and features: %i != %i" % (k, len(v['weights']), len(v['features'])) )

            self.enumeration = np.concatenate( [ np.array( [i_base_point for _ in training_data[tuple(base_point)]['features']]) for i_base_point, base_point in enumerate( self.base_points)] , axis=0)
            self.features    = np.concatenate( [ training_data[tuple(base_point)]['features'] for i_base_point, base_point in enumerate( self.base_points)] , axis=0)
            self.weights     = np.concatenate( [ training_data[tuple(base_point)]['weights'] for i_base_point, base_point in enumerate( self.base_points)] , axis=0)

        # Will hold the trees
        self.trees              = []

    @staticmethod 
    def sort_comb( comb ):
        return tuple(sorted(comb))

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls( None,  
                    n_trees             = old_instance.n_trees, 
                    learning_rate       = old_instance.learning_rate,
                    nominal_base_point  = old_instance.nominal_base_point,
                    parameters          = old_instance.parameters,
                    combinations        = old_instance.combinations,
                    base_points         = old_instance.base_points,
                    learn_global_param  = old_instance.learn_global_param if hasattr( old_instance, "learn_global_param") else False,
                    feature_names       = old_instance.feature_names if hasattr( old_instance, "feature_names") else None,

                    )
            new_instance.trees = old_instance.trees

            #new_instance.derivatives = old_instance.trees[0].derivatives[1:]

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
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        weak_learner_time = 0.0
        update_time = 0.0

        # reweight only the non-base point events
        reweight_mask = self.enumeration!=self.nominal_base_point_index
        
        #snapshot1 = tracemalloc.take_snapshot()

        for n_tree in range(self.n_trees):

            training_time = 0

            # store the param vector in the first tree:
            _get_only_param = ( (n_tree==0) and self.cfg["learn_global_param"] )
            self.node_cfg["_get_only_param"] = _get_only_param 

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
                 **self.node_cfg )

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

            #snapshot2 = tracemalloc.take_snapshot()
            #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            #print ()
            #for stat in top_stats[:10]:
            #    print(stat) 
            #snapshot1 = snapshot2

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

