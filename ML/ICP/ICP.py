#!/usr/bin/env python
# Standard imports
import cProfile
import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
import time
import pickle
import copy
import itertools
import numpy as np
import operator
import functools

from data_loader.data_loader_2 import H5DataLoader

class InclusiveCrosssectionParametrization:
    def __init__( self, config=None, combinations=None, nominal_base_point=None, base_points=None, parameters=None):

        if config is not None:
            self.config      = config
            self.config_name = config.__name__ 
            self.base_points   = np.array(config.base_points)
            self.n_base_points = len(self.base_points)
    
            self.nominal_base_point = np.array( config.nominal_base_point, dtype='float')
            self.combinations       = config.combinations
            self.parameters         = config.parameters
        elif (combinations is not None) and (nominal_base_point is not None) and (base_points is not None) and (parameters is not None):
            #print("Config not provided, the ICP can only be used for prediction.")
            self.config_name   = None
            self.base_points   = np.array(base_points)
            self.n_base_points = len(self.base_points)
    
            self.nominal_base_point = np.array( nominal_base_point, dtype='float')
            self.combinations       = combinations
            self.parameters         = parameters
        else:
            raise Exception("Please provide either a config or all other parameters (combinations, nominal_base_point, base_points, parameters).")

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

        # Compute matrix VkA from non-nominal base_points
        self._VKA = np.zeros( (len(self.masked_base_points), len(self.combinations)) )
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_combination, combination in enumerate(self.combinations):
                res=1
                for var in combination:
                    res*=base_point[self.parameters.index(var)]

                self._VKA[i_base_point, i_combination ] = res

    def load_training_data( self, datasets_hephy, selection, process=None, n_split=10):
        self.training_data = {}
        for base_point in self.base_points:
            base_point  = tuple(base_point)
            values      = self.config.get_alpha(base_point)
            data_loader = datasets_hephy.get_data_loader( selection=selection, values=values, process=process, selection_function=None, n_split=n_split)
            print ("ICP training data: Base point nu = %r, alpha = %r, file = %s"%( base_point, values, data_loader.file_path)) 
            self.training_data[base_point] = data_loader

    def train( self, small=False, train_ratio=True, yields={}, selection=None):

        # We might pass yields externally
        self.yields = yields

        # Fetch from data loader if not externally provided
        if len(yields)==0:
            for base_point, loader in self.training_data.items():
                self.yields[base_point] = H5DataLoader.get_weight_sum(self.training_data[base_point], small=small, selection=selection)

        # The default case: We devide by the nominal base-point yield and 
        if train_ratio:
            self.DeltaA = np.dot( self.CInv, sum([ self._VKA[i_base_point]*np.log(self.yields[tuple(base_point)]/self.yields[self.nominal_base_point_key]) for i_base_point, base_point in enumerate(self.masked_base_points)]))
        # Parametrize including a constant offset
        else:
            if ( tuple() not in self.combinations ):
                raise RuntimeError("We must have the constant term (an empty tuple) in the combinations unless we're training ratios. Please add it.")
 
            self.DeltaA = np.dot( self.CInv, sum([ self._VKA[i_base_point]*np.log(self.yields[tuple(base_point)]) for i_base_point, base_point in enumerate(self.masked_base_points)])) 

    def __str__( self ):
        return " ".join( [(f"{deltaA:+.2e}")+( "*"+c if c!="" else "") for deltaA, c  in zip( self.DeltaA, [ "*".join( comb ) for comb in self.combinations])] )

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            import importlib
            old_instance = pickle.load(file_)
            if old_instance.config_name is not None:
                config_ = importlib.import_module(old_instance.config_name) 
                new_instance = cls(   
                        config               = config_,
                        )
            else:
                new_instance = cls(   
                        nominal_base_point  = old_instance.nominal_base_point,
                        parameters          = old_instance.parameters,
                        combinations        = old_instance.combinations,
                        base_points         = old_instance.base_points,
                        )

            # Everything we need for prediction should come from the old instance, not the config
            new_instance.DeltaA         = old_instance.DeltaA
            new_instance.combinations   = old_instance.combinations
            new_instance.parameters     = old_instance.parameters

            return new_instance  

    def save(self, filename):
        _config = self.config
        self.config=None
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )
        self.config = _config

    def __setstate__(self, state):
        self.__dict__ = state

    def nu_A(self, nu):
        return np.array( [ functools.reduce(operator.mul, [nu[self.parameters.index(c)] for c in list(comb)], 1) for comb in self.combinations] )

    def log_predict( self, nu):
        return np.dot( self.nu_A(nu), self.DeltaA )

    def predict( self, nu):
        return np.exp( self.log_predict(nu) )

    # Predictor of log-DCR for usage in PNN
    def get_predictor( self ):

        def predictor(  DeltaA=self.DeltaA, parameters=self.parameters, combinations=self.combinations, **kwargs,):
            
            nu = [ 0 for _ in range(3)]

            for p in parameters:
                if p not in kwargs:
                    raise RuntimeError(f"Must set all parameters: {parameters}")

            for k, v in kwargs.items():
                nu[parameters.index(k)] = v

            nu_A = np.array( [ functools.reduce(operator.mul, [nu[parameters.index(c)] for c in list(comb)], 1) for comb in combinations] )

            return np.exp(np.dot( nu_A, DeltaA ))

        return predictor
