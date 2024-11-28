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

default_cfg = {
}

class ICP:
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

        for (key, val) in self.cfg.items():
                setattr( self, key, val )

        self.base_points   = np.array(base_points)
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
        self.masked_base_points = self.base_points[nu_mask]

        # computing base-point matrix
        C    = np.zeros( [len(self.combinations), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                for i_comb2, comb2 in enumerate(self.combinations):
                    C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

        assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

        self.CInv = np.linalg.inv(C)

        # Compute matrix Mkk from non-nominal base_points
        self._VKA = np.zeros( (len(self.masked_base_points), len(self.combinations)) )
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_combination, combination in enumerate(self.combinations):
                res=1
                for var in combination:
                    res*=base_point[parameters.index(var)]

                self._VKA[i_base_point, i_combination ] = res

        self.yields = {}
        if training_data is not None:
            if 'weights' not in training_data[self.nominal_base_point_key]:
                self.yields[self.nominal_base_point_key] = training_data[self.nominal_base_point_key]['features'].shape[0]
            else:
                self.yields[self.nominal_base_point_key] = training_data[self.nominal_base_point_key]['weights'].sum()

            for k, v in training_data.items():
                if "features" not in v and "weights" not in v:
                    raise RuntimeError( "Key %r has neither features nor weights" %k  )
                if k == self.nominal_base_point_key:
                    if 'features' not in v:
                        raise RuntimeError( "Nominal base point does not have features!" )
                else:
                    if not 'features' in v:
                        # we must have weights
                        self.yields[k] = v['weights'].sum()
                    if (not 'weights' in training_data[self.nominal_base_point_key].keys()) and 'weights' in v:
                        raise RuntimeError( "Found no weights for nominal base point, but for a variation. This is not allowed" )

                    if not 'weights' in v:
                        self.yields[k] = len( v['features'] ) 

            self.DeltaA = np.dot( self.CInv, sum([ self._VKA[i_base_point]*np.log(self.yields[tuple(base_point)]/self.yields[self.nominal_base_point_key]) for i_base_point, base_point in enumerate(self.masked_base_points)])) 

    @staticmethod 
    def sort_comb( comb ):
        return tuple(sorted(comb))

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls( None,  
                    nominal_base_point  = old_instance.nominal_base_point,
                    parameters          = old_instance.parameters,
                    combinations        = old_instance.combinations,
                    base_points         = old_instance.base_points,
                    )
            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

    def nu_A(self, nu):
        return np.array( [ functools.reduce(operator.mul, [nu[self.parameters.index(c)] for c in list(comb)], 1) for comb in self.combinations] )

    def predict( self, nu):
        return np.exp(np.dot( self.nu_A(nu), self.DeltaA ))
