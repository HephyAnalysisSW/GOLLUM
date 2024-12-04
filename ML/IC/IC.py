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

import common.data_structure as data_structure

class InclusiveCrosssection:
    def __init__( self ):
        self.selection = None

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as file_:
            old_instance = pickle.load(file_)
            new_instance = cls()
            new_instance.weight_sums = old_instance.weight_sums 
            new_instance.selection   = old_instance.selection if hasattr(old_instance, "selection") else None
            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

    def load_training_data( self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader( selection=selection, selection_function=None, n_split=n_split)
        self.selection   = selection

    def train(self, datasets, selection, small=True):
        from collections import defaultdict
        from tqdm import tqdm
        weight_sums = defaultdict(float)  # Use defaultdict for easy updates
        for batch in tqdm(self.data_loader, desc="Computing weight sum", unit="batch"):
            weights = batch[:, -2]  # One-but-last column
            classes = batch[:, -1]  # Last column
            for cls, weight in zip(classes, weights):
                weight_sums[cls] += weight
            if small: break

        self.weight_sums = {int(k):v for k,v in dict(weight_sums).items()}

    def __str__( self ):
        prefix = ("IC: "+'\033[1m'+self.selection+'\033[0m'+" X-sec: ") if hasattr(self, "selection") and self.selection is not None else "X-Sec: "
        return ( prefix+" ".join([l+": "+"%8.2f"%self.weight_sums[data_structure.label_encoding[l]] for l in data_structure.labels ]))

    def predict( self, sample):
        if type(sample)==str:
            return self.weight_sums[data_structure.label_encoding[sample]]
        elif sample in self.weight_sums:
            return self.weight_sums[sample]
        else:
            raise RuntimeError("IC: Don't know what to do with sample %r"%sample)
