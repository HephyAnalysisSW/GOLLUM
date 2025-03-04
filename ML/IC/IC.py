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
            print("filename", filename, old_instance.__dict__)
            new_instance.weight_sums = old_instance.weight_sums 
            new_instance.selection   = old_instance.selection if hasattr(old_instance, "selection") else None
            new_instance.unweighted_sums = old_instance.unweighted_sums if hasattr(old_instance, "unweighted_sums") else None
            return new_instance  

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename,'wb') as file_:
            pickle.dump( self, file_ )

    def load_training_data( self, datasets_hephy, selection, n_split=10):
        self.data_loader = datasets_hephy.get_data_loader( selection=selection, selection_function=None, n_split=n_split)
        self.selection   = selection

    def train(self, datasets_hephy, selection=None, small=True):
        from collections import defaultdict
        from tqdm import tqdm
        import numpy as np

        weight_sums = defaultdict(float)  # Sum of weights for each class
        unweighted_sums = defaultdict(float)  # Count of elements (unweighted) for each class

        for batch in tqdm(self.data_loader, desc="Computing weight sums", unit="batch"):
    
            if selection is not None:
                batch_ = selection(batch)
            else:
                batch_ = batch
 
            weights = batch_[:, -2]  # One-but-last column
            classes = batch_[:, -1]  # Last column

            # Use NumPy to compute weighted and unweighted sums
            unique_classes, indices = np.unique(classes, return_inverse=True)
            summed_weights = np.bincount(indices, weights=weights)
            unweighted_counts = np.bincount(indices)

            # Update the dictionaries
            for cls, weight, count in zip(unique_classes, summed_weights, unweighted_counts):
                weight_sums[cls] += weight
                unweighted_sums[cls] += count

            if small:
                break

        # Convert defaultdicts to regular dictionaries with int keys
        self.weight_sums = {int(k): v for k, v in dict(weight_sums).items()}
        self.unweighted_sums = {int(k): v for k, v in dict(unweighted_sums).items()}

    def __str__( self ):
        prefix = ("IC: "+'\033[1m'+self.selection+'\033[0m') if hasattr(self, "selection") and self.selection is not None else "X-Sec: "
        S = self.weight_sums[data_structure.label_encoding['htautau']]
        B = sum( [self.weight_sums[data_structure.label_encoding[l]] for l in data_structure.labels if l!='htautau' ])
        SoverB = " S/B = %8.6f "%(S/B)
        if hasattr( self, "unweighted_sums") and self.unweighted_sums is not None:
            unweighed_str = "\n"+" ".ljust(59)+"count: "+" ".join([l+": "+"%8i"%self.unweighted_sums[data_structure.label_encoding[l]] for l in data_structure.labels ])
        else:
            unweighed_str = ""
        return ( prefix.ljust(50)+SoverB+" yield: "+" ".join([l+": "+"%8.2f"%self.weight_sums[data_structure.label_encoding[l]] for l in data_structure.labels ])+unweighed_str)

    def predict( self, sample):
        if type(sample)==str:
            return self.weight_sums[data_structure.label_encoding[sample]]
        elif sample in self.weight_sums:
            return self.weight_sums[sample]
        else:
            raise RuntimeError("IC: Don't know what to do with sample %r"%sample)
