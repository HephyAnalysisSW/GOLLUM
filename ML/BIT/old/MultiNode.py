#!/usr/bin/env python

import numpy as np
import operator 
from math import sqrt
import itertools
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')

import functools

default_cfg = {
    "max_depth":        4,
    "min_size" :        50,
    "max_n_split":      -1,     # similar to TMVA: test only max_n_split values in the node-split search. (Not thoroughly tested yet and usually not needed.)
    "base_points":      None,
    "feature_names":    None,
    "positive":         False,  # only perform node split when the resulting yields are positive definite. I do not recommend this, the bias-variance tradeoff is not favourable.
    "min_node_size_neg_adjust": False, # Increase the min_size parameter by (1+f)/(1-f) where f is n-/n+ where n- (n+) are the number of events with negative (positive) weight 
    "loss" : "MSE", # or "CrossEntropy" # MSE for toys is fine, in real life CrossEntropy is a bit more stable against outliers
    "_get_only_score":False, 
}

class MultiNode:
    def __init__( self, features, training_weights, _depth=0, **kwargs):

        ## basic BDT configuration + kwargs
        self.cfg = default_cfg
        self.cfg.update( kwargs )
        for attr, val in self.cfg.items():
            setattr( self, attr, val )

        # data set
        self.features           = features
        self.size               = len(self.features)

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

        # keep track of recursion depth
        self._depth             = _depth

        self.split(_depth=_depth)
        self.prune()

        # Let's not leak the dataset.
        del self.training_weights
        del self.features 
        del self.split_left_group 

    def get_split_vectorized( self ):
        ''' determine where to split the features, first vectorized version of FI maximization
        '''

        # loop over the features ... assume the features consists of rows with [x1, x2, ...., xN]
        self.split_i_feature, self.split_value, self.split_gain, self.split_left_group = 0, -float('inf'), 0, None

        # for a valid binary split, we need at least twice the mean size
        assert self.size >= 2*self.min_size

        # loop over features
        #print "len(self.features[0]))",len(self.features[0])

        for i_feature in range(len(self.features[0])):
            feature_values = self.features[:,i_feature]

            feature_sorted_indices = np.argsort(feature_values)
            sorted_weight_sums     = np.cumsum(self.training_weights[feature_sorted_indices],axis=0) # FIXME cumsum does not implement max_n_split -> inefficient?
 
            # respect min size for split
            if self.max_n_split<2:
                plateau_and_split_range_mask = np.ones(self.size-1, dtype=np.dtype('bool'))
            else:
                min_, max_ = min(feature_values), max(feature_values) 
                #print "_depth",self._depth, "len(feature_values)",len(feature_values), "min_, max_", min_, max_
                plateau_and_split_range_mask  = np.digitize(feature_values[feature_sorted_indices], np.arange (min_, max_, (max_-min_)/(self.max_n_split+1)))
                #print len(plateau_and_split_range_mask), plateau_and_split_range_mask
                plateau_and_split_range_mask = plateau_and_split_range_mask[1:]-plateau_and_split_range_mask[:-1]
                plateau_and_split_range_mask = np.insert( plateau_and_split_range_mask, 0, 0).astype('bool')[:-1]
                #print "plateau_and_split_range_mask", plateau_and_split_range_mask
                #print "CUTS", feature_values[feature_sorted_indices][:-1][plateau_and_split_range_mask] 

            if self.min_size > 1:
                plateau_and_split_range_mask[0:self.min_size-1] = False
                plateau_and_split_range_mask[-self.min_size+1:] = False
            plateau_and_split_range_mask &= (np.diff(feature_values[feature_sorted_indices]) != 0)

            total_weight_sum         = sorted_weight_sums[-1]
            sorted_weight_sums       = sorted_weight_sums[0:-1]
            sorted_weight_sums_right = total_weight_sum-sorted_weight_sums

            # mask negative definite splits
            if self.cfg['positive']:
                pos       = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums,self.base_point_const_for_pos.transpose())>=0)
                pos_right = np.apply_along_axis(all, 1, np.dot(sorted_weight_sums_right,self.base_point_const_for_pos.transpose())>=0)

                all_pos = np.concatenate((pos, pos_right))
                #if not np.all(all_pos):
                #    print ("Warning! Found negative node splits {:.2%}".format(1-float(np.count_nonzero(all_pos))/len(all_pos)) )

                plateau_and_split_range_mask &= pos
                plateau_and_split_range_mask &= pos_right

            plateau_and_split_range_mask = plateau_and_split_range_mask.astype(int)

            # Never allow negative yields
            plateau_and_split_range_mask &= (sorted_weight_sums[:,0]>0)
            plateau_and_split_range_mask &= (sorted_weight_sums_right[:,0]>0)

            if self.cfg['loss'] == 'MSE':
                neg_loss_gains = np.sum(np.dot( sorted_weight_sums, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums[:,0]
                neg_loss_gains+= np.sum(np.dot( sorted_weight_sums_right, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums_right[:,0]
            elif self.cfg['loss'] == 'CrossEntropy':
                with np.errstate(divide='ignore', invalid='ignore'):
                    r       = np.dot( sorted_weight_sums, self.base_point_const.transpose())/sorted_weight_sums[:,0].reshape(-1,1)
                    r_right = np.dot( sorted_weight_sums_right, self.base_point_const.transpose())/sorted_weight_sums_right[:,0].reshape(-1,1)
                    #neg_loss_gains  = sorted_weight_sums[:,0]*np.sum( ( r*0.5*np.log(r**2) + (1.-r)*0.5*np.log((1.-r)**2) ), axis=1)
                    #neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( r_right*0.5*np.log(r_right**2) + (1.-r_right)*0.5*np.log((1.-r_right)**2) ), axis=1)
                    neg_loss_gains  = sorted_weight_sums[:,0]*      np.sum( ( 0.5*np.log((1./(1.+r))**2) + r*0.5*np.log((r/(1.+r))**2) ), axis=1)
                    neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( 0.5*np.log((1./(1.+r_right))**2) + r_right*0.5*np.log((r_right/(1.+r_right))**2) ), axis=1)
                    neg_loss_gains -= min(neg_loss_gains)# make loss positive

            with np.errstate(divide='ignore', invalid='ignore'):
                if self.cfg['min_node_size_neg_adjust']:
                    # Defining the negative weight fraction as f=n^+/n^- the relative statistical MC uncertainty in a yield is 1/sqrt(n) sqrt(1+f)/sqrt(1-f). 
                    # The min node size should therefore be increased to (1+f)/(1-f)
                    sorted_pos_sums       = np.cumsum( (self.training_weights[:,0]>0).astype('int')[feature_sorted_indices])
                    total_pos_sum         = sorted_pos_sums[-1]
                    sorted_pos_sums       = sorted_pos_sums[0:-1]
                    sorted_pos_sums_right = total_pos_sum-sorted_pos_sums
                    sorted_neg_sum       = np.array(range(1, len(self.training_weights[:,0])))     - sorted_pos_sums
                    sorted_neg_sum_right = np.array(range(len(self.training_weights[:,0])-1,0,-1)) - sorted_pos_sums_right
                    f      = sorted_neg_sum/sorted_pos_sums.astype(float)
                    f_right= sorted_neg_sum_right/sorted_pos_sums_right.astype(float)

                    plateau_and_split_range_mask &= range(1, len(self.training_weights[:,0]))>(1+f)/(1-f)*self.min_size
                    plateau_and_split_range_mask &= np.array(range(len(self.training_weights[:,0])-1,0,-1)) >(1+f_right)/(1-f_right)*self.min_size

            gain_masked = np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask
            argmax_fi   = np.argmax(gain_masked)
            gain        = gain_masked[argmax_fi]

            #argmax_fi   = np.argmax(np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask)
            #gain        = neg_loss_gains[argmax_fi]
            
            value = feature_values[feature_sorted_indices[argmax_fi]]

            debug_self_split_gain = self.split_gain
            if gain > self.split_gain: 
                self.split_i_feature = i_feature
                self.split_value     = value
                self.split_gain      = gain

            if np.count_nonzero(self.features[:,self.split_i_feature]<=self.split_value) == 1: #self.split_value <= 3.26e-05:
                print ("sorted_weight_sums[:,0]      ", sorted_weight_sums[:,0])
                print ("sorted_weight_sums_right[:,0]", sorted_weight_sums_right[:,0])
                print ("plateau_and_split_range_mask", plateau_and_split_range_mask)
                print ("neg_loss_gains left", np.sum(np.dot( sorted_weight_sums, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums[:,0] )
                print ("neg_loss_gains right", np.sum(np.dot( sorted_weight_sums_right, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums_right[:,0])
                print ("neg_loss_gains", neg_loss_gains)
                print ("np.nan_to_num(neg_loss_gains)", np.nan_to_num(neg_loss_gains))
                print ("np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask", np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask)
                print ("argmax_fi", np.argmax(np.nan_to_num(neg_loss_gains)*plateau_and_split_range_mask) )
                print ("gain", gain )
                print ("found split?", gain > debug_self_split_gain, "gain", gain, "self.split_gain (before)",debug_self_split_gain, "self.split_gain(after)",self.split_gain)
                print ("self.split_left_group", self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool'))
                print ("non_zero", np.count_nonzero(self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')))
                print ()
                assert False, "single-entry node!!"

        assert not np.isnan(self.split_value)

        self.split_left_group = self.features[:,self.split_i_feature]<=self.split_value if not  np.isnan(self.split_value) else np.ones(self.size, dtype='bool')

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
        #tic = time.time()
        if self.cfg["_get_only_score"]:
            # This is the first time we see the data. We store just store the derivatives in the left box and do not split further 
            self.split_i_feature, self.split_value, self.split_left_group = 0, +float('inf'), None
            self.left        = ResultNode(derivatives=self.derivatives, **self.__store(np.ones(self.size,dtype=bool)))
            self.right       = ResultNode(derivatives=self.derivatives, **self.__store(np.zeros(self.size,dtype=bool)))
            return

        self.get_split_vectorized()

        # check for max depth or a 'no' split
        if  self.max_depth <= _depth+1 or (not any(self.split_left_group)) or all(self.split_left_group): # Jason Brownlee starts counting depth at 1, we start counting at 0, hence the +1
            #print ("Choice2", _depth, result_func(self.split_left_group), result_func(~self.split_left_group) )
            # The split was good, but we stop splitting further. Put everything in the left node! 
            self.split_value = float('inf')
            self.left        = ResultNode(derivatives=self.derivatives, **self.__store(np.ones(self.size,dtype=bool)))
            self.right       = ResultNode(derivatives=self.derivatives, **self.__store(np.zeros(self.size,dtype=bool)))
            return

        # process left child
        if np.count_nonzero(self.split_left_group) < 2*self.min_size:
            #print ("Choice3", _depth, result_func(self.split_left_group) )
            # Too few events in the left box. We stop.
            self.left             = ResultNode(derivatives=self.derivatives, **self.__store(self.split_left_group) )
        else:
            #print ("Choice4", _depth )
            # Continue splitting left box.
            self.left             = MultiNode(self.features[self.split_left_group], training_weights = self.training_weights[self.split_left_group], _depth=self._depth+1, **self.cfg)
        # process right child
        if np.count_nonzero(~self.split_left_group) < 2*self.min_size:
            #print ("Choice5", _depth, result_func(~self.split_left_group) )
            # Too few events in the right box. We stop.
            self.right            = ResultNode(derivatives=self.derivatives, **self.__store(~self.split_left_group) )
        else:
            #print ("Choice6", _depth  )
            # Continue splitting right box. 
            self.right            = MultiNode( self.features[~self.split_left_group], training_weights = self.training_weights[~self.split_left_group], _depth=self._depth+1, **self.cfg)

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
        """Create numpy logical expressions from all paths to results nodes, associate with prediction defined by key, and return predictions for given feature matrix
           Should be faster for shallow trees due to numpy being implemented in C, despite going over feature vectors multiple times."""

        emmitted_expressions_with_predictions = []

        def emit_expressions_with_predictions(node, logical_expression):
            if isinstance(node, ResultNode):
                emmitted_expressions_with_predictions.append((logical_expression, node.coefficient_sum))
            else:
                if node == self:
                    prepend = ""
                else:
                    prepend = " & "
                if np.isinf(node.split_value):
                    split_value_str = 'np.inf'
                else:
                    split_value_str = format(node.split_value, '.32f')
                emit_expressions_with_predictions(node.left, logical_expression + "%s(feature_matrix[:,%d] <= %s)" % (prepend, node.split_i_feature, split_value_str))
                emit_expressions_with_predictions(node.right, logical_expression + "%s(feature_matrix[:,%d] > %s)" % (prepend, node.split_i_feature, split_value_str))

        emit_expressions_with_predictions(self, "")
        predictions = np.zeros((len(feature_matrix), len(self.derivatives)))

        for expression, prediction in emmitted_expressions_with_predictions:
            predictions[eval(expression)] = prediction

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
        #f_poly_str = " f={:3.0%} ".format(self.f)

        #print_string = '%s(%5i) r = %s   c = %s' % ((_depth)*' ', self.size, r_poly_str, c_poly_str)
        try:
            unc = 1./sqrt(self.size)*sqrt((1+self.f)/(1-self.f))
        except ZeroDivisionError:
            unc = 0 
        print_string = '%s(%6i, unc=%1.3f) r = %s   c = %s' % ((_depth)*' ', self.size, 1./sqrt(self.size)*sqrt((1+self.f)/(1-self.f)), r_poly_str, c_poly_str)
        print(print_string)

    def get_list(self):
        ''' recursively obtain all thresholds (bottom of recursion)'''
        return self.coefficient_sum 
