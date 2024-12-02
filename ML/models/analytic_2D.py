import numpy as np
import random

from math import pi

''' Analytic toy model from paper
'''

base_point_index = {
    0 : (0., 0.),
    1 : (.5, 0.),
    2 : (0., .5),
    3 : (1., 0.),
    4 : (0., 1.),
    5 : (.5, .5),
    #6 : (.45, .45),
    #7 : (.55, .55),
}
base_point_index.update ({val:key for key, val in base_point_index.items()})

base_points        = [ base_point_index[i] for i in [0,1,2,3,4,5,] ] 
parameters         = ['nu1', 'nu2']
combinations       = [('nu1',),  ('nu1', 'nu1'), ('nu2',), ('nu2', 'nu2'), ('nu1', 'nu2')] 
tex                = {"nu1":"#nu_{1}", "nu2":"#nu_{2}"}
nominal_base_point = base_point_index[0]

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

feature_names =  ['x']

from scipy.stats.sampling import NumericalInversePolynomial

class PDF:
    def __init__( self, parameters):
        self.parameters = parameters
        self.rng = NumericalInversePolynomial(self, domain=(-pi, pi), random_state=np.random.default_rng())

    # for external use
    @staticmethod
    def generic_pdf( x, parameters):
        return np.exp( .25*(parameters[0]*np.sin(x)+ parameters[1]*np.cos(.5*x))**2 ) #THIS 
        #return np.exp( .25*(parameters[0]*np.sin(x)+ parameters[1]*np.cos(.5*x)) ) 

    def pdf(self, x: float) -> float:
        # note that the normalization constant isn't required
        return self.generic_pdf( x, self.parameters )

    def getFeatures( self, N_events_requested ):
        return np.array( self.rng.rvs(N_events_requested) ).reshape( -1, 1)

nominal_PDF = PDF(nominal_base_point)

def getEvents( N_events_requested, weighted=True, systematic=None):

    if weighted:
        x  = nominal_PDF.getFeatures( N_events_requested )
        res = {tuple(bp):{'weights':nominal_PDF.generic_pdf( x[:,0], bp)} for bp in base_points}
        res[nominal_base_point]['features'] = x
        for key, val in res.items():
            val['weights']/=(val['weights'].sum())
            val['weights']*=100
    else:
        res = {tuple(bp):{'features': PDF(bp).getFeatures( N_events_requested )} for bp in base_points}

    return res

plot_options = {
    'x': {'binning':[20,-pi,pi],      'tex':"x",},
    }

bpt_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "CrossEntropy", 
    "learn_global_param": False,
    "min_size": 50,
}
pnn_cfg = {
    "learning_rate" : 0.01,
    "n_epochs":1000, 
}
