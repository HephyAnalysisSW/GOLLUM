import numpy as np
import random

from math import pi

''' Analytic toy model from paper
'''

base_point_index = {
    0 : (0.,0.),
    1 : (1.,0.),
    2 : (0,1.),
    #4 : (0.8,),
    #3 : (1.3,),
}
base_point_index.update ({val:key for key, val in base_point_index.items()})

base_points        = [ base_point_index[i] for i in [0,1,2] ] 
parameters         = ['nu1','nu2']
combinations       = [ ('nu1',), ('nu2',)]
tex                = {"nu1":"#nu_{1}", "nu2":"#nu_{2}"}
nominal_base_point = base_point_index[0]

default_parameters = {  }
default_parameters.update( {var:0. for var in parameters} )

def make_parameters(**kwargs):
    result = { key:val for key, val in default_parameters.items() }
    for key, val in kwargs.items():
        if not key in parameters:
            raise RuntimeError ("Parameter not known.")
        else:
            result[key] = float(val)
    return result

random_parameters = make_parameters(**{v:random.random() for v in parameters} )
sm         = make_parameters()

feature_names =  ['x']

#from scipy.stats.sampling import NumericalInversePolynomial
#
#class PDF:
#    def __init__( self, parameters):
#        self.parameters = parameters
#        self.rng = NumericalInversePolynomial(self, domain=(-pi, pi), random_state=np.random.default_rng())
#
#    # for external use
#    @staticmethod
#    def generic_pdf( x, parameters):
#        return np.exp( .25*(parameters[0]*np.sin(x)+ parameters[1]*np.cos(.5*x))**2 ) 
#
#    def pdf(self, x: float) -> float:
#        # note that the normalization constant isn't required
#        return self.generic_pdf( x, self.parameters )
#
#    def getFeatures( self, N_events_requested ):
#        return np.array( self.rng.rvs(N_events_requested) ).reshape( -1, 1)
#
#nominal_PDF = PDF(nominal_base_point)

def getEvents( N_events_requested, weighted=True, systematic=None):

    if weighted:
        x  = np.array( [ [ (random.random()-0.5)] for _ in range(N_events_requested)] )
        res = {tuple(bp):{'weights':np.ones(N_events_requested)*np.exp((bp[0]+2*bp[1]))} for bp in base_points}
        res[nominal_base_point]['features'] = x
        #for key, val in res.items():
        #    val['weights']/=(val['weights'].sum())
        #    val['weights']*=100
    else:
        raise NotImplementedError
    #    res = {tuple(bp):{'features': PDF(bp).getFeatures( N_events_requested )} for bp in base_points}

    return res

plot_options = {
    'x': {'binning':[20,-0.5,0.5],      'tex':"x",},
    }

bpt_cfg = {
    "n_trees" : 100,
    "learning_rate" : 0.2, 
    "loss" : "CrossEntropy", 
    "learn_global_param": False,
    "min_size": 50,
}
bpn_cfg = {
    "learning_rate" : 0.01,
    "n_epochs": 1000, 
}
