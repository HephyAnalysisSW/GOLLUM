import ROOT
import numpy as np

import ZH_Nakamura as process

# translate weights to matrix
def getEvents( nTraining ):
    features, derivatives = process.getEents(nTraining)
    # We get the derivatives. Therefore, the coefficient of the quadratic term is 0.5 of the derivative because of the 1/2 in the Taylor expansion
    weights = np.column_stack([(0.5 if len(der)==2 else 1)*derivatives[der] for der in process.derivatives]) 
    return features, weights 


