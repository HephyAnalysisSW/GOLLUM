import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

# Tensorflow Multiclassifier
from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_5/v5") 

# Parametric Neural Network for JES uncertaintiy 
from ML.PNN.PNN import PNN 
jes_pnn = PNN.load( "/groups/hephy/cms/robert.schoefbeck/Challenge/models/PNN/lowMT_VBFJet/pnn_quad_jes/v3")

# dSigma(mu,nu)/dSigma(SM)  = (mu*dSigmaS + dSigmaB)/(dSigmaB)*dSigmaTot(nu)/dSigma(SM) = (mu*dSigmaS/dSigmaB + 1) * dSigmaTot(nu)/dSigma(SM)
# = (mu*tfmc[:,0]/tfmc[:,1::3].sum(axis=1) + 1)*jes_pnn.predict(data, nu=nu_jec)

def dSigmaOverDSigmaSM( features, mu=1, nu_jec=0 ):
    p_tfmc = tfmc.predict( features )
    p_jes_pnn = jes_pnn.predict( features, nu=( nu_jec, ) )
    return (mu*p_tfmc[:,0]/(p_tfmc[:,1:].sum(axis=1)) + 1)*p_jes_pnn

import common.datasets_hephy as datasets_hephy

data_loader = datasets_hephy.get_data_loader( selection="lowMT_VBFJet", n_split=100)

max_n_batch = 1

for i_batch, batch in enumerate(data_loader):

    features, _, _ = data_loader.split(batch)

    # an example
    dSoDS = dSigmaOverDSigmaSM( features, mu=1, nu_jec = 1 )
    print( dSoDS )
    
    if i_batch>=max_n_batch:
        break

