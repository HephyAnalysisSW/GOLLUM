import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# Tensorflow Multiclassifier
from ML.TFMC.TFMC import TFMC
tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_VBFJet/tfmc_5/v5")

# Import all the data sets
import common.datasets_hephy as datasets_hephy

data_loader = datasets_hephy.get_data_loader( selection="lowMT_VBFJet", n_split=100 )

max_n_batch = 1

for i_batch, batch in enumerate(data_loader):

    features, _, _ = data_loader.split(batch)

    # an example
    xsec                = tfmc.predict( features ) 
    class_probabilities = tfmc.predict( features, ic_scaling=False) 

    if i_batch>=max_n_batch:
        break
