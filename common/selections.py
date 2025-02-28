import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# It is much faster to NOT user the FeatureSelector. 
import common.data_structure as data_structure

MT_ind  = data_structure.feature_names.index("DER_mass_transverse_met_lep")
LJ_ind  = data_structure.feature_names.index("PRI_jet_leading_pt")
SLJ_ind = data_structure.feature_names.index("PRI_jet_subleading_pt")
ptH_ind = data_structure.feature_names.index("DER_pt_h")


#from ML.TFMC.TFMC import TFMC
## https://schoef.web.cern.ch/schoef/Challenge/TFMC/lowMT_noVBFJet_ptH0to100/tfmc_2_reg/predicted_probabilities/predicted_class_htautau.png
#tfmc = TFMC.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/lowMT_noVBFJet_ptH0to100/tfmc_2_reg/v6")

# Put the elements of your selections here, i.e., cuts you want to apply
selections = {
        "inclusive"   : lambda data: data,
        "lowMT"       : lambda data: data[      data[:,MT_ind]<70  ],
        "highMT"      : lambda data: data[      data[:,MT_ind]>=70 ],
        "VBFJet"      : lambda data: data[     (data[:,LJ_ind]>50) & (data[:,SLJ_ind]>30)],
        "noVBFJet"    : lambda data: data[   ~((data[:,LJ_ind]>50) & (data[:,SLJ_ind]>30))],
        "ptH0to50"    : lambda data: data[      data[:,ptH_ind]<50],
        "ptH0to100"   : lambda data: data[      data[:,ptH_ind]<100],
        "ptH50to100"  : lambda data: data[(50 <=data[:,ptH_ind]) & (data[:,ptH_ind]<100)],
        "ptH100to200" : lambda data: data[(100<=data[:,ptH_ind]) & (data[:,ptH_ind]<200)],
        "ptH100"      : lambda data: data[ 100<=data[:,ptH_ind]],
        "ptH200"      : lambda data: data[ 200<=data[:,ptH_ind]],
#        "GGHMVA"      : lambda data: data[ tfmc.predict(data[:,:28], ic_scaling=False)[:, 0]>0.5 ],
    }

# Define all the selections here
update={}
for s in [
# The following are the current analysis selections:
    "lowMT_VBFJet",
    "highMT_VBFJet",          
    "highMT_noVBFJet",          
    "lowMT_noVBFJet_ptH0to100",
    "lowMT_noVBFJet_ptH100", 
    "highMT_noVBFJet_ptH0to100",
    "highMT_noVBFJet_ptH100",
    "highMT",          
#    "GGHMVA_lowMT_noVBFJet_ptH0to100",
# other selections:
    #"lowMT_noVBFJet_ptH0to50",
    #"lowMT_noVBFJet_ptH50to100",
    "lowMT_noVBFJet_ptH100to200", 
    "lowMT_noVBFJet_ptH200",
    #"highMT_noVBFJet_ptH0to50",
    #"highMT_noVBFJet_ptH50to100",
    "highMT_noVBFJet_ptH100to200",
    "highMT_noVBFJet_ptH200",
    "noVBFJet_ptH0to100",
    "noVBFJet_ptH50to100",
    "noVBFJet_ptH100to200",
    "noVBFJet_ptH100to200",
    "noVBFJet_ptH100",
    "noVBFJet_ptH200",
    ]:
    c_fs = [selections[k] for k in s.split('_')]
    def selector(data, c_fs=c_fs):
        for c_f in c_fs:
            data = c_f(data)
        return data
    update[s] = selector

selections.update(update)

all_selections = sorted(list(selections.keys()))

def print_all():
    print("All selections: "+", ".join(all_selections)) 

# from selections import lowMT_VBFJet
globals().update(selections)
