import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# It is much faster to NOT user the FeatureSelector. 
import common.common as common

MT_ind  = common.feature_names.index("DER_mass_transverse_met_lep")
LJ_ind  = common.feature_names.index("PRI_jet_leading_pt")
SLJ_ind = common.feature_names.index("PRI_jet_subleading_pt")
ptH_ind = common.feature_names.index("DER_pt_h")

# Put the elements of your selections here, i.e., cuts you want to apply
selections = {
        "inclusive"   : lambda data: data,
        "lowMT"       : lambda data: data[data[:,MT_ind]<70],
        "highMT"      : lambda data: data[data[:,MT_ind]>=70],
        "VBFJet"      : lambda data: data[(data[:,LJ_ind]>50) & (data[:,SLJ_ind]>30)],
        "noVBFJet"    : lambda data: data[~( (data[:,LJ_ind]>50) & (data[:,SLJ_ind]>30))],
        "ptH0to50"    : lambda data: data[ data[:,ptH_ind]<50],
        "ptH50to100"  : lambda data: data[ (50<=data[:,ptH_ind]) & (data[:,ptH_ind]<100)],
        "ptH100to200" : lambda data: data[ (100<=data[:,ptH_ind])&(data[:,ptH_ind]<200)],
        "ptH100"      : lambda data: data[ (100<=data[:,ptH_ind])],
        "ptH200"      : lambda data: data[ 200<=data[:,ptH_ind]],
    }

# Define all the selections here
update={}
for s in [
    "lowMT_VBFJet",
    "highMT_VBFJet",          
    "lowMT_noVBFJet_ptH0to50",
    "lowMT_noVBFJet_ptH50to100",
    "lowMT_noVBFJet_ptH100to200", 
    "lowMT_noVBFJet_ptH100", 
    "lowMT_noVBFJet_ptH200",
    "highMT_noVBFJet_ptH0to50",
    "highMT_noVBFJet_ptH50to100",
    "highMT_noVBFJet_ptH100to200",
    "highMT_noVBFJet_ptH100",
    "highMT_noVBFJet_ptH200",
    ]:
    c_fs = [selections[k] for k in s.split('_')]
    def selector(data, c_fs=c_fs):
        for c_f in c_fs:
            data = c_f(data)
        return data
    update[s] = selector

selections.update(update)

# This allows to do
# from selections import lowMT_VBFJet
globals().update(selections)
