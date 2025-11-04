import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# Put the elements of your selections here, i.e., cuts you want to apply
selections = {
        "inclusive"   : lambda data: data,
#        "lowMT"       : lambda data: data[      data[:,MT_ind]<70  ],
    }

# Define all the selections here
update={}
for s in [
## The following are the current analysis selections:
#    "xyz",
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

globals().update(selections)
