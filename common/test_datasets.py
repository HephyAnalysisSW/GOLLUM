import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import glob

if __name__=="__main__":
    from common.logger import get_logger
    logger = get_logger("INFO", logFile = None)
else:
    import logging
    logger = logging.getLogger("UNC")

from data_loader.data_loader_2 import H5DataLoader
import common.user as user

import selections
import Dataset
import common.data_structure as data_structure

subdirectories = glob.glob(os.path.join(user.derived_test_data_directory, "*/"))
data = {}
for subdir in subdirectories:
    selection = os.path.basename(subdir.rstrip('/'))
    if selection not in data:
        data[selection]={}
    for filename in glob.glob(os.path.join(os.path.normpath(subdir), "*.h5")):
        process, values =  Dataset.parse_filename(os.path.basename(filename))
        data[selection][(process, values)] = filename
        #data[selection][filename] = process

def print_all(specific_selection=None, verbose=False): 
    logger.info("All data sets I have:")
    for selection in selections.all_selections:
        if selection not in data: continue
        if specific_selection is not None and specific_selection!=selection: continue
        if verbose:
            logger.info( "selection: "+'\033[1m'+selection+'\033[0m')
            for (s, v), f in data[selection].items():
                sstr = "  "+(s if s is not None else "combined")+" "+", ".join( [ data_structure.systematics[i_v]+"="+str(v) for i_v, v in enumerate( v )])
                logger.info(sstr.ljust(50), f)
        else:
            len_=len(data[selection])
            logger.info( "selection: "+'\033[1m'+selection+'\033[0m'+f" {len_} dataset(s) found")
    #print() 

print_all()

## Initialize the data loader
def get_data_loader( selection="inclusive", process=None, values=data_structure.default_values, n_split=10, batch_size=None, selection_function=None):
    if selection not in selections.all_selections:
        logger.info(f"I know nothing about selection {selection}")
        selections.print_all()
    if (process, values) not in data[selection]:
        logger.warning("I don't have the file for this choice: process: %s values:%r"%((process if process is not None else "combined"), values))
        print_all(specific_selection=selection, verbose=True)
    return H5DataLoader(
        file_path          = data[selection][(process, values)], 
        batch_size         = batch_size,
        n_split            = n_split,
        selection_function = selection_function,
    ) 
   
if __name__=="__main__":
    import numpy as np
    # Iterate through the dataset
    loader = get_data_loader(selection="lowMT_VBFJet", n_split=1)
    for batch in loader:
        data, weights, labels = H5DataLoader.split(batch)
        print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )
