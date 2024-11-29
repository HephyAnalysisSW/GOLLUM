import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from data_loader.data_loader_2 import H5DataLoader
import common.user as user

batch_size = None #64**2
n_split    = 10

# Initialize the data loader
def get_data_loader( name = "nominal", n_split=10, selection_function=None, data_directory = user.data_directory):
    return H5DataLoader(
        os.path.join( data_directory, name+'.h5'), 
        datasets, 
        batch_size         = batch_size,
        n_split            = n_split, 
        selection_function = selection_function )
   
if __name__=="__main__":
    # Iterate through the dataset
    for batch in get_data_loader(n_split=100):
        data = batch['data']
        weights = batch['weights']
        labels = batch['detailed_labels']
        print(data.shape, weights.shape, labels.shape)

        break
