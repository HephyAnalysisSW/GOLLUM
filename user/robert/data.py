import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from data_loader.data_loader_2 import H5DataLoader
import common.user as user

datasets = ['data', 'weights', 'detailed_labels']
batch_size = None #64**2
n_split    = 10

# Initialize the data loader
def get_data_loader( name = "nominal", n_split=10):
    return H5DataLoader(os.path.join( user.data_directory, name+'.h5') , datasets, batch_size=batch_size, n_split=n_split)

if __name__=="__main__":
    # Iterate through the dataset
    for batch in get_data_loader(n_split=1000):
        data = batch['data']
        weights = batch['weights']
        labels = batch['detailed_labels']
        print(data.shape, weights.shape, labels.shape)

        break
