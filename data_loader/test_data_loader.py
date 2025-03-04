import numpy as np
from data_loader import create_dataloader

# Example Usage
file_path_single = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/htautau_nominal.h5"
file_path_multiple = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/nominal.h5"

single_dataset_name = 'htautau'  # Modify as needed
datasets_multiple = ['data', 'labels', 'detailed_labels', 'weights']
batch_size = 1024

# Create DataLoaders for single dataset
train_loader_single, test_loader_single = create_dataloader(file_path_single, None, batch_size, load_mode='single', single_dataset_name=single_dataset_name)

# Create DataLoaders for multiple datasets
train_loader_multiple, test_loader_multiple = create_dataloader(file_path_multiple, datasets_multiple, batch_size, load_mode='multiple')

# Create lightweight DataLoaders for single dataset
light_train_loader_single, light_test_loader_single = create_dataloader(file_path_single, None, batch_size, load_mode='single', single_dataset_name=single_dataset_name, light=True)

# Create lightweight DataLoaders for multiple datasets
light_train_loader_multiple, light_test_loader_multiple = create_dataloader(file_path_multiple, datasets_multiple, batch_size, load_mode='multiple', light=True)

if __name__ == "__main__":

    # Iterate through the multiple classes dataset (train)
#    for batch in train_loader_multiple:
#        data = batch['data']
#        labels = batch['labels']
#        detailed_labels = batch['detailed_labels']
#        weights = batch['weights']
#        print(data.shape, labels.shape, weights.shape)
#        break

#    for batch in test_loader_multiple:
#        data = batch['data']
#        labels = batch['labels']
#        detailed_labels = batch['detailed_labels']
#        weights = batch['weights']
#        print(data.shape, labels.shape, weights.shape)
#        break

    for batch in train_loader_single:
        print(batch.shape)  # Should print the shape of the batch (batch_size, 30)
        break  # Only print the first batch

    for batch in test_loader_single:
        print(batch.shape)  # Should print the shape of the batch (batch_size, 30)
        break  # Only print the first batch

    # Iterate through the lightweight single class dataset
    for batch in light_train_loader_single:
        print(batch.shape)  # Should print the shape of the batch (batch_size, 30)
        break  # Only print the first batch

    for batch in light_test_loader_single:
        print(batch.shape)  # Should print the shape of the batch (batch_size, 30)
        break  # Only print the first batch

    # Iterate through the lightweight multiple classes dataset
    for batch in light_train_loader_multiple:
        data = batch['data']
        labels = batch['labels']
        detailed_labels = batch['detailed_labels']
        weights = batch['weights']
        print(data.shape, labels.shape, weights.shape)
        break  # Only print the first batch
