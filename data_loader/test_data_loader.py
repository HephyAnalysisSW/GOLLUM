import torch
from torch.utils.data import DataLoader
from data_loader import get_higgs_dataloader

path_to_h5_file = "/eos/vbc/group/mlearning/data/PUdata/syst_train_set_test.h5"

batch_size = 32
return_label = True
shuffle = True

def main():

    dataloader = get_higgs_dataloader(
        path_to_file=path_to_h5_file,
        batch_size=batch_size,
        return_label=return_label,
        shuffle=shuffle
    )

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print("Data:", batch['data'])
        print("Weights:", batch['weights'])
        print("Detailed Labels:", batch['detailed_labels'])
        if return_label:
            print("Labels:", batch['label'])
        print("\n")

        if i == 1:
            break

if __name__ == "__main__":
    main()
