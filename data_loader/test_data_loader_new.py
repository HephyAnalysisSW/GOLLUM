import torch
from data_loader import create_higgs_dataloaders, create_lightweight_dataloaders

path_to_h5_file = "/eos/vbc/group/mlearning/data/PUdata/syst_train_set_0.h5"
batch_size = 32

def main():
    # 创建完整数据集的训练和测试 DataLoaders
    (
        train_htautau_data, test_htautau_data,
        train_ztautau_data, test_ztautau_data,
        train_diboson_data, test_diboson_data,
        train_ttbar_data, test_ttbar_data
    ) = create_higgs_dataloaders(path_to_h5_file, batch_size=batch_size)

    # 创建轻量级（1% 数据）的训练和测试 DataLoaders
    (
        train_htautau_light, test_htautau_light,
        train_ztautau_light, test_ztautau_light,
        train_diboson_light, test_diboson_light,
        train_ttbar_light, test_ttbar_light
    ) = create_lightweight_dataloaders(path_to_h5_file, batch_size=batch_size)

    # 将所有加载器存储在字典中，便于统一迭代
    datasets = {
        'htautau': (train_htautau_data, test_htautau_data),
        'ztautau': (train_ztautau_data, test_ztautau_data),
        'diboson': (train_diboson_data, test_diboson_data),
        'ttbar': (train_ttbar_data, test_ttbar_data)
    }

    light_datasets = {
        'htautau_light': (train_htautau_light, test_htautau_light),
        'ztautau_light': (train_ztautau_light, test_ztautau_light),
        'diboson_light': (train_diboson_light, test_diboson_light),
        'ttbar_light': (train_ttbar_light, test_ttbar_light)
    }

    # 遍历完整数据集并输出信息
    print("Processing Full Datasets:")
    for label, (train_loader, test_loader) in datasets.items():
        print(f"\n{label} dataset:")
        for i, batch in enumerate(train_loader):
            print(f"Train Batch {i+1}:")
            print("Data:", batch)
            if i == 1:
                break

        for i, batch in enumerate(test_loader):
            print(f"Test Batch {i+1}:")
            print("Data:", batch)
            if i == 1:
                break

    # 遍历轻量级数据集并输出信息
    print("\nProcessing Lightweight Datasets (1% of the data):")
    for label, (train_loader, test_loader) in light_datasets.items():
        print(f"\n{label} dataset:")
        for i, batch in enumerate(train_loader):
            print(f"Train Batch {i+1}:")
            print("Data:", batch)
            if i == 1:
                break

        for i, batch in enumerate(test_loader):
            print(f"Test Batch {i+1}:")
            print("Data:", batch)
            if i == 1:
                break

if __name__ == "__main__":
    main()
