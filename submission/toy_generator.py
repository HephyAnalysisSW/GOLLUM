import os
import pandas as pd
from pseudo import generate_pseudo_experiments

# 固定的数据路径
DATA_PATHS = {
    "ztautau": "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/test/data/ztautau_data.parquet",
    "diboson": "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/test/data/diboson_data.parquet",
    "ttbar": "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/test/data/ttbar_data.parquet",
    "htautau": "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/test/data/htautau_data.parquet",
}


def run_pseudo_experiments(
    tes,
    jes,
    soft_met,
    ttbar_scale,
    diboson_scale,
    bkg_scale,
    num_pseudo_experiments,
    num_of_sets,
    ground_truth_mus,
    seed_input = None
):
    """
    Simplified interface for generating pseudo experiments.

    Args:
        tes (float): Tau Energy Scale factor.
        jes (float): Jet Energy Scale factor.
        soft_met (float): Soft MET adjustment.
        ttbar_scale (float): TTBar scaling factor.
        diboson_scale (float): Diboson scaling factor.
        bkg_scale (float): Background scaling factor.
        num_pseudo_experiments (int): Number of pseudo experiments per set.
        num_of_sets (int): Number of sets.
        ground_truth_mus (list): Ground truth mu values for each set.
        output_dir (str): Directory to save the output files.

    Returns:
        None
    """
    # 固定数据集读取
    full_test_set = {}
    for key, path in DATA_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        full_test_set[key] = pd.read_parquet(path)

    # 构建测试设置
    test_settings = {
        "custom_uncertainties": {
            "tes": tes,
            "jes": jes,
            "soft_met": soft_met,
            "ttbar_scale": ttbar_scale,
            "diboson_scale": diboson_scale,
            "bkg_scale": bkg_scale,
        },
        "num_pseudo_experiments": num_pseudo_experiments,
        "num_of_sets": num_of_sets,
        "ground_truth_mus": ground_truth_mus,
    }

    # 调用生成函数
    if seed_input is not None:
        test_set, pkl_data = generate_pseudo_experiments(test_settings, full_test_set, initial_seed=seed_input)
    else:
        test_set, pkl_data = generate_pseudo_experiments(test_settings, full_test_set, initial_seed=31415)
    return test_set, pkl_data
