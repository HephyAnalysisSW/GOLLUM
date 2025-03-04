import numpy as np
import os
from itertools import product
import logging
import pickle
import pandas as pd
import h5py

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

# DEFAULT_INGESTION_SEED = 31415

def process_final_data(final_data):
    # 指定列的排列顺序
    column_order = [
        'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi',
        'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
        'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_n_jets', 'PRI_jet_all_pt',
        'PRI_met', 'PRI_met_phi', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h',
        'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_had_lep',
        'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_had', 'DER_met_phi_centrality', 'DER_lep_eta_centrality'
    ]

    # 检查final_data是否包含所需列
    if not all(col in final_data.columns for col in column_order):
        missing_cols = [col for col in column_order if col not in final_data.columns]
        raise ValueError(f"Missing columns in final_data: {missing_cols}")

    # 按顺序重新排列列
    ordered_data = final_data[column_order].copy()
    ordered_data.insert(len(column_order), 'Constant_1', 1)

    # 转换为 NumPy 数组并返回
    data_array = ordered_data.to_numpy()
    return {'data': data_array}

def _generate_pseudo_exp_data(data, set_mu=1, tes=1.0, jes=1.0, soft_met=0.0, ttbar_scale=None, diboson_scale=None, bkg_scale=None, seed=31415):
    from systematics import get_bootstrapped_dataset, get_systematics_dataset
    pseudo_exp_data = get_bootstrapped_dataset(
        data,
        mu=set_mu,
        ttbar_scale=ttbar_scale,
        diboson_scale=diboson_scale,
        bkg_scale=bkg_scale,
        seed = seed,
    )
    pseudo_exp_data = {'data': pseudo_exp_data}
    syst_test_set = get_systematics_dataset(
        pseudo_exp_data,
        tes=tes,
        jes=jes,
        soft_met=soft_met,
        dopostprocess=True,
        save_to_hdf5=False,
        seed = seed,
    )
    final_data = syst_test_set['data'].copy()

    # 调用处理函数
    processed_data = process_final_data(final_data)
    return processed_data

def generate_pseudo_experiments(test_settings, full_test_set, initial_seed):
    """
    Args:
        test_settings (dict): The test settings.
        full_test_set (dict): The full dataset to generate pseudo experiments from.
        initial_seed (int): The initial seed for random number generation.
    """

    custom_uncertainties = test_settings["custom_uncertainties"]
    num_pseudo_experiments = test_settings["num_pseudo_experiments"]
    num_of_sets = test_settings["num_of_sets"]

    set_indices = np.arange(0, num_of_sets)
    test_set_indices = np.arange(0, num_pseudo_experiments)

    all_combinations = list(product(set_indices, test_set_indices))
    random_state_initial = np.random.RandomState(initial_seed)
    random_state_initial.shuffle(all_combinations)

    results_dict = {}
    for set_index, test_set_index in all_combinations:

        seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

        set_mu = test_settings["ground_truth_mus"][set_index]

        random_state = np.random.RandomState(seed)

        # 使用用户设置或随机采样生成不确定性因子
        tes = (
            custom_uncertainties["tes"]
            if "tes" in custom_uncertainties and custom_uncertainties["tes"] is not None
            else np.clip(random_state.normal(loc=1.0, scale=0.01), 0.9, 1.1)
        )

        jes = (
            custom_uncertainties["jes"]
            if "jes" in custom_uncertainties and custom_uncertainties["jes"] is not None
            else np.clip(random_state.normal(loc=1.0, scale=0.01), 0.9, 1.1)
        )

        soft_met = (
            custom_uncertainties["soft_met"]
            if "soft_met" in custom_uncertainties and custom_uncertainties["soft_met"] is not None
            else np.clip(random_state.lognormal(mean=0.0, sigma=1.0), 0.0, 5.0)
        )

        ttbar_scale = (
            custom_uncertainties["ttbar_scale"]
            if "ttbar_scale" in custom_uncertainties and custom_uncertainties["ttbar_scale"] is not None
            else np.clip(random_state.normal(loc=1.0, scale=0.02), 0.8, 1.2)
        )

        diboson_scale = (
            custom_uncertainties["diboson_scale"]
            if "diboson_scale" in custom_uncertainties and custom_uncertainties["diboson_scale"] is not None
            else np.clip(random_state.normal(loc=1.0, scale=0.25), 0.0, 2.0)
        )

        bkg_scale = (
            custom_uncertainties["bkg_scale"]
            if "bkg_scale" in custom_uncertainties and custom_uncertainties["bkg_scale"] is not None
            else np.clip(random_state.normal(loc=1.0, scale=0.001), 0.99, 1.01)
        )

        full_data = _generate_pseudo_exp_data(
            full_test_set,
            set_mu=set_mu,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            seed = initial_seed
        )
        full_data = full_data["data"]
        test_data = pd.DataFrame(full_data[:, :28])  # (N, 28)
        test_weights = full_data[:, 28]  # (N,)
        test_set = {
            "data": test_data,
            "weights": test_weights
        }

        pkl_data = {
            "mu": set_mu,
            "tes": tes,
            "jes": jes,
            "soft_met": soft_met,
            "ttbar_scale": ttbar_scale,
            "diboson_scale": diboson_scale,
            "bkg_scale": bkg_scale,
        }

        return test_set, pkl_data
