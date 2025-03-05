import numpy as np
import os
from itertools import product
import logging
import pickle
import pandas as pd
import h5py

import logging
logger = logging.getLogger('UNC')

DEFAULT_INGESTION_SEED = 31415

def _generate_pseudo_exp_data(data, set_mu=1, tes=1.0, jes=1.0, soft_met=0.0, ttbar_scale=None, diboson_scale=None, bkg_scale=None):
    from systematics import get_bootstrapped_dataset, get_systematics_dataset

    pesudo_exp_data = get_bootstrapped_dataset(
        data,
        mu=set_mu,
        ttbar_scale=ttbar_scale,
        diboson_scale=diboson_scale,
        bkg_scale=bkg_scale,
    )
    
    pesudo_exp_data = {'data': pesudo_exp_data}

    syst_test_set = get_systematics_dataset(
        pesudo_exp_data,
        tes=tes,
        jes=jes,
        soft_met=soft_met,
        dopostprocess=True,
        save_to_hdf5=False,  
    )

    return syst_test_set

def generate_pseudo_experiments(test_settings, full_test_set,output_dir="./output_pseudo_experiments", initial_seed=DEFAULT_INGESTION_SEED):
    """
    Args:
        test_settings (dict): The test settings.
        full_test_set (dict): The full dataset to generate pseudo experiments from.
        initial_seed (int): The initial seed for random number generation.
    """

    os.makedirs(output_dir, exist_ok=True)
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

        hdf5_filename = os.path.join(output_dir, f"set_{set_mu}_pseudo_exp_{test_set_index}.h5")
        pkl_filename = os.path.join(output_dir, f"set_{set_mu}_pseudo_exp_{test_set_index}.pkl")

        if not os.path.basename(hdf5_filename).endswith(".h5"):
            raise ValueError(f"Invalid HDF5 filename: {hdf5_filename}")
        if not os.path.basename(pkl_filename).endswith(".pkl"):
            raise ValueError(f"Invalid PKL filename: {pkl_filename}")

        test_set = _generate_pseudo_exp_data(
            full_test_set,
            set_mu=set_mu,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
        )

        with h5py.File(hdf5_filename, "w") as hf:
            for key, value in test_set.items():
                if isinstance(value, pd.DataFrame):
                    hf.create_dataset(key, data=value.to_numpy())
                else:
                    hf.create_dataset(key, data=value)
        pkl_data = {
            "mu": set_mu,
            "tes": tes,
            "jes": jes,
            "soft_met": soft_met,
            "ttbar_scale": ttbar_scale,
            "diboson_scale": diboson_scale,
            "bkg_scale": bkg_scale,
        }
        with open(pkl_filename, "wb") as pkl_file:
            pickle.dump(pkl_data, pkl_file)

        results_dict[(set_index, test_set_index)] = hdf5_filename

    return results_dict
