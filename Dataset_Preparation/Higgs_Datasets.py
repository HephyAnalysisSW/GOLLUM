import numpy as np
import pyarrow.parquet as pq
import json
import os
import pandas as pd
import requests
from zipfile import ZipFile
import logging
import io
import h5py


# Get the logging level from an environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)

test_set_settings = None

class Data:
    """
    A class to represent a dataset.

    Parameters:
        * input_dir (str): The directory path of the input data.

    Attributes:
        * __train_set (dict): A dictionary containing the train dataset.
        * __test_set (dict): A dictionary containing the test dataset.
        * input_dir (str): The directory path of the input data.

    Methods:
        * load_train_set(): Loads the train dataset.
        * load_test_set(): Loads the test dataset.
        * get_train_set(): Returns the train dataset.
        * get_test_set(): Returns the test dataset.
        * delete_train_set(): Deletes the train dataset.
        * get_syst_train_set(): Returns the train dataset with systematic variations.
    """

    def __init__(self, input_dir):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """

        self.__train_set = None
        self.__test_set = None
        self.input_dir = input_dir

    def load_train_set(self, sample_size=None, selected_indices=None):

        train_data_file = os.path.join(self.input_dir, "train", "data", "data.parquet")
        train_labels_file = os.path.join(
            self.input_dir, "train", "labels", "data.labels"
        )
        train_settings_file = os.path.join(
            self.input_dir, "train", "settings", "data.json"
        )
        train_weights_file = os.path.join(
            self.input_dir, "train", "weights", "data.weights"
        )
        train_detailed_labels_file = os.path.join(
            self.input_dir, "train", "detailed_labels", "data.detailed_labels"
        )

        parquet_file = pq.ParquetFile(train_data_file)

        # Step 1: Determine the total number of rows
        total_rows = sum(parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups))

        if sample_size is not None:
            if isinstance(sample_size, int):
                sample_size = min(sample_size, total_rows)
            elif isinstance(sample_size, float):
                if 0.0 <= sample_size <= 1.0:
                    sample_size = int(sample_size * total_rows)
                else:
                    raise ValueError("Sample size must be between 0.0 and 1.0")
            else:
                raise ValueError("Sample size must be an integer or a float")
        elif selected_indices is not None:
            if isinstance(selected_indices, list):
                selected_indices = np.array(selected_indices)
            elif isinstance(selected_indices, np.ndarray):
                pass
            else:
                raise ValueError("Selected indices must be a list or a numpy array")
            sample_size = len(selected_indices)
        else:
            sample_size = total_rows

        if selected_indices is None:
            selected_indices = np.sort(np.random.choice(total_rows, size=sample_size, replace=False))

        selected_indices_set = set(selected_indices)

        def get_sampled_data(data_file):
            selected_list = []
            with open(data_file, "r") as f:
                for i, line in enumerate(f):
                    # Check if the current line index is in the selected indices
                    if i not in selected_indices_set:
                        continue
                    if data_file.endswith(".detailed_labels"):
                        selected_list.append(line.strip())
                    else:
                        selected_list.append(float(line.strip()))
                    # Optional: stop early if all indices are found
                    if len(selected_list) == len(selected_indices):
                        break
            return np.array(selected_list)

        current_row = 0
        sampled_data = []
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_size = len(row_group)

            # Determine indices within the current row group that fall in the selected range
            within_group_indices = selected_indices[(selected_indices >= current_row) & (selected_indices < current_row + row_group_size)] - current_row
            sampled_data.append(row_group.iloc[within_group_indices].to_numpy())

            # Update the current row count
            current_row += row_group_size

        sampled_data = np.concatenate(sampled_data, axis=0)

        selected_train_labels = get_sampled_data(train_labels_file)
        selected_train_weights = get_sampled_data(train_weights_file)
        selected_train_detailed_labels = get_sampled_data(train_detailed_labels_file)

        logger.info(f"Sampled train data shape: {sampled_data.shape}")
        logger.info(f"Sampled train labels shape: {selected_train_labels.shape}")
        logger.info(f"Sampled train weights shape: {selected_train_weights.shape}")
        logger.info(f"Sampled train detailed labels shape: {selected_train_detailed_labels.shape}")

        self.__train_set = {
            "data": sampled_data,
            "labels": selected_train_labels,
            "settings": selected_train_labels,
            "weights": selected_train_weights,
            "detailed_labels": selected_train_detailed_labels,
        }

        del sampled_data, selected_train_labels, selected_train_weights, selected_train_detailed_labels

        logger.info("Train data loaded successfully")

    def load_test_set(self):
        from derived_quantities import DER_data
        test_data_dir = os.path.join(self.input_dir, "test", "data")

        # read test setting
        test_set = {
            "ztautau": None,
            "diboson": None,
            "ttbar": None,
            "htautau": None,
        }
        weights_set = {}
        for key in test_set.keys():

            test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
            data = pq.read_table(test_data_path).to_pandas()
            weights = data['weights'].to_numpy()
            weights_set[key] = weights
            columns_reordered = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']
            data_reordered = data[columns_reordered]
            data_with_der_features = DER_data(data_reordered)
            test_set[key] = data_with_der_features.to_numpy()

        self.__test_set = test_set
        self.__weights_set = weights_set 

        test_settings_file = os.path.join(
            self.input_dir, "test", "settings", "data.json"
        )
        with open(test_settings_file) as f:
            test_settings = json.load(f)

        self.ground_truth_mus = test_settings["ground_truth_mus"]

        for key in self.__test_set.keys():
            logger.info(f"{key} data shape: {self.__test_set[key].shape}")
        logger.info("Test data loaded successfully")

    def get_train_set(self):
        """
        Returns the train dataset.

        Returns:
            dict: The train dataset.
        """
        return self.__train_set

    def get_test_set(self):
        """
        Returns the test dataset.

        Returns:
            dict: The test dataset.
        """
        return self.__test_set

    def delete_train_set(self):
        """
        Deletes the train dataset.
        """
        self.__train_set = None

    def get_syst_train_set0(
        self,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        dopostprocess=False,
    ):
        from systematics import systematics

        if self.__train_set is None:
            self.load_train_set()
        return systematics(
            self.__train_set,
            tes,
            jes,
            soft_met,
            ttbar_scale,
            diboson_scale,
            bkg_scale,
            dopostprocess=dopostprocess,
        )

    def get_syst_train_set(
        self,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        dopostprocess=False,
        save_to_hdf5=False,
        hdf5_filename=None
    ):
        from systematics import systematics
        logger.info("Entered get_syst_train_set function.")

        if self.__train_set is None:
            logger.info("Loading training set because it is currently None.")
            self.load_train_set()
        
        logger.info("Copying training set.")
        train_set = self.__train_set.copy()

        # Convert train_set['data'] to DataFrame if it is a numpy array
        columns = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']
        if isinstance(train_set['data'], np.ndarray):
            logger.info("Converting train_set['data'] to DataFrame.")
            train_set['data'] = pd.DataFrame(train_set['data'], columns=columns)
        
        # Call systematics function
        syst_train_set = systematics(
            train_set,
            tes,
            jes,
            soft_met,
            ttbar_scale,
            diboson_scale,
            bkg_scale,
            dopostprocess=dopostprocess,
        )
        del train_set
        logger.info("Systematics function called successfully.")
        
        # Check if systematics function returned a valid result
        if syst_train_set is None:
            logger.error("Systematics function returned None.")
            raise ValueError("Systematics function returned None. Please check the systematics function for errors.")
        
        # Save to HDF5 files if required
        if save_to_hdf5:
            # Extract all data, labels, detailed_labels, and weights
            data = syst_train_set['data'].copy()
            detailed_labels = syst_train_set['detailed_labels'].copy()
            weights = syst_train_set['weights'].copy()

            if isinstance(data, pd.DataFrame):
                data_array = data.to_numpy(dtype='float32')
            weights = np.array(weights, dtype=np.float32)
            label_mapping = {
                b'htautau': 0.0,
                b'ztautau': 1.0,
                b'ttbar': 2.0,
                b'diboson': 3.0
            }
            detailed_labels = np.array([label.encode() if isinstance(label, str) else label for label in detailed_labels])
            label_values = np.array([label_mapping[label] for label in detailed_labels], dtype=np.float32)
            combined_data = np.hstack((data, weights[:, np.newaxis], label_values[:, np.newaxis]))

            # Save to the specified HDF5 file
            with h5py.File(hdf5_filename, "w") as hf:
                logger.info(f"Saving combined dataset to file '{hdf5_filename}' with shape {combined_data.shape}")
                hf.create_dataset('data', data=combined_data, compression="gzip")

            logger.info(f"Systematic train set saved to {hdf5_filename}")

        return syst_train_set

