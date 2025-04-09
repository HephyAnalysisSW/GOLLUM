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
        self.label_mapping = {
            "diboson": 3,
            "ttbar": 2,
            "ztautau": 1,
            "htautau": 0
        }

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
        detailed_labels_set = {}
        for key in test_set.keys():
            test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
            data = pq.read_table(test_data_path).to_pandas()
            weights = data['weights'].to_numpy()
            weights_set[key] = weights
            columns_reordered = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']
            data_reordered = data[columns_reordered]
     #       data_with_der_features = DER_data(data_reordered)
            detailed_labels = np.array([key] * len(data), dtype=object)
            detailed_labels_set[key] = detailed_labels
            test_set[key] = data_reordered.to_numpy()

        self.__test_set = test_set
        self.__weights_set = weights_set 
        self.__detailed_labels_set = detailed_labels_set
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

    def get_syst_test_set(
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
        logger.info("Entered get_syst_test_set function.")

        if self.__test_set is None:
            logger.info("Loading test set because it is currently None.")
            self.load_test_set()

        logger.info("Copying test set.")
        test_set = self.__test_set.copy()

        # Convert test_set data to DataFrame if it is a numpy array
        columns = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi','PRI_had_pt', 'PRI_had_eta', 'PRI_had_phi','PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi','PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi','PRI_n_jets','PRI_jet_all_pt','PRI_met', 'PRI_met_phi']
        for key in test_set:
            if isinstance(test_set[key], np.ndarray):
                logger.info(f"Converting test_set['{key}'] to DataFrame.")
                test_set[key] = pd.DataFrame(test_set[key], columns=columns)

        # Call systematics function
        syst_test_set = {
            key: systematics(
		    {"data": test_set[key], "weights": self.__weights_set[key], "detailed_labels": self.__detailed_labels_set[key]},
                tes,
                jes,
                soft_met,
                ttbar_scale,
                diboson_scale,
                bkg_scale,
                dopostprocess=dopostprocess,
            )
            for key in test_set
        }

        logger.info("Systematics function called successfully.")

        # Construct output data with shape (N, 30)
        processed_test_set = {}
        for key, syst_data in syst_test_set.items():
            if syst_data is None:
                logger.error(f"Systematics function returned None for {key}.")
                raise ValueError(f"Systematics function returned None for {key}. Please check the systematics function for errors.")

            data = syst_data["data"].to_numpy()
            weights = syst_data["weights"].to_numpy()
            labels = np.full((len(weights),), self.label_mapping[key], dtype=int)

            # Extend data to 30 columns
            num_features = data.shape[1]
            additional_columns = np.zeros((data.shape[0], 30 - num_features - 2))

            processed_data = np.hstack([
                data,
                additional_columns,
                weights.reshape(-1, 1),
                labels.reshape(-1, 1),
            ])

            processed_test_set[key] = processed_data

        # Save to HDF5 files if required
        if save_to_hdf5:
            combined_data = []
            with h5py.File(hdf5_filename, "w") as hf:
                for key, processed_data in processed_test_set.items():
                    combined_data.append(processed_data)
                    logger.info(f"Saved processed {key} data to HDF5.")
                combined_data = np.vstack(combined_data)
                hf.create_dataset("data", data=combined_data, compression="gzip")
                logger.info("Combined dataset 'data' saved to HDF5.")
        logger.info("Systematic test set processing completed.")

        return processed_test_set

