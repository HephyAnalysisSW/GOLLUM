import os
import h5py
import numpy as np
from tqdm import tqdm


class HDF5Processor:
    def __init__(self, input_file, output_file, batch_size=1000, max_batches=None, overwrite=False):
        """
        Initialize the HDF5Processor.

        Parameters:
        - input_file: str, path to the input HDF5 file.
        - output_file: str, path to the output file.
        - batch_size: int, number of entries to process at a time.
        - max_batches: int, optional, maximum number of batches to process.
        - overwrite: bool, if True, deletes the output file if it already exists.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.overwrite = overwrite

        # Hardcoded label map
        self.label_map = {
            "htautau": 0,
            "ztautau": 1,
            "ttbar": 2,
            "diboson": 3
        }

    def process_file(self):
        """
        Processes the HDF5 file and writes the concatenated data to a new HDF5 file.
        """
        # Handle overwrite option
        if os.path.exists(self.output_file):
            if self.overwrite:
                print(f"File {self.output_file} already exists. Deleting it as 'overwrite' is enabled.")
                os.remove(self.output_file)
            else:
                print(f"File {self.output_file} already exists. Skipping as 'overwrite' is disabled.")
                return

        # Determine if the file name starts with a key in the label map
        filename_key = os.path.basename(self.input_file).split('_')[0]
        is_single_label_file = filename_key in self.label_map

        with h5py.File(self.input_file, 'r') as infile, h5py.File(self.output_file, 'w') as outfile:
            if is_single_label_file:
                # Process files with a single dataset and fixed structure
                self._process_single_label_file(infile, outfile, filename_key)
            else:
                # Process regular multi-dataset files
                self._process_multi_label_file(infile, outfile)

    def _process_single_label_file(self, infile, outfile, label_key):
        """
        Processes a file containing only one dataset, labeled according to the filename.
        """
        print(label_key)
        dataset = infile[label_key]
        label_index = self.label_map[label_key]
        num_entries = dataset.shape[0]
        total_batches = (num_entries + self.batch_size - 1) // self.batch_size
        if self.max_batches:
            total_batches = min(total_batches, self.max_batches)

        # Prepare dataset for output file
        output_data = outfile.create_dataset(
            "data",
            shape=(0, dataset.shape[1]),  # Keep the same structure as the input
            maxshape=(None, dataset.shape[1]),
            dtype=dataset.dtype
        )

        # Initialize tqdm progress bar
        with tqdm(total=total_batches, desc=f"Processing {label_key}", unit="batch") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_entries)

                # Load a batch
                batch = dataset[start_idx:end_idx]

                # Write the batch to the output file directly
                current_size = output_data.shape[0]
                output_data.resize(current_size + batch.shape[0], axis=0)
                output_data[current_size:] = batch

                # Update progress bar
                pbar.update(1)

        print(f"Processed single-label data for {label_key} saved to {self.output_file}")

    def _process_multi_label_file(self, infile, outfile):
        """
        Processes a file with multiple datasets and translates string labels to indices.
        """
        data = infile['data']
        weights = infile['weights']
        detailed_labels = infile['detailed_labels']

        num_entries = data.shape[0]
        total_batches = (num_entries + self.batch_size - 1) // self.batch_size
        if self.max_batches:
            total_batches = min(total_batches, self.max_batches)

        # Prepare dataset for output file
        output_data = outfile.create_dataset(
            "data",
            shape=(0, data.shape[1] + 2),  # Add 2 extra columns for weights and numeric labels
            maxshape=(None, data.shape[1] + 2),
            dtype=data.dtype
        )

        # Initialize tqdm progress bar
        with tqdm(total=total_batches, desc="Processing multi-label", unit="batch") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_entries)

                # Load a batch
                data_batch = data[start_idx:end_idx]
                weights_batch = weights[start_idx:end_idx]
                labels_batch = detailed_labels[start_idx:end_idx]

                # Convert labels to numeric using the label map
                numeric_labels = np.array([
                    self.label_map[label.decode('utf-8')] for label in labels_batch
                ])

                # Concatenate the batch data
                batch_output = np.column_stack((data_batch, weights_batch, numeric_labels))

                # Resize and write the current batch to the output file
                current_size = output_data.shape[0]
                output_data.resize(current_size + batch_output.shape[0], axis=0)
                output_data[current_size:] = batch_output

                # Update progress bar
                pbar.update(1)

        print(f"Processed multi-label data saved to {self.output_file}")


# Example usage:
if __name__ == "__main__":

    output_path = "/eos/vbc/group/cms/robert.schoefbeck/Higgs_uncertainty/processed"

    for input_file in [
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/ztautau_tes_1p03.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/nominal.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_0p97.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_0p98.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_0p99.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_1p01.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_1p02.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/jes_1p03.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/met_1p5.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/met_3.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/met_4p5.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/met_6.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p97.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p98.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99_jes_0p99_met_3.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99_jes_0p99_met_6.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99_jes_1p01.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99_jes_1p01_met_3.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_0p99_jes_1p01_met_6.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_0p99.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_0p99_met_3.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_0p99_met_6.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_1p01.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_1p01_met_3.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p01_jes_1p01_met_6.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p02.h5",
         "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/tes_1p03.h5",
            ]:

        out_file = os.path.join( output_path, input_file.split('/')[-1])
        processor = HDF5Processor(
            input_file=input_file,
            output_file=out_file,
            batch_size=1000000,
            max_batches=None,
            overwrite=False
        )
        processor.process_file()
