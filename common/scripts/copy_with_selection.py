import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")
import glob
from tqdm import tqdm
import os
import argparse
import h5py
import numpy as np
from common.FeatureSelector import FeatureSelector
from data_loader.data_loader_2 import H5DataLoader

def copy_with_selection(input_files, target_dir, selection_function, datasets, n_batches=100, overwrite=False):
    """
    Apply a selection function to events and copy the selected data to a target directory, writing each batch.

    Parameters:
    - input_files (list of str): List of input HDF5 files to process.
    - target_dir (str): Directory where selected files will be copied.
    - selection_function (callable): A function that takes a numpy array and returns a boolean mask for selection.
    - datasets (list of str): Names of datasets to include in the output files.
    - n_batches (int): Number of batches to divide the dataset into (default: 100).
    - overwrite (bool): Whether to overwrite existing target files (default: False).
    """
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    for file_path in input_files:
        print(f"Processing file: {file_path}")

        # Initialize data loader
        data_loader = H5DataLoader(
            file_path=file_path,
            datasets=datasets,
            n_split=n_batches,
            selection_function=selection_function,
        )

        # Define target file
        target_file = os.path.join(target_dir, os.path.basename(file_path))
        
        # Handle overwrite logic
        if os.path.exists(target_file) and not overwrite:
            print(f"Skipping file: {target_file} (already exists). Use --overwrite to overwrite.")
            continue
        if os.path.exists(target_file) and overwrite:
            print(f"Delete file: {target_file} (exists).")
            os.remove(target_file)

        # Create the output file
        print(f"Creating target file: {target_file}")
        with h5py.File(target_file, "w") as f_out:
            # Create empty datasets in the output file
            dataset_shapes = {}
            for ds in datasets:
                first_batch = next(iter(data_loader))  # Peek at the first batch to infer dataset shapes
                dataset_shapes[ds] = first_batch[ds].shape[1:]  # Exclude batch dimension
                f_out.create_dataset(
                    ds,
                    shape=(0, *dataset_shapes[ds]),
                    maxshape=(None, *dataset_shapes[ds]),
                    dtype=first_batch[ds].dtype,
                )

            # Reset the data loader iterator after peeking
            data_loader = iter(data_loader)

            # Write each batch to the output file with progress bar
            total_batches = len(data_loader)  # Total number of batches
            with tqdm(total=total_batches, desc=f"Writing to {os.path.basename(target_file)}", unit="batch") as pbar:
                for batch in data_loader:
                    for ds in datasets:
                        data = batch[ds]
                        # Resize and write data to the output file
                        f_out[ds].resize(f_out[ds].shape[0] + data.shape[0], axis=0)
                        f_out[ds][-data.shape[0]:] = data
                    pbar.update(1)  # Update progress bar

        print(f"Copied selected events to: {target_file}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply event selection and copy HDF5 files to a target directory.")
    parser.add_argument("--files", type=str, required=True,
                        help="Input files with wildcards (e.g., '/path/to/files/*.h5').")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Directory to copy selected files.")
    parser.add_argument("--selection", type=str, required=True,
                        help="Selection condition for filtering events.")
    parser.add_argument("--n-batches", type=int, default=1000,
                        help="Number of batches to divide the dataset into (default: 1000).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing target files.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Load selection mask
    print(f"Loading selection: {args.selection}")
    exec( f"from eventSelection import {args.selection} as selection_function")

    # Expand input files (support for wildcards)
    input_files = glob.glob(args.files)
    if not input_files:
        raise FileNotFoundError(f"No files matched the pattern: {args.files}")

    # Datasets to include in the output
    datasets = ['data', 'weights', 'detailed_labels']

    # Apply selection and copy files
    copy_with_selection(input_files, args.target_dir, selection_function, datasets, args.n_batches, args.overwrite)

