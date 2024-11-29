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
from data_loader.data_loader_2 import H5DataLoader

def copy_with_selection(input_files, target_dir, selection_function, n_batches=100, max_batch = None, overwrite=False):
    """
    Apply a selection function to events and copy the selected data to a target directory, writing each batch.

    Parameters:
    - input_files (list of str): List of input HDF5 files to process.
    - target_dir (str): Directory where selected files will be copied.
    - selection_function (callable): A function that takes a numpy array and returns a boolean mask for selection.
    - n_batches (int): Number of batches to divide the dataset into (default: 100).
    - overwrite (bool): Whether to overwrite existing target files (default: False).
    """
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    for file_path in input_files:
        print(f"Processing file: {file_path}")

        # Initialize data loader
        data_loader = H5DataLoader(
            file_path=file_path,
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
            first_batch = next(iter(data_loader))  # Peek at the first batch to infer dataset shapes
            dataset_shape = first_batch.shape[1:]  # Exclude batch dimension
            f_out.create_dataset(
                'data',
                shape=(0, *dataset_shape),
                maxshape=(None, *dataset_shape),
                dtype=first_batch.dtype,
            )

            # Reset the data loader iterator after peeking
            data_loader = iter(data_loader)

            # Write each batch to the output file with progress bar
            total_batches = len(data_loader)  # Total number of batches
            with tqdm(total=total_batches, desc=f"Writing to {os.path.basename(target_file)}", unit="batch") as pbar:
                i_batch=0
                for data in data_loader:
                    f_out['data'].resize(f_out['data'].shape[0] + data.shape[0], axis=0)
                    f_out['data'][-data.shape[0]:] = data
                    pbar.update(1)  # Update progress bar
                    i_batch+=1
                    if max_batch is not None:
                        if i_batch>=max_batch: break

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
    exec( f"from common.selections import {args.selection} as selection_function")

    # Expand input files (support for wildcards)
    input_files = glob.glob(args.files)
    if not input_files:
        raise FileNotFoundError(f"No files matched the pattern: {args.files}")

    # Apply selection and copy files
    copy_with_selection(input_files, target_dir=args.target_dir, selection_function=selection_function, n_batches=args.n_batches, overwrite=args.overwrite)

