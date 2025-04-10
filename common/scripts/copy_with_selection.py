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
import common.data_structure as data_structure
import common.user as user
import shutil
import uuid

def copy_with_selection(input_files, target_dir, selection_function, process=None, n_batches=100, max_batch = None, overwrite=False, skip_tmp_copy=False):
    """
    Apply a selection function to events and copy the selected data to a target directory, writing each batch.

    Parameters:
    - input_files (list of str): List of input HDF5 files to process.
    - target_dir (str): Directory where selected files will be copied.
    - selection_function (callable): A function that takes a numpy array and returns a boolean mask for selection.
    - process (str): Identifier of the process to be copied.
    - n_batches (int): Number of batches to divide the dataset into (default: 100).
    - overwrite (bool): Whether to overwrite existing target files (default: False).
    """
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    for file_path in input_files:
        print(f"Processing file: {file_path}")

        # Copy input to tmp directory
        if not skip_tmp_copy:
            tmp_dir = os.path.join( user.tmp_mem_directory, str(uuid.uuid4()) )
            os.makedirs(tmp_dir, exist_ok=True)
            new_file_path = os.path.join(tmp_dir, os.path.basename( file_path ))
            print( f"Copy {file_path} to {new_file_path}")
            shutil.copy(file_path, new_file_path)
            file_path = new_file_path

        # Initialize data loader
        data_loader = H5DataLoader(
            file_path=file_path,
            n_split=n_batches,
            selection_function=selection_function,
            process=process,
        )

        # Define target file
        base_name = os.path.basename(file_path)
        if process is None:
            target_file = os.path.join(target_dir, base_name)
        else:

            if process not in data_structure.labels:
                raise RuntimeError( "Don't know this process: %s" % process )

            # the nominal file
            if base_name.startswith('nominal'):
                file_name = base_name.replace('nominal', process)
            # the input file was already process specific
            elif any( [base_name.startswith(l) for l in data_structure.labels] ):
                file_name = base_name
            elif any( [base_name.startswith(s) for s in data_structure.systematics] ):
                file_name = process+'_'+base_name
            else:
                raise RuntimeError( "Don't know what the file name should be." )
            
            target_file = os.path.join(target_dir, file_name)

        # Handle overwrite logic
        if os.path.exists(target_file) and not overwrite:
            print(f"Skipping file: {target_file} (already exists). Use --overwrite to overwrite.")
            continue
        if os.path.exists(target_file) and overwrite:
            print(f"Delete file: {target_file} (exists).")
            os.remove(target_file)

        # Create the output file
        try:
            print(f"Creating target file: {target_file}")
            with h5py.File(target_file, "w") as f_out:
                first_batch = next(iter(data_loader))  # Peek at the first batch to infer dataset shapes
                dataset_shape = first_batch.shape[1:]  # Exclude batch dimension
                f_out.create_dataset(
                    'data',
                    shape=(0, *dataset_shape),
                    maxshape=(None, *dataset_shape),
                    dtype=first_batch.dtype,
                    compression="gzip",  # Use gzip compression
                    compression_opts=4   # Compression level (1: fastest, 9: smallest)
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
        except Exception as e:
            print(f"An error occurred: {e}")
            raise  # Re-raise the exception after logging or handling
        finally:
            # Ensure the file is deleted in all cases
            if not skip_tmp_copy:
                try:
                    if os.path.exists(new_file_path):
                        os.remove(new_file_path)
                        print(f"Cleanup. File deleted: {new_file_path}")
                except Exception as cleanup_error:
                    print(f"Failed to clean up {new_file_path}: {cleanup_error}")

        print(f"Copied selected events to: {target_file}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply event selection and copy HDF5 files to a target directory.")
    parser.add_argument("--files", type=str, nargs='+', required=True,
                        help="Input files (can use wildcards, e.g., '/path/to/files/*.h5').")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Directory to copy selected files.")
    parser.add_argument("--selection", type=str, required=True,
                        help="Selection condition for filtering events.")
    parser.add_argument("--process", type=str, default=None,
                        help="Copy only events from this process.")
    parser.add_argument("--n-batches", type=int, default=1000,
                        help="Number of batches to divide the dataset into (default: 1000).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing target files.")
    parser.add_argument("--skip_tmp_copy", action="store_true",
                        help="Do not copy to tmp before processing.")
    parser.add_argument("--cmds", action="store_true",
                        help="Write single jobs to jobs.sh.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Load selection mask
    print(f"Loading selection: {args.selection}")
    exec( f"from common.selections import {args.selection} as selection_function")

    # Handle input files
    input_files = []
    for file_pattern in args.files:
        expanded_files = glob.glob(file_pattern)  # Expand wildcards
        if expanded_files:
            input_files.extend(expanded_files)
        else:
            # If glob fails, assume it's a direct file path
            input_files.append(file_pattern)

    # Ensure files exist
    if not input_files:
        raise FileNotFoundError(f"No input files matched the patterns or were provided: {args.files}")
    missing_files = [file for file in input_files if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"The following files were not found: {missing_files}")

    # write jobs.sh if required
    if args.cmds:
        with open('jobs.sh', 'a+') as job_file:
            for i_input_file, input_file in enumerate(input_files):

                cmds = ["python", "copy_with_selection.py", "--files", input_file, "--target-dir", args.target_dir, "--selection", args.selection, "--n-batches", str(args.n_batches)]
                if args.process is not None:
                    cmds.extend( ["--process", args.process] )
                if args.overwrite:
                    cmds.append("--overwrite")
                if args.skip_tmp_copy:
                    cmds.append("--skip_tmp_copy")
                job_file.write(" ".join(cmds)+ '\n')
        print("Appended %i jobs to jobs.sh."%len(input_files))
        sys.exit(0)

    # Apply selection and copy files
    copy_with_selection(input_files, target_dir=args.target_dir, selection_function=selection_function, process=args.process, n_batches=args.n_batches, overwrite=args.overwrite, skip_tmp_copy=args.skip_tmp_copy)
