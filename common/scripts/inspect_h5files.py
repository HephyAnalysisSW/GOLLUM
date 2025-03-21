import os
import h5py
import argparse
def inspect_h5files(directory):
    """
    Inspect all HDF5 files in a given directory and print details or warnings.

    Parameters:
    - directory: str, path to the directory containing HDF5 files.
    """
    # Print header for aligned output
    print(f"{'Filename':<40} {'File Size (MB)':<15} {'libver':<15} {'Dataset Name':<25} {'Shape':<30} {'Dtype':<20} {'Length':<10}")
    print("=" * 160)

    # Loop over all files in the directory
    for file in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file)

        # Check if the file is an HDF5 file
        if not file.endswith(".h5"):
            print(f"Skipping non-HDF5 file: {file}")
            continue

        try:
            # Get file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            with h5py.File(file_path, 'r') as f:
                # Fetch HDF5 library version
                libver = f.swmr_mode
                libver_str = "SWMR-enabled" if libver else "Standard"

                # Track if datasets are found
                datasets_found = False

                # Loop through datasets and print details
                for dataset_name in f.keys():
                    datasets_found = True
                    dataset = f[dataset_name]
                    shape = dataset.shape
                    dtype = dataset.dtype
                    length = shape[0] if len(shape) > 0 else 0

                    if length == 0:
                        print(f"Warning: Dataset '{dataset_name}' in file '{file}' is empty. Skipping.")
                        continue

                    print(f"{file:<40} {file_size_mb:<15.2f} {libver_str:<15} {dataset_name:<25} {str(shape):<30} {str(dtype):<20} {length:<10}")

                # Warn if no datasets were found in the file
                if not datasets_found:
                    print(f"Warning: File '{file}' contains no datasets. Skipping.")

        except Exception as e:
            print(f"{file:<40} {'-':<15} {'Error':<15} {'-':<25} {'-':<30} {'-':<20} {'Error':<10}")
            print(f"Error while inspecting {file}: {e}")

#def inspect_h5files(directory):
#    for filename in os.listdir(directory):
#        file_path = os.path.join(directory, filename)
#        
#        # Skip non-HDF5 files
#        if not filename.endswith('.h5'):
#            print(f"Skipping non-HDF5 file: {filename}")
#            continue
#        
#        try:
#            # Open the file and check its contents
#            with h5py.File(file_path, 'r') as f:
#                # Check for datasets
#                if len(f.keys()) == 0:
#                    print(f"Warning: File '{filename}' has no datasets. Skipping.")
#                    continue
#
#                # Check for specific dataset "data"
#                if "data" not in f.keys():
#                    print(f"Warning: File '{filename}' does not contain a 'data' dataset. Skipping.")
#                    continue
#                
#                # Check the dataset size
#                data = f["data"]
#                if data.shape[0] == 0:
#                    print(f"Warning: File '{filename}' contains an empty 'data' dataset. Skipping.")
#                    continue
#
#                print(f"File '{filename}' is valid with shape {data.shape} and dtype {data.dtype}.")
#        
#        except Exception as e:
#            # Handle any errors while trying to read the file
#            print(f"Error reading file '{filename}': {str(e)}. Skipping.")
#            continue
#
#def inspect_h5files(directory):
#    """
#    Inspect all HDF5 files in a given directory.
#
#    Parameters:
#    - directory: str, path to the directory containing HDF5 files.
#    """
#    # Print header for aligned output
#    print(f"{'Filename':<40} {'File Size (MB)':<15} {'libver':<15} {'Dataset Name':<25} {'Shape':<30} {'Dtype':<20} {'Length':<10}")
#    print("=" * 160)
#
#    # Loop over all files in the directory
#    for file in sorted(os.listdir(directory)):
#        file_path = os.path.join(directory, file)
#        
#        # Check if the file is an HDF5 file
#        if not file.endswith(".h5"):
#            continue
#
#        try:
#            # Get file size in MB
#            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
#
#            with h5py.File(file_path, 'r') as f:
#                # Fetch HDF5 library version (requires SWMR for precise versioning)
#                libver = f.swmr_mode
#                libver_str = "SWMR-enabled" if libver else "Standard"
#
#                # Loop through datasets and print details
#                for dataset_name in f.keys():
#                    dataset = f[dataset_name]
#                    shape = dataset.shape
#                    dtype = dataset.dtype
#                    length = shape[0] if len(shape) > 0 else 0
#                    print(f"{file:<40} {file_size_mb:<15.2f} {libver_str:<15} {dataset_name:<25} {str(shape):<30} {str(dtype):<20} {length:<10}")
#
#        except Exception as e:
#            print(f"{file:<40} {'-':<15} {'Error':<15} {'-':<25} {'-':<30} {'-':<20} {'Error':<10}")
#            print(f"Error while inspecting {file}: {e}")

def main():
    """Main function to parse arguments and call the inspection function."""
    parser = argparse.ArgumentParser(description="Inspect HDF5 files in a directory.")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory containing HDF5 files to inspect."
    )
    args = parser.parse_args()

    # Inspect the specified directory
    inspect_h5files(args.directory)

if __name__ == "__main__":
    main()

