import h5py

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--file', action='store', default=None, help='input file')
args = argParser.parse_args()

if args.file is None:
    print( "No file defined. Use --file=")
    sys.exit(0)

print(f'Content of {args.file}:')
with h5py.File(args.file, 'r') as h5_file:
    for key in h5_file.keys():
        print(f" -- Dataset: {key}, shape: {h5_file[key].shape}")
