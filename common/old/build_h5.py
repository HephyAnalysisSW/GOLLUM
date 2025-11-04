#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import h5py
import uproot
from tqdm import tqdm

# Import your structure (must be on PYTHONPATH or same dir)
from data_structure import feature_names, label_encoding, input_data

def find_root_files(dirs):
    files = []
    for d in dirs:
        # pick only .root files (recursively)
        files.extend(glob.glob(os.path.join(d, "**", "*.root"), recursive=True))
    # unique + sorted for stability
    return sorted(set(files))

#default_selection = "met_pt>30&&(nJetGood==2&&nBTag==2)&&l1_pt>0&&(!(l2_pt>0))"
default_selection = "(met_pt>30) & (nJetGood==2) & (nBTag==2) & (nlep==1)"
sel_branches = ["met_pt", "nJetGood", "nBTag", "nlep"]

def main():
    parser = argparse.ArgumentParser(
        description="Stream ROOT branches to HDF5 with weight and numeric label."
    )
    parser.add_argument("dataset", help="Dataset key (e.g. t_sch, t_tch, tWch, TT)")
    parser.add_argument("output_h5", help="Output HDF5 file path")
    parser.add_argument("--tree", default="Events", help="TTree name (default: Events)")
    parser.add_argument("--weight-name", default="weight", help='Weight branch name (default: "weight")')
    parser.add_argument("--batch", type=int, default=200_000, help="Approx. events per batch (default: 200000)")
    parser.add_argument("--dataset-name", default="data", help="HDF5 dataset name (default: data)")
    parser.add_argument("--dtype", default="float32", choices=["float32","float64"], help="Output dtype")
    parser.add_argument("--selection",    default=default_selection,    help='String selection applied while reading.',
    #parser.add_argument("--selection",    default=None,    help='String selection applied while reading.',
)
    args = parser.parse_args()

    ds_key = args.dataset
    if ds_key not in input_data:
        sys.exit(f"[ERROR] Unknown dataset '{ds_key}'. Valid options: {list(input_data.keys())}")

    # numeric label as float (carry the integer value in a float)
    if ds_key not in label_encoding:
        sys.exit(f"[ERROR] No label encoding for '{ds_key}'.")
    label_value = float(label_encoding[ds_key])

    dirs = input_data[ds_key]
    files = find_root_files(dirs)
    if not files:
        sys.exit(f"[ERROR] No ROOT files found under directories: {dirs}")

    read_branches = feature_names + [args.weight_name] + sel_branches

    ncols = len(feature_names) + 2  # + weight + label

    # Prepare HDF5 dataset (appendable)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5)), exist_ok=True)
    f5 = h5py.File(args.output_h5, "w")
    dset = f5.create_dataset(
        args.dataset_name,
        shape=(0, ncols),
        maxshape=(None, ncols),
        dtype=args.dtype,
        chunks=(max(1, min(args.batch, 1_000_000)), ncols),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    # Some metadata
    dset.attrs["columns"] = np.array(feature_names + [args.weight_name, "label"]).astype("S")
    dset.attrs["dataset_key"] = ds_key
    dset.attrs["label_value"] = label_value
    dset.attrs["tree"] = args.tree

    total_written = 0
    # uproot.iterate streams across many files & batches
    # treepath syntax: "filename:tree" OR pass file list + tree name
    try:
        from tqdm import tqdm
        treepaths = [f"{f}:{args.tree}" for f in files]

        print (args.selection)
        with tqdm(desc=f"Processing {ds_key}", unit="events", dynamic_ncols=True) as pbar:
            for batch in uproot.iterate(
                treepaths,
                expressions=read_branches,   # branches to keep
                cut=args.selection,           # <--- ROOT-like selection string
                step_size=args.batch,
                library="np",
                allow_missing=False,
            ):
                # batch is already filtered to the cut
                cols = [batch[b].astype(args.dtype, copy=False) for b in feature_names]
                w = batch[args.weight_name].astype(args.dtype, copy=False)
                cols.append(w)
                cols.append(np.full(len(w), float(label_encoding[ds_key]), dtype=args.dtype))

                X = np.column_stack(cols)
                old = dset.shape[0]
                dset.resize(old + X.shape[0], axis=0)
                dset[old:old + X.shape[0], :] = X
                total_written += X.shape[0]
                pbar.update(X.shape[0])
    except Exception as e:
        f5.close()
        # Clean up partial file on failure to avoid confusion
        try:
            os.remove(args.output_h5)
        except Exception:
            pass
        raise SystemExit(f"[ERROR] Failed while streaming ROOT -> HDF5: {e}")

    f5.flush()
    f5.close()
    print(f"[OK] Wrote {total_written} events to '{args.output_h5}' "
          f"with shape ({total_written}, {ncols}).")

if __name__ == "__main__":
    main()

