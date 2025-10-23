#!/usr/bin/env python3
import argparse
import os
import sys
import h5py
import numpy as np

def find_2d_dataset(f, prefer_name=None):
    """Return (dset, path). Prefer `prefer_name` if present; else auto-detect a single 2D dataset."""
    if prefer_name and prefer_name in f:
        return f[prefer_name], prefer_name
    candidates = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
            candidates.append(name)
    f.visititems(visitor)
    if prefer_name:
        raise KeyError(f"Dataset '{prefer_name}' not found. 2D candidates: {candidates or 'none'}")
    if len(candidates) == 1:
        return f[candidates[0]], candidates[0]
    if len(candidates) == 0:
        raise ValueError("No 2D datasets found.")
    raise ValueError(f"Multiple 2D datasets found: {candidates}. Use --dataset-name.")

def split_slices(nrows, n_batches):
    """Evenly split [0, nrows) into n_batches slices."""
    edges = np.linspace(0, nrows, n_batches + 1, dtype=np.int64)
    return list(zip(edges[:-1], edges[1:]))

def main():
    ap = argparse.ArgumentParser(
        description="Merge HDF5 2D datasets in synchronized batches, shuffle per batch, write one output."
    )
    ap.add_argument("inputs", nargs="+", help="Input .h5 files (any number).")
    ap.add_argument("--output", required=True, help="Output HDF5 filename (name only; written next to first input).")
    ap.add_argument("--dataset-name", default="data",
                    help="Dataset name to read (default: 'data'); if absent, auto-detect a single 2D dataset.")
    ap.add_argument("--output-dset", default="data", help="Output dataset name (default: 'data').")
    ap.add_argument("--n-batches", type=int, default=100, help="Number of synchronized batches (default: 100).")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for shuffling (default: 12345).")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float64"],
                    help="Force output dtype; default is input dtype of first file.")
    args = ap.parse_args()

    inputs = [os.path.abspath(p) for p in args.inputs]
    if not inputs:
        sys.exit("[error] No input files provided.")

    first_dir = os.path.dirname(inputs[0])
    out_name = os.path.basename(args.output)     # ignore any directory the user typed
    output_path = os.path.join(first_dir, out_name)
    if os.path.dirname(args.output) not in ("", ".", first_dir):
        print(f"[info] Ignoring directory in --output; writing to: {output_path}")

    # Ignore the output file if it appears among inputs
    filtered_inputs = []
    for p in inputs:
        if os.path.abspath(p) == os.path.abspath(output_path) or os.path.basename(p) == out_name:
            print(f"[info] Ignoring output file listed among inputs: {p}")
            continue
        filtered_inputs.append(p)
    if not filtered_inputs:
        sys.exit("[error] After ignoring the output filename, no input files remain.")

    # Scan inputs: dataset path, shape, dtype, columns attr (if consistent)
    file_info = []
    ncols_ref = None
    dtype_ref = None
    columns_attr = None
    for path in filtered_inputs:
        if not os.path.isfile(path):
            sys.exit(f"[error] Not a file: {path}")
        with h5py.File(path, "r") as f:
            dset, dname = find_2d_dataset(f, args.dataset_name)
            if dset.ndim != 2:
                sys.exit(f"[error] Dataset '{dname}' in {path} is not 2D (shape={dset.shape}).")
            nrows, ncols = dset.shape
            if ncols_ref is None:
                ncols_ref = ncols
            elif ncols != ncols_ref:
                sys.exit(f"[error] Column mismatch: expected {ncols_ref}, got {ncols} in {path}:{dname}.")
            if dtype_ref is None:
                dtype_ref = dset.dtype
            # track columns attribute consistency
            cols = dset.attrs.get("columns")
            if cols is not None:
                if columns_attr is None:
                    columns_attr = cols
                elif not np.array_equal(columns_attr, cols):
                    print(f"[warn] 'columns' attribute differs in {path}; dropping it in output.")
                    columns_attr = None
            file_info.append({"path": path, "dpath": dname, "nrows": nrows})

    total_rows = sum(fi["nrows"] for fi in file_info)
    print(f"[info] Inputs: {len(file_info)} files | total rows: {total_rows} | cols: {ncols_ref}")
    print(f"[info] Writing output to: {output_path} (dataset='{args.output_dset}')")
    if total_rows == 0:
        sys.exit("[error] No rows to merge.")

    # Prepare per-file batch slices
    n_batches = max(1, int(args.n_batches))
    per_file_slices = [split_slices(fi["nrows"], n_batches) for fi in file_info]

    # Decide output dtype
    out_dtype = np.dtype(args.dtype) if args.dtype else dtype_ref

    # Create output file and dataset
    os.makedirs(first_dir, exist_ok=True)
    with h5py.File(output_path, "w") as fout:
        dset_out = fout.create_dataset(
            args.output_dset,
            shape=(total_rows, ncols_ref),
            dtype=out_dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=True,
        )
        if columns_attr is not None:
            dset_out.attrs["columns"] = columns_attr
        dset_out.attrs["merged_from"] = np.array([os.path.basename(fi["path"]).encode() for fi in file_info])

        rng = np.random.default_rng(args.seed)
        write_offset = 0

        for b in range(n_batches):
            batch_arrays = []
            rows_this_batch = 0

            # Read the b-th slice from each input
            for fi, slices in zip(file_info, per_file_slices):
                start, stop = slices[b]
                if stop > start:  # non-empty slice
                    with h5py.File(fi["path"], "r") as fin:
                        arr = fin[fi["dpath"]][start:stop]
                    if arr.ndim != 2 or arr.shape[1] != ncols_ref:
                        sys.exit(f"[error] Unexpected shape from {fi['path']} slice {start}:{stop}: {arr.shape}")
                    if arr.dtype != out_dtype:
                        arr = arr.astype(out_dtype, copy=False)
                    batch_arrays.append(arr)
                    rows_this_batch += arr.shape[0]
                else:
                    # empty slice: contribute nothing (keeps batches synchronized)
                    pass

            if rows_this_batch == 0:
                # nothing in this batch across all files; keep going
                continue

            # Concatenate and shuffle within this batch
            combined = np.vstack(batch_arrays)
            perm = rng.permutation(combined.shape[0])
            combined = combined[perm]

            # Write sequentially
            dset_out[write_offset:write_offset + combined.shape[0], :] = combined
            write_offset += combined.shape[0]

            # Optional lightweight progress
            print(f"[info] batch {b+1}/{n_batches}: wrote {combined.shape[0]} rows (cum {write_offset}/{total_rows})")

        if write_offset != total_rows:
            print(f"[warn] Wrote {write_offset} rows, expected {total_rows}. "
                  f"(Likely some trailing empty batches due to small inputs.)")

    print(f"[ok] Done. Output shape: ({total_rows}, {ncols_ref}).")

if __name__ == "__main__":
    main()

