import h5py
import sys
import numpy as np

def list_h5_contents(filename, head=5):
    with h5py.File(filename, "r") as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                # Print column names if stored as an attribute (e.g. our build script does this)
                cols = obj.attrs.get("columns")
                if cols is not None:
                    try:
                        cols = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in cols]
                        print("  columns:", ", ".join(cols))
                    except Exception:
                        pass

                # Print first 'head' events/rows without loading everything
                try:
                    n = int(head)
                except Exception:
                    n = 5
                n = max(0, min(n, obj.shape[0] if obj.shape else 0))

                if n == 0:
                    print("  (empty)")
                    return

                arr = obj[:n]  # slice reads only the needed part
                np.set_printoptions(suppress=True, linewidth=200)

                if obj.dtype.names:  # structured/record array
                    print("  head (structured):")
                    for i in range(n):
                        row = tuple(arr[i][name] for name in obj.dtype.names)
                        print(f"  [{i}] {row}")
                elif arr.ndim == 1:
                    print("  head:", arr)
                elif arr.ndim == 2:
                    print("  head (rows):")
                    is_num = np.issubdtype(arr.dtype, np.number)
                    for i in range(n):
                        row = arr[i]
                        if is_num:
                            row_str = " ".join(f"{x:.6g}" for x in row)
                        else:
                            row_str = " ".join(str(x) for x in row)
                        print(f"  [{i}] {row_str}")
                else:
                    print(f"  head slice shape: {arr.shape} (not printed)")
        f.visititems(print_structure)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file.h5> [n_head]")
        sys.exit(1)
    filename = sys.argv[1]
    head = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    list_h5_contents(filename, head=head)

