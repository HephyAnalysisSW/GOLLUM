"""
RDataLoader — lightweight ROOT/UPROOT data loader with group-aware branch handling.

Key features
------------
- Instantiate on a directory (or list of files/dirs) containing ROOT files.
- Select a TTree by name (default: "Events").
- Provide an explicit branch allowlist; unknown branches are ignored with a warning (optional strict mode).
- Optional selection callable: `selection(ar) -> mask` applied to the loaded awkward Array.
- Split data by files or events for simple parallelization (`n_split`, `splitting_strategy`).
- Access helpers to extract scalar and vector branches into numpy/awkward.
- Minimal dependencies: uproot, awkward, numpy.

Usage
-----
from tools.RDataLoader import RDataLoader
ldr = RDataLoader(
    input_paths=["/path/to/dirA", "/path/to/file.root"],
    tree_name="Events",
    branches=["pt", "eta", "phi"],
    selection=lambda ar: (ar["pt"] > 0),
    n_split=1,
    splitting_strategy="events",
)
arr = ldr[0]                      # awkward.Array (possibly filtered)
X   = ldr.scalar_branches(arr, ["pt","eta","phi"])  # (N,3) numpy

"""
from __future__ import annotations
import os
import glob
import math
import warnings
from typing import Callable, Iterable, List, Optional, Sequence, Union, Dict

import numpy as np
import awkward as ak
from awkward import types as aktypes
import uproot

PathLike = Union[str, os.PathLike]
SelectionFn = Optional[Callable[[ak.Array], Union[ak.Array, np.ndarray]]]


class RDataLoader:
    def __init__(
        self,
        input_paths: Union[PathLike, Sequence[PathLike]],
        tree_name: str = "Events",
        branches: Optional[Sequence[str]] = None,
        selection: SelectionFn = None,
        file_pattern: str = "*.root",
        n_split: int = 1,
        splitting_strategy: str = "files",  # "files" or "events"
        strict_branches: bool = False,
        max_files: Optional[int] = None,
    ) -> None:
        self.tree_name = tree_name
        self.selection = selection
        self.strict_branches = strict_branches
        self.splitting_strategy = splitting_strategy
        self.n_split = max(1, int(n_split))

        # Resolve file list
        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [str(input_paths)]
        files: List[str] = []
        for p in input_paths or []:
            p = os.path.expanduser(str(p))
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, file_pattern))))
            elif os.path.isfile(p):
                files.append(p)
            else:
                warnings.warn(f"RDataLoader: path not found: {p}")
        if not files:
            raise FileNotFoundError("RDataLoader: no ROOT files found.")
        if max_files is not None:
            files = files[: int(max_files)]
        self._all_files = files

        # Discover branches if not provided
        self._requested_branches = list(branches) if branches else None
        if self._requested_branches is None:
            with uproot.open(self._all_files[0]) as f:
                t = f[self.tree_name]
                self._requested_branches = list(t.keys())
                warnings.warn("RDataLoader: 'branches' not provided, loading all tree branches.")

        # Compute split layout
        if self.splitting_strategy not in ("files", "events"):
            raise ValueError("splitting_strategy must be 'files' or 'events'")
        self._file_splits: List[List[str]] = self._make_file_splits(self._all_files, self.n_split)

    # ---------------------------- public API ----------------------------
    def __len__(self) -> int:
        """Number of shards/splits."""
        return len(self._file_splits) if self.splitting_strategy == "files" else self.n_split

    def __getitem__(self, idx: int) -> ak.Array:
        """Load one shard as an awkward Array, apply selection if present."""
        if self.splitting_strategy == "files":
            files = self._file_splits[self._wrap_index(idx)]
            ar = self._load_files(files)
        else:  # events
            # load all, then slice by event ranges
            ar_all = self._load_files(self._all_files)
            n = len(ar_all)
            # compute contiguous event chunks
            bounds = self._event_bounds(n, self.n_split)
            lo, hi = bounds[self._wrap_index(idx)]
            ar = ar_all[lo:hi]
        # selection after split to keep shard sizes balanced
        if self.selection is not None:
            mask = self.selection(ar)
            # Be robust to numpy or awkward boolean masks
            mask = ak.to_numpy(mask) if isinstance(mask, ak.Array) else mask
            if mask is None:
                warnings.warn("RDataLoader: selection returned None; skipping.")
            else:
                ar = ar[mask]
        return ar

    @property
    def files(self) -> List[str]:
        return list(self._all_files)

    @property
    def branches(self) -> List[str]:
        return list(self._requested_branches or [])

    # ---- helpers for extracting data ----
    def scalar_branches(self, ar: ak.Array, names: Sequence[str]) -> np.ndarray:
        """Return a 2D numpy array stack of scalar branches `names` from awkward record array `ar`.
        If a branch is jagged (variable-length), raise an informative error.
        """
        cols = []
        for n in names:
            if n not in ar.fields:
                if self.strict_branches:
                    raise KeyError(f"Requested branch '{n}' not in array fields.")
                warnings.warn(f"scalar_branches: missing '{n}', filling with zeros.")
                cols.append(np.zeros(len(ar), dtype=np.float32))
                continue
            v = ar[n]
            # In Awkward v2, inspect types via awkward.types
            t = ak.type(v)
            # Unwrap optional types
            while isinstance(t, aktypes.OptionType):
                t = t.type
            # Treat ListType and RegularType as non-scalar (vectors/jagged/fixed-length)
            if isinstance(t, (aktypes.ListType, aktypes.RegularType)):
                raise ValueError(
                    f"Branch '{n}' is vector-like (List/Regular), not scalar. Use vector_branch()."
                )
            cols.append(ak.to_numpy(v))
        if not cols:
            return np.empty((len(ar), 0), dtype=np.float32)
        return np.stack(cols, axis=1)

    def vector_branch(self, ar: ak.Array, name: str) -> ak.Array:
        """Return a single jagged/vector branch as awkward Array."""
        if name not in ar.fields:
            if self.strict_branches:
                raise KeyError(f"Requested branch '{name}' not in array fields.")
            warnings.warn(f"vector_branch: missing '{name}', returning empty jagged array.")
            return ak.Array([[] for _ in range(len(ar))])
        return ar[name]

    # --------------------------- internals ----------------------------
    def _wrap_index(self, idx: int) -> int:
        if not isinstance(idx, int):
            raise TypeError("RDataLoader indices must be integers")
        n = len(self)
        return idx % n

    def _make_file_splits(self, files: List[str], n_split: int) -> List[List[str]]:
        if n_split <= 1:
            return [files]
        # round-robin assignment for better balancing when file sizes vary
        splits: List[List[str]] = [[] for _ in range(n_split)]
        for i, f in enumerate(files):
            splits[i % n_split].append(f)
        return splits

    def _event_bounds(self, n_events: int, n_split: int) -> List[tuple]:
        # contiguous chunks (nearly equal sizes)
        base = n_events // n_split
        rem = n_events % n_split
        bounds = []
        start = 0
        for i in range(n_split):
            extra = 1 if i < rem else 0
            end = start + base + extra
            bounds.append((start, end))
            start = end
        return bounds

    def _load_files(self, files: Sequence[str]) -> ak.Array:
        if len(files) == 1:
            f = files[0]
            with uproot.open(f) as rf:
                t = rf[self.tree_name]
                available = set(t.keys())
                use_branches = self._filter_branches(available, self._requested_branches)
                # Return an awkward record array
                return t.arrays(expressions=use_branches, library="ak")
        else:
            # uproot.concatenate keeps event order per file
            # peek first file for available branches
            with uproot.open(files[0]) as rf0:
                t0 = rf0[self.tree_name]
                available0 = set(t0.keys())
                use_branches = self._filter_branches(available0, self._requested_branches)
            # Return a single awkward record array across all files
            return uproot.concatenate(
                {f: self.tree_name for f in files},
                expressions=use_branches,
                library="ak",
            )

    def _filter_branches(self, available: set, requested: Sequence[str]) -> List[str]:
        missing = [b for b in requested if b not in available]
        if missing:
            if self.strict_branches:
                raise KeyError(f"Missing branches in tree: {missing}")
            warnings.warn(f"RDataLoader: missing branches will be skipped: {missing}")
        return [b for b in requested if b in available]
