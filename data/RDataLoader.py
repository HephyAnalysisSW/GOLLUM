"""
RDataLoader — lightweight ROOT/UPROOT data loader with group-aware branch handling.
"""
from __future__ import annotations
import os
import glob
import warnings
from typing import Callable, List, Optional, Sequence, Union, Tuple

import numpy as np
import awkward as ak
from awkward import types as aktypes
import uproot

import SelectionView

PathLike = Union[str, os.PathLike]
SelectionFn = Optional[Callable[[ak.Array], Union[ak.Array, np.ndarray]]]
WeightFn = Callable[[dict[str, np.ndarray]], np.ndarray]

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
        # ---- NEW (optional, does not break callers) ----
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        # ---- NEW: weight handling ----
        # allow: str | list[str] (product) | callable
        weight: Union[str, Sequence[str], WeightFn] = "weight",
        weight_branches: Optional[Sequence[str]] = None,
    ) -> None:
        self.tree_name = tree_name
        self.selection = selection
        self.strict_branches = strict_branches
        self.splitting_strategy = splitting_strategy
        self.n_split = max(1, int(n_split))

        # Persist configured features/observers (optional)
        self.feature_names: Optional[List[str]] = list(feature_names) if feature_names else None
        self.observer_names: Optional[List[str]] = list(observer_names) if observer_names else None

        # NEW: weight config
        self.weight: Union[str, Sequence[str], WeightFn] = weight
        self.weight_branches: List[str] = list(weight_branches) if weight_branches else []

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
        _requested = list(branches) if branches else None
        if _requested is None:
            with uproot.open(self._all_files[0]) as f:
                t = f[self.tree_name]
                _requested = list(t.keys())
                warnings.warn("RDataLoader: 'branches' not provided, loading all tree branches.")

        # Ensure requested branches include configured features/observers if any
        if self.feature_names:
            _requested = list(dict.fromkeys(list(_requested) + list(self.feature_names)))
        if self.observer_names:
            _requested = list(dict.fromkeys(list(_requested) + list(self.observer_names)))

        # Ensure requested branches include what the weight needs
        if isinstance(self.weight, str):
            if self.weight:
                _requested = list(dict.fromkeys(list(_requested) + [self.weight]))
        elif isinstance(self.weight, (list, tuple)):
            # product of these branches
            _requested = list(dict.fromkeys(list(_requested) + list(self.weight)))
        else:
            # callable -> ensure needed inputs are present
            if self.weight_branches:
                _requested = list(dict.fromkeys(list(_requested) + list(self.weight_branches)))

        self._requested_branches = _requested

        # Compute split layout
        if self.splitting_strategy not in ("files", "events"):
            raise ValueError("splitting_strategy must be 'files' or 'events'")
        self._file_splits: List[List[str]] = self._make_file_splits(self._all_files, self.n_split)

        # -------- caches (per shard) --------
        self._arr_cache: dict[int, ak.Array] = {}
        self._mask_cache: dict[str, dict[int, np.ndarray]] = {}

    # ---------------------------- public API ----------------------------
    def __len__(self) -> int:
        return len(self._file_splits) if self.splitting_strategy == "files" else self.n_split

    def __getitem__(self, idx: int) -> ak.Array:
        """Load one shard as an awkward Array, apply selection if present. Result is cached."""
        key = self._wrap_index(idx)
        if key in self._arr_cache:
            return self._arr_cache[key]

        if self.splitting_strategy == "files":
            files = self._file_splits[key]
            ar = self._load_files(files)
        else:  # events
            ar_all = self._load_files(self._all_files)
            n = len(ar_all)
            lo, hi = self._event_bounds(n, self.n_split)[key]
            ar = ar_all[lo:hi]

        # selection after split
        if self.selection is not None:
            mask = self.selection(ar)
            mask = ak.to_numpy(mask) if isinstance(mask, ak.Array) else mask
            if mask is None:
                warnings.warn("RDataLoader: selection returned None; skipping.")
            else:
                ar = ar[mask]

        self._arr_cache[key] = ar
        return ar

    @property
    def files(self) -> List[str]:
        return list(self._all_files)

    @property
    def branches(self) -> List[str]:
        return list(self._requested_branches or [])

    # ---- helpers for extracting data ----
    def scalar_branches(self, ar: ak.Array, names: Sequence[str]) -> np.ndarray:
        """Return a 2D numpy array stack of scalar branches `names` from awkward record array `ar`."""
        cols = []
        for n in names:
            if n not in ar.fields:
                if self.strict_branches:
                    raise KeyError(f"Requested branch '{n}' not in array fields.")
                warnings.warn(f"scalar_branches: missing '{n}', filling with zeros.")
                cols.append(np.zeros(len(ar), dtype=np.float32))
                continue
            v = ar[n]
            t = ak.type(v)
            while isinstance(t, aktypes.OptionType):
                t = t.type
            if isinstance(t, (aktypes.ListType, aktypes.RegularType)):
                raise ValueError(f"Branch '{n}' is vector-like (List/Regular), not scalar. Use vector_branch().")
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

    # -------- direct feature/observer access (no awkward escapes) ----------
    def observers(self, shard: int = 0, n: Optional[int] = None,
                  observer_names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(observer_names) if observer_names is not None else (self.observer_names or [])
        if not names:
            raise ValueError("No observer names configured. Pass observer_names=... or set observer_names in the constructor.")
        ar = self[shard]
        G = self.scalar_branches(ar, names)
        return G if n is None else G[:n]

    def features(self, shard: int = 0, n: Optional[int] = None,
                 feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(feature_names) if feature_names is not None else (self.feature_names or [])
        if not names:
            raise ValueError("No feature names configured. Pass feature_names=... or set feature_names in the constructor.")
        ar = self[shard]
        X = self.scalar_branches(ar, names)
        return X if n is None else X[:n]

    def features_and_observers(self, shard: int = 0, n: Optional[int] = None,
                               feature_names: Optional[Sequence[str]] = None,
                               observer_names: Optional[Sequence[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        fnames = list(feature_names) if feature_names is not None else (self.feature_names or [])
        onames = list(observer_names) if observer_names is not None else (self.observer_names or [])
        if not fnames:
            raise ValueError("No feature names configured. Pass feature_names=... or set feature_names in the constructor.")
        if not onames:
            raise ValueError("No observer names configured. Pass observer_names=... or set observer_names in the constructor.")
        ar = self[shard]
        X = self.scalar_branches(ar, fnames)
        G = self.scalar_branches(ar, onames)
        if n is not None:
            X = X[:n]
            G = G[:n]
        return X, G

    def weight_vector(self, shard: int = 0, n: Optional[int] = None) -> np.ndarray:
        """Compute weight vector. Missing inputs now raise KeyError."""
        ar = self[shard]

        if isinstance(self.weight, str):
            # nominal weight must exist
            if self.weight not in ar.fields:
                raise KeyError(f"Weight branch '{self.weight}' not found in loaded branches. "
                               f"Add it to 'branches' or 'observer_names'.")
            w = ak.to_numpy(ar[self.weight]).astype(np.float32, copy=False)

        else:
            # callable — ALL required inputs must exist
            needed = list(self.weight_branches or [])
            if not needed:
                raise ValueError("Callable weight requires non-empty 'weight_branches'.")
            data: dict[str, np.ndarray] = {}
            missing = [nme for nme in needed if nme not in ar.fields]
            if missing:
                raise KeyError(f"Weight input branches missing: {missing}. "
                               f"Add them to 'branches' or 'observer_names'.")
            for nme in needed:
                data[nme] = ak.to_numpy(ar[nme]).astype(np.float32, copy=False)

            w = np.asarray(self.weight(data))
            if w.ndim != 1 or len(w) != len(ar):
                raise ValueError("Weight function must return a 1D array of length equal to the shard length.")
            w = w.astype(np.float32, copy=False)

        return w if n is None else w[:n]

    def materialize(
        self,
        shard: int = 0,
        what: str = "fo",
        n: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Materialize any combination (and order) of Features/Observers/Weights.
        `what` is a string composed of letters in {'f','o','w'}; order is preserved (e.g. 'fwo', 'wof').
        """
        outputs: list[np.ndarray] = []
        for ch in what:
            if ch == 'f':
                outputs.append(self.features(shard=shard, n=None, feature_names=feature_names))
            elif ch == 'o':
                outputs.append(self.observers(shard=shard, n=None, observer_names=observer_names))
            elif ch == 'w':
                outputs.append(self.weight_vector(shard=shard, n=None))
            else:
                raise ValueError(f"materialize: unknown spec letter '{ch}' (allowed: 'f','o','w').")
        if n is not None:
            outputs = [arr[:n] for arr in outputs]
        return tuple(outputs)

    # -------- NEW: masked materialization helpers (for views) ----------
    def features_from_mask(self, shard: int, mask: np.ndarray,
                           feature_names: Optional[Sequence[str]] = None,
                           n: Optional[int] = None) -> np.ndarray:
        X = self.features(shard=shard, n=None, feature_names=feature_names)
        if mask is not None:
            X = X[mask]
        return X if n is None else X[:n]

    def observers_from_mask(self, shard: int, mask: np.ndarray,
                            observer_names: Optional[Sequence[str]] = None,
                            n: Optional[int] = None) -> np.ndarray:
        G = self.observers(shard=shard, n=None, observer_names=observer_names)
        if mask is not None:
            G = G[mask]
        return G if n is None else G[:n]

    def weight_from_mask(self, shard: int, mask: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        w = self.weight_vector(shard=shard, n=None)
        if mask is not None:
            w = w[mask]
        return w if n is None else w[:n]

    def iter_features(self, shard: int = 0, batch_size: int = 8192,
                      feature_names: Optional[Sequence[str]] = None):
        X = self.features(shard=shard, n=None, feature_names=feature_names)
        N = len(X)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for i in range(0, N, batch_size):
            yield X[i:i + batch_size]

    # -------- NEW: view helpers (mask computation & minimal shard reuse) --------
    def load_selection_shard(self, shard: int) -> ak.Array:
        return self[shard]

    def compute_mask(self, selection_name: str, selection_fn, shard: int,
                     observer_names: Optional[Sequence[str]] = None) -> np.ndarray:
        if selection_name not in self._mask_cache:
            self._mask_cache[selection_name] = {}
        if shard in self._mask_cache[selection_name]:
            return self._mask_cache[selection_name][shard]

        G = self.observers(shard=shard, n=None, observer_names=observer_names)
        names = list(observer_names) if observer_names is not None else (self.observer_names or [])
        mask = selection_fn(G, names)
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 1:
            raise ValueError(f"Selection '{selection_name}' did not return a 1D boolean mask.")
        if len(mask) != len(G):
            raise ValueError(f"Selection '{selection_name}' mask length mismatch: {len(mask)} vs {len(G)}.")

        self._mask_cache[selection_name][shard] = mask
        return mask

    # --------------------------- internals ----------------------------
    def _wrap_index(self, idx: int) -> int:
        if not isinstance(idx, int):
            raise TypeError("RDataLoader indices must be integers")
        n = len(self)
        return idx % n

    def _make_file_splits(self, files: List[str], n_split: int) -> List[List[str]]:
        if n_split <= 1:
            return [files]
        splits: List[List[str]] = [[] for _ in range(n_split)]
        for i, f in enumerate(files):
            splits[i % n_split].append(f)
        return splits

    def _event_bounds(self, n_events: int, n_split: int) -> List[tuple]:
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
                return t.arrays(expressions=use_branches, library="ak")
        else:
            with uproot.open(files[0]) as rf0:
                t0 = rf0[self.tree_name]
                available0 = set(t0.keys())
                use_branches = self._filter_branches(available0, self._requested_branches)
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

    def view(
        self,
        name: str,
        selection_fn,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        selection_feature_names: Optional[Sequence[str]] = None,
    ) -> 'SelectionView':
        return SelectionView(
            base=self,
            name=name,
            selection_fn=selection_fn,
            feature_names=feature_names,
            observer_names=observer_names,
            selection_feature_names=selection_feature_names,
        )


# -----------------------------
# Minimal in-memory test (no file I/O)
# -----------------------------
if __name__ == "__main__":
    # Build a tiny awkward shard in memory
    N = 8
    ar = ak.Array({
        "f1": np.linspace(0, 1, N).astype(np.float32),
        "o1": np.arange(N).astype(np.int32),
        "weight": np.ones(N, dtype=np.float32) * 2.0,
        "a": np.linspace(1.0, 2.0, N).astype(np.float32),
        "b": np.linspace(0.5, 1.5, N).astype(np.float32),
    })

    # Construct a loader without touching I/O: allocate then set required fields
    dummy = object.__new__(RDataLoader)  # bypass __init__
    # minimally required public config
    dummy.tree_name = "Events"
    dummy.selection = None
    dummy.strict_branches = False
    dummy.splitting_strategy = "files"
    dummy.n_split = 1
    dummy.feature_names = ["f1"]
    dummy.observer_names = ["o1"]
    dummy.weight = "weight"
    dummy.weight_branches = []
    # internal caches / lists used by helpers
    dummy._all_files = ["dummy.root"]
    dummy._requested_branches = ["f1", "o1", "weight", "a", "b"]
    dummy._file_splits = [dummy._all_files]
    dummy._arr_cache = {0: ar}
    dummy._mask_cache = {}

    print("\n[TEST] materialize order & defaults")
    F, O, W = dummy.materialize(shard=0, what="fow")
    print(" shapes:", F.shape, O.shape, W.shape, "| W[:3] =", W[:3])

    W_only, = dummy.materialize(shard=0, what="w")
    print(" w only:", W_only[:5])

    print("\n[TEST] callable weight")
    def wfn(d: dict[str, np.ndarray]) -> np.ndarray:
        # simple function of branches ('a','b')
        return 3.0 * d["a"] + 2.0 * d["b"]

    dummy.weight = wfn
    dummy.weight_branches = ["a", "b"]
    Wc, Fo = dummy.materialize(shard=0, what="wf")
    print(" callable W[:3] =", Wc[:3], "| f[:3] =", Fo[:3, 0])

    print("\n[TEST] list-of-branches product")
    dummy.weight = ["weight", "a"]
    Wp, = dummy.materialize(shard=0, what="w")
    print(" product W[:3] =", Wp[:3])

    print("\n[TEST] ordering preserved")
    O2, W2, F2 = dummy.materialize(shard=0, what="owf")
    print(" order ok:", O2.shape, W2.shape, F2.shape)

