
# RDataLoader.py
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

# A single selection function: ar -> boolean mask (or awkward array mask)
SelectionFn = Callable[[ak.Array], Union[ak.Array, np.ndarray]]
# For the public interface we now allow: None, a single SelectionFn, or a list of SelectionFn
SelectionLike = Union[SelectionFn, Sequence[SelectionFn], None]


class RDataLoader:
    def __init__(
        self,
        input_paths: Union[PathLike, Sequence[PathLike]],
        tree_name: str = "Events",
        branches: Optional[Sequence[str]] = None,
        selection: SelectionLike = None,
        file_pattern: str = "*.root",
        n_split: int = 1,
        splitting_strategy: str = "files",  # "files" or "events"
        strict_branches: bool = False,
        max_files: Optional[int] = None,
        # ---- optional feature/observer names ----
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        # ---- NEW & SIMPLE: explicit weight branches (product). If None/empty -> weights = 1.
        weight_branches: Optional[Sequence[str]] = None,
    ) -> None:
        self.tree_name = tree_name
        # keep the original attribute for backwards-compatibility / introspection
        self.selection = selection
        self.strict_branches = strict_branches
        self.splitting_strategy = splitting_strategy
        self.n_split = max(1, int(n_split))

        # normalize selection(s) to an internal list of callables
        self._selection_fns: List[SelectionFn] = []
        if selection is None:
            self._selection_fns = []
        else:
            # allow single callable or iterable of callables
            if callable(selection):
                self._selection_fns = [selection]  # type: ignore[arg-type]
            else:
                self._selection_fns = list(selection)  # type: ignore[arg-type]

        # Persist configured features/observers (optional)
        self.feature_names: Optional[List[str]] = list(feature_names) if feature_names else None
        self.observer_names: Optional[List[str]] = list(observer_names) if observer_names else None

        # NEW: purely explicit list of weight branches (product)
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

        # Ensure requested branches include weight_branches (if any)
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

    # ----------------------- reset features post-init----------------------
    def setFeatures(
        self,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        extra_branches: Optional[Sequence[str]] = None,
    ) -> "RDataLoader":
        """
        Reconfigure feature/observer names and ensure the requested-branch list
        contains them (plus optional extra_branches). Must be called *before*
        any shard has been loaded; otherwise raises.

        Returns self to allow chaining.
        """
        if getattr(self, "_arr_cache", None) and len(self._arr_cache):
            raise RuntimeError("RDataLoader.setFeatures: data already materialized; "
                               "call only right after initialization.")

        # Update configured names if provided
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if observer_names is not None:
            self.observer_names = list(observer_names)

        # Start from current requested list
        req = list(self._requested_branches or [])

        # Ensure features/observers are present
        if self.feature_names:
            req += list(self.feature_names)
        if self.observer_names:
            req += list(self.observer_names)

        # Always ensure weight branches are included
        if self.weight_branches:
            req += list(self.weight_branches)

        # Optional extra branches
        if extra_branches:
            req += list(extra_branches)

        # De-duplicate while preserving order
        seen = set()
        req_dedup = []
        for b in req:
            if b not in seen:
                seen.add(b)
                req_dedup.append(b)

        self._requested_branches = req_dedup
        # Reset any selection-mask caches (paranoia; should be empty anyway)
        self._mask_cache = {}
        return self

    # ---------------------------- public API ----------------------------
    def __len__(self) -> int:
        return len(self._file_splits) if self.splitting_strategy == "files" else self.n_split

    def __getitem__(self, idx: int) -> ak.Array:
        """Load one shard as an awkward Array, apply all configured selections if present. Result is cached."""
        #key = self._wrap_index(idx)
        key = idx
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

        # apply all base selections consecutively
        for i, sel in enumerate(getattr(self, "_selection_fns", []) or []):
            if sel is None:
                continue
            mask = sel(ar)
            mask = ak.to_numpy(mask) if isinstance(mask, ak.Array) else mask
            if mask is None:
                warnings.warn(f"RDataLoader: selection function {i} returned None; skipping.")
                continue
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
        #print(f"[RDataloader.features] X {X.shape} n {n}") 
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

    # -------- NEW: explicit weight product or ones --------
    def weight_vector(self, shard: int = 0, n: Optional[int] = None) -> np.ndarray:
        """
        Return event weights for this shard.
        - If self.weight_branches is empty -> all ones
        - Else product of listed scalar branches (must exist)
        """
        ar = self[shard]
        if not self.weight_branches:
            w = np.ones(len(ar), dtype=np.float32)
            return w if n is None else w[:n]

        missing = [bn for bn in self.weight_branches if bn not in ar.fields]
        if missing:
            raise KeyError(
                f"Weight branches missing: {missing}. Include them in 'branches' (and usually 'observer_names')."
            )
        Wcols = self.scalar_branches(ar, self.weight_branches).astype(np.float32, copy=False)
        w = np.prod(Wcols, axis=1)
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

        #print(f"[RDataloader.materialize] outputs {outputs[0].shape}") 
        return tuple(outputs)

    # -------- view helpers --------
    def load_selection_shard(self, shard: int) -> ak.Array:
        return self[shard]

    # --------------------------- internals ----------------------------
    #def _wrap_index(self, idx: int) -> int:
    #    if not isinstance(idx, int):
    #        raise TypeError("RDataLoader indices must be integers")
    #    n = len(self)
    #    return idx % n

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

    def __str__(self) -> str:
        files = getattr(self, "_all_files", [])
        first = files[0] if files else None
        feat = self.feature_names or []
        obs = self.observer_names or []
        wbs = getattr(self, "weight_branches", []) or []
        weight_expr = "1" if not wbs else " * ".join(wbs)
        n_sel = len(getattr(self, "_selection_fns", []) or [])
        lines = [
            "RDataLoader(",
            f"  tree_name='{self.tree_name}', splitting='{self.splitting_strategy}', n_split={self.n_split},",
            #f"  files={len(files)}" + (f", first='{os.path.basename(first)}'" if first is not None else ""),
            f"  files={len(files)}" + (f", files='{files}'" if first is not None else ""),
            f"  features ({len(feat)}): {feat}",
            f"  observers ({len(obs)}): {obs}",
            f"  weights (product): {weight_expr}",
            f"  selections: {n_sel} function(s) applied consecutively",
            ")",
        ]
        return "\n".join(lines)

    def clone_from_files(self, input_paths: Union[PathLike, Sequence[PathLike]], weight_branches: Sequence[str] = None) -> "RDataLoader":
        """
        Shallow clone of this loader, but reading from a different file or list of files.

        - Copies all configuration (tree_name, branches, selection(s), features, observers, weights, splits, etc.)
        - Does NOT copy caches; the clone starts "fresh".
        """
        # Use the fully resolved branch list (including weights, added branches, etc.)
        branches = list(self._requested_branches) if getattr(self, "_requested_branches", None) is not None else None

        # Reuse the normalized list of selection functions; if empty, pass None
        selection = list(getattr(self, "_selection_fns", [])) or None

        # Reuse original file_pattern if we have it, else default
        file_pattern = getattr(self, "file_pattern", "*.root")

        clone = RDataLoader(
            input_paths=input_paths,
            tree_name=self.tree_name,
            branches=branches,
            selection=selection,
            file_pattern=file_pattern,
            n_split=self.n_split,
            splitting_strategy=self.splitting_strategy,
            strict_branches=self.strict_branches,
            max_files=None,  # usually not meaningful for a clone; change if you want
            feature_names=self.feature_names,
            observer_names=self.observer_names,
            weight_branches=self.weight_branches if weight_branches is None else weight_branches,
        )
        return clone

    def view(
        self,
        name: str,
        selection_fn: SelectionLike,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        selection_feature_names: Optional[Sequence[str]] = None,
    ) -> 'SelectionView.SelectionView':
        return SelectionView.SelectionView(
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
        "w":  np.ones(N, dtype=np.float32) * 2.0,
        "a":  np.linspace(1.0, 2.0, N).astype(np.float32),
        "b":  np.linspace(0.5, 1.5, N).astype(np.float32),
    })

    dummy = object.__new__(RDataLoader)  # bypass __init__
    dummy.tree_name = "Events"
    dummy.selection = None
    dummy.strict_branches = False
    dummy.splitting_strategy = "files"
    dummy.n_split = 1
    dummy.feature_names = ["f1"]
    dummy.observer_names = ["o1", "w", "a", "b"]
    dummy.weight_branches = ["w", "a"]
    dummy._all_files = ["dummy.root"]
    dummy._requested_branches = ["f1", "o1", "w", "a", "b"]
    dummy._file_splits = [dummy._all_files]
    dummy._arr_cache = {0: ar}
    dummy._mask_cache = {}
    dummy._selection_fns = []  # since we bypassed __init__

    F, O, W = dummy.materialize(shard=0, what="fow")
    print("shapes:", F.shape, O.shape, W.shape, "| W[:3] =", W[:3])

    dummy.weight_branches = []  # -> ones
    W1, = dummy.materialize(shard=0, what="w")
    print("ones W[:3] =", W1[:3])
    print()
    print(dummy)

