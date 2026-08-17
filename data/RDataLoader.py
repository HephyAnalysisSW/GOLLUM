# RDataLoader.py
"""
RDataLoader — lightweight ROOT/UPROOT data loader with group-aware branch handling.
"""

from __future__ import annotations
import os
import glob
import time
from typing import Callable, List, Optional, Sequence, Union, Tuple

import numpy as np
import awkward as ak
from awkward import types as aktypes
import uproot

import SelectionView

# -----------------------------------------------------------------------------
# Module-level debug switch (timing + memory prints)
# -----------------------------------------------------------------------------
VERBOSE = False

PathLike = Union[str, os.PathLike]

SelectionAkFn = Callable[[ak.Array], Union[ak.Array, np.ndarray]]
SelectionNpFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Union[np.ndarray, ak.Array]]
SelectionItem = Union[str, SelectionAkFn, SelectionNpFn]
SelectionLike = Union[SelectionItem, Sequence[SelectionItem], None]


def _rss_mb() -> float:
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _ak_mb(ar: ak.Array) -> float:
    # Awkward 2.6.x compatible payload-bytes estimate
    return sum(buf.nbytes for buf in ak.to_buffers(ar)[2].values()) / 1024.0 / 1024.0


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
        weight_rescale: float = None,
    ) -> None:
        self.tree_name = tree_name
        self.selection = selection
        self.strict_branches = strict_branches
        self.splitting_strategy = splitting_strategy
        self.n_split = max(1, int(n_split))
        self.file_pattern = file_pattern

        # Persist configured features/observers (optional)
        self.feature_names: Optional[List[str]] = list(feature_names) if feature_names else None
        self.observer_names: Optional[List[str]] = list(observer_names) if observer_names else None
        self.weight_branches: List[str] = list(weight_branches) if weight_branches else []
        self.weight_rescale = weight_rescale

        # Resolve file list
        if isinstance(input_paths, (str, os.PathLike)):
            input_paths = [str(input_paths)]
        self.input_paths = input_paths
        files: List[str] = []
        for p in input_paths or []:
            p = os.path.expanduser(str(p))
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, file_pattern))))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"RDataLoader: path not found: {p}")
        if not files:
            raise FileNotFoundError("RDataLoader: no ROOT files found.")
        if max_files is not None:
            files = files[: int(max_files)]
        self._all_files = files

        # Discover branches if not provided
        _requested = list(branches) if branches else None
        if _requested is None:
            with uproot.open(self._all_files[0], object_cache=None, array_cache=None) as f:
                t = f[self.tree_name]
                _requested = list(t.keys())

        # Ensure requested branches include configured features/observers/weights
        if self.feature_names:
            _requested = list(dict.fromkeys(list(_requested) + list(self.feature_names)))
        if self.observer_names:
            _requested = list(dict.fromkeys(list(_requested) + list(self.observer_names)))
        if self.weight_branches:
            _requested = list(dict.fromkeys(list(_requested) + list(self.weight_branches)))
        self._requested_branches = _requested

        # Split layout
        if self.splitting_strategy not in ("files", "events"):
            raise ValueError("splitting_strategy must be 'files' or 'events'")
        self._file_splits: List[List[str]] = self._make_file_splits(self._all_files, self.n_split)

        # Selection storage
        self._selection_items: List[SelectionItem] = []
        self._sel_exprs: List[SelectionAkFn] = []
        self._sel_ak_fns: List[SelectionAkFn] = []
        self._sel_np_fns: List[SelectionNpFn] = []

        # Probe available branches once; used for filtering and later re-filter after setFeatures/addSelection
        with uproot.open(self._all_files[0], object_cache=None, array_cache=None) as f0:
            t0 = f0[self.tree_name]
            self._available0 = set(t0.keys())
        self._use_branches = self._filter_branches(self._available0, self._requested_branches)

        # Persistent file/tree handle (single-file fast path)
        self._open_path: Optional[str] = None
        self._open_file = None
        self._open_tree = None

        # Single-shard cache (awkward + materialized)
        self._cache_shard: Optional[int] = None
        self._cache_ar: Optional[ak.Array] = None
        self._cache_X: Optional[np.ndarray] = None
        self._cache_G: Optional[np.ndarray] = None
        self._cache_w: Optional[np.ndarray] = None
        self._cache_X_names: Optional[Tuple[str, ...]] = None
        self._cache_G_names: Optional[Tuple[str, ...]] = None
        self._cache_w_branches: Optional[Tuple[str, ...]] = None

        # Kept for compatibility (unused here)
        self._mask_cache: dict[str, dict[int, np.ndarray]] = {}

        # Apply constructor selection(s)
        if selection is not None:
            items = [selection] if isinstance(selection, (str,)) or callable(selection) else list(selection)
            for it in items:
                self.addSelection(it)

    def set_max_files( self, max_files ):
        self._all_files = self._all_files[: int(max_files)]
        # Split layout
        if self.splitting_strategy not in ("files", "events"):
            raise ValueError("splitting_strategy must be 'files' or 'events'")
        self._file_splits: List[List[str]] = self._make_file_splits(self._all_files, self.n_split)

    # ----------------------- selections -----------------------
    def addSelection(
        self,
        selection: Union[SelectionAkFn, SelectionNpFn, str],
        required_branches: Optional[Sequence[str]] = None,
        execute_on_numpy: bool = False,
    ) -> "RDataLoader":
        """
        Add an additional selection.

        - Must be called before any shard has been loaded/materialized.
        - required_branches extend requested branches (order-preserving de-dup).
        - string selections: evaluated early on awkward record array fields.
        - callable selections: default evaluated on awkward (execute_on_numpy=False).
          If execute_on_numpy=True: evaluated on materialized (f,o,w) numpy.
        """
        if self._cache_shard is not None or self._cache_ar is not None:
            raise RuntimeError("RDataLoader.addSelection: data already materialized; call only right after initialization.")

        req = list(required_branches) if required_branches else []
        if req:
            curr = list(self._requested_branches or [])
            seen = set(curr)
            for b in req:
                if b not in seen:
                    seen.add(b)
                    curr.append(b)
            self._requested_branches = curr
            self._use_branches = self._filter_branches(self._available0, self._requested_branches)

        if selection not in self._selection_items:
            self._selection_items.append(selection)
        else:
            print(f"Selection {selection} already applied. Skip.")

        if isinstance(selection, str):
            expr = selection.strip()
            code = compile(expr, "<RDataLoader.selection>", "eval")

            def _sel(ar: ak.Array) -> Union[ak.Array, np.ndarray]:
                # locals are required_branches if provided, else all fields
                fields = req if req else list(ar.fields)
                loc = {}
                for b in fields:
                    if b in ar.fields:
                        loc[b] = ar[b]
                    else:
                        if self.strict_branches:
                            raise KeyError(f"Selection requires missing branch '{b}'.")
                        loc[b] = np.zeros(len(ar), dtype=np.float32)
                out = eval(code, {"__builtins__": {}, "np": np, "ak": ak}, loc)
                if isinstance(out, (bool, np.bool_)):
                    return np.full(len(ar), bool(out), dtype=bool)
                if isinstance(out, ak.Array):
                    return out
                return np.asarray(out, dtype=bool)

            setattr(_sel, "_rdata_expr", expr)
            self._sel_exprs.append(_sel)
            return self

        if execute_on_numpy:
            self._sel_np_fns.append(selection)  # type: ignore[arg-type]
        else:
            self._sel_ak_fns.append(selection)  # type: ignore[arg-type]
        return self
    
    def clearSelections(self) -> RDataLoader:
        """
        Clear selections. Must be called before any shard has been loaded/materialized.
        """
        if self._cache_shard is not None or self._cache_ar is not None:
            raise RuntimeError("RDataLoader.addSelection: data already materialized; call only right after initialization.")

        self._selection_items.clear()
        self._sel_exprs.clear()
        self._sel_ak_fns.clear()
        self._sel_np_fns.clear()

        return self

    def set_n_split(self, n_split: int) -> "RDataLoader":
        if self._cache_shard is not None or self._cache_ar is not None:
            raise RuntimeError("RDataLoader.set_n_split: data already materialized; call only right after initialization.")

        self.n_split = max(1, int(n_split))

        if self.splitting_strategy == "files":
            self._file_splits = self._make_file_splits(self._all_files, self.n_split)

        # defensive: make sure no stale shard state survives
        self.clear_cache()
        return self


    # ----------------------- reset features post-init----------------------
    def setFeatures(
        self,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        extra_branches: Optional[Sequence[str]] = None,
    ) -> "RDataLoader":
        if self._cache_shard is not None or self._cache_ar is not None:
            raise RuntimeError("RDataLoader.setFeatures: data already materialized; call only right after initialization.")

        if feature_names is not None:
            self.feature_names = list(feature_names)
        if observer_names is not None:
            self.observer_names = list(observer_names)

        req = list(self._requested_branches or [])
        if self.feature_names:
            req += list(self.feature_names)
        if self.observer_names:
            req += list(self.observer_names)
        if self.weight_branches:
            req += list(self.weight_branches)
        if extra_branches:
            req += list(extra_branches)

        seen = set()
        req_dedup = []
        for b in req:
            if b not in seen:
                seen.add(b)
                req_dedup.append(b)

        self._requested_branches = req_dedup
        self._use_branches = self._filter_branches(self._available0, self._requested_branches)
        self._mask_cache = {}
        return self

    # ---------------------------- public API ----------------------------
    def __len__(self) -> int:
        return len(self._file_splits) if self.splitting_strategy == "files" else self.n_split

    def __getitem__(self, idx: int) -> ak.Array:
        key = idx
        if self._cache_shard == key and self._cache_ar is not None:
            return self._cache_ar

        t0 = time.perf_counter() if VERBOSE else None
        rss0 = _rss_mb() if VERBOSE else None

        # evict single-shard caches
        self._cache_shard = key
        self._cache_ar = None
        self._cache_X = None
        self._cache_G = None
        self._cache_w = None
        self._cache_X_names = None
        self._cache_G_names = None
        self._cache_w_branches = None

        # load shard
        if self.splitting_strategy == "files":
            files = self._file_splits[key]
            if not files:
                raise ValueError("RDataLoader: empty file split (n_split too large for number of files?)")
            ar = self._load_files(files)
        else:
            if len(self._all_files) == 1:
                self._ensure_open(self._all_files[0])
                assert self._open_tree is not None
                n_entries = int(self._open_tree.num_entries)
                lo, hi = self._event_bounds(n_entries, self.n_split)[key]
                ar = self._open_tree.arrays(expressions=self._use_branches, entry_start=lo, entry_stop=hi, library="ak")
            else:
                ar_all = self._load_files(self._all_files)
                n = len(ar_all)
                lo, hi = self._event_bounds(n, self.n_split)[key]
                ar = ar_all[lo:hi]

        # selections: early string exprs
        for sel in self._sel_exprs:
            m = sel(ar)
            m = m if isinstance(m, ak.Array) else np.asarray(m, dtype=bool)
            ar = ar[m]

        # selections: ak callables (default)
        for sel in self._sel_ak_fns:
            m = sel(ar)
            m = m if isinstance(m, ak.Array) else np.asarray(m, dtype=bool)
            ar = ar[m]

        # selections: numpy callables (opt-in)
        if self._sel_np_fns:
            fnames = tuple(self.feature_names or [])
            onames = tuple(self.observer_names or [])
            wbs = tuple(self.weight_branches or [])

            f = self.scalar_branches(ar, fnames) if fnames else np.empty((len(ar), 0), dtype=np.float32)
            o = self.scalar_branches(ar, onames) if onames else np.empty((len(ar), 0), dtype=np.float32)
            if wbs:
                Wcols = self.scalar_branches(ar, wbs).astype(np.float32, copy=False)
                w = np.prod(Wcols, axis=1)
            else:
                w = np.ones(len(ar), dtype=np.float32)

            for fn in self._sel_np_fns:
                m = fn(f, o, w)
                m = ak.to_numpy(m) if isinstance(m, ak.Array) else np.asarray(m, dtype=bool)
                ar = ar[m]
                f = f[m]
                o = o[m]
                w = w[m]

            # cache materialized results produced for selection
            self._cache_X = f if fnames else None
            self._cache_G = o if onames else None
            self._cache_w = w
            self._cache_X_names = fnames if fnames else None
            self._cache_G_names = onames if onames else None
            self._cache_w_branches = wbs

        self._cache_ar = ar

        if VERBOSE:
            dt_ms = 1e3 * (time.perf_counter() - t0)  # type: ignore[operator]
            print(
                f"[RDataLoader] shard={key:3d} n={len(ar):7d} "
                f"ak={_ak_mb(ar):6.1f}MB rss={_rss_mb():7.1f}MB dt={dt_ms:7.1f}ms (rss0={rss0:7.1f}MB)"
            )

        return ar

    def clear_cache(self) -> None:
        self._cache_shard = None
        self._cache_ar = None
        self._cache_X = None
        self._cache_G = None
        self._cache_w = None
        self._cache_X_names = None
        self._cache_G_names = None
        self._cache_w_branches = None
        self._mask_cache.clear()

    def close(self) -> None:
        self.clear_cache()
        if self._open_file is not None:
            self._open_file.close()
        self._open_file = None
        self._open_tree = None
        self._open_path = None

    @property
    def files(self) -> List[str]:
        return list(self._all_files)

    @property
    def branches(self) -> List[str]:
        return list(self._requested_branches or [])

    # ---- helpers for extracting data ----
    def scalar_branches(self, ar: ak.Array, names: Sequence[str]) -> np.ndarray:
        cols = []
        for n in names:
            if n not in ar.fields:
                if self.strict_branches:
                    raise KeyError(f"Requested branch '{n}' not in array fields.")
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
        if name not in ar.fields:
            if self.strict_branches:
                raise KeyError(f"Requested branch '{name}' not in array fields.")
            return ak.Array([[] for _ in range(len(ar))])
        return ar[name]

    # -------- direct feature/observer access ----------
    def observers(self, shard: int = 0, n: Optional[int] = None,
                  observer_names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(observer_names) if observer_names is not None else (self.observer_names or [])
        if not names:
            raise ValueError("No observer names configured. Pass observer_names=... or set observer_names in the constructor.")
        names_t = tuple(names)

        if n is None and self._cache_shard == shard and self._cache_G is not None and self._cache_G_names == names_t:
            return self._cache_G

        ar = self[shard]
        G = self.scalar_branches(ar, names)
        if n is None and self._cache_shard == shard and observer_names is None:
            self._cache_G = G
            self._cache_G_names = names_t
        return G if n is None else G[:n]

    def features(self, shard: int = 0, n: Optional[int] = None,
                 feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(feature_names) if feature_names is not None else (self.feature_names or [])
        if not names:
            raise ValueError("No feature names configured. Pass feature_names=... or set feature_names in the constructor.")
        names_t = tuple(names)

        if n is None and self._cache_shard == shard and self._cache_X is not None and self._cache_X_names == names_t:
            return self._cache_X

        ar = self[shard]
        X = self.scalar_branches(ar, names)
        if n is None and self._cache_shard == shard and feature_names is None:
            self._cache_X = X
            self._cache_X_names = names_t
        return X if n is None else X[:n]

    def features_and_observers(self, shard: int = 0, n: Optional[int] = None,
                               feature_names: Optional[Sequence[str]] = None,
                               observer_names: Optional[Sequence[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        X = self.features(shard=shard, n=None, feature_names=feature_names)
        G = self.observers(shard=shard, n=None, observer_names=observer_names)
        if n is not None:
            X = X[:n]
            G = G[:n]
        return X, G

    # -------- Explicit weight product or ones --------
    def weight_vector(self, shard: int = 0, n: Optional[int] = None) -> np.ndarray:
        if not self.weight_branches:
            ar = self[shard]
            w = np.ones(len(ar), dtype=np.float32)
            return w if n is None else w[:n]

        wbs_t = tuple(self.weight_branches)
        if n is None and self._cache_shard == shard and self._cache_w is not None and self._cache_w_branches == wbs_t:
            return self._cache_w

        ar = self[shard]
        missing = [bn for bn in self.weight_branches if bn not in ar.fields]
        if missing and self.strict_branches:
            raise KeyError(f"Weight branches missing: {missing}. Include them in 'branches'.")
        Wcols = self.scalar_branches(ar, [w for w in self.weight_branches if w not in missing]).astype(np.float32, copy=False)
        w = np.prod(Wcols, axis=1) if Wcols.shape[1] else np.ones(len(ar), dtype=np.float32)

        # rescale the weights
        if self.weight_rescale is not None:
            w*=self.weight_rescale
        if n is None and self._cache_shard == shard:
            self._cache_w = w
            self._cache_w_branches = wbs_t
        return w if n is None else w[:n]

    def materialize(
        self,
        shard: int = 0,
        what: str = "fo",
        n: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
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

    # -------- view helpers --------
    def load_selection_shard(self, shard: int) -> ak.Array:
        return self[shard]

    # --------------------------- internals ----------------------------
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

    def _ensure_open(self, path: str) -> None:
        if self._open_path == path and self._open_tree is not None:
            return
        if self._open_file is not None:
            self._open_file.close()
        self._open_file = uproot.open(path, object_cache=None, array_cache=None)
        self._open_tree = self._open_file[self.tree_name]
        self._open_path = path

    def _load_files(self, files: Sequence[str]) -> ak.Array:
        if len(files) == 1:
            self._ensure_open(files[0])
            assert self._open_tree is not None
            return self._open_tree.arrays(expressions=self._use_branches, library="ak")

        # multi-file shard: concatenate (keeps existing behavior)
        if self._open_file is not None:
            self._open_file.close()
            self._open_file = None
            self._open_tree = None
            self._open_path = None

        return uproot.concatenate(
            {f: self.tree_name for f in files},
            expressions=self._use_branches,
            library="ak",
        )

    def _filter_branches(self, available: set, requested: Sequence[str]) -> List[str]:
        missing = [b for b in requested if b not in available]
        if missing and self.strict_branches:
            raise KeyError(f"Missing branches in tree: {missing}")
        return [b for b in requested if b in available]

    def __str__(self) -> str:
        files = getattr(self, "_all_files", [])
        feat = self.feature_names or []
        obs = self.observer_names or []
        wbs = getattr(self, "weight_branches", []) or []
        weight_expr = "1" if not wbs else " * ".join(wbs)

        sels = []
        for it in (getattr(self, "_selection_items", []) or []):
            if isinstance(it, str):
                sels.append(it)
            else:
                expr = getattr(it, "_rdata_expr", None)
                sels.append(expr if expr is not None else getattr(it, "__name__", repr(it)))

        lines = [
            "RDataLoader(",
            f"  tree_name='{self.tree_name}', splitting='{self.splitting_strategy}', n_split={self.n_split},",
            f"  files={len(files)}" + (f", files='{files}'" if files else ""),
            f"  features ({len(feat)}): {feat}",
            f"  observers ({len(obs)}): {obs}",
            f"  weights (product): {weight_expr} (rescale: {self.weight_rescale})",
            f"  selections ({len(sels)}): {sels}",
            ")",
        ]
        return "\n".join(lines)

    def clone_from_files(self, input_paths: Union[PathLike, Sequence[PathLike]], weight_branches: Sequence[str] = None) -> "RDataLoader":
        branches = list(self._requested_branches) if getattr(self, "_requested_branches", None) is not None else None
        selection = list(getattr(self, "_selection_items", [])) or None
        file_pattern = getattr(self, "file_pattern", "*.root")
        return RDataLoader(
            input_paths=input_paths,
            tree_name=self.tree_name,
            branches=branches,
            selection=selection,
            file_pattern=file_pattern,
            n_split=self.n_split,
            splitting_strategy=self.splitting_strategy,
            strict_branches=self.strict_branches,
            max_files=None,
            feature_names=self.feature_names,
            observer_names=self.observer_names,
            weight_branches=self.weight_branches if weight_branches is None else weight_branches,
            weight_rescale=self.weight_rescale,
        )

    def clone(self) -> "RDataLoader":
        branches = list(self._requested_branches) if getattr(self, "_requested_branches", None) is not None else None
        selection = list(getattr(self, "_selection_items", [])) or None
        file_pattern = getattr(self, "file_pattern", "*.root")
        return RDataLoader(
            input_paths=self.input_paths,
            tree_name=self.tree_name,
            branches=branches,
            selection=selection,
            file_pattern=file_pattern,
            n_split=self.n_split,
            splitting_strategy=self.splitting_strategy,
            strict_branches=self.strict_branches,
            max_files=None,
            feature_names=self.feature_names,
            observer_names=self.observer_names,
            weight_branches=self.weight_branches,
            weight_rescale=self.weight_rescale
        )

    def view(
        self,
        name: str,
        selection_fn: SelectionLike,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        selection_feature_names: Optional[Sequence[str]] = None,
    ) -> "SelectionView.SelectionView":
        return SelectionView.SelectionView(
            base=self,
            name=name,
            selection_fn=selection_fn,
            feature_names=feature_names,
            observer_names=observer_names,
            selection_feature_names=selection_feature_names,
        )


# -----------------------------
# Usage / micro-test
# -----------------------------
if __name__ == "__main__":
    VERBOSE = True

    from samples_RunII import TTLep_pow_2016

    # String selection (early, ak fields)
    TTLep_pow_2016.addSelection(
        "(tr_ttbar_mass >= 1000) & (tr_ttbar_eta > 1)",
        required_branches=["tr_ttbar_mass", "tr_ttbar_eta"],
    )

    # Callable selection (default: ak-based)
    #TTLep_pow_2016.addSelection(
    #    lambda ar: (ar["tr_ttbar_mass"] >= 1000) & (ar["tr_ttbar_eta"] > 1),
    #    required_branches=["tr_ttbar_mass", "tr_ttbar_eta"],
    #)

    # Callable selection on numpy (explicit opt-in)
    # i_mass = TTLep_pow_2016.feature_names.index("tr_ttbar_mass")
    # i_eta  = TTLep_pow_2016.feature_names.index("tr_ttbar_eta")
    # TTLep_pow_2016.addSelection(
    #     lambda f, o, w: (f[:, i_mass] >= 1000) & (f[:, i_eta] > 1),
    #     execute_on_numpy=True,
    # )

    print(TTLep_pow_2016)

    TTLep_pow_2016.set_n_split(20)
    for shard in range(len(TTLep_pow_2016)):
        t0 = time.perf_counter()
        f, o, w = TTLep_pow_2016.materialize(shard, "fow")
        histos = [np.histogram(f[:, i], weights=w) for i in range(f.shape[1])]
        dt = time.perf_counter() - t0
        print(f"[main] shard={shard:2d}  n={f.shape[0]:7d}  dt={dt:7.3f}s  rss={_rss_mb():7.1f}MB  nh={len(histos)}")

    TTLep_pow_2016.close()

