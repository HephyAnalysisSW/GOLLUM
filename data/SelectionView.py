# SelectionView.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Callable, Union

import numpy as np

SelectionFnView = Callable[[np.ndarray, Sequence[str]], Union[np.ndarray, None]]
SelectionLikeView = Union[SelectionFnView, Sequence[SelectionFnView], None]


class SelectionView:
    """
    First-class loader view over a base RDataLoader.

    - selection_fn can be:
        * None                -> no additional selection
        * a single callable   -> one selection
        * a sequence of callables -> multiple selections applied consecutively

      Each selection_fn(G_sel, names) -> 1D boolean mask (same length as shard),
      or None for "no additional cut".

    - feature_names / observer_names: optional per-view overrides
    - selection_feature_names: observer names used to compute mask (defaults to base.observer_names)

    Weight semantics (explicit):
      * weight is Optional[List[str]] of branch names to multiply.
      * If weight is None  -> inherit base loader weights as-is.
      * If weight is []    -> replaced by ones.
      * If weight is ['a','b',...] -> replaced by product a*b*...

    No implicit multiplication with base weights.
    No awkward-based selection functions here (views can be many; keep it numpy).
    """

    def __init__(
        self,
        base,
        name: str,
        selection_fn: SelectionLikeView = None,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        selection_feature_names: Optional[Sequence[str]] = None,
        weight: Optional[Sequence[str]] = None,
    ):
        self.base = base
        self.name = name

        # Keep original attribute for backwards compatibility / introspection
        self.selection_fn = selection_fn

        # Normalize selection(s) to a list of callables
        if selection_fn is None:
            self._selection_fns: list[SelectionFnView] = []
        else:
            self._selection_fns = [selection_fn] if callable(selection_fn) else list(selection_fn)

        self._feature_names = list(feature_names) if feature_names is not None else None
        self._observer_names = list(observer_names) if observer_names is not None else None
        self._sel_feats = list(selection_feature_names) if selection_feature_names is not None else None

        # single-shard mask cache (prevents unbounded growth across shards)
        self._mask_cache_shard: Optional[int] = None
        self._mask_cache: Optional[np.ndarray] = None

        # weight override list or None (inherit)
        self._w_override = None if weight is None else list(weight)

        # Inform base which branches we will need (union-add, no renaming)
        if getattr(self.base, "_cache_shard", None) is not None or getattr(self.base, "_cache_ar", None) is not None:
            raise RuntimeError("SelectionView: base already materialized; create views only right after initialization.")

        need = []
        if self._feature_names is not None:
            need += self._feature_names
        if self._observer_names is not None:
            need += self._observer_names
        if self._sel_feats is not None:
            need += self._sel_feats
        if self._w_override:
            need += self._w_override

        if need:
            curr = list(getattr(self.base, "_requested_branches", []) or [])
            seen = set(curr)
            for b in need:
                if b not in seen:
                    seen.add(b)
                    curr.append(b)
            self.base._requested_branches = curr
            self.base._use_branches = self.base._filter_branches(self.base._available0, self.base._requested_branches)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names if self._feature_names is not None else list(self.base.feature_names or [])

    @property
    def observer_names(self) -> list[str]:
        return self._observer_names if self._observer_names is not None else list(self.base.observer_names or [])

    def setFeatures(
        self,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        extra_branches: Optional[Sequence[str]] = None,
    ) -> "SelectionView":
        """
        Reconfigure this view's feature/observer names and ensure the BASE
        loader requests the necessary branches. Must be called before any shard
        was read (base cache or view mask cache); otherwise raises.

        Returns self to allow chaining.
        """
        if getattr(self.base, "_cache_shard", None) is not None or getattr(self.base, "_cache_ar", None) is not None:
            raise RuntimeError("SelectionView.setFeatures: base already materialized; call only right after initialization.")
        if self._mask_cache_shard is not None:
            raise RuntimeError("SelectionView.setFeatures: selection mask already built; call only right after initialization.")

        if feature_names is not None:
            self._feature_names = list(feature_names)
        if observer_names is not None:
            self._observer_names = list(observer_names)

        need = []
        if self._feature_names is not None:
            need += self._feature_names
        if self._observer_names is not None:
            need += self._observer_names
        if self._sel_feats is not None:
            need += self._sel_feats
        if self._w_override:
            need += self._w_override
        if extra_branches:
            need += list(extra_branches)

        if need:
            curr = list(getattr(self.base, "_requested_branches", []) or [])
            seen = set(curr)
            for b in need:
                if b not in seen:
                    seen.add(b)
                    curr.append(b)
            self.base._requested_branches = curr
            self.base._use_branches = self.base._filter_branches(self.base._available0, self.base._requested_branches)

        return self

    def clear_cache(self) -> None:
        self._mask_cache_shard = None
        self._mask_cache = None

    def close(self) -> None:
        self.clear_cache()

    def _mask(self, shard: int) -> np.ndarray:
        """
        Compute (or fetch cached) boolean mask for this shard.

        - If no selection functions are configured, returns all-True mask.
        - Otherwise applies each selection fn and ANDs masks.
        """
        # If you truly want "no mask caching", delete this if-block.
        if self._mask_cache_shard == shard and self._mask_cache is not None:
            return self._mask_cache

        # No selection -> identity
        if not self._selection_fns:
            n = len(self.base.load_selection_shard(shard))
            m = np.ones(n, dtype=bool)
            self._mask_cache_shard, self._mask_cache = shard, m
            return m

        sel_feats = self._sel_feats if self._sel_feats is not None else (self.base.observer_names or [])
        if not sel_feats:
            raise RuntimeError(f"SelectionView[{self.name}]: no selection features/observers to compute mask.")

        # Only observers are needed for the selection; avoid fetching features.
        G_sel = self.base.observers(shard=shard, n=None, observer_names=sel_feats)

        m = np.ones(len(G_sel), dtype=bool)
        for i, fn in enumerate(self._selection_fns):
            if fn is None:
                continue
            m_i = fn(G_sel, sel_feats)
            if m_i is None:
                continue
            m_i = np.asarray(m_i)
            if m_i.dtype != bool or m_i.ndim != 1 or len(m_i) != len(G_sel):
                raise RuntimeError(
                    f"SelectionView[{self.name}]: selection_fn[{i}] must return a 1D boolean mask of length {len(G_sel)}."
                )
            m &= m_i

        self._mask_cache_shard, self._mask_cache = shard, m
        return m

    def _weight_vector_view(self, shard: int) -> np.ndarray:
        """
        Return the weight vector for this view.

        - If no override is provided -> use the base loader's weight as-is.
        - If an override list is provided -> REPLACE base weight with product of those branches.
        """
        if self._w_override is None:
            return self.base.weight_vector(shard=shard, n=None)

        ar = self.base.load_selection_shard(shard)
        if len(self._w_override) == 0:
            return np.ones(len(ar), dtype=np.float32)

        # Use base.observers to get weight columns (scalar-only, as before)
        O = self.base.observers(shard=shard, n=None, observer_names=self._w_override).astype(np.float32, copy=False)
        return np.prod(O, axis=1)

    def features(self, shard: int = 0, n: Optional[int] = None) -> np.ndarray:
        names = self.feature_names
        if not names:
            raise RuntimeError(f"SelectionView[{self.name}]: no feature_names configured.")

        if not self._selection_fns:
            X = self.base.features(shard=shard, n=None, feature_names=names)
            return X if n is None else X[:n]

        X = self.base.features(shard=shard, n=None, feature_names=names)
        X = X[self._mask(shard)]
        return X if n is None else X[:n]

    def features_and_observers(self, shard: int = 0, n: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        fnames = self.feature_names
        onames = self.observer_names
        if not fnames:
            raise RuntimeError(f"SelectionView[{self.name}]: no feature_names configured.")
        if not onames:
            raise RuntimeError(f"SelectionView[{self.name}]: no observer_names configured.")

        if not self._selection_fns:
            X, G = self.base.features_and_observers(shard=shard, n=None, feature_names=fnames, observer_names=onames)
            if n is not None:
                X = X[:n]
                G = G[:n]
            return X, G

        X, G = self.base.features_and_observers(shard=shard, n=None, feature_names=fnames, observer_names=onames)
        m = self._mask(shard)
        X = X[m]
        G = G[m]
        if n is not None:
            X = X[:n]
            G = G[:n]
        return X, G

    def materialize(self, shard: int = 0, what: str = "fo", n: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        outs: list[np.ndarray] = []

        if not self._selection_fns:
            for ch in what:
                if ch == "f":
                    X = self.base.features(shard=shard, n=None, feature_names=self.feature_names)
                    outs.append(X if n is None else X[:n])
                elif ch == "o":
                    G = self.base.observers(shard=shard, n=None, observer_names=self.observer_names)
                    outs.append(G if n is None else G[:n])
                elif ch == "w":
                    w = self._weight_vector_view(shard=shard)
                    outs.append(w if n is None else w[:n])
                else:
                    raise ValueError("materialize(view): unknown spec letter (allowed: 'f','o','w').")
            return tuple(outs)

        m = self._mask(shard)
        for ch in what:
            if ch == "f":
                X = self.base.features(shard=shard, n=None, feature_names=self.feature_names)
                X = X[m]
                outs.append(X if n is None else X[:n])
            elif ch == "o":
                G = self.base.observers(shard=shard, n=None, observer_names=self.observer_names)
                G = G[m]
                outs.append(G if n is None else G[:n])
            elif ch == "w":
                w = self._weight_vector_view(shard=shard)
                w = w[m]
                outs.append(w if n is None else w[:n])
            else:
                raise ValueError("materialize(view): unknown spec letter (allowed: 'f','o','w').")

        return tuple(outs)

    def __str__(self) -> str:
        feats = self.feature_names
        obs = self.observer_names

        if self._w_override is None:
            weight_info = "inherit base"
        elif len(self._w_override) == 0:
            weight_info = "ones"
        else:
            weight_info = " * ".join(self._w_override)

        sel_feats = self._sel_feats if self._sel_feats is not None else (self.base.observer_names or [])
        sel_names = [getattr(fn, "__name__", "<lambda>") for fn in self._selection_fns]

        lines = [
            f"SelectionView('{self.name}')",
            f"  features ({len(feats)}): {feats}",
            f"  observers ({len(obs)}): {obs}",
            f"  selection_features: {list(sel_feats) if sel_feats else []}",
            f"  weight: {weight_info}",
            f"  selections: {len(self._selection_fns)} function(s)"
            + (f", names={sel_names}" if sel_names else ""),
        ]
        return "\n".join(lines)


# -----------------------------
# Usage / micro-test
# -----------------------------
if __name__ == "__main__":
    import time

    def rss_mb():
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        return float("nan")

    from samples_RunII import TTLep_pow_2016

    base = TTLep_pow_2016  # already configured in your framework
    base.set_n_split(10)

    # Pick selection features (prefer observers; else fall back to features)
    sel_feats = list(base.observer_names or base.feature_names or [])
    if not sel_feats:
        raise RuntimeError("Example needs base.observer_names or base.feature_names to be configured.")

    # A simple numpy-level selection on the first selection feature
    def cut_pos(G, names):
        return G[:, 0] > 0

    v_proxy = SelectionView(base=base, name="proxy")  # no selection, pure proxy (no extra copies)
    v_cut = SelectionView(
        base=base,
        name="cut_pos",
        selection_fn=cut_pos,
        selection_feature_names=sel_feats[:1],
    )
    v_ones = SelectionView(base=base, name="ones_weight", weight=[])

    print(base)
    print(v_proxy)
    print(v_cut)
    print(v_ones)

    n_shards = min(len(base))
    print(f"RSS start: {rss_mb():.1f} MB")
    print("-" * 110)
    print("shard  base_ms  proxy_ms  cut_ms  ones_ms   n_base   n_cut   rss_MB  proxy_X_same_id")
    print("-" * 110)

    for shard in range(n_shards):
        t0 = time.perf_counter()
        _ = base[shard]  # load shard once
        base_ms = 1e3 * (time.perf_counter() - t0)

        t1 = time.perf_counter()
        Xp, wp = v_proxy.materialize(shard, "fw")
        proxy_ms = 1e3 * (time.perf_counter() - t1)

        t2 = time.perf_counter()
        Xc, wc = v_cut.materialize(shard, "fw")
        cut_ms = 1e3 * (time.perf_counter() - t2)

        t3 = time.perf_counter()
        X1, w1 = v_ones.materialize(shard, "fw")
        ones_ms = 1e3 * (time.perf_counter() - t3)

        # proxy has no selection => should return base.features array object (same id) for same names
        Xb = base.features(shard, feature_names=v_proxy.feature_names)
        same_id = (id(Xp) == id(Xb))

        print(f"{shard:5d}  {base_ms:7.1f}  {proxy_ms:8.1f}  {cut_ms:6.1f}  {ones_ms:7.1f}  "
              f"{len(Xb):7d}  {len(Xc):6d}  {rss_mb():7.1f}  {same_id}")

    # Keep base lifecycle explicit (evict caches / close files)
    base.close()
    print("-" * 110)
    print(f"RSS end: {rss_mb():.1f} MB")

