# SelectionView.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Callable, Union, Iterable

import numpy as np
import awkward as ak  # needed to read raw arrays from base shard

SelectionFnView = Callable[[np.ndarray, Sequence[str]], np.ndarray]
SelectionLikeView = Union[SelectionFnView, Sequence[SelectionFnView], None]

class SelectionView:
    """
    First-class loader view over a base RDataLoader.

    - selection_fn can be:
        * None                -> no additional selection
        * a single callable   -> one selection
        * a sequence of callables -> multiple selections applied consecutively

      Each selection_fn(G_sel, names) -> 1D boolean mask (same length as shard), or None for no selection.

    - feature_names / observer_names: optional per-view overrides
    - selection_feature_names: observer names used to compute mask (defaults to base.observer_names)

    Weight semantics (simplified & explicit):
      * View 'weight' is Optional[List[str]] of branch names to multiply.
      * If weight is None  -> inherit base loader weights as-is.
      * If weight is []    -> replaced by ones (rare, but defined).
      * If weight is ['a','b',...] -> replaced by product a*b*...

    No implicit multiplication with base weights.
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

        # Keep original attribute for backwards compatibility
        self.selection_fn = selection_fn

        # Normalize selection(s) to a list of callables
        if selection_fn is None:
            self._selection_fns: list[SelectionFnView] = []
        else:
            if callable(selection_fn):
                self._selection_fns = [selection_fn]  # type: ignore[list-item]
            else:
                self._selection_fns = list(selection_fn)  # type: ignore[arg-type]

        self._feature_names = list(feature_names) if feature_names is not None else None
        self._observer_names = list(observer_names) if observer_names is not None else None
        self._sel_feats = list(selection_feature_names) if selection_feature_names is not None else None
        self._mask_cache: dict[int, np.ndarray] = {}

        # weight override list or None (inherit)
        self._w_override = None if weight is None else list(weight)

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
        Reconfigure this view's feature/observer names and make sure the BASE
        loader is updated to request the necessary branches. Must be called
        before any shard was read (base cache or view mask cache); otherwise raises.

        Returns self to allow chaining.
        """
        # If either the base has loaded data or this view has built a mask -> forbid
        if getattr(self.base, "_arr_cache", None) and len(self.base._arr_cache):
            raise RuntimeError("SelectionView.setFeatures: base data already materialized; "
                               "call only right after initialization.")
        if getattr(self, "_mask_cache", None) and len(self._mask_cache):
            raise RuntimeError("SelectionView.setFeatures: selection mask already built; "
                               "call only right after initialization.")

        # Update own overrides (if provided)
        if feature_names is not None:
            self._feature_names = list(feature_names)
        if observer_names is not None:
            self._observer_names = list(observer_names)

        # Propagate to base so its requested-branch list is consistent
        # (Use the view's current names if caller passed None.)
        self.base.setFeatures(
            feature_names=self._feature_names if self._feature_names is not None else self.base.feature_names,
            observer_names=self._observer_names if self._observer_names is not None else self.base.observer_names,
            extra_branches=extra_branches,
        )

        return self

    def _mask(self, shard: int) -> np.ndarray:
        """
        Compute (or fetch cached) boolean mask for this shard.

        - If no selection functions are configured, return an all-True mask.
        - Otherwise, apply each selection in self._selection_fns consecutively
          (logical AND of all masks).
        """
        if shard in self._mask_cache:
            return self._mask_cache[shard]

        # No selection -> identity mask
        if not self._selection_fns:
            ar = self.base.load_selection_shard(shard)
            m = np.ones(len(ar), dtype=bool)
            self._mask_cache[shard] = m
            return m

        sel_feats = self._sel_feats if self._sel_feats is not None else (self.base.observer_names or [])
        if not sel_feats:
            raise RuntimeError(f"SelectionView[{self.name}]: no selection features/observers to compute mask.")

        _, G_sel = self.base.features_and_observers(
            shard=shard, n=None,
            feature_names=self.base.feature_names,
            observer_names=sel_feats
        )

        # start from all-True and AND all selection masks
        m = np.ones(len(G_sel), dtype=bool)
        for i, fn in enumerate(self._selection_fns):
            if fn is None:
                continue
            m_i = fn(G_sel, sel_feats)
            m_i = np.asarray(m_i)
            if m_i.dtype != bool or m_i.ndim != 1 or len(m_i) != len(G_sel):
                raise RuntimeError(
                    f"SelectionView[{self.name}]: selection_fn[{i}] must return a 1D boolean mask "
                    f"of length {len(G_sel)}."
                )
            m &= m_i

        self._mask_cache[shard] = m
        return m

    def _weight_vector_view(self, shard: int) -> np.ndarray:
        """
        Return the weight vector for this view.

        - If no override is provided -> use the base loader's weight as-is.
        - If an override list is provided -> REPLACE base weight with product of those branches.
        """
        if self._w_override is None:
            return self.base.weight_vector(shard=shard, n=None)

        # replacement path
        ar = self.base.load_selection_shard(shard)
        if len(self._w_override) == 0:
            return np.ones(len(ar), dtype=np.float32)

        missing = [bn for bn in self._w_override if bn not in ar.fields]
        if missing:
            raise KeyError(
                f"{self.name}: override weight branches missing: {missing}. "
                f"Include them in the base loader's 'branches'."
            )
        O = self.base.observers(shard=shard, n=None, observer_names=self._w_override).astype(np.float32, copy=False)
        rw = np.prod(O, axis=1)
        return rw

    def features(self, shard: int = 0, n: Optional[int] = None) -> np.ndarray:
        names = self.feature_names
        if not names:
            raise RuntimeError(f"SelectionView[{self.name}]: no feature_names configured.")
        X = self.base.features(shard=shard, n=None, feature_names=names)
        #print(f"[SeletionView.features] (1) X {X.shape} n {n}")                          
        m = self._mask(shard)
        X = X[m]
        #print(f"[SeletionView.features] (2) X {X.shape} n {n}")                          
        return X if (n is None) else X[:n]

    def features_and_observers(self, shard: int = 0, n: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        fnames = self.feature_names
        onames = self.observer_names
        if not fnames:
            raise RuntimeError(f"SelectionView[{self.name}]: no feature_names configured.")
        if not onames:
            raise RuntimeError(f"SelectionView[{self.name}]: no observer_names configured.")
        X, G = self.base.features_and_observers(shard=shard, n=None, feature_names=fnames, observer_names=onames)
        m = self._mask(shard)
        X = X[m]; G = G[m]
        if n is not None:
            X = X[:n]; G = G[:n]
        return X, G

    def materialize(self, shard: int = 0, what: str = "fo", n: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        m = self._mask(shard)
        outs: list[np.ndarray] = []
        for ch in what:
            if ch == 'f':
                X = self.base.features(shard=shard, n=None, feature_names=self.feature_names)
                outs.append(X[m])
            elif ch == 'o':
                G = self.base.observers(shard=shard, n=None, observer_names=self.observer_names)
                outs.append(G[m])
            elif ch == 'w':
                w = self._weight_vector_view(shard=shard)
                outs.append(w[m])
            else:
                raise ValueError(f"materialize(view): unknown spec letter '{ch}' (allowed: 'f','o','w').")
        if n is not None:
            outs = [arr[:n] for arr in outs]
        #print(f"[SelectionView.materialize] outs {outs[0].shape}")
        return tuple(outs)

    def __str__(self) -> str:
        try:
            base_name = getattr(self.base, "name", None)
        except Exception:
            base_name = None
        base_name = base_name or "RDataLoader"

        feats = self.feature_names
        obs = self.observer_names

        if self._w_override is None:
            weight_info = "inherit base"
        else:
            if len(self._w_override) == 0:
                weight_info = "ones"
            else:
                weight_info = " * ".join(self._w_override)

        sel_count = len(self._selection_fns)
        sel_names = [getattr(fn, "__name__", "<lambda>") for fn in self._selection_fns]

        lines = [
            f"SelectionView('{self.name}')",
            f"  base: {base_name}",
            f"  features ({len(feats)}): {feats}",
            f"  observers ({len(obs)}): {obs}",
            f"  weight: {weight_info}",
            f"  selections: {sel_count} function(s) applied consecutively"
            + (f", names={sel_names}" if sel_names else ""),
        ]
        return "\n".join(lines)

