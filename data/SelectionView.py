from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import awkward as ak  # needed to read raw arrays from base shard

class SelectionView:
    """
    First-class loader view over a base RDataLoader.

    - selection_fn(G_sel, names) -> 1D boolean mask (same length as shard), or None for no selection
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
        selection_fn: Optional[callable] = None,
        feature_names: Optional[Sequence[str]] = None,
        observer_names: Optional[Sequence[str]] = None,
        selection_feature_names: Optional[Sequence[str]] = None,
        weight: Optional[Sequence[str]] = None,
    ):
        self.base = base
        self.name = name
        self.selection_fn = selection_fn
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

    def _mask(self, shard: int) -> np.ndarray:
        """Compute (or fetch cached) boolean mask for this shard. If selection_fn is None, return all-True."""
        if shard in self._mask_cache:
            return self._mask_cache[shard]

        # No selection -> identity mask
        if self.selection_fn is None:
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
        m = self.selection_fn(G_sel, sel_feats)
        m = np.asarray(m)
        if m.dtype != bool or m.ndim != 1 or len(m) != len(G_sel):
            raise RuntimeError(
                f"SelectionView[{self.name}]: selection_fn must return a 1D boolean mask of length {len(G_sel)}."
            )
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
        m = self._mask(shard)
        X = X[m]
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
        return tuple(outs)

    def __str__(self) -> str:
        sel = getattr(self, "selection_fn", None)
        sel_name = getattr(sel, "__name__", None) if sel is not None else None
        try:
            base_name = getattr(self.base, "name", None)
        except Exception:
            base_name = None
        parts = [
            "SelectionView(",
            f"  name='{self.name}', base={base_name or 'RDataLoader'},",
            f"  features={len(self.feature_names or [])}, observers={len(self.observer_names or [])},",
            f"  weight_override={'inherit' if self._w_override is None else self._w_override},",
            f"  selection={sel_name or ('<lambda>' if sel else 'None')}",
            ")",
        ]
        return " ".join(parts)

