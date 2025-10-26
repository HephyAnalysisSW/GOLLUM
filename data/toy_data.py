# File: data/toy_data.py
from __future__ import annotations
import sys
import numpy as np

# project roots
sys.path.insert(0, '..')

# --- Toy in-memory loaders for debugging TFMC --------------------------------

import numpy as _np

# Adjustable knobs
_TOY_N_EVENTS   = 10_000     # total events per class
_TOY_STEP_X0    = 0.5        # location of the step
_TOY_W_LOW      = 1.0        # weight for x < STEP_X0 in the step sample
_TOY_W_HIGH     = 3.0        # weight for x >= STEP_X0 in the step sample
_TOY_SEED       = 1          # RNG seed

class _ArrayLoader:
    """
    Minimal in-memory loader with the same front-end API used by training:
      - feature_names = ["x"]
      - observer_names = ["weight"]
      - __len__ -> 1 shard
      - features(...), observers(...), weight_vector(...), materialize(...)
    """
    def __init__(self, X: _np.ndarray, W: _np.ndarray):
        assert X.ndim == 2 and X.shape[1] == 1, "X must be (N,1) with column 'x'."
        assert W.ndim == 1 and len(W) == len(X), "W must be (N,) and match X rows."
        self._X = X.astype(_np.float32, copy=False)
        self._W = W.astype(_np.float32, copy=False)
        self._feature_names  = ["x"]
        self._observer_names = ["weight"]
        # match big loaders: default weight is the "weight" observer branch
        self.weight = "weight"

    def __len__(self) -> int:
        return 1  # single shard

    @property
    def feature_names(self):
        return list(self._feature_names)

    @property
    def observer_names(self):
        return list(self._observer_names)

    # --- minimal helpers to mirror main loader API ---
    def features(self, shard: int = 0, n: int | None = None):
        if shard != 0:
            raise IndexError("Toy loader has a single shard: shard must be 0.")
        X = self._X
        return X if n is None else X[:n]

    def observers(self, shard: int = 0, n: int | None = None):
        if shard != 0:
            raise IndexError("Toy loader has a single shard: shard must be 0.")
        # observers matrix with one column "weight"
        G = self._W[:, None]
        return G if n is None else G[:n]

    def weight_vector(self, shard: int = 0, n: int | None = None):
        if shard != 0:
            raise IndexError("Toy loader has a single shard: shard must be 0.")
        w = self._W
        return w if n is None else w[:n]

    def materialize(self, shard: int = 0, what: str = "fo", n: int | None = None):
        """
        Return any ordered combo of Features/Observers/Weights per `what`:
          'f','o','w' — e.g. 'fw', 'fow', 'wof' etc.
        """
        if shard != 0:
            raise IndexError("Toy loader has a single shard: shard must be 0.")
        outs: list[_np.ndarray] = []
        for ch in what:
            if ch == 'f':
                outs.append(self.features(shard=shard, n=None))
            elif ch == 'o':
                outs.append(self.observers(shard=shard, n=None))
            elif ch == 'w':
                outs.append(self.weight_vector(shard=shard, n=None))
            else:
                raise ValueError(f"materialize: unknown spec letter '{ch}' (allowed: 'f','o','w').")
        if n is not None:
            outs = [a[:n] for a in outs]
        return tuple(outs)

# Build toy datasets
_rng = _np.random.default_rng(_TOY_SEED)

# Uniform sample: x ~ U[0,1], weight = 1
_x_uniform = _rng.uniform(0.0, 1.0, size=_TOY_N_EVENTS).astype(_np.float64)
_w_uniform = _np.ones_like(_x_uniform, dtype=_np.float64)

# Step-weight sample: same x ~ U[0,1], but weights jump at x0
_x_step = _rng.uniform(0.0, 1.0, size=_TOY_N_EVENTS).astype(_np.float64)
_w_step = _np.where(_x_step < _TOY_STEP_X0, _TOY_W_LOW, _TOY_W_HIGH).astype(_np.float64)

# Shape to (N,1) for features
_X_uniform = _x_uniform.reshape(-1, 1)
_X_step    = _x_step.reshape(-1, 1)

# Public loaders (use these names in YAML)
toy_uniform = _ArrayLoader(_X_uniform, _w_uniform)
toy_step    = _ArrayLoader(_X_step,    _w_step)

# Public export
__all__ = ["toy_uniform", "toy_step"]

