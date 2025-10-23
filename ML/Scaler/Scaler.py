import pickle
import numpy as np
import sys

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

class Scaler:
    def __init__(self):
        # learned stats
        self.feature_means = None          # np.ndarray [F]
        self.feature_variances = None      # np.ndarray [F]

        # meta (for pretty-printing)
        self.selection = None              # str
        self.process   = None              # str
        self.feature_names = None          # list[str]

        # accumulators (internal, not serialized strictly necessary but fine)
        self._sum = None                   # np.ndarray [F]
        self._sqsum = None                 # np.ndarray [F]
        self._wsum = 0.0

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        # Backward compatible re-wrap
        new = cls()
        for k in ["feature_means","feature_variances","selection","process","feature_names","_sum","_sqsum","_wsum"]:
            if hasattr(obj, k):
                setattr(new, k, getattr(obj, k))
        return new

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # --------- new minimal API ----------
    def reset(self):
        self._sum = None
        self._sqsum = None
        self._wsum = 0.0

    def accumulate(self, X: np.ndarray, w: np.ndarray | None = None):
        """Add a chunk of features X (N,F) with optional weights w (N,)."""
        if X is None or len(X) == 0:
            return
        if w is None:
            w = np.ones(len(X), dtype=np.float64)
        else:
            w = w.astype(np.float64, copy=False)

        if self._sum is None:
            F = X.shape[1]
            self._sum = np.zeros(F, dtype=np.float64)
            self._sqsum = np.zeros(F, dtype=np.float64)

        self._sum   += (w[:, None] * X).sum(axis=0)
        self._sqsum += (w[:, None] * (X**2)).sum(axis=0)
        self._wsum  += float(w.sum())

    def finalize(self):
        if self._wsum <= 0:
            raise RuntimeError("No events accumulated. Did you call accumulate()?")
        self.feature_means = self._sum / self._wsum
        self.feature_variances = self._sqsum / self._wsum - self.feature_means**2
        # Drop accumulators to keep object lean
        self._sum = None
        self._sqsum = None
        self._wsum = 0.0

    def fit_from_arrays(self, X: np.ndarray, w: np.ndarray | None = None):
        self.reset()
        self.accumulate(X, w)
        self.finalize()

    def __str__(self):
        sel = '\033[1m' + (self.selection or "(not set)") + '\033[0m'
        proc = '(' + '\033[1m' + (self.process or "not set") + '\033[0m' + ')'
        lines = [f"Scaler: selection {sel} process {proc}"]
        if self.feature_names is not None and self.feature_means is not None:
            for i, name in enumerate(self.feature_names):
                mean = self.feature_means[i]
                var = self.feature_variances[i]
                lines.append(f"{name}: mean={mean:.3f}, variance={var:.3f}")
        return "\n".join(lines)

    def normalize(self, data):
        """Normalize data using the computed (weighted) mean and variance."""
        return (data - self.feature_means) / np.sqrt(self.feature_variances)

