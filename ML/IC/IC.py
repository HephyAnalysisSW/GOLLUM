#!/usr/bin/env python
import pickle
import numpy as np
import sys

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

class InclusiveCrosssection:
    """
    Minimal IC accumulator: sums event weights (and counts events).
    The training loop lives outside; this class only accumulates and stores results.
    """

    def __init__(self):
        # learned stats
        self.total_weight = 0.0
        self.total_count  = 0

        # meta (for pretty-printing)
        self.selection = None   # str (optional, extra top-level selection name)
        self.process   = None   # str (process/view name)
        self.note      = None   # optional free text

    # -------- persistence --------
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        # Backward compatible re-wrap
        new = cls()
        for k in ["total_weight", "total_count", "selection", "process", "note"]:
            if hasattr(obj, k):
                setattr(new, k, getattr(obj, k))
        return new

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # -------- minimal API --------
    def reset(self):
        self.total_weight = 0.0
        self.total_count  = 0

    def accumulate(self, w: np.ndarray):
        """
        Add a chunk of event weights (shape: [N]).
        """
        if w is None or len(w) == 0:
            return
        w = np.asarray(w, dtype=np.float64)
        self.total_weight += float(w.sum())
        self.total_count  += int(w.shape[0])

    def finalize(self):
        # nothing to compute beyond sums, kept for symmetry
        pass

    def fit_from_weights(self, w: np.ndarray):
        self.reset()
        self.accumulate(w)
        self.finalize()

    # -------- presentation --------
    def __str__(self):
        sel = (self.selection if self.selection not in (None, "",) else "None")
        proc = (self.process  if self.process  not in (None, "",) else "None")
        return (f"IC — process: \033[1m{proc}\033[0m, selection: \033[1m{sel}\033[0m\n"
                f"  total weighted yield: {self.total_weight:.6f}\n"
                f"  total #events:        {self.total_count:d}")

