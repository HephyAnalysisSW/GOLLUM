from __future__ import annotations

import numpy as np


class RandomSplitter:
    def __init__(self, fraction: float = 0.5, seed: int = 0):
        self.fraction = float(fraction)
        self.seed = int(seed)
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"RandomSplitter: fraction must satisfy 0 < fraction <= 1, got {self.fraction}")

    def mask(self, size: int, shard: int = 0) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(shard))
        return rng.random(int(size)) < self.fraction
