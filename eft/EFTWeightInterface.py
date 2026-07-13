from __future__ import annotations

import itertools

import numpy as np


class EFTWeightInterface:
    def __init__(self, parameters):
        self.parameters = list(parameters or [])
        if not self.parameters:
            raise RuntimeError("EFTWeightInterface requires a non-empty operator list.")

        self._combinations = [()]
        self._combinations.extend((op,) for op in self.parameters)
        self._combinations.extend(itertools.combinations_with_replacement(self.parameters, 2))

        self.base_points = []
        for comb in itertools.combinations_with_replacement(self.parameters, 1):
            self.base_points.append({op: comb.count(op) for op in self.parameters})
        for comb in itertools.combinations_with_replacement(self.parameters, 2):
            self.base_points.append({op: comb.count(op) for op in self.parameters})

        self.required_observers = ["EFTWeight_SM"]
        self.required_observers.extend(f"der_{op}" for op in self.parameters)
        # lower triangular matrix in (op0,op1)
        self.required_observers.extend(
            f"der_{op1}_{op0}" for op0, op1 in itertools.combinations_with_replacement(self.parameters, 2)
        )

    @property
    def combinations(self):
        return self._combinations

    def make_weight_matrix(self, observers: np.ndarray, observer_names, nominal_weight: np.ndarray) -> np.ndarray:
        idx = {name: i for i, name in enumerate(observer_names)}
        missing = [name for name in self.required_observers if name not in idx]
        if missing:
            raise RuntimeError(f"Observer_names missing EFT targets: {missing}")

        sm = observers[:, idx["EFTWeight_SM"]].astype(np.float32, copy=False)
        safe_sm = sm.copy()
        safe_sm[safe_sm == 0] = 1.0

        out = [nominal_weight.astype(np.float32, copy=False)]
        for op in self.parameters:
            der = observers[:, idx[f"der_{op}"]].astype(np.float32, copy=False)
            out.append(nominal_weight * der / safe_sm)
        for op0, op1 in itertools.combinations_with_replacement(self.parameters, 2):
            der = observers[:, idx[f"der_{op1}_{op0}"]].astype(np.float32, copy=False)
            out.append(nominal_weight * der / safe_sm)

        return np.stack(out, axis=1)
