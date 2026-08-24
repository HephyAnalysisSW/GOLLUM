from __future__ import annotations

import itertools

import numpy as np

from data.samples_eft import wc_names, GENERATION_POINT


def _derivative_branch_name(op0: str, op1: str) -> str:
    """Name of the der_{later}_{earlier} branch for a pair of operators.

    Branch names are fixed at ROOT-production time by the global order of
    wc_names in data/samples_eft.py (der_{wc_j}_{wc_k} for k <= j), which need
    not match the order operators are listed in a job's `eft.parameters`.
    """
    j0, j1 = wc_names.index(op0), wc_names.index(op1)
    later, earlier = (op0, op1) if j0 >= j1 else (op1, op0)
    return f"der_{later}_{earlier}"


class EFTWeightInterface:
    def __init__(self, parameters):
        self.parameters = list(parameters or [])
        if not self.parameters:
            raise RuntimeError("EFTWeightInterface requires a non-empty operator list.")

        # Point the samples were generated at. The expansion below is rebased here
        # instead of at the SM, and covers all 16 operators regardless of
        # self.parameters, since every entry of GENERATION_POINT is nonzero.
        self.reference_point = dict(GENERATION_POINT)

        self._combinations = [()]
        self._combinations.extend((op,) for op in self.parameters)
        self._combinations.extend(itertools.combinations_with_replacement(self.parameters, 2))

        self.base_points = []
        for comb in itertools.combinations_with_replacement(self.parameters, 1):
            self.base_points.append({op: comb.count(op) for op in self.parameters})
        for comb in itertools.combinations_with_replacement(self.parameters, 2):
            self.base_points.append({op: comb.count(op) for op in self.parameters})

        self.required_observers = ["EFTWeight_gen"]
        self.required_observers.extend(f"der_{op}" for op in self.parameters)
        # Full row of the Hessian for each fitted operator, against every operator in
        # wc_names: the derivative shift D'_i = D_i + sum_j H_ij r_j sums over all 16
        # operators, since GENERATION_POINT has no zero entries.
        row_branches = {
            _derivative_branch_name(op, other) for op in self.parameters for other in wc_names
        }
        # lower triangular matrix in (op0,op1) among the fitted parameters only, needed
        # for the quadratic block of make_weight_matrix.
        quad_branches = {
            _derivative_branch_name(op0, op1)
            for op0, op1 in itertools.combinations_with_replacement(self.parameters, 2)
        }
        self.required_observers.extend(sorted(row_branches | quad_branches))

    @property
    def combinations(self):
        return self._combinations

    def make_weight_matrix(self, observers: np.ndarray, observer_names, nominal_weight: np.ndarray) -> np.ndarray:
        idx = {name: i for i, name in enumerate(observer_names)}
        missing = [name for name in self.required_observers if name not in idx]
        if missing:
            raise RuntimeError(f"Observer_names missing EFT targets: {missing}")

        gen = observers[:, idx["EFTWeight_gen"]].astype(np.float32, copy=False)
        safe_gen = gen.copy()
        safe_gen[safe_gen == 0] = 1.0

        out = [nominal_weight.astype(np.float32, copy=False)]
        for op in self.parameters:
            der = observers[:, idx[f"der_{op}"]].astype(np.float32, copy=False)
            # D'_op = D_op + sum_j H_{op,j} r_j, the first derivative at the generation
            # point rather than at the SM. No factor on the diagonal: der_{op}_{op} is
            # already the raw Hessian entry (see make_ntuple.py).
            shift = np.zeros_like(der)
            for other in wc_names:
                h = observers[:, idx[_derivative_branch_name(op, other)]].astype(np.float32, copy=False)
                shift += h * self.reference_point[other]
            out.append(nominal_weight * (der + shift) / safe_gen)
        for op0, op1 in itertools.combinations_with_replacement(self.parameters, 2):
            der = observers[:, idx[_derivative_branch_name(op0, op1)]].astype(np.float32, copy=False)
            out.append(nominal_weight * der / safe_gen)

        return np.stack(out, axis=1)
