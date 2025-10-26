#!/usr/bin/env python
from __future__ import annotations
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import pickle
import functools, operator
import numpy as np

class InclusiveCrosssectionParametrization:
    """
    Log-polynomial parametrization of inclusive cross section ratios:
      σ(nu) / σ(nominal) = exp( sum_c A_c * prod_{p in c} nu_p )

    Supports multi-parameter and mixed-combination terms.
    """
    def __init__(
        self,
        config=None,
        combinations=None,
        nominal_base_point=None,
        base_points=None,
        parameters=None,
    ):
        if config is not None:
            # legacy-style config module
            self.config_name        = config.__name__
            self.base_points        = np.array(config.base_points, dtype=float)
            self.n_base_points      = len(self.base_points)
            self.nominal_base_point = np.array(config.nominal_base_point, dtype=float)
            self.combinations       = [tuple(c) for c in config.combinations]
            self.parameters         = list(config.parameters)
        elif (combinations is not None) and (nominal_base_point is not None) and (base_points is not None) and (parameters is not None):
            self.config_name        = None
            self.base_points        = np.array(base_points, dtype=float)
            self.n_base_points      = len(self.base_points)
            self.nominal_base_point = np.array(nominal_base_point, dtype=float)
            self.combinations       = [tuple(c) for c in combinations]
            self.parameters         = list(parameters)
        else:
            raise RuntimeError("Provide either a legacy config module or (combinations, nominal_base_point, base_points, parameters).")

        # Locate nominal index
        nom_idx = np.where(np.all(self.base_points == self.nominal_base_point, axis=1))[0]
        if len(nom_idx) == 0:
            raise RuntimeError(f"Nominal base point {self.nominal_base_point} not found in base_points.")
        self.nominal_base_point_index = int(nom_idx[0])
        self.nominal_base_point_key   = tuple(self.nominal_base_point.tolist())

        # Masked (non-nominal) base points
        mask = np.ones(self.n_base_points, dtype=bool)
        mask[self.nominal_base_point_index] = False
        self.masked_base_points = self.base_points[mask]

        # Build C matrix: C_{ab} = sum_k ( Π nu_k^{comb_a+comb_b} ) over non-nominal base points
        # Implemented via explicit products from the combination tuples.
        nC = len(self.combinations)
        C  = np.zeros((nC, nC), dtype=np.float64)
        for bp in self.masked_base_points:
            for ia, ca in enumerate(self.combinations):
                for ib, cb in enumerate(self.combinations):
                    prod_terms = list(ca) + list(cb)
                    val = 1.0
                    for p in prod_terms:
                        val *= bp[self.parameters.index(p)]
                    C[ia, ib] += val

        if np.linalg.matrix_rank(C) != C.shape[0]:
            raise RuntimeError("Base point matrix does not have full rank. Check base_points & combinations.")
        self.CInv = np.linalg.inv(C)

        # Precompute V matrix over non-nominal base points: V_{k,a} = Π nu_k^{comb_a}
        self._V = np.zeros((len(self.masked_base_points), len(self.combinations)), dtype=np.float64)
        for k, bp in enumerate(self.masked_base_points):
            for a, comb in enumerate(self.combinations):
                val = 1.0
                for p in comb:
                    val *= bp[self.parameters.index(p)]
                self._V[k, a] = val

        # Filled during training
        self.DeltaA = None

    # ---------------- training (from already-computed yields) ----------------
    def train(self, small=False, train_ratio=True, yields=None, selection=None):
        """
        Parameters
        ----------
        yields : dict { base_point_tuple -> total_weight }
            e.g. { (-1.0,): Y_down, (0.0,): Y_nom, (1.0,): Y_up }
        train_ratio : bool
            True: fit log ratios log σ(nu)/σ(nom).
            False: fit absolute log σ(nu); requires inclusion of the constant term () in combinations.
        """
        if not isinstance(yields, dict) or not yields:
            raise RuntimeError("ICP.train requires a non-empty dict of yields keyed by base_point tuples.")

        # Ordered list of yields for the masked (non-nominal) base points
        y_masked = []
        for bp in self.masked_base_points:
            key = tuple(bp.tolist())
            if key not in yields:
                raise RuntimeError(f"Missing yield for base point {key}.")
            y_masked.append(float(yields[key]))
        y_masked = np.asarray(y_masked, dtype=np.float64)

        y_nom = float(yields[self.nominal_base_point_key])

        if train_ratio:
            # log σ(nu_k)/σ(nom)
            rhs = np.log(np.maximum(y_masked, 1e-300)) - np.log(max(y_nom, 1e-300))
        else:
            # absolute: must include constant term ()
            if tuple() not in self.combinations:
                raise RuntimeError("Absolute fit requires the constant term () in combinations.")
            rhs = np.log(np.maximum(y_masked, 1e-300))

        # Solve (V C^{-1}) * DeltaA = rhs  -> DeltaA = C^{-1} * sum_k V_k * rhs_k
        # Here implemented as DeltaA = CInv · sum_k V[k,:] * rhs[k]
        accum = np.zeros(len(self.combinations), dtype=np.float64)
        for k in range(self._V.shape[0]):
            accum += self._V[k, :] * rhs[k]
        self.DeltaA = self.CInv.dot(accum)

    # ---------------- prediction ----------------
    def _nu_A(self, nu_vec):
        """Return vector [ Π nu^{comb} ] for all combinations."""
        out = np.zeros(len(self.combinations), dtype=np.float64)
        for a, comb in enumerate(self.combinations):
            val = 1.0
            for p in comb:
                val *= nu_vec[self.parameters.index(p)]
            out[a] = val
        return out

    def log_predict(self, nu: dict | list | tuple):
        if self.DeltaA is None:
            raise RuntimeError("ICP not trained.")
        # Accept dict {param: value} or plain vector in parameter order
        if isinstance(nu, dict):
            vec = [float(nu[p]) for p in self.parameters]
        else:
            vec = [float(x) for x in nu]
        return float(self._nu_A(vec).dot(self.DeltaA))

    def predict(self, nu):
        return np.exp(self.log_predict(nu))

    # Handy derivative if you need it downstream
    def nu_A_diff(self, nu, param):
        param_idx = self.parameters.index(param)
        if isinstance(nu, dict):
            vec = [float(nu[p]) for p in self.parameters]
        else:
            vec = [float(x) for x in nu]

        def diff_term(comb):
            cnt = comb.count(param)
            if cnt == 0:
                return 0.0
            total = 0.0
            for i, p in enumerate(comb):
                if p != param:
                    continue
                prod = 1.0
                for j, q in enumerate(comb):
                    if j == i:  # skip one occurrence
                        continue
                    prod *= vec[self.parameters.index(q)]
                total += prod
            return total

        return np.array([diff_term(c) for c in self.combinations], dtype=np.float64)

    # ---------------- persistence ----------------
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fh:
            old = pickle.load(fh)
        if getattr(old, "config_name", None):
            # Try to rebuild from the stored module name, then copy learned params
            import importlib
            cfg_mod = importlib.import_module(old.config_name)
            new = cls(config=cfg_mod)
        else:
            new = cls(
                combinations=old.combinations,
                nominal_base_point=old.nominal_base_point,
                base_points=old.base_points,
                parameters=old.parameters,
            )
        new.DeltaA = old.DeltaA
        return new

    def save(self, filename):
        cfg = getattr(self, "config", None)
        self.config = None
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)
        self.config = cfg

    def __str__(self):
        labels = ["*".join(c) if len(c) else "" for c in self.combinations]
        terms  = [f"{d:+.3e}{('*'+lab) if lab else ''}" for d, lab in zip(self.DeltaA, labels)]
        return " ".join(terms)

