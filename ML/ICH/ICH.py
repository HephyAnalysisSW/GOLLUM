#!/usr/bin/env python
from __future__ import annotations

import pickle
import numpy as np
import sys
from typing import List, Sequence, Tuple, Union, Optional

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')


ArrayLike = Union[np.ndarray, Sequence[float]]


class InclusiveCrosssectionHistogram:
    """
    Inclusive Cross Section (Histogram) – ICH

    Stores binned coefficient histograms for a polynomial PDF parametrization:

        σ_bin(c) = h_()           (constant term)
                   + ∑_a c_a h_(a)
                   + 1/2 ∑_{a,b} c_a c_b h_(a,b)

    where:
      - combinations = [(), ('c0',), ..., ('c_i','c_j'), ...]
      - each combination has an associated histogram h_(comb)
      - binning: 1D or 2D, defined at construction time

    This class is purely “linalg + histograms”; data iteration & derivatives
    are handled in ich_training.py.
    """

    def __init__(
        self,
        variables: Sequence[str],
        combinations: Sequence[Tuple[str, ...]],
        axis_names: Sequence[str],
        bin_edges: Sequence[ArrayLike],
        process: Optional[str] = None,
        selection: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        variables   : list of coefficient names, e.g. ['c0', 'c1', ..., 'cN'].
        combinations: list of tuples, e.g. [(), ('c0',), ..., ('c0','c1'), ...].
        axis_names  : list of observable names along each axis; length 1 or 2.
        bin_edges   : list of bin edges arrays (same length as axis_names).
        process     : optional process/view name (for pretty-printing).
        selection   : optional top-level selection label.
        note        : optional free text.
        """
        self.variables   = list(variables)
        self.combinations = [tuple(c) for c in combinations]

        self.axis_names = list(axis_names)
        if len(self.axis_names) not in (1, 2):
            raise ValueError("ICH: only 1D or 2D binning is supported.")

        self.bin_edges = [np.asarray(be, dtype=float) for be in bin_edges]
        if len(self.bin_edges) != len(self.axis_names):
            raise ValueError("ICH: len(bin_edges) must match len(axis_names).")

        n_bins1 = len(self.bin_edges[0]) - 1
        if n_bins1 <= 0:
            raise ValueError("ICH: first axis needs at least 1 bin.")

        if len(self.axis_names) == 1:
            # shape: (n_combinations, n_bins)
            self.histograms = np.zeros(
                (len(self.combinations), n_bins1), dtype=np.float64
            )
        else:
            n_bins2 = len(self.bin_edges[1]) - 1
            if n_bins2 <= 0:
                raise ValueError("ICH: second axis needs at least 1 bin.")
            # shape: (n_combinations, n_bins1, n_bins2)
            self.histograms = np.zeros(
                (len(self.combinations), n_bins1, n_bins2), dtype=np.float64
            )

        # meta
        self.process   = process or None
        self.selection = selection or None
        self.note      = note or None

    # ---------------- persistence ----------------

    @classmethod
    def load(cls, filename: str) -> "InclusiveCrosssectionHistogram":
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        # Allow for older pickles if we ever had them
        if isinstance(obj, cls):
            return obj
        # If not our class, try to reconstruct (best-effort)
        raise TypeError(f"File '{filename}' does not contain an ICH object.")

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __setstate__(self, state):
        self.__dict__ = state

    # ---------------- accumulation ----------------

    def accumulate(
        self,
        axis1: ArrayLike,
        weights_per_comb: np.ndarray,
        axis2: Optional[ArrayLike] = None,
    ) -> None:
        """
        Accumulate a chunk of events.

        Parameters
        ----------
        axis1          : array-like of shape (N,), values along first binning axis.
        weights_per_comb : array-like of shape (N, M),
                           where M = len(self.combinations).
                           Each column is the per-event weight for that combination.
        axis2          : optional array-like of shape (N,) for 2D binning.
        """
        axis1 = np.asarray(axis1, dtype=float)
        if axis1.size == 0:
            return

        wpc = np.asarray(weights_per_comb, dtype=np.float64)
        if wpc.ndim != 2 or wpc.shape[0] != axis1.shape[0]:
            raise ValueError("ICH.accumulate: weights_per_comb must have shape (N, M) with same N as axis1.")
        if wpc.shape[1] != len(self.combinations):
            raise ValueError("ICH.accumulate: second dim of weights_per_comb must equal len(combinations).")

        if len(self.axis_names) == 1:
            # 1D
            for i_comb in range(len(self.combinations)):
                hist, _ = np.histogram(axis1, bins=self.bin_edges[0], weights=wpc[:, i_comb])
                self.histograms[i_comb, :] += hist
        else:
            # 2D
            if axis2 is None:
                raise ValueError("ICH.accumulate: axis2 must be provided for 2D binning.")
            axis2 = np.asarray(axis2, dtype=float)
            if axis2.shape != axis1.shape:
                raise ValueError("ICH.accumulate: axis1 and axis2 must have same shape.")

            for i_comb in range(len(self.combinations)):
                hist2d, _, _ = np.histogram2d(
                    axis1, axis2,
                    bins=(self.bin_edges[0], self.bin_edges[1]),
                    weights=wpc[:, i_comb],
                )
                self.histograms[i_comb, :, :] += hist2d

    def finalize(self) -> None:
        """Placeholder for potential post-processing; currently a no-op."""
        pass

    # ---------------- prediction ----------------

    def _combo_weight(self, comb: Tuple[str, ...], coeffs: np.ndarray) -> float:
        """
        Compute the scalar factor multiplying the histogram for `comb`:

           ()            -> 1
           ('c_a',)      -> c_a
           ('c_a','c_b') -> c_a c_b       (a != b)
           ('c_a','c_a') -> 1/2 c_a^2     (diagonal quadratic)
        """
        if len(comb) == 0:
            return 1.0

        # map 'c_i' -> coeffs[i]
        prod = 1.0
        for name in comb:
            try:
                idx = self.variables.index(name)
            except ValueError:
                raise KeyError(f"Coefficient name '{name}' not found in variables={self.variables}.")
            prod *= coeffs[idx]

        if len(comb) == 2 and comb[0] == comb[1]:
            prod *= 0.5

        return float(prod)

    def predict(self, coeffs: Sequence[float]) -> np.ndarray:
        """
        Build the predicted histogram for a given set of Chebyshev coefficients `c`.

        Parameters
        ----------
        coeffs : sequence of floats, e.g. [c0, c1, ..., cN].

        Returns
        -------
        hist : np.ndarray
            - shape (n_bins,)  for 1D
            - shape (n_bins1, n_bins2) for 2D
        """
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.shape[0] != len(self.variables):
            raise ValueError(f"ICH.predict: expected {len(self.variables)} coeffs, got {coeffs.shape[0]}.")

        alphas = np.array([self._combo_weight(c, coeffs) for c in self.combinations], dtype=float)
        # contracted over combination axis (0)
        return np.tensordot(alphas, self.histograms, axes=(0, 0))

    # ---------------- nice printing ----------------

    def __str__(self) -> str:
        proc = self.process or "None"
        sel  = self.selection or "None"
        axes = ", ".join(self.axis_names)
        shape = self.histograms.shape
        return (
            f"ICH — process: \033[1m{proc}\033[0m, selection: \033[1m{sel}\033[0m\n"
            f"  axes: {axes}\n"
            f"  combinations: {len(self.combinations)}\n"
            f"  histogram shape: {shape}\n"
        )


# Optional alias for backward-compat naming if ever needed
InclusiveCrosssection = InclusiveCrosssectionHistogram

