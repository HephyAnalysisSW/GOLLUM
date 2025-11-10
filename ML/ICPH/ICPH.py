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


class InclusiveCrosssectionParametrizationHistogram:
    """
    ICPH — Inclusive Cross Section Parametrization (Histogram-valued)

    Stores histogram-valued coefficients Δ_A for the exponential model

        σ_bin(ν) = exp( ∑_A ν_A Δ_A(bin) )

    where:
      - each Δ_A(bin) is the histogram value for parameter combination A
      - no constant term is stored (by construction, exp(0)=1)

    The predict() method returns a 1D or 2D histogram of values exp(∑_A ν_A Δ_A),
    such that ν=0 yields a histogram of ones.
    """

    def __init__(
        self,
        combinations: Sequence[Tuple[str, ...]],
        nominal_base_point: Tuple[float, ...],
        base_points: Sequence[Tuple[float, ...]],
        parameters: Sequence[str],
        axis_names: Sequence[str],
        bin_edges: Sequence[ArrayLike],
        process: Optional[str] = None,
        note: Optional[str] = None,
    ):
        self.combinations = [tuple(c) for c in combinations]
        self.nominal_base_point = tuple(nominal_base_point)
        self.base_points = [tuple(bp) for bp in base_points]
        self.parameters = list(parameters)

        self.axis_names = list(axis_names)
        if len(self.axis_names) not in (1, 2):
            raise ValueError("ICPH: only 1D or 2D binning is supported.")
        self.bin_edges = [np.asarray(be, dtype=float) for be in bin_edges]
        if len(self.bin_edges) != len(self.axis_names):
            raise ValueError("ICPH: len(bin_edges) must match len(axis_names).")

        n_bins1 = len(self.bin_edges[0]) - 1
        if len(self.axis_names) == 1:
            self.deltas = np.zeros((len(self.combinations), n_bins1), dtype=np.float64)
        else:
            n_bins2 = len(self.bin_edges[1]) - 1
            self.deltas = np.zeros((len(self.combinations), n_bins1, n_bins2), dtype=np.float64)

        self.process = process or None
        self.note = note or None

    # ---------------- persistence ----------------
    @classmethod
    def load(cls, filename: str) -> "InclusiveCrosssectionParametrizationHistogram":
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        raise TypeError(f"File '{filename}' does not contain an ICPH object.")

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __setstate__(self, state):
        self.__dict__ = state

    # ---------------- training interface ----------------
    def train(
        self,
        yields: dict[Tuple[float, ...], np.ndarray],
        train_ratio: bool = True,
        small: bool = False,
        selection: Optional[str] = None,
    ):
        """
        Given per-base-point histograms (yields), determine Δ_A histograms by
        solving the same linear system as the scalar ICP but bin-by-bin.

        Parameters
        ----------
        yields : dict
            Mapping from base_point (tuple of coords) → histogram (1D or 2D array).
        """
        base_pts = np.array(self.base_points, dtype=float)
        nominal = np.array(self.nominal_base_point, dtype=float)
        n_bp = len(base_pts)
        n_comb = len(self.combinations)

        # Flatten histograms to shape (n_bp, n_bins_flat)
        sample_hist = next(iter(yields.values()))
        hist_shape = sample_hist.shape
        n_bins_flat = np.prod(hist_shape)
        Y = np.zeros((n_bp, n_bins_flat), dtype=np.float64)
        for i, bp in enumerate(self.base_points):
            Y[i, :] = np.asarray(yields[bp], dtype=np.float64).reshape(-1)

        # Construct design matrix M (same as scalar ICP)
        M = np.zeros((n_bp, n_comb), dtype=np.float64)
        for i_bp, coords in enumerate(self.base_points):
            ν = np.array(coords) - nominal
            for j, comb in enumerate(self.combinations):
                if len(comb) == 1:
                    idx = self.parameters.index(comb[0])
                    M[i_bp, j] = ν[idx]
                elif len(comb) == 2:
                    i1 = self.parameters.index(comb[0])
                    i2 = self.parameters.index(comb[1])
                    M[i_bp, j] = 0.5 * ν[i1] * ν[i2] if i1 == i2 else ν[i1] * ν[i2]
                else:
                    M[i_bp, j] = 0.0

        # Solve per-bin linear regression: log(Y / Y_nominal) = M * Δ
        Y_nom = yields[self.nominal_base_point].reshape(-1)
        Y_ratio = np.maximum(Y / np.maximum(Y_nom, 1e-15), 1e-15)
        T = np.log(Y_ratio)

        MTM_inv = np.linalg.pinv(M.T @ M)
        MT = M.T
        coeff = MTM_inv @ MT @ T  # (n_comb, n_bins_flat)

        self.deltas = coeff.reshape((n_comb,) + hist_shape)

    # ---------------- prediction ----------------
    def predict(self, nu: Sequence[float]) -> np.ndarray:
        """
        Return histogram exp(∑_A ν_A Δ_A).

        ν=0 returns histogram of ones (by construction).
        """
        nu = np.asarray(nu, dtype=float)
        if nu.shape[0] != len(self.parameters):
            raise ValueError(f"ICPH.predict: expected {len(self.parameters)} coefficients.")

        alphas = np.zeros(len(self.combinations), dtype=np.float64)
        for i, comb in enumerate(self.combinations):
            if len(comb) == 1:
                idx = self.parameters.index(comb[0])
                alphas[i] = nu[idx]
            elif len(comb) == 2:
                i1 = self.parameters.index(comb[0])
                i2 = self.parameters.index(comb[1])
                alphas[i] = 0.5 * nu[i1] * nu[i2] if i1 == i2 else nu[i1] * nu[i2]
            else:
                alphas[i] = 0.0

        exponent = np.tensordot(alphas, self.deltas, axes=(0, 0))
        return np.exp(exponent)

    def __str__(self) -> str:
        proc = self.process or "None"
        shape = self.deltas.shape
        return (
            f"ICPH — process: \033[1m{proc}\033[0m\n"
            f"  combinations: {len(self.combinations)}\n"
            f"  histogram shape: {shape}\n"
        )


#InclusiveCrosssectionParametrizationHistogram = InclusiveCrosssectionParametrizationHistogram
