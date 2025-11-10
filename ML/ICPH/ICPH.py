#!/usr/bin/env python
from __future__ import annotations
import pickle
import numpy as np
import sys
from typing import Sequence, Tuple, Union, Optional

# project roots
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

ArrayLike = Union[np.ndarray, Sequence[float]]


class InclusiveCrosssectionParametrizationHistogram:
    """
    ICPH — Inclusive Cross Section Parametrization (Histogram-valued)

    Histogram-valued analogue of ICP:
        σ_bin(ν) = exp( ∑_A (Π_p∈A ν_p) Δ_A(bin) )

    where combinations A correspond exactly to the ICP combination list.
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

    # ---------------- helpers ----------------
    def _nu_A(self, nu_vec: Sequence[float]) -> np.ndarray:
        """Return vector [ Π_p∈comb ν_p ] for all combinations."""
        out = np.ones(len(self.combinations), dtype=np.float64)
        for a, comb in enumerate(self.combinations):
            val = 1.0
            for p in comb:
                val *= nu_vec[self.parameters.index(p)]
            out[a] = val
        return out

    # ---------------- training ----------------
    def train(
        self,
        yields: dict[Tuple[float, ...], np.ndarray],
        train_ratio: bool = True,
        small: bool = False,
        selection: Optional[str] = None,
    ):
        """
        Given per-base-point histograms (yields), determine Δ_A histograms
        by solving the same linear system as the scalar ICP, bin-by-bin.
        """
        base_pts = np.array(self.base_points, dtype=float)
        nominal = np.array(self.nominal_base_point, dtype=float)
        n_bp = len(base_pts)
        n_comb = len(self.combinations)

        # flatten histograms to shape (n_bp, n_bins_flat)
        sample_hist = next(iter(yields.values()))
        hist_shape = sample_hist.shape
        n_bins_flat = np.prod(hist_shape)
        Y = np.zeros((n_bp, n_bins_flat), dtype=np.float64)
        for i, bp in enumerate(self.base_points):
            Y[i, :] = np.asarray(yields[bp], dtype=np.float64).reshape(-1)

        # Construct design matrix M using combinations, like in ICP
        M = np.zeros((n_bp, n_comb), dtype=np.float64)
        for i_bp, coords in enumerate(self.base_points):
            nu_vec = np.array(coords) - nominal
            for j, comb in enumerate(self.combinations):
                val = 1.0
                for p in comb:
                    val *= nu_vec[self.parameters.index(p)]
                M[i_bp, j] = val

        # Solve log(Y/Y_nominal) = M * Δ, per bin
        Y_nom = yields[self.nominal_base_point].reshape(-1)
        Y_ratio = np.maximum(Y / np.maximum(Y_nom, 1e-15), 1e-15)
        T = np.log(Y_ratio)

        MTM_inv = np.linalg.pinv(M.T @ M)
        coeff = MTM_inv @ M.T @ T  # (n_comb, n_bins_flat)
        self.deltas = coeff.reshape((n_comb,) + hist_shape)

    # ---------------- prediction ----------------
    def predict(self, nu: Sequence[float]) -> np.ndarray:
        """
        Return histogram exp(∑_A Π_p∈A ν_p Δ_A).
        """
        nu_vec = np.asarray(nu, dtype=float)
        if nu_vec.shape[0] != len(self.parameters):
            raise ValueError(f"ICPH.predict: expected {len(self.parameters)} parameters.")
        alphas = self._nu_A(nu_vec)
        exponent = np.tensordot(alphas, self.deltas, axes=(0, 0))
        return np.exp(exponent)

    # ---------------- nice printing ----------------
    def __str__(self) -> str:
        """
        Human-readable per-bin printout of the learned exponent:

            log σ_bin(ν) = ∑_A Π_p∈A ν_p Δ_A(bin)
        """
        labels = ["*".join(c) if len(c) else "" for c in self.combinations]
        lines = []
        proc = self.process or "None"
        lines.append(
            f"ICPH — process: \033[1m{proc}\033[0m, "
            f"axes: {', '.join(self.axis_names)}, "
            f"combinations: {len(self.combinations)}, "
            f"histogram shape: {self.deltas.shape}"
        )

        if self.deltas.ndim == 2:
            n_bins = self.deltas.shape[1]
            edges1 = self.bin_edges[0]
            for ib in range(n_bins):
                coeffs = self.deltas[:, ib]
                terms = [
                    f"{d:+.3e}{('*' + lab) if lab else ''}"
                    for d, lab in zip(coeffs, labels)
                ]
                expr = " ".join(terms)
                lo, hi = edges1[ib], edges1[ib + 1]
                lines.append(f"bin[{ib}] [{lo:.3g}, {hi:.3g}): exp({expr})")

        elif self.deltas.ndim == 3:
            n_bins1, n_bins2 = self.deltas.shape[1], self.deltas.shape[2]
            edges1, edges2 = self.bin_edges
            for i1 in range(n_bins1):
                for i2 in range(n_bins2):
                    coeffs = self.deltas[:, i1, i2]
                    terms = [
                        f"{d:+.3e}{('*' + lab) if lab else ''}"
                        for d, lab in zip(coeffs, labels)
                    ]
                    expr = " ".join(terms)
                    lo1, hi1 = edges1[i1], edges1[i1 + 1]
                    lo2, hi2 = edges2[i2], edges2[i2 + 1]
                    lines.append(
                        f"bin[{i1},{i2}] "
                        f"[{lo1:.3g}, {hi1:.3g}) x [{lo2:.3g}, {hi2:.3g}): exp({expr})"
                    )

        else:
            lines.append("Unsupported deltas.ndim (expected 2 or 3).")

        return "\n".join(lines)

