#!/usr/bin/env python3
"""Standalone native loader for the 250503 POD basis.

This intentionally does not import PODBasis.py and does not use max_amplitudes.
The basis vectors are the native LHAPDF shifts

    member_i(x, Q, pid) - member_0(x, Q, pid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import lhapdf


PDF_SET = "250503_pod_basis_40k"


@dataclass(frozen=True)
class NativePODBasis:
    pdf_set: str
    reference_pdf: object
    basis_pdfs: Sequence[object]
    variations: Sequence[int]
    flavors: Sequence[int]

    @classmethod
    def load(
        cls,
        pdf_set: str = PDF_SET,
        variations: Iterable[int] | None = None,
        flavors: Iterable[int] | None = None,
    ) -> "NativePODBasis":
        pdf_info = lhapdf.getPDFSet(pdf_set)
        size = pdf_info.size
        n_members = size() if callable(size) else size

        if variations is None:
            variations = range(1, n_members)
        variations = tuple(int(var) for var in variations)

        reference_pdf = lhapdf.mkPDF(pdf_set, 0)
        basis_pdfs = tuple(lhapdf.mkPDF(pdf_set, var) for var in variations)

        if flavors is None:
            flavors = reference_pdf.flavors()
        flavors = tuple(int(pid) for pid in flavors)

        return cls(
            pdf_set=pdf_set,
            reference_pdf=reference_pdf,
            basis_pdfs=basis_pdfs,
            variations=variations,
            flavors=flavors,
        )

    @property
    def nvariations(self) -> int:
        return len(self.variations)

    def xfx_grid(self, pdf: object, x_grid: Sequence[float], q: float) -> np.ndarray:
        x_grid = np.asarray(x_grid, dtype=float)
        out = np.empty((len(self.flavors), len(x_grid)), dtype=float)

        for i_pid, pid in enumerate(self.flavors):
            out[i_pid] = np.array([pdf.xfxQ(pid, x, q) for x in x_grid], dtype=float)

        return out

    def reference_grid(self, x_grid: Sequence[float], q: float) -> np.ndarray:
        return self.xfx_grid(self.reference_pdf, x_grid, q)

    def native_shift_grid(self, x_grid: Sequence[float], q: float) -> np.ndarray:
        reference = self.reference_grid(x_grid, q)
        shifts = np.empty((self.nvariations, len(self.flavors), len(x_grid)), dtype=float)

        for i_var, pdf in enumerate(self.basis_pdfs):
            shifts[i_var] = self.xfx_grid(pdf, x_grid, q) - reference

        return shifts

    def combine(
        self,
        coeffs: Sequence[float],
        x_grid: Sequence[float],
        q: float,
    ) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.shape != (self.nvariations,):
            raise ValueError(
                f"Expected {self.nvariations} coefficients, got shape {coeffs.shape}"
            )

        reference = self.reference_grid(x_grid, q)
        shifts = self.native_shift_grid(x_grid, q)
        return reference + np.einsum("i,ifx->fx", coeffs, shifts)


def main() -> None:
    basis = NativePODBasis.load()

    x_grid = np.geomspace(1e-4, 0.8, 50)
    q = 1.65
    reference = basis.reference_grid(x_grid, q)
    shifts = basis.native_shift_grid(x_grid, q)

    print(f"Loaded {basis.pdf_set}")
    print(f"  reference member : 0")
    print(f"  basis members    : {basis.variations[0]}..{basis.variations[-1]}")
    print(f"  nvariations      : {basis.nvariations}")
    print(f"  flavors          : {list(basis.flavors)}")
    print(f"  reference shape  : {reference.shape}  # flavors, x")
    print(f"  shifts shape     : {shifts.shape}  # variations, flavors, x")
    print("  native shifts    : member_i - member_0, no max_amplitudes")


if __name__ == "__main__":
    main()
