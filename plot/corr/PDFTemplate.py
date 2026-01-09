#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import lhapdf

from PDFParametrization import PDFParametrization
from TemplateBase import TemplateBase

class PDFShapeTemplate(TemplateBase):
    """
    Provides PDF "templates" for a single flavor as a function of x at fixed Q.

    Template definition:
      template[m][i] = x * f_pid(x_i, Q)  (via LHAPDF xfxQ)

    Binning:
      - you give x_edges (thresholds)
      - we evaluate in each bin at a bin-center in log(x):
            x_center = exp(0.5*(log(x_lo)+log(x_hi)))
        (keeps things stable over wide x ranges)

    This is intended for:
      - PDF auto-correlations (same provider twice),
      - or correlations vs some other binned observable later.
    """

    def __init__(self, pdf: PDFParametrization, pid: int, x_edges, Q: float, name: str = ""):
        super().__init__(name=name or f"PDF(pid={pid})")
        self.pdf = pdf
        self.pid = int(pid)
        self.Q = float(Q)

        self.x_edges = np.asarray(x_edges, dtype=float)
        # log-mid bin centers (simple, effective)
        lo = self.x_edges[:-1]
        hi = self.x_edges[1:]
        self.x_centers = np.exp(0.5 * (np.log(lo) + np.log(hi)))

    @property
    def n_members(self) -> int:
        return self.pdf.n_members

    def get_template(self, member: int) -> np.ndarray:
        vals = np.array([self.pdf.xfxQ(member, self.pid, float(x), self.Q) for x in self.x_centers],
                        dtype=float)
        return vals

    # Convenience accessors (useful for plotting later)
    def get_x_edges(self) -> np.ndarray:
        return self.x_edges

    def get_x_centers(self) -> np.ndarray:
        return self.x_centers

# -------------------------------------------------------------------------
# Minimal usage sketch (not executed; keep top-level lean as requested):
#
if __name__ == "__main__":
    pdf = PDFParametrization("NNPDF31_nnlo_hessian_pdfas", include_alphas_members=False)
    x_edges_fine = np.logspace(-4, 0, 201)   # thresholds; fine binning
    Q = 100.0
    gluon_provider = PDFShapeTemplate(pdf, pid=21, x_edges=x_edges_fine, Q=Q, name="g(x,Q)")
    quark_provider = PDFShapeTemplate(pdf, pid=2, x_edges=x_edges_fine, Q=Q, name="u(x,Q)")
    
    A_m = gluon_provider.get_template(0)  # numpy array over x-bins

