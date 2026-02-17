#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import lhapdf


class PDFParametrization:
    """
    Thin LHAPDF wrapper that:
      - loads a PDF set once,
      - keeps a list of members (central + variations),
      - provides xfxQ access for (member, pid, x, Q),
      - can be reused for event reweighting later.

    Notes:
      * LHAPDF's xfxQ(pid, x, Q) returns x * f(x, Q).
      * For NNPDF31_nnlo_hessian_pdfas:
          mem=0 central (as=0.118)
          mem=1..100 symmhessian eigenvectors (unpaired)
          mem=101,102 alternative central fits (as=0.116,0.120)
    """

    def __init__(self, setname: str, include_alphas_members: bool = False, member_ids=None):
        self.setname = setname

        # Decide which LHAPDF member IDs we keep.
        # By default: central + symmhessian eigenvectors; optionally include alpha_s members too.
        pdfset = lhapdf.getPDFSet(setname)
        n_members_total = pdfset.size  # includes central

        if member_ids is None:
            # Heuristic tailored to the common "symmhessian+as" layout:
            # keep 0..100 by default, optionally append the tail.
            if include_alphas_members:
                member_ids = list(range(n_members_total))
            else:
                # keep central + first 100 variations if available; otherwise keep everything
                member_ids = list(range(min(n_members_total, 101)))
        self.member_ids = list(member_ids)

        # Load the chosen members once.
        self.pdfs = [lhapdf.mkPDF(setname, mid) for mid in self.member_ids]

    @property
    def n_members(self) -> int:
        return len(self.pdfs)

    def xfxQ(self, member: int, pid: int, x: float, Q: float) -> float:
        return self.pdfs[member].xfxQ(pid, x, Q)

    def ratio_xfxQ(self, member: int, pid: int, x: float, Q: float, ref_member: int = 0) -> float:
        # Simple reweighting primitive; deliberately light on checks.
        denom = self.pdfs[ref_member].xfxQ(pid, x, Q)
        if denom == 0.0:
            return 0.0
        return self.pdfs[member].xfxQ(pid, x, Q) / denom

    def event_pdf_ratio(self, member: int, id1: int, x1: float, id2: int, x2: float, Q: float,
                        ref_member: int = 0) -> float:
        # For later use: LO-style ratio for 2->X with incoming partons (id1,id2).
        f0_1 = self.pdfs[ref_member].xfxQ(id1, x1, Q)
        f0_2 = self.pdfs[ref_member].xfxQ(id2, x2, Q)
        denom = f0_1 * f0_2
        if denom == 0.0:
            return 0.0
        f1 = self.pdfs[member].xfxQ(id1, x1, Q)
        f2 = self.pdfs[member].xfxQ(id2, x2, Q)
        return (f1 * f2) / denom

if __name__=="__main__":
    pdf = PDFParametrization("NNPDF31_nnlo_hessian_pdfas", include_alphas_members=False)
    x_edges_fine = np.logspace(-4, 0, 201)   # thresholds; fine binning
    Q = 100.0
    #gluon_provider = PDFShapeTemplate(pdf, pid=21, x_edges=x_edges_fine, Q=Q, name="g(x,Q)")
