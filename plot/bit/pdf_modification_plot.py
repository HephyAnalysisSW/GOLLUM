#!/usr/bin/env python
"""Plot the modifications induced in kinematic distributions by PDF coefficients.

PDF-specific entry point: it builds the PDF derivative provider from the
``pdf`` block of the job (via ``PDFParametrization``), computing the derivatives
on the fly from generator-level information, and hands it to the shared plotting
boilerplate in ``modification_plotter.py``.

Truth is available directly from the sample generator info, so no trained model
is required. Pass ``--with-bit`` to overlay the trained BIT (truth solid, BIT
dashed); at ``--value 1`` the curves equal the raw BIT coefficients, making this
a BIT closure plot.

Examples
--------
    # truth only (no BIT model needed)
    python plot/bit/pdf_modification_plot.py configs/unbinned_v6/unbinned_2018.yaml \
        --job bit_NG_PDF4LHC21_6_...

    # overlay the trained BIT
    python plot/bit/pdf_modification_plot.py configs/unbinned_v6/unbinned_2018.yaml \
        --job bit_NG_PDF4LHC21_6_... --with-bit
"""

from __future__ import annotations

import os
import sys

import numpy as np

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common.syncer as syncer
from pdf.PDFParametrization import PDFParametrization

import modification_plotter as mp


class PDFDerivativeProvider:
    """PDF derivative provider: evaluates the parametrization from generator info."""

    GEN_OBSERVERS = [
        "Generator_x1",
        "Generator_x2",
        "Generator_id1",
        "Generator_id2",
        "Generator_scalePDF",
    ]

    def __init__(self, pdf_cfg):
        self._pdf = PDFParametrization(
            n=pdf_cfg.get("pdf_n"),
            typ=pdf_cfg.get("pdf_type"),
            basis=pdf_cfg.get("pdf_basis"),
            rescale_pod_amplitudes=pdf_cfg.get("rescale_pod_amplitudes", True),
        )
        self.combinations = [mp.canonical_combination(c) for c in self._pdf.combinations]
        self.parameters = [c[0] for c in self.combinations if len(c) == 1]
        self.required_observers = list(self.GEN_OBSERVERS)

    def truth_weight_matrix(self, G, w, observer_names):
        idx = {name: i for i, name in enumerate(observer_names)}
        missing = [name for name in self.GEN_OBSERVERS if name not in idx]
        if missing:
            raise RuntimeError(f"observer_names missing generator branches: {missing}")

        Q = G[:, idx["Generator_scalePDF"]].astype(np.float32, copy=False)
        x1 = G[:, idx["Generator_x1"]].astype(np.float32, copy=False)
        x2 = G[:, idx["Generator_x2"]].astype(np.float32, copy=False)
        id1 = G[:, idx["Generator_id1"]].astype(np.int32, copy=False)
        id2 = G[:, idx["Generator_id2"]].astype(np.int32, copy=False)

        deriv = self._pdf.derivatives(x1=x1, x2=x2, id1=id1, id2=id2, Q=Q).astype(np.float32, copy=False)
        return deriv * w.reshape(-1, 1)


def main():
    args = mp.build_arg_parser(__doc__).parse_args()
    cfg, job, samples_mod, _ = mp.load_cfg_and_job(args)

    if not job.get("pdf"):
        raise RuntimeError(f"Job '{job['id']}' has no 'pdf' block; use eft_modification_plot.py for EFT jobs.")

    provider = PDFDerivativeProvider(job["pdf"])
    mp.make_modification_plots(cfg, job, samples_mod, args, provider)


if __name__ == "__main__":
    main()
    syncer.sync()
