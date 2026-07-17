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

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common.syncer as syncer
from common.derivative_providers import PDFDerivativeProvider

import modification_plotter as mp


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
