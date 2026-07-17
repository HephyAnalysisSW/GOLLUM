#!/usr/bin/env python
"""Save BIT calibration truth/prediction arrays for PDF coefficients (YAML-driven).

PDF-specific entry point: it builds the PDF derivative provider from the ``pdf`` block
of the job (via ``PDFParametrization``), computing the derivatives on the fly from
generator-level information, and hands it to the shared calibration pipeline in
``calibration_runner.py``.

Example
-------
    python ML/Calibration/pdf_calibration.py configs/unbinned_v6/unbinned_2018.yaml \
        --job bit_NG_PDF4LHC21_6_...
"""

from __future__ import annotations

import os
import sys

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.derivative_providers import PDFDerivativeProvider

import calibration_runner as cr


def main():
    args = cr.build_arg_parser(__doc__).parse_args()
    cfg, job, samples_mod = cr.load_cfg_and_job(args)

    if not job.get("pdf"):
        raise RuntimeError(f"Job '{job['id']}' has no 'pdf' block; use eft_calibration.py for EFT jobs.")

    provider = PDFDerivativeProvider(job["pdf"])
    cr.run_calibration(cfg, job, samples_mod, args, provider)


if __name__ == "__main__":
    main()
