#!/usr/bin/env python
"""Save BIT calibration truth/prediction arrays for EFT coefficients (YAML-driven).

EFT-specific entry point: it builds the EFT derivative provider from the precomputed
``EFTWeight_SM`` / ``der_*`` branches (via ``EFTWeightInterface``) and hands it to the
shared calibration pipeline in ``calibration_runner.py``.

Example
-------
    python ML/Calibration/eft_calibration.py configs/eft/unbinned_gen_ND.yaml \
        --job bit_TT01j2l_EFT_2016APV
"""

from __future__ import annotations

import os
import sys

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.derivative_providers import EFTDerivativeProvider

import calibration_runner as cr


def main():
    args = cr.build_arg_parser(__doc__).parse_args()
    cfg, job, samples_mod = cr.load_cfg_and_job(args)

    if not job.get("eft"):
        raise RuntimeError(f"Job '{job['id']}' has no 'eft' block; use pdf_calibration.py for PDF jobs.")

    provider = EFTDerivativeProvider(job["eft"].get("parameters", []))
    cr.run_calibration(cfg, job, samples_mod, args, provider)


if __name__ == "__main__":
    main()
