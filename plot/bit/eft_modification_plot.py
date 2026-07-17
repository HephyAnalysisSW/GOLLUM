#!/usr/bin/env python
"""Plot the modifications induced in kinematic distributions by EFT coefficients.

EFT-specific entry point: it builds the EFT derivative provider from the
precomputed ``EFTWeight_SM`` / ``der_*`` branches (via ``EFTWeightInterface``)
and hands it to the shared plotting boilerplate in ``modification_plotter.py``.

Truth is available directly from the sample derivatives, so no trained model is
required. Pass ``--with-bit`` to overlay the trained BIT (truth solid, BIT
dashed); at ``--value 1`` the curves equal the raw BIT coefficients, making this
a BIT closure plot.

Examples
--------
    # truth only (no BIT model needed)
    python plot/bit/eft_modification_plot.py configs/eft/unbinned_gen_ND.yaml \
        --job bit_TT01j2l_EFT_2016APV

    # overlay the trained BIT on the validation split
    python plot/bit/eft_modification_plot.py configs/eft/unbinned_gen_ND.yaml \
        --job bit_TT01j2l_EFT_2016APV --with-bit --split valid
"""

from __future__ import annotations

import os
import sys

# project roots (repo root + this script's directory for sibling imports)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common.syncer as syncer
from common.derivative_providers import EFTDerivativeProvider

import modification_plotter as mp


def main():
    args = mp.build_arg_parser(__doc__).parse_args()
    cfg, job, samples_mod, _ = mp.load_cfg_and_job(args)

    if not job.get("eft"):
        raise RuntimeError(f"Job '{job['id']}' has no 'eft' block; use pdf_modification_plot.py for PDF jobs.")

    provider = EFTDerivativeProvider(job["eft"].get("parameters", []))
    mp.make_modification_plots(cfg, job, samples_mod, args, provider)


if __name__ == "__main__":
    main()
    syncer.sync()
