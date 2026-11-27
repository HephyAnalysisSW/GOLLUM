#!/usr/bin/env python3
"""One-off: append the STRIPPER NNLO PDF columns (absolute and normalised) to the files with scale variations.

Rene's files with individual scale variations have no PDF columns. The PDF band cannot be rebuilt from the
absolute up/down columns for the normalised distributions (each PDF member was normalised before combining), so
both the absolute PDF columns (`data-for-CMS-PAS-TOP-20-006-mt172.5`) and the normalised ones (`...-norm`) are
appended, in this order: pdf_up, pdf_down, pdf_up_normalized, pdf_down_normalized. The existing columns are kept
byte-identical (the new values are appended as text); files that already have the extra columns are skipped.

    python user/ricardo/nnlo_reweighting/diagnostics/add_pdf_columns_to_stripper_variations.py
"""
import glob
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # nnlo_reweighting (samples)
from samples import STRIPPER_ABSOLUTE_DIR, STRIPPER_NORMALIZED_DIR, STRIPPER_VARIATIONS_DIR

logger = logging.getLogger(__name__)

# Per row of a variations file: 2 edges per axis, central + 6 scale variations, their 7 MC errors.
VARIATION_COLUMNS = 14
PDF_COLUMNS = 4


def main():
    """Append the PDF columns to every NNLO variations file."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    paths = sorted(glob.glob(os.path.join(STRIPPER_VARIATIONS_DIR, 'incl_NNLO_*.dat')))
    if not paths:
        raise RuntimeError(f'no NNLO files in {STRIPPER_VARIATIONS_DIR}')
    for path in paths:
        name = os.path.basename(path)
        variations = np.loadtxt(path, ndmin=2)
        absolute = np.loadtxt(os.path.join(STRIPPER_ABSOLUTE_DIR, name), ndmin=2)
        normalized = np.loadtxt(os.path.join(STRIPPER_NORMALIZED_DIR, name), ndmin=2)
        edge_columns = absolute.shape[1] - 6  # envelope files: edges, central, up, down, mcerr, pdf_up, pdf_down
        if variations.shape[1] == edge_columns + VARIATION_COLUMNS + PDF_COLUMNS:
            logger.info('%s already has the PDF columns, skipped', name)
            continue
        if variations.shape[1] != edge_columns + VARIATION_COLUMNS:
            raise ValueError(f'{name}: unexpected number of columns {variations.shape[1]}')
        for other, label in [(absolute, 'absolute'), (normalized, 'normalised')]:
            if other.shape[0] != variations.shape[0] or not np.allclose(other[:, :edge_columns],
                                                                        variations[:, :edge_columns]):
                raise ValueError(f'{name}: bins differ from the {label} envelope file')
        if not np.allclose(variations[:, edge_columns], absolute[:, edge_columns]):
            raise ValueError(f'{name}: central values differ from the absolute envelope file')
        extra = np.column_stack([absolute[:, edge_columns + 4:edge_columns + 6],
                                 normalized[:, edge_columns + 4:edge_columns + 6]])
        with open(path) as variations_file:
            lines = [line.rstrip('\n') for line in variations_file if line.strip()]
        with open(path, 'w') as variations_file:
            for line, row in zip(lines, extra):
                variations_file.write(line + ' ' + ' '.join(f'{value:.5e}' for value in row) + '\n')
        logger.info('%s: appended %d PDF columns', name, PDF_COLUMNS)


if __name__ == '__main__':
    main()
