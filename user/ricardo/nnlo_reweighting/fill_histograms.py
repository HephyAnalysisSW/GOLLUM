#!/usr/bin/env python3
"""Fill parton-level tt-bar histograms (in the fixed-order binnings) from the gen ntuples of one sample.

Output: `histograms/<sample>.npz` with `sum_w_all` (all events, for the normalisation), `sum_w_tops`
(events with two gen tops) and, per `<variable>__<binning>`, `edges<axis>` (one per axis), `sum_w`, `sum_w2`.
`sum_w`/`sum_w2` have shape (n_1 + 2, ..., n_d + 2): every axis has [underflow, bins..., overflow].
The nine LHE scale variations (`LHEScaleWeight`, w_var / w_nominal) are stored as `sum_w_scale` with an extra
leading axis of length 9, plus `sum_w_all_scale` and `sum_w_tops_scale`.

    python user/ricardo/nnlo_reweighting/fill_histograms.py --sample nom_13 --jobs 12
"""
import argparse
import glob
import logging
import multiprocessing as mp
import os

import numpy as np
import uproot

from samples import HISTOGRAM_DIR, SIM_SAMPLES, VARIABLES, histogram_path
from theory import binnings

logger = logging.getLogger(__name__)

NUMBER_OF_SCALE_WEIGHTS = 9

BRANCHES = ['parton_hasGenTops', 'Generator_weight', 'LHEScaleWeight'] + sorted(
    {axis.branch for variable in VARIABLES.values() for axis in variable.axes})


def all_binnings() -> dict:
    """{'<variable>__<binning>': (axes, edges)} for all variables and their fixed-order binnings."""
    result = {}
    for variable_key, variable in VARIABLES.items():
        for binning_name, edges in binnings(variable_key).items():
            result[f'{variable_key}__{binning_name}'] = (variable.axes, edges)
    return result


BINNINGS = all_binnings()


def accumulate_file(path: str) -> dict:
    """Weighted sums for one ntuple file."""
    arrays = uproot.open(path)['Events'].arrays(BRANCHES, library='np')
    weights = arrays['Generator_weight'].astype(np.float64)  # float32 in the ntuples; sum in double precision
    has_tops = arrays['parton_hasGenTops'] == 1
    scale_ratios = np.stack(arrays['LHEScaleWeight']).astype(np.float64)
    if scale_ratios.shape != (len(weights), NUMBER_OF_SCALE_WEIGHTS):
        raise ValueError(f'{path}: expected {NUMBER_OF_SCALE_WEIGHTS} LHEScaleWeight entries per event, '
                         f'got shape {scale_ratios.shape}')
    scale_weights = weights[:, None] * scale_ratios  # (events, 9)
    result = {'sum_w_all': weights.sum(), 'sum_w_tops': weights[has_tops].sum(),
              'sum_w_all_scale': scale_weights.sum(axis=0), 'sum_w_tops_scale': scale_weights[has_tops].sum(axis=0)}
    for name, (axes, edges) in BINNINGS.items():
        padded_edges = [np.concatenate([[-np.inf], axis_edges, [np.inf]]) for axis_edges in edges]
        values = np.column_stack([np.abs(arrays[axis.branch][has_tops]) if axis.absolute
                                  else arrays[axis.branch][has_tops] for axis in axes])
        result[f'{name}__sum_w'] = np.histogramdd(values, padded_edges, weights=weights[has_tops])[0]
        result[f'{name}__sum_w2'] = np.histogramdd(values, padded_edges, weights=weights[has_tops] ** 2)[0]
        result[f'{name}__sum_w_scale'] = np.stack([
            np.histogramdd(values, padded_edges, weights=scale_weights[has_tops, variation])[0]
            for variation in range(NUMBER_OF_SCALE_WEIGHTS)])
    return result


def main():
    """Accumulate all files of a sample and write the npz."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sample', required=True, choices=sorted(SIM_SAMPLES))
    parser.add_argument('--jobs', type=int, default=1, help='worker processes over input files')
    parser.add_argument('--max-files', type=int, default=None, help='only read the first N files (quick test)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    sample = SIM_SAMPLES[args.sample]
    files = sorted(glob.glob(os.path.join(sample.ntuple_dir, '*.root')))
    if not files:
        raise RuntimeError(f'No ROOT files under {sample.ntuple_dir}')
    if args.max_files:
        files = files[:args.max_files]
    logger.info('%s: %d files in %s', args.sample, len(files), sample.ntuple_dir)

    totals = None
    with mp.Pool(args.jobs) as pool:
        for file_index, partial in enumerate(pool.imap_unordered(accumulate_file, files)):
            if totals is None:
                totals = partial
            else:
                for key in totals:
                    totals[key] = totals[key] + partial[key]
            if (file_index + 1) % 20 == 0:
                logger.info('processed %d / %d files', file_index + 1, len(files))

    for name, (_, edges) in BINNINGS.items():
        for axis, axis_edges in enumerate(edges):
            totals[f'{name}__edges{axis}'] = axis_edges
        sum_w = totals[f'{name}__sum_w']
        in_range = sum_w[(slice(1, -1),) * len(edges)].sum()
        logger.info('%-45s outside range %.4f  (fraction of two-top weight)',
                    name, 1 - in_range / totals['sum_w_tops'])
    logger.info('fraction of weight without two gen tops: %.4f', 1 - totals['sum_w_tops'] / totals['sum_w_all'])

    os.makedirs(HISTOGRAM_DIR, exist_ok=True)
    output_path = histogram_path(args.sample)
    np.savez(output_path, n_files=len(files), **totals)
    logger.info('wrote %s', output_path)


if __name__ == '__main__':
    main()
