#!/usr/bin/env python3
"""Fill parton-level tt-bar histograms (in the fixed-order binnings) from the gen ntuples of one sample.

Output: `histograms/<sample>.npz` with `sum_w_all` (all events, for the normalisation), `sum_w_tops`
(events with two gen tops) and, per `<variable>__<binning>`, `edges`, `sum_w`, `sum_w2`. The `sum_w`/`sum_w2`
arrays have len(edges)+1 entries: [underflow, bins..., overflow].

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
j
logger = logging.getLogger(__name__)

BRANCHES = ['parton_hasGenTops', 'Generator_weight'] + [variable.branch for variable in VARIABLES.values()]


def all_binnings() -> dict:
    """{'<variable>__<binning>': (branch, edges)} for all variables and their fixed-order binnings."""
    result = {}
    for variable_key, variable in VARIABLES.items():
        for binning_name, edges in binnings(variable_key).items():
            result[f'{variable_key}__{binning_name}'] = (variable.branch, edges)
    return result


BINNINGS = all_binnings()


def accumulate_file(path: str) -> dict:
    """Weighted sums for one ntuple file."""
    arrays = uproot.open(path)['Events'].arrays(BRANCHES, library='np')
    weights = arrays['Generator_weight']
    has_tops = arrays['parton_hasGenTops'] == 1
    result = {'sum_w_all': weights.sum(), 'sum_w_tops': weights[has_tops].sum()}
    for name, (branch, edges) in BINNINGS.items():
        padded_edges = np.concatenate([[-np.inf], edges, [np.inf]])
        values = arrays[branch][has_tops]
        result[f'{name}__sum_w'] = np.histogram(values, padded_edges, weights=weights[has_tops])[0]
        result[f'{name}__sum_w2'] = np.histogram(values, padded_edges, weights=weights[has_tops] ** 2)[0]
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
        totals[f'{name}__edges'] = edges
        sum_w = totals[f'{name}__sum_w']
        logger.info('%-22s underflow %.4f  overflow %.4f  (fraction of two-top weight)',
                    name, sum_w[0] / totals['sum_w_tops'], sum_w[-1] / totals['sum_w_tops'])
    logger.info('fraction of weight without two gen tops: %.4f', 1 - totals['sum_w_tops'] / totals['sum_w_all'])

    os.makedirs(HISTOGRAM_DIR, exist_ok=True)
    output_path = histogram_path(args.sample)
    np.savez(output_path, n_files=len(files), **totals)
    logger.info('wrote %s', output_path)


if __name__ == '__main__':
    main()
