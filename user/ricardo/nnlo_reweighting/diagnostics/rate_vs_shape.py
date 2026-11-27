"""Split scale variations into rate (total) and shape (per bin, normalised) parts, POWHEG and STRIPPER."""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, 'user/ricardo/nnlo_reweighting')
from samples import LHE_SCALE_VARIATIONS, histogram_path  # noqa: E402
from theory import bin_volumes, integral, load_prediction, scale_envelope  # noqa: E402

logger = logging.getLogger(__name__)
# LHEScaleWeight indices of the 7-point envelope and their (muR, muF) labels
SCALE_ENVELOPE_INDICES = tuple(LHE_SCALE_VARIATIONS.values())
SCALE_LABELS = {index: f'({renormalization:g},{factorization:g})'
                for (renormalization, factorization), index in LHE_SCALE_VARIATIONS.items()}


def powheg_rate_and_shape(variable: str, sample: str) -> None:
    """Per variation: total change (rate), per-bin absolute change, per-bin normalised change (shape)."""
    accumulated = np.load(histogram_path(sample))
    name = f'{variable}__stripper'
    edges = accumulated[f'{name}__edges0']
    nominal_bins = accumulated[f'{name}__sum_w'][1:-1]
    nominal_total = float(accumulated['sum_w_tops'])  # incl. under/overflow, as in plot_comparison.py
    header = ' '.join(f'{low:.4g}-{high:.4g}'.rjust(11) for low, high in zip(edges[:-1], edges[1:]))
    logger.info('POWHEG %s %s, relative change w.r.t. nominal [%%]', sample, variable)
    logger.info('%-10s %-6s %8s %s', 'variation', 'type', 'total', header)
    for index in SCALE_ENVELOPE_INDICES:
        variation_bins = accumulated[f'{name}__sum_w_scale'][index][1:-1]
        rate = accumulated['sum_w_tops_scale'][index] / nominal_total
        absolute = variation_bins / nominal_bins
        normalised = absolute / rate  # each variation normalised to its own total
        for kind, values in (('abs', absolute), ('norm', normalised)):
            row = ' '.join(f'{100 * (value - 1):11.1f}' for value in values)
            total = f'{100 * (rate - 1):8.1f}' if kind == 'abs' else f'{0.0:8.1f}'
            logger.info('%-10s %-6s %s %s', SCALE_LABELS[index], kind, total, row)
    powheg_envelope(accumulated, name, nominal_bins, nominal_total)


def powheg_envelope(accumulated, name: str, nominal_bins: np.ndarray, nominal_total: float) -> None:
    """Per-bin 7-point envelope (as scale_envelope in plot_comparison.py), absolute and normalised, and the
    integral of each envelope edge relative to the nominal: != 0 means the envelope carries a rate difference."""
    variation_bins = np.stack([nominal_bins] + [accumulated[f'{name}__sum_w_scale'][index][1:-1]
                                                for index in SCALE_ENVELOPE_INDICES])
    variation_totals = np.array([nominal_total] + [accumulated['sum_w_tops_scale'][index]
                                                   for index in SCALE_ENVELOPE_INDICES])
    labels = ['nominal'] + [SCALE_LABELS[index] for index in SCALE_ENVELOPE_INDICES]
    for kind, contents in (('abs', variation_bins / nominal_total),
                           ('norm', variation_bins / variation_totals[:, None])):
        # contents: fraction of the nominal (abs) or of the own total (norm) per bin; widths cancel in ratios
        nominal = contents[0]
        down, up = contents.min(axis=0), contents.max(axis=0)
        logger.info('envelope %-4s integral of edges relative to nominal: down %+.2f%%, up %+.2f%%', kind,
                    100 * (down.sum() / nominal.sum() - 1), 100 * (up.sum() / nominal.sum() - 1))
        logger.info('  per bin [%%]: %s', ' '.join(f'{100 * (low / central - 1):+.1f}/{100 * (high / central - 1):+.1f}'
                                                  for low, high, central in zip(down, up, nominal)))
        logger.info('  edge from:   %s', ' '.join(f'{labels[low]}|{labels[high]}' for low, high in
                                                  zip(contents.argmin(axis=0), contents.argmax(axis=0))))


def stripper_bands(variable: str, order: str) -> None:
    """Absolute vs normalised band edges per bin, and the totals of the band-edge histograms."""
    absolute = load_prediction(f'stripper_{order}', variable, normalized=False)
    normalised = load_prediction(f'stripper_{order}', variable, normalized=True)
    edges = absolute['edges'][0]
    envelopes = {id(histogram): scale_envelope(histogram) for histogram in (absolute, normalised)}
    absolute_down, absolute_up = envelopes[id(absolute)]
    volumes = bin_volumes(absolute['edges'])
    logger.info('STRIPPER %s %s: band edges relative to central [%%]', order.upper(), variable)
    logger.info('  totals of band-edge histograms (absolute): down %+.1f%%, up %+.1f%%  (central %.1f pb)',
                100 * (np.sum(absolute_down * volumes) / integral(absolute) - 1),
                100 * (np.sum(absolute_up * volumes) / integral(absolute) - 1), integral(absolute))
    logger.info('%13s %15s %15s', 'bin', 'absolute', 'normalised')
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        cells = []
        for histogram in (absolute, normalised):
            central = histogram['central'][bin_index]
            down, up = envelopes[id(histogram)]
            cells.append(f'{100 * (down[bin_index] / central - 1):+6.1f}/'
                         f'{100 * (up[bin_index] / central - 1):+6.1f}')
        logger.info('%6.4g-%-6.4g %15s %15s', low, high, *cells)


def main():
    """Run for the requested 1D variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variables', nargs='+', default=['y_ttbar', 'pt_ttbar'])
    parser.add_argument('--sample', default='nom_13')
    parser.add_argument('--order', default='nlo', choices=['lo', 'nlo', 'nnlo'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if not os.path.isdir('user/ricardo/nnlo_reweighting'):
        raise RuntimeError('run from the repo root')
    for variable in args.variables:
        powheg_rate_and_shape(variable, args.sample)
        stripper_bands(variable, args.order)


if __name__ == '__main__':
    main()
