#!/usr/bin/env python3
"""Plot parton-level y_tt, m_tt, pT_tt of POWHEG hvq / bb4l against STRIPPER LO/NLO/NNLO and MATRIX NNLO.

One figure per (variable, fixed-order binning), with a ratio panel to the NNLO reference of that binning.
Simulations are scaled to the inclusive tt-bar cross section via sigma_sample / BR_DILEPTON / sum_w_all;
`--normalized` shows 1/sigma dsigma/dX instead.

    python user/ricardo/nnlo_reweighting/plot_comparison.py [--normalized] [--pdf-band]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common.user as user
from common.helpers import copyIndexPHP
import common.syncer as syncer  # noqa: F401  (import hooks plt.savefig for the EOS sync)

from samples import BR_DILEPTON, PREDICTIONS, RATIO_REFERENCE, SIM_SAMPLES, VARIABLES, histogram_path
from theory import integral, load_prediction, rebin

logger = logging.getLogger(__name__)

plt.rcParams.update({'font.size': 18, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})


def load_simulation(sample_key: str, variable_key: str, binning_name: str, normalized: bool) -> dict:
    """Differential distribution of a simulated sample in the given binning (in-range bins only)."""
    sample = SIM_SAMPLES[sample_key]
    accumulated = np.load(histogram_path(sample_key))
    name = f'{variable_key}__{binning_name}'
    edges = accumulated[f'{name}__edges']
    widths = np.diff(edges)
    sum_w = accumulated[f'{name}__sum_w'][1:-1]
    sum_w2 = accumulated[f'{name}__sum_w2'][1:-1]
    if normalized:
        # Normalise to the total (incl. under/overflow), as the STRIPPER normalised files are.
        scale = 1.0 / float(accumulated['sum_w_tops'])
    else:
        scale = sample.sigma / BR_DILEPTON / float(accumulated['sum_w_all'])
    return {
        'edges': edges,
        'central': sum_w * scale / widths,
        'stat_err': np.sqrt(sum_w2) * scale / widths,
        'sigma_total': sample.sigma / BR_DILEPTON * float(accumulated['sum_w_tops'] / accumulated['sum_w_all']),
    }


def draw_band(axis, edges, low, high, color, label=None, hatch=None):
    """Draw a step-wise band between low and high."""
    axis.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]), step='post',
                      facecolor='none' if hatch else color, edgecolor=color if hatch else 'none',
                      alpha=0.6 if hatch else 0.25, hatch=hatch, linewidth=0, label=label)


def plot_variable(variable_key: str, binning_name: str, normalized: bool, show_pdf_band: bool, plot_dir: str):
    """Make one comparison figure for a variable in one fixed-order binning."""
    variable = VARIABLES[variable_key]
    reference_key = RATIO_REFERENCE[binning_name]
    predictions = {key: load_prediction(key, variable_key, normalized) for key in PREDICTIONS}
    predictions = {key: histogram for key, histogram in predictions.items() if histogram is not None}
    # Predictions in another binning are rebinned onto the reference binning; target bins that are not an
    # exact union of their bins stay NaN (drawn as gaps). Predictions without any such bin are dropped.
    reference = predictions[reference_key]
    edges = reference['edges']
    rebinned_keys = set()
    for key in list(predictions):
        histogram = predictions[key]
        if len(histogram['edges']) == len(edges) and np.allclose(histogram['edges'], edges):
            continue
        rebinned = rebin(histogram, edges)
        if not rebinned['compatible'].any():
            logger.warning('%-14s %-9s %-14s cannot be rebinned onto this binning, not drawn',
                        variable_key, binning_name, key)
            del predictions[key]
            continue
        if not rebinned['compatible'].all():
            logger.warning('%-14s %-9s %-14s only %d/%d bins rebinnable, the others are drawn as gaps',
                           variable_key, binning_name, key, rebinned['compatible'].sum(), len(edges) - 1)
        predictions[key] = rebinned
        rebinned_keys.add(key)

    figure, (axis_main, axis_ratio) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    centers = 0.5 * (edges[1:] + edges[:-1])

    for key, histogram in predictions.items():
        prediction = PREDICTIONS[key]
        label = prediction.label + (' (rebinned)' if key in rebinned_keys else '')
        # No baseline for rebinned curves, so NaN gaps are not closed by vertical lines down to the axis.
        baseline = None if key in rebinned_keys else 0
        axis_main.stairs(histogram['central'], edges, color=prediction.color, linestyle=prediction.linestyle,
                         linewidth=2, label=label, baseline=baseline)
        axis_ratio.stairs(histogram['central'] / reference['central'], edges, color=prediction.color,
                          linestyle=prediction.linestyle, linewidth=2, baseline=baseline)
        if prediction.order == 'NNLO':
            draw_band(axis_main, edges, histogram['band_down'], histogram['band_up'], prediction.color,
                      label=f'{label} scale unc.')
            draw_band(axis_ratio, edges, histogram['band_down'] / reference['central'],
                      histogram['band_up'] / reference['central'], prediction.color)
            if show_pdf_band and 'pdf_down' in histogram:
                draw_band(axis_main, edges, histogram['pdf_down'], histogram['pdf_up'], prediction.color,
                          label=f'{prediction.label} PDF unc.', hatch='////')
                draw_band(axis_ratio, edges, histogram['pdf_down'] / reference['central'],
                          histogram['pdf_up'] / reference['central'], prediction.color, hatch='////')
        logger.info('%-14s %-9s %-14s sigma in range = %.2f', variable_key, binning_name, key, integral(histogram))

    number_of_samples = len(SIM_SAMPLES)
    for sample_index, (sample_key, sample) in enumerate(SIM_SAMPLES.items()):
        simulation = load_simulation(sample_key, variable_key, binning_name, normalized)
        # Offset markers horizontally inside each bin so the error bars stay separable.
        offset = (sample_index - (number_of_samples - 1) / 2) * 0.15 * np.diff(edges)
        axis_main.errorbar(centers + offset, simulation['central'], yerr=simulation['stat_err'], fmt=sample.marker,
                           color=sample.color, markersize=6, label=sample.label)
        axis_ratio.errorbar(centers + offset, simulation['central'] / reference['central'],
                            yerr=simulation['stat_err'] / reference['central'], fmt=sample.marker,
                            color=sample.color, markersize=6)
        logger.info('%-14s %-9s %-14s sigma in range = %.2f (two-top total %.2f)', variable_key, binning_name,
                    sample_key, integral(simulation), simulation['sigma_total'])

    unit_suffix = f'/{variable.unit}' if variable.unit else ''
    if normalized:
        axis_main.set_ylabel(r'$1/\sigma\, d\sigma/dx$' + (f' [1{unit_suffix}]' if variable.unit else ''))
    else:
        axis_main.set_ylabel(r'$d\sigma/dx$' + f' [pb{unit_suffix}]')
    if variable.log_y:
        axis_main.set_yscale('log')
    axis_main.legend(fontsize=14, title_fontsize=14, loc='best', ncol=1, title='parton level, inclusive phase space')
    axis_main.text(0.0, 1.01, r'$\bf{CMS}$ $\it{Simulation\ Work\ in\ progress}$', transform=axis_main.transAxes,
                   fontsize=20, va='bottom')
    axis_main.text(1.0, 1.01, '13 TeV', transform=axis_main.transAxes, fontsize=18, va='bottom', ha='right')

    axis_ratio.axhline(1.0, color='grey', linewidth=1)
    axis_ratio.set_ylabel(f'X / {PREDICTIONS[reference_key].label.replace(" ", chr(10), 1)}', fontsize=16)
    axis_ratio.set_xlabel(variable.axis_label + (f' [{variable.unit}]' if variable.unit else ''))
    axis_ratio.set_xlim(edges[0], edges[-1])
    axis_ratio.set_ylim(0.6, 1.4)

    file_name = os.path.join(plot_dir, f'{variable_key}_{binning_name}')
    plt.savefig(file_name + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close(figure)
    logger.info('saved %s.png', file_name)


def main():
    """Plot all variables in all available fixed-order binnings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--normalized', action='store_true', help='plot 1/sigma dsigma/dX')
    parser.add_argument('--pdf-band', action='store_true', help='also draw the STRIPPER NNLO PDF band')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    plot_dir = os.path.join(user.plot_directory, 'nnlo_reweighting', 'parton_level',
                            'normalized' if args.normalized else 'absolute')
    copyIndexPHP(plot_dir)
    for variable_key in VARIABLES:
        binning_names = ['stripper'] + (['matrix'] if VARIABLES[variable_key].matrix_file else [])
        for binning_name in binning_names:
            plot_variable(variable_key, binning_name, args.normalized, args.pdf_band, plot_dir)


if __name__ == '__main__':
    main()
