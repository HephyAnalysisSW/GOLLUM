#!/usr/bin/env python3
"""Compare the individual STRIPPER scale variations with the envelope built from them and the original envelope.

One figure per STRIPPER variable and mode (absolute, normalised), one row per order (LO, NLO, NNLO). Everything is
relative to the central value: each scale variation as a line, the envelope built from them as a band, and the
envelope of the original files (absolute: data-for-CMS-PAS-TOP-20-006-mt172.5; normalised: Rene's -norm files) as
dashed black lines. Multi-dimensional distributions are unrolled (last axis fastest; vertical lines separate the
outer bins). The largest relative difference between the two envelopes is logged per order.

    python user/ricardo/nnlo_reweighting/diagnostics/plot_stripper_variations.py [--variables m_ttbar ...]
"""
import argparse
import logging
import os
import sys

# Repo root (for common.*) and the parent nnlo_reweighting directory (for samples, theory, plot_comparison).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import common.user as user
from common.helpers import copyIndexPHP
import common.syncer as syncer  # noqa: F401  (import hooks plt.savefig for the EOS sync)

from plot_comparison import draw_band
from samples import STRIPPER_SCALE_VARIATIONS, VARIABLES, scale_key
from theory import load_prediction, ratio, read_stripper_envelope, scale_envelope, stripper_envelope_path

logger = logging.getLogger(__name__)

hep.style.use('CMS')

ORDERS = ('LO', 'NLO', 'NNLO')
VARIATION_COLORS = ('#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6', '#a96b59')
VARIATION_LABELS = {scale_key(renormalization, factorization): rf'$(\mu_R, \mu_F) = ({renormalization:g}, '
                                                                 rf'{factorization:g})\,\mu_0$'
                    for renormalization, factorization in STRIPPER_SCALE_VARIATIONS}


def plot_variable(variable_key: str, normalized: bool, plot_dir: str):
    """One figure (one row per order) for a variable in one mode."""
    variable = VARIABLES[variable_key]
    dimension = len(variable.axes)
    figure, axes = plt.subplots(len(ORDERS), 1, sharex=True, figsize=(12, 12), gridspec_kw={'hspace': 0.08})
    for row, (axis, order) in enumerate(zip(axes, ORDERS)):
        histogram = load_prediction(f'stripper_{order.lower()}', variable_key, normalized)
        original = read_stripper_envelope(stripper_envelope_path(order, variable.stripper_observable, normalized),
                                          dimension)
        central = histogram['central'].ravel()
        # Unroll multi-dimensional distributions onto the bin index (last axis fastest).
        x_edges = histogram['edges'][0] if dimension == 1 else np.arange(central.size + 1)
        for color, (key, label) in zip(VARIATION_COLORS, VARIATION_LABELS.items()):
            hep.histplot(ratio(histogram[key].ravel(), central), x_edges, histtype='step', edges=False, ax=axis,
                         color=color, linewidth=1.5, label=label if row == 0 else None)
        low, high = (edge.ravel() for edge in scale_envelope(histogram))
        draw_band(axis, x_edges, ratio(central, central), ratio(low, central), ratio(high, central), 'grey',
                  label='envelope of the variations' if row == 0 else None)
        original_low, original_high = original['envelope_down'].ravel(), original['envelope_up'].ravel()
        for edge, label in [(original_low, 'envelope of the original files'), (original_high, None)]:
            hep.histplot(ratio(edge, central), x_edges, histtype='step', edges=False, ax=axis, color='black',
                         linestyle='--', linewidth=1.5, label=label if row == 0 else None)
        with np.errstate(divide='ignore', invalid='ignore'):
            difference = np.nanmax(np.abs(np.concatenate([low / original_low - 1, high / original_high - 1])))
        logger.info('%-35s %-10s %-4s max rel. difference new vs original envelope: %.1e', variable_key,
                    'normalised' if normalized else 'absolute', order, difference)
        if dimension > 1:
            inner_bins = len(histogram['edges'][-1]) - 1
            for boundary in range(inner_bins, central.size, inner_bins):
                axis.axvline(boundary, color='grey', linewidth=0.5)
        axis.axhline(1.0, color='grey', linewidth=1)
        axis.text(0.02, 0.9, f'STRIPPER {order}', transform=axis.transAxes, fontsize=16, va='top')
        axis.set_ylabel('variation / central', fontsize=16)
    if dimension == 1:
        axis_label = variable.axes[0].label + (f' [{variable.axes[0].unit}]' if variable.axes[0].unit else '')
    else:
        axis_label = 'bin index (' + r' $\times$ '.join(axis.label for axis in variable.axes) + ', last axis fastest)'
    axes[-1].set_xlabel(axis_label, fontsize=16)
    axes[-1].set_xlim(x_edges[0], x_edges[-1])
    hep.cms.label('Work in progress', data=False, ax=axes[0], rlabel='13 TeV')
    figure.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3, fontsize=13,
                  title=('normalised' if normalized else 'absolute') + ', parton level, inclusive phase space',
                  title_fontsize=14)
    figure.subplots_adjust(top=0.86)
    file_name = os.path.join(plot_dir, variable_key)
    plt.savefig(file_name + '.png', dpi=120, bbox_inches='tight')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close(figure)
    logger.info('saved %s.png', file_name)


def main():
    """Plot all (or the selected) STRIPPER variables in both modes."""
    stripper_variables = [key for key, variable in VARIABLES.items() if variable.stripper_observable is not None]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--variables', nargs='+', choices=stripper_variables, default=stripper_variables)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    for normalized in (False, True):
        plot_dir = os.path.join(user.plot_directory, 'nnlo_reweighting', 'stripper_variations',
                                'normalized' if normalized else 'absolute')
        copyIndexPHP(plot_dir)
        for variable_key in args.variables:
            plot_variable(variable_key, normalized, plot_dir)


if __name__ == '__main__':
    main()
