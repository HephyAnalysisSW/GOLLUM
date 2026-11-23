#!/usr/bin/env python3
"""Plot parton-level 1D/2D/3D tt-bar distributions of POWHEG hvq / bb4l against STRIPPER LO/NLO/NNLO and MATRIX NNLO.

One figure per (variable, fixed-order binning), with a ratio panel to the NNLO reference of that binning (to the
nominal POWHEG sample for simulation-only variables such as the top mass, which use their own binning). Multi-
dimensional variables get one panel per bin of the outer axes, showing the innermost (last) axis.
Simulations are scaled to the inclusive tt-bar cross section via sigma_sample / BR_DILEPTON / sum_w_all;
`--normalized` shows 1/sigma dsigma/dX instead.

    python user/ricardo/nnlo_reweighting/plot_comparison.py [--normalized] [--pdf-band] [--variables m_ttbar ...]
        [--truncate m_ttbar=1500] [--rebin m_ttbar=2] [--rebin m_ttbar_x_pt_ttbar=1,2]
"""
import argparse
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import common.user as user
from common.helpers import copyIndexPHP
import common.syncer as syncer  # noqa: F401  (import hooks plt.savefig for the EOS sync)

from samples import BR_DILEPTON, PREDICTIONS, RATIO_REFERENCE, SIM_SAMPLES, VARIABLES, histogram_path
from theory import bin_volumes, binnings, coarsen_edges, integral, load_prediction, rebin, same_edges

logger = logging.getLogger(__name__)

hep.style.use('CMS')

PANELS_PER_ROW = 4
RATIO_RANGE = (0.6, 1.4)


def load_simulation(sample_key: str, variable_key: str, binning_name: str, normalized: bool) -> dict:
    """Differential distribution of a simulated sample in the given binning (in-range bins only)."""
    sample = SIM_SAMPLES[sample_key]
    accumulated = np.load(histogram_path(sample_key))
    name = f'{variable_key}__{binning_name}'
    edges = binnings(variable_key)[binning_name]
    stored_keys = [f'{name}__edges{axis}' for axis in range(len(edges))]
    if not all(key in accumulated.files for key in stored_keys) or not all(
            np.array_equal(accumulated[key], axis_edges) for key, axis_edges in zip(stored_keys, edges)):
        raise RuntimeError(f'{histogram_path(sample_key)}: {name} missing or filled in an outdated binning, '
                           f'rerun fill_histograms.py --sample {sample_key}')
    in_range = (slice(1, -1),) * len(edges)
    sum_w = accumulated[f'{name}__sum_w'][in_range]
    sum_w2 = accumulated[f'{name}__sum_w2'][in_range]
    if normalized:
        # Normalise to the total (incl. under/overflow), as the STRIPPER normalised files are.
        scale = 1.0 / float(accumulated['sum_w_tops'])
    else:
        scale = sample.sigma / BR_DILEPTON / float(accumulated['sum_w_all'])
    volumes = bin_volumes(edges)
    return {
        'edges': edges,
        'central': sum_w * scale / volumes,
        'stat_err': np.sqrt(sum_w2) * scale / volumes,
        'sigma_total': sample.sigma / BR_DILEPTON * float(accumulated['sum_w_tops'] / accumulated['sum_w_all']),
    }


def draw_band(axis, edges, central, low, high, color, label=None, hatch=None):
    """Draw an asymmetric band [low, high] around central (filled, or hatched if hatch is given)."""
    hep.histplot(central, edges, yerr=np.array([central - low, high - central]), histtype='band', ax=axis,
                 facecolor='none' if hatch else color, edgecolor=color if hatch else 'none',
                 alpha=0.6 if hatch else 0.25, hatch=hatch, label=label)


def density_label(axes: tuple, normalized: bool) -> str:
    """Axis label d^n sigma / dX dY ... with its unit, e.g. 'd^2 sigma / dm d|y| [pb/GeV]'."""
    dimension = len(axes)
    differential = ' '.join('d' + axis.label.strip('$') for axis in axes)
    sigma_order = 'd' if dimension == 1 else f'd^{{{dimension}}}'
    prefix = r'1/\sigma\, ' if normalized else ''
    units = [axis.unit for axis in axes if axis.unit]
    if not units:
        unit = ''
    elif len(set(units)) == 1 and len(units) > 1:
        unit = f'{units[0]}$^{{{len(units)}}}$'
    else:
        unit = '/'.join(units)
    numerator = '1' if normalized else 'pb'
    unit_text = f' [{numerator}/{unit}]' if unit else ('' if normalized else ' [pb]')
    return rf'${prefix}{sigma_order}\sigma / {differential}$' + unit_text


def outer_bin_text(axes: tuple, edges: tuple, index: tuple) -> str:
    """One line per outer axis describing the outer bin of a panel, e.g. '400 < m(tt) < 500 GeV'."""
    lines = []
    for axis, axis_edges, bin_index in zip(axes, edges, index):
        unit = f' {axis.unit}' if axis.unit else ''
        lines.append(f'{axis_edges[bin_index]:g} < {axis.label} < {axis_edges[bin_index + 1]:g}{unit}')
    return '\n'.join(lines)


def draw_panel(axis_main, axis_ratio, edges: np.ndarray, index: tuple, predictions: dict, rebinned_keys: set,
               reference: dict, simulations: dict, show_pdf_band: bool):
    """Draw predictions and simulations of one outer bin (index) along the innermost axis (edges)."""
    reference_central = reference['central'][index]
    for key, histogram in predictions.items():
        prediction = PREDICTIONS[key]
        label = prediction.label + (' (rebinned)' if key in rebinned_keys else '')
        central = histogram['central'][index]
        style = {'color': prediction.color, 'linestyle': prediction.linestyle, 'linewidth': 2, 'edges': False}
        hep.histplot(central, edges, histtype='step', ax=axis_main, label=label, **style)
        hep.histplot(central / reference_central, edges, histtype='step', ax=axis_ratio, **style)
        if prediction.order != 'NNLO':
            continue
        bands = [('band_down', 'band_up', 'scale unc.', None)]
        if show_pdf_band and 'pdf_down' in histogram:
            bands.append(('pdf_down', 'pdf_up', 'PDF unc.', '////'))
        for low_key, high_key, band_label, hatch in bands:
            low, high = histogram[low_key][index], histogram[high_key][index]
            draw_band(axis_main, edges, central, low, high, prediction.color, f'{label} {band_label}', hatch)
            draw_band(axis_ratio, edges, central / reference_central, low / reference_central,
                      high / reference_central, prediction.color, hatch=hatch)

    samples = [SIM_SAMPLES[sample_key] for sample_key in simulations]
    centrals = [simulation['central'][index] for simulation in simulations.values()]
    errors = [simulation['stat_err'][index] for simulation in simulations.values()]
    style = {'histtype': 'errorbar', 'xoffsets': True, 'color': [sample.color for sample in samples],
             'marker': [sample.marker for sample in samples], 'markersize': 6}
    hep.histplot(centrals, edges, yerr=errors, ax=axis_main, label=[sample.label for sample in samples], **style)
    hep.histplot([central / reference_central for central in centrals], edges,
                 yerr=[error / reference_central for error in errors], ax=axis_ratio, **style)
    axis_ratio.axhline(1.0, color='grey', linewidth=1)
    axis_ratio.set_xlim(edges[0], edges[-1])
    axis_ratio.set_ylim(*RATIO_RANGE)


def plot_variable(variable_key: str, binning_name: str, normalized: bool, show_pdf_band: bool, plot_dir: str,
                  upper_edges: tuple, rebin_factors: tuple):
    """Make one comparison figure for a variable in one fixed-order binning.

    Each axis of the reference binning is optionally truncated at the first edge >= upper_edges[axis] (None: no
    truncation) and coarsened by rebin_factors[axis].
    """
    variable = VARIABLES[variable_key]
    reference_key = RATIO_REFERENCE[binning_name]
    predictions = {key: load_prediction(key, variable_key, normalized) for key in PREDICTIONS}
    predictions = {key: histogram for key, histogram in predictions.items() if histogram is not None}
    reference_edges = binnings(variable_key)[binning_name]
    edges = tuple(coarsen_edges(axis_edges, upper_edge, factor)
                  for axis_edges, upper_edge, factor in zip(reference_edges, upper_edges, rebin_factors))
    if not same_edges(edges, reference_edges):
        logger.info('%-14s %-9s plotting %s bins (truncate at %s, rebin factors %s)', variable_key, binning_name,
                    'x'.join(str(len(axis_edges) - 1) for axis_edges in edges), upper_edges, rebin_factors)
    # Predictions are rebinned onto the plotted binning; target bins that are not an exact union of their bins
    # stay NaN (drawn as gaps). Predictions without any such bin are dropped. "(rebinned)" in the legend marks
    # predictions whose native binning differs from the reference one.
    rebinned_keys = set()
    for key in list(predictions):
        histogram = predictions[key]
        if not same_edges(histogram['edges'], reference_edges):
            rebinned_keys.add(key)
        if same_edges(histogram['edges'], edges):
            continue
        rebinned = rebin(histogram, edges)
        if not rebinned['compatible'].any():
            logger.warning('%-14s %-9s %-14s cannot be rebinned onto this binning, not drawn',
                           variable_key, binning_name, key)
            del predictions[key]
            continue
        if not rebinned['compatible'].all():
            logger.warning('%-14s %-9s %-14s only %d/%d bins rebinnable, the others are drawn as gaps',
                           variable_key, binning_name, key, rebinned['compatible'].sum(), rebinned['compatible'].size)
        predictions[key] = rebinned
    for key, histogram in predictions.items():
        logger.info('%-14s %-9s %-14s sigma in range = %.3f', variable_key, binning_name, key, integral(histogram))

    simulations = {}
    for sample_key in SIM_SAMPLES:
        simulation = load_simulation(sample_key, variable_key, binning_name, normalized)
        if not same_edges(simulation['edges'], edges):
            simulation.update(rebin(simulation, edges))
        simulations[sample_key] = simulation
        logger.info('%-14s %-9s %-14s sigma in range = %.3f (two-top total %.2f)', variable_key, binning_name,
                    sample_key, integral(simulation), simulation['sigma_total'])
    reference = predictions[reference_key] if reference_key in PREDICTIONS else simulations[reference_key]

    # One panel per outer bin (a single panel for 1D), wrapped into rows of PANELS_PER_ROW.
    panel_indices = list(np.ndindex(*(len(axis_edges) - 1 for axis_edges in edges[:-1])))
    number_of_columns = min(len(panel_indices), PANELS_PER_ROW)
    number_of_rows = math.ceil(len(panel_indices) / number_of_columns)
    single_panel = len(panel_indices) == 1
    figure_size = (10, 10) if single_panel else (5.5 * number_of_columns, 7.5 * number_of_rows)
    figure = plt.figure(figsize=figure_size)
    # Fixed margin in inches above the panels for the CMS label (above the axis offset texts in the multi-panel
    # case); the multi-panel legend goes on top of it.
    top_margin_inches = 0.8 if single_panel else 1.2
    outer_grid = figure.add_gridspec(number_of_rows, number_of_columns, hspace=0.3, wspace=0.35,
                                     top=1 - top_margin_inches / figure_size[1])
    inner_axis = variable.axes[-1]
    panel_axes = []
    for panel_number, index in enumerate(panel_indices):
        row, column = divmod(panel_number, number_of_columns)
        cell = outer_grid[row, column].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        axis_main = figure.add_subplot(cell[0])
        axis_ratio = figure.add_subplot(cell[1], sharex=axis_main)
        axis_main.tick_params(labelbottom=False)
        draw_panel(axis_main, axis_ratio, edges[-1], index, predictions, rebinned_keys, reference, simulations,
                   show_pdf_band)
        if inner_axis.log_y:
            axis_main.set_yscale('log')
        else:
            axis_main.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
        if not single_panel:
            axis_main.text(0.05, 0.95, outer_bin_text(variable.axes, edges, index), transform=axis_main.transAxes,
                           va='top', fontsize=13)
            axis_main.set_ylim(top=axis_main.get_ylim()[1] * (10 if inner_axis.log_y else 1.3))
        axis_ratio.set_xlabel(inner_axis.label + (f' [{inner_axis.unit}]' if inner_axis.unit else ''),
                              fontsize=18 if single_panel else 16)
        if column == 0:
            axis_main.set_ylabel(density_label(variable.axes, normalized), fontsize=20 if single_panel else 16)
            reference_label = (PREDICTIONS[reference_key].label if reference_key in PREDICTIONS
                               else SIM_SAMPLES[reference_key].label.split(' (')[0]).replace(' ', '\n', 1)
            axis_ratio.set_ylabel(f'X / {reference_label}', fontsize=14)
        panel_axes.append(axis_main)

    legend_title = 'parton level, inclusive phase space'
    if single_panel:
        hep.cms.label('Work in progress', data=False, ax=panel_axes[0], rlabel='13 TeV')
        panel_axes[0].legend(fontsize=14, title=legend_title, title_fontsize=14, loc='best')
    else:
        # Invisible header axis spanning all columns, 0.45 inch above the panels, carrying the CMS label.
        left = panel_axes[0].get_position().x0
        right = panel_axes[number_of_columns - 1].get_position().x1
        top = panel_axes[0].get_position().y1
        header_axis = figure.add_axes([left, top + 0.45 / figure_size[1], right - left, 1e-3])
        header_axis.axis('off')
        hep.cms.label('Work in progress', data=False, ax=header_axis, rlabel='13 TeV', fontsize=20)
        handles, labels = panel_axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 4),
                      fontsize=16, title=legend_title, title_fontsize=16)

    suffix = ''.join(f'_{axis.key}upto{axis_edges[-1]:g}'
                     for axis, axis_edges, upper_edge in zip(variable.axes, edges, upper_edges) if upper_edge is not None)
    if any(factor != 1 for factor in rebin_factors):
        suffix += '_rebin' + 'x'.join(str(factor) for factor in rebin_factors)
    file_name = os.path.join(plot_dir, f'{variable_key}_{binning_name}{suffix}')
    plt.savefig(file_name + '.png', dpi=150 if single_panel else 100, bbox_inches='tight')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close(figure)
    logger.info('saved %s.png', file_name)


def parse_axis_options(options: list, value_type: type) -> dict:
    """Parse 'VARIABLE=V1[,V2,...]' (one value per axis, 'none' to skip an axis) into {variable: tuple}."""
    result = {}
    for option in options:
        variable_key, separator, values = option.partition('=')
        if not separator or variable_key not in VARIABLES:
            raise ValueError(f'expected VARIABLE=VALUE[,VALUE...] with VARIABLE in {sorted(VARIABLES)}, got {option!r}')
        values = values.split(',')
        if len(values) != len(VARIABLES[variable_key].axes):
            raise ValueError(f'{option!r}: {variable_key} has {len(VARIABLES[variable_key].axes)} axes, '
                             f'got {len(values)} values')
        result[variable_key] = tuple(None if value == 'none' else value_type(value) for value in values)
    return result


def main():
    """Plot all (or the selected) variables in all available fixed-order binnings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--normalized', action='store_true', help='plot 1/sigma dsigma/dX')
    parser.add_argument('--pdf-band', action='store_true', help='also draw the STRIPPER NNLO PDF band')
    parser.add_argument('--variables', nargs='+', choices=sorted(VARIABLES), default=list(VARIABLES),
                        help='only plot these variables (default: all)')
    parser.add_argument('--truncate', action='append', default=[], metavar='VARIABLE=VALUE[,VALUE...]',
                        help='per axis, cut the binning at the first edge >= VALUE (\'none\' to keep an axis), '
                             'e.g. m_ttbar=1500 or m_ttbar_x_pt_ttbar=none,150 (repeatable)')
    parser.add_argument('--rebin', action='append', default=[], metavar='VARIABLE=FACTOR[,FACTOR...]',
                        help='per axis, merge every FACTOR bins after truncation, e.g. m_ttbar=2 (repeatable)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    upper_edges = parse_axis_options(args.truncate, float)
    rebin_factors = parse_axis_options(args.rebin, int)

    plot_dir = os.path.join(user.plot_directory, 'nnlo_reweighting', 'parton_level',
                            'normalized' if args.normalized else 'absolute')
    copyIndexPHP(plot_dir)
    for variable_key in args.variables:
        dimension = len(VARIABLES[variable_key].axes)
        variable_upper_edges = upper_edges.get(variable_key, (None,) * dimension)
        variable_rebin_factors = tuple(factor or 1 for factor in rebin_factors.get(variable_key, (1,) * dimension))
        for binning_name in binnings(variable_key):
            plot_variable(variable_key, binning_name, args.normalized, args.pdf_band, plot_dir,
                          variable_upper_edges, variable_rebin_factors)


if __name__ == '__main__':
    main()
