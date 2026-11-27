"""Readers for the fixed-order predictions (STRIPPER from Rene Poncelet, MATRIX from Niels), scale envelopes and
rebinning.

Every histogram is a dict with `edges`, a tuple of per-axis bin-edge arrays (one entry for 1D), and per-bin arrays of
shape (n_1, ..., n_d), differential in every axis (pb/[X]/[Y]/..., or 1/[X]/... when normalised):
- `central` and `mc_err`;
- the individual scale variations `scale_<muR>_<muF>` (keys from samples.scale_key). The scale uncertainty is their
  envelope together with the central value (`scale_envelope`), taken only when drawing, so rebinning and
  normalisation act on every scale choice separately. MATRIX only provides the envelope edges, kept as the two
  pseudo-variations `scale_matrix_down` and `scale_matrix_up`;
- STRIPPER NNLO: `pdf_down`, `pdf_up`.
"""
import functools
import itertools
import os

import numpy as np

from samples import (MATRIX_DIR, PREDICTIONS, STRIPPER_ABSOLUTE_DIR, STRIPPER_NORMALIZED_DIR,
                     STRIPPER_SCALE_VARIATIONS, STRIPPER_TOTAL_OBSERVABLE, STRIPPER_VARIATIONS_DIR, VARIABLES,
                     scale_key)

FB_TO_PB = 1e-3
# Absolute tolerance for two bin edges to count as identical when rebinning.
EDGE_TOLERANCE = 1e-6
# Columns per row of a STRIPPER variations file after the bin edges: central + 6 variations, their 7 MC errors, and at
# NNLO the appended pdf_up, pdf_down, pdf_up_normalized, pdf_down_normalized.
STRIPPER_VARIATION_COLUMNS = 14
STRIPPER_PDF_COLUMNS = 4


def stripper_grid(table: np.ndarray, dimension: int, path: str) -> tuple[tuple, tuple]:
    """Per-axis bin edges and shape of a STRIPPER table; checks the bins form a regular grid in C order."""
    edge_columns = 2 * dimension
    edges = tuple(np.unique(table[:, 2 * axis:2 * axis + 2]) for axis in range(dimension))
    bins_per_axis = [np.column_stack([axis_edges[:-1], axis_edges[1:]]) for axis_edges in edges]
    grid = np.array([np.concatenate(cell) for cell in itertools.product(*bins_per_axis)])
    if grid.shape != table[:, :edge_columns].shape or not np.allclose(grid, table[:, :edge_columns]):
        raise ValueError(f'{path}: bins do not form a contiguous regular grid in C order')
    return edges, tuple(len(axis_edges) - 1 for axis_edges in edges)


def read_stripper(path: str, dimension: int, normalized_pdf: bool) -> dict:
    """Read a STRIPPER file with the individual scale variations (absolute, pb/[X]...) on a regular grid.

    Columns after the edges: central, scale_var1..6, mcerr_central, mcerr_var1..6 [, pdf_up, pdf_down,
    pdf_up_normalized, pdf_down_normalized] (PDF columns appended at NNLO, see the README of the directory).
    `normalized_pdf` selects the normalised PDF columns.
    """
    table = np.loadtxt(path, ndmin=2)
    edge_columns = 2 * dimension
    expected = (edge_columns + STRIPPER_VARIATION_COLUMNS,
                edge_columns + STRIPPER_VARIATION_COLUMNS + STRIPPER_PDF_COLUMNS)
    if table.shape[1] not in expected:
        raise ValueError(f'{path}: expected {expected} columns for a {dimension}D STRIPPER histogram with scale '
                         f'variations, found {table.shape[1]}')
    edges, shape = stripper_grid(table, dimension, path)
    values = table[:, edge_columns:].reshape(shape + (table.shape[1] - edge_columns,))
    histogram = {'edges': edges, 'central': values[..., 0], 'mc_err': values[..., 7]}
    for column, (renormalization_factor, factorization_factor) in enumerate(STRIPPER_SCALE_VARIATIONS, start=1):
        histogram[scale_key(renormalization_factor, factorization_factor)] = values[..., column]
    if values.shape[-1] == STRIPPER_VARIATION_COLUMNS + STRIPPER_PDF_COLUMNS:
        pdf = values[..., 16:18] if normalized_pdf else values[..., 14:16]
        histogram['pdf_down'] = pdf.min(axis=-1)
        histogram['pdf_up'] = pdf.max(axis=-1)
    return histogram


def read_stripper_envelope(path: str, dimension: int) -> dict:
    """Read an original STRIPPER envelope file (edges, central, scale_up, scale_down, mcerr [, pdf_up, pdf_down]).

    Only used to cross-check the envelopes built from the individual variations; returns `central`,
    `envelope_down`, `envelope_up`.
    """
    table = np.loadtxt(path, ndmin=2)
    edges, shape = stripper_grid(table, dimension, path)
    values = table[:, 2 * dimension:].reshape(shape + (table.shape[1] - 2 * dimension,))
    return {'edges': edges, 'central': values[..., 0], 'envelope_down': values[..., 1:3].min(axis=-1),
            'envelope_up': values[..., 1:3].max(axis=-1)}


def read_matrix(path: str) -> dict:
    """Read a 1D MATRIX histogram (fb/[X]; columns edges, central+err, scale-min+err, scale-max+err, rel. down/up).

    MATRIX only gives the envelope edges; they are stored as the pseudo-variations `scale_matrix_down` / `_up`.
    """
    rows = []
    with open(path) as matrix_file:
        for line in matrix_file:
            if line.lstrip().startswith('#') or not line.strip():
                continue
            rows.append([float(token) for token in line.replace('\\', ' ').replace('%', ' ').split()])
    table = np.array(rows)
    if not np.allclose(table[1:, 0], table[:-1, 1]):
        raise ValueError(f'{path}: bins are not contiguous')
    edges = np.append(table[:, 0], table[-1, 1])
    scale_edges = table[:, [4, 6]] * FB_TO_PB
    return {
        'edges': (edges,),
        'central': table[:, 2] * FB_TO_PB,
        'scale_matrix_down': scale_edges.min(axis=1),
        'scale_matrix_up': scale_edges.max(axis=1),
        'mc_err': table[:, 3] * FB_TO_PB,
    }


def stripper_path(order: str, observable: str) -> str:
    """File with the individual scale variations of an inclusive STRIPPER prediction (absolute)."""
    return os.path.join(STRIPPER_VARIATIONS_DIR, f'incl_{order}_mt172.5_HT4_NNPDF31_{observable}.dat')


def stripper_envelope_path(order: str, observable: str, normalized: bool) -> str:
    """Original STRIPPER envelope file (absolute or Rene's normalised version), for cross-checks."""
    directory = STRIPPER_NORMALIZED_DIR if normalized else STRIPPER_ABSOLUTE_DIR
    return os.path.join(directory, f'incl_{order}_mt172.5_HT4_NNPDF31_{observable}.dat')


def bin_volumes(edges: tuple) -> np.ndarray:
    """Bin volumes (product of the per-axis widths), shape (n_1, ..., n_d)."""
    return functools.reduce(np.multiply.outer, [np.diff(axis_edges) for axis_edges in edges])


def integral(histogram: dict, key: str = 'central') -> float:
    """Integral of a differential histogram over its bins; NaN bins (not rebinnable) are skipped."""
    return float(np.nansum(histogram[key] * bin_volumes(histogram['edges'])))


def scale_keys(histogram: dict) -> list:
    """Keys of the individual scale variations of a histogram."""
    return [key for key in histogram if key.startswith('scale_')]


@functools.lru_cache(maxsize=None)
def stripper_totals(order: str) -> dict:
    """Total cross section of the central value and of every scale choice at one order, from the full-phase-space
    observable STRIPPER_TOTAL_OBSERVABLE."""
    histogram = read_stripper(stripper_path(order, STRIPPER_TOTAL_OBSERVABLE), 1, normalized_pdf=False)
    return {key: integral(histogram, key) for key in ['central'] + scale_keys(histogram)}


def load_prediction(prediction_key: str, variable_key: str, normalized: bool) -> dict | None:
    """Load a prediction for a variable; None if the generator does not provide the variable.

    Normalised as in Rene's normalised files: every scale choice is divided by its own total cross section (for
    STRIPPER the full-phase-space total of that scale choice, which reproduces Rene's files to 1e-4; the normalised
    PDF band is taken from his files). MATRIX only provides the envelope edges, which are each normalised to their own
    integral over the MATRIX range, as if each edge were a single scale choice.
    """
    prediction = PREDICTIONS[prediction_key]
    variable = VARIABLES[variable_key]
    if prediction.generator == 'stripper':
        if variable.stripper_observable is None:
            return None
        histogram = read_stripper(stripper_path(prediction.order, variable.stripper_observable), len(variable.axes),
                                  normalized_pdf=normalized)
        totals = stripper_totals(prediction.order)
    elif prediction.generator == 'matrix':
        if variable.matrix_file is None:
            return None
        histogram = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))
        totals = {key: integral(histogram, key) for key in ['central'] + scale_keys(histogram)}
    else:
        raise ValueError(f'Unknown generator {prediction.generator}')
    if normalized:
        histogram['mc_err'] = histogram['mc_err'] / totals['central']
        for key in ['central'] + scale_keys(histogram):
            histogram[key] = histogram[key] / totals[key]
    return histogram


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Bin-wise numerator / denominator, NaN (drawn as a gap) where the denominator is zero, e.g. LO pT(tt) > 0."""
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan), where=denominator != 0)


def scale_envelope(histogram: dict, index: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge of the scale envelope (central value and all scale variations) in one panel."""
    keys = scale_keys(histogram)
    if not keys:
        raise ValueError('histogram has no scale variations')
    variations = np.stack([histogram['central'][index]] + [histogram[key][index] for key in keys])
    return variations.min(axis=0), variations.max(axis=0)


def correlated_ratio_envelope(numerator: dict, denominator: dict, index: tuple = ()) -> tuple | None:
    """Envelope of numerator / denominator with the same (muR, muF) in both (fully correlated scale variations),
    including the central ratio; None if the two histograms share no scale variation (e.g. MATRIX)."""
    common = [key for key in scale_keys(numerator) if key in denominator]
    if not common:
        return None
    ratios = np.stack([ratio(numerator['central'][index], denominator['central'][index])] +
                      [ratio(numerator[key][index], denominator[key][index]) for key in common])
    return ratios.min(axis=0), ratios.max(axis=0)


def same_edges(edges: tuple, other_edges: tuple) -> bool:
    """True if two (multi-dimensional) binnings are identical within EDGE_TOLERANCE."""
    return len(edges) == len(other_edges) and all(
        len(axis_edges) == len(other_axis_edges)
        and np.allclose(axis_edges, other_axis_edges, rtol=0, atol=EDGE_TOLERANCE)
        for axis_edges, other_axis_edges in zip(edges, other_edges))


def coarsen_edges(edges: np.ndarray, upper_edge: float | None = None, factor: int = 1) -> np.ndarray:
    """Truncate one axis at the first edge >= upper_edge, then merge every `factor` bins (last group may be smaller)."""
    edges = np.asarray(edges)
    if upper_edge is not None:
        stop = int(np.searchsorted(edges, upper_edge - EDGE_TOLERANCE))
        if stop == 0 or stop == len(edges):
            raise ValueError(f'truncation value {upper_edge} outside the binning range [{edges[0]}, {edges[-1]}]')
        edges = edges[:stop + 1]
    if factor < 1:
        raise ValueError(f'rebinning factor must be >= 1, got {factor}')
    coarse_edges = edges[::factor]
    if coarse_edges[-1] != edges[-1]:
        coarse_edges = np.append(coarse_edges, edges[-1])
    return coarse_edges


def rebinnable_bins(source_edges: np.ndarray, target_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one axis: mask over target bins that are an exact union of source bins, and the source index of each
    target edge (-1 for target edges that are not source edges)."""
    source_index = np.full(len(target_edges), -1)
    for target_position, edge in enumerate(target_edges):
        matches = np.flatnonzero(np.isclose(source_edges, edge, rtol=0, atol=EDGE_TOLERANCE))
        if len(matches):
            source_index[target_position] = matches[0]
    compatible = (source_index[:-1] >= 0) & (source_index[1:] >= 0)
    return compatible, source_index


def merge_matrix(source_edges: np.ndarray, target_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one axis: 0/1 matrix (n_target x n_source) summing source bins into target bins, and the compatible mask."""
    compatible, source_index = rebinnable_bins(source_edges, target_edges)
    matrix = np.zeros((len(target_edges) - 1, len(source_edges) - 1))
    for target_bin in np.flatnonzero(compatible):
        matrix[target_bin, source_index[target_bin]:source_index[target_bin + 1]] = 1.0
    return matrix, compatible


def merge_axes(matrices: list, array: np.ndarray) -> np.ndarray:
    """Apply one merge matrix per axis to an N-dimensional array of bin contents."""
    for axis, matrix in enumerate(matrices):
        array = np.moveaxis(np.tensordot(matrix, array, axes=(1, axis)), 0, axis)
    return array


def rebin(histogram: dict, target_edges: tuple) -> dict:
    """Rebin a differential histogram onto target_edges (one array per axis); target bins that are not an exact
    union of source bins (in every axis) or that contain NaN source bins are NaN.

    All per-bin arrays (same shape as `central`) are rebinned: values and bands as volume-weighted sums (bands
    treated as fully correlated across the merged bins), the errors `mc_err` (theory) and `stat_err` (simulation)
    in quadrature. Other entries are not carried over. The result has the boolean mask `compatible` over target bins.
    """
    target_edges = tuple(np.asarray(axis_edges) for axis_edges in target_edges)
    if len(target_edges) != len(histogram['edges']):
        raise ValueError(f'cannot rebin a {len(histogram["edges"])}D histogram onto a {len(target_edges)}D binning')
    matrices, masks = zip(*(merge_matrix(source, target) for source, target in zip(histogram['edges'], target_edges)))
    compatible = functools.reduce(np.logical_and.outer, masks)
    source_volumes = bin_volumes(histogram['edges'])
    target_volumes = bin_volumes(target_edges)
    result = {'edges': target_edges, 'compatible': compatible}
    for key, histogram_values in histogram.items():
        # Every per-bin array is rebinned (central, errors, every scale variation `scale_*`, PDF band edges).
        if key in ('edges', 'compatible') or np.shape(histogram_values) != np.shape(histogram['central']):
            continue
        content = histogram_values * source_volumes
        contains_nan = merge_axes(matrices, np.isnan(content).astype(float)) > 0
        content = np.nan_to_num(content, nan=0.0)
        if key in ('mc_err', 'stat_err'):
            merged = np.sqrt(merge_axes(matrices, content ** 2))
        else:
            merged = merge_axes(matrices, content)
        values = merged / target_volumes
        values[~compatible | contains_nan] = np.nan
        result[key] = values
    return result


def binnings(variable_key: str) -> dict:
    """Bin edges (tuple per axis) per binning name: 'stripper' (and 'matrix' where MATRIX provides the variable),
    or 'simulation' (the variable's own edges) for variables without fixed-order prediction."""
    variable = VARIABLES[variable_key]
    if variable.stripper_observable is None:
        return {'simulation': variable.edges}
    result = {'stripper': read_stripper(stripper_path('NNLO', variable.stripper_observable), len(variable.axes),
                                        normalized_pdf=False)['edges']}
    if variable.matrix_file is not None:
        result['matrix'] = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))['edges']
    return result
