"""Readers for the fixed-order predictions (STRIPPER from Rene Poncelet, MATRIX from Niels), and rebinning.

Every histogram is a dict with `edges`, a tuple of per-axis bin-edge arrays (one entry for 1D), and per-bin
arrays of shape (n_1, ..., n_d) `central`, `mc_err`, `band_down`, `band_up` (scale-variation envelope), all
differential in every axis (pb/[X]/[Y]/...). STRIPPER NNLO additionally provides `pdf_down`, `pdf_up`.
"""
import functools
import itertools
import os

import numpy as np

from samples import MATRIX_DIR, PREDICTIONS, STRIPPER_ABSOLUTE_DIR, STRIPPER_NORMALIZED_DIR, VARIABLES

FB_TO_PB = 1e-3
# Absolute tolerance for two bin edges to count as identical when rebinning.
EDGE_TOLERANCE = 1e-6


def read_stripper(path: str, dimension: int) -> dict:
    """Read a STRIPPER histogram of the given dimension on a regular grid (last axis running fastest).

    Columns: (low, high) per axis, central, scale_up, scale_down, mcerr [, pdf_up, pdf_down]. The PDF columns are
    only present at NNLO. The dimension cannot be inferred from the file: 8 columns is 1D with PDF or 2D without.
    """
    table = np.loadtxt(path, ndmin=2)
    edge_columns = 2 * dimension
    if table.shape[1] not in (edge_columns + 4, edge_columns + 6):
        raise ValueError(f'{path}: expected {edge_columns + 4} or {edge_columns + 6} columns for a {dimension}D '
                         f'STRIPPER histogram, found {table.shape[1]}')
    edges = tuple(np.unique(table[:, 2 * axis:2 * axis + 2]) for axis in range(dimension))
    bins_per_axis = [np.column_stack([axis_edges[:-1], axis_edges[1:]]) for axis_edges in edges]
    grid = np.array([np.concatenate(cell) for cell in itertools.product(*bins_per_axis)])
    if grid.shape != table[:, :edge_columns].shape or not np.allclose(grid, table[:, :edge_columns]):
        raise ValueError(f'{path}: bins do not form a contiguous regular grid in C order')
    shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    values = table[:, edge_columns:].reshape(shape + (table.shape[1] - edge_columns,))
    histogram = {
        'edges': edges,
        'central': values[..., 0],
        'band_down': values[..., 1:3].min(axis=-1),
        'band_up': values[..., 1:3].max(axis=-1),
        'mc_err': values[..., 3],
    }
    if values.shape[-1] == 6:
        histogram['pdf_down'] = values[..., 4:6].min(axis=-1)
        histogram['pdf_up'] = values[..., 4:6].max(axis=-1)
    return histogram


def read_matrix(path: str) -> dict:
    """Read a 1D MATRIX histogram (fb/[X]; columns edges, central+err, scale-min+err, scale-max+err, rel. down/up)."""
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
    scale_variations = table[:, [4, 6]] * FB_TO_PB
    return {
        'edges': (edges,),
        'central': table[:, 2] * FB_TO_PB,
        'band_down': scale_variations.min(axis=1),
        'band_up': scale_variations.max(axis=1),
        'mc_err': table[:, 3] * FB_TO_PB,
    }


def stripper_path(order: str, observable: str, normalized: bool) -> str:
    """File name of an inclusive STRIPPER prediction."""
    directory = STRIPPER_NORMALIZED_DIR if normalized else STRIPPER_ABSOLUTE_DIR
    return os.path.join(directory, f'incl_{order}_mt172.5_HT4_NNPDF31_{observable}.dat')


def bin_volumes(edges: tuple) -> np.ndarray:
    """Bin volumes (product of the per-axis widths), shape (n_1, ..., n_d)."""
    return functools.reduce(np.multiply.outer, [np.diff(axis_edges) for axis_edges in edges])


def integral(histogram: dict, key: str = 'central') -> float:
    """Integral of a differential histogram over its bins; NaN bins (not rebinnable) are skipped."""
    return float(np.nansum(histogram[key] * bin_volumes(histogram['edges'])))


def load_prediction(prediction_key: str, variable_key: str, normalized: bool) -> dict | None:
    """Load a prediction for a variable; None if the generator does not provide the variable.

    Normalized STRIPPER uses Rene's normalised files (bands with correlated normalisation). Normalized MATRIX
    divides central, band and errors by the central integral over its own range (uncorrelated approximation
    for the band).
    """
    prediction = PREDICTIONS[prediction_key]
    variable = VARIABLES[variable_key]
    if prediction.generator == 'stripper':
        if variable.stripper_observable is None:
            return None
        return read_stripper(stripper_path(prediction.order, variable.stripper_observable, normalized),
                             len(variable.axes))
    if prediction.generator == 'matrix':
        if variable.matrix_file is None:
            return None
        histogram = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))
    else:
        raise ValueError(f'Unknown generator {prediction.generator}')
    if normalized:
        total = integral(histogram)
        for key in ('central', 'band_down', 'band_up', 'mc_err'):
            histogram[key] = histogram[key] / total
    return histogram


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

    Central values and bands are merged as volume-weighted sums (bands treated as fully correlated across the
    merged bins), the errors `mc_err` (theory) and `stat_err` (simulation) in quadrature. The result carries
    the boolean mask `compatible` over target bins.
    """
    target_edges = tuple(np.asarray(axis_edges) for axis_edges in target_edges)
    if len(target_edges) != len(histogram['edges']):
        raise ValueError(f'cannot rebin a {len(histogram["edges"])}D histogram onto a {len(target_edges)}D binning')
    matrices, masks = zip(*(merge_matrix(source, target) for source, target in zip(histogram['edges'], target_edges)))
    compatible = functools.reduce(np.logical_and.outer, masks)
    source_volumes = bin_volumes(histogram['edges'])
    target_volumes = bin_volumes(target_edges)
    result = {'edges': target_edges, 'compatible': compatible}
    for key in ('central', 'band_down', 'band_up', 'mc_err', 'stat_err', 'pdf_down', 'pdf_up'):
        if key not in histogram:
            continue
        content = histogram[key] * source_volumes
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
    result = {'stripper': read_stripper(stripper_path('NNLO', variable.stripper_observable, False),
                                        len(variable.axes))['edges']}
    if variable.matrix_file is not None:
        result['matrix'] = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))['edges']
    return result
