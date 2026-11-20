"""Readers for the fixed-order predictions (STRIPPER from Rene Poncelet, MATRIX from Niels).

Every reader returns a dict with bin `edges` and per-bin arrays `central`, `mc_err`, `band_down`, `band_up`
(scale-variation envelope), all in pb/[X]. STRIPPER additionally provides `pdf_down`, `pdf_up`.
"""
import os

import numpy as np

from samples import (MATRIX_DIR, PREDICTIONS, STRIPPER_ABSOLUTE_DIR, STRIPPER_NORMALIZED_DIR, VARIABLES)

FB_TO_PB = 1e-3
# Absolute tolerance for two bin edges to count as identical when rebinning.
EDGE_TOLERANCE = 1e-6


def read_stripper(path: str) -> dict:
    """Read a 1D STRIPPER histogram (edge-low edge-high central up down mcerr [pdf_up pdf_down]).

    The PDF columns are only present at NNLO.
    """
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] not in (6, 8):
        raise ValueError(f'{path}: expected 6 or 8 columns for a 1D STRIPPER histogram, found {table.shape[1]}')
    edges = np.append(table[:, 0], table[-1, 1])
    if not np.allclose(table[1:, 0], table[:-1, 1]):
        raise ValueError(f'{path}: bins are not contiguous')
    scale_variations = table[:, 3:5]
    histogram = {
        'edges': edges,
        'central': table[:, 2],
        'band_down': scale_variations.min(axis=1),
        'band_up': scale_variations.max(axis=1),
        'mc_err': table[:, 5],
    }
    if table.shape[1] == 8:
        histogram['pdf_down'] = np.minimum(table[:, 6], table[:, 7])
        histogram['pdf_up'] = np.maximum(table[:, 6], table[:, 7])
    return histogram


def read_matrix(path: str) -> dict:
    """Read a MATRIX histogram (fb/[X]; columns edges, central+err, scale-min+err, scale-max+err, rel. down/up)."""
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
        'edges': edges,
        'central': table[:, 2] * FB_TO_PB,
        'band_down': scale_variations.min(axis=1),
        'band_up': scale_variations.max(axis=1),
        'mc_err': table[:, 3] * FB_TO_PB,
    }


def stripper_path(order: str, observable: str, normalized: bool) -> str:
    """File name of an inclusive STRIPPER prediction."""
    directory = STRIPPER_NORMALIZED_DIR if normalized else STRIPPER_ABSOLUTE_DIR
    return os.path.join(directory, f'incl_{order}_mt172.5_HT4_NNPDF31_{observable}.dat')


def integral(histogram: dict, key: str = 'central') -> float:
    """Integral of a differential histogram over its bins; NaN bins (not rebinnable) are skipped."""
    return float(np.nansum(histogram[key] * np.diff(histogram['edges'])))


def load_prediction(prediction_key: str, variable_key: str, normalized: bool) -> dict | None:
    """Load a prediction for a variable; None if the generator does not provide the variable.

    Normalized STRIPPER uses Rene's normalised files (bands with correlated normalisation). Normalized MATRIX
    divides central, band and errors by the central integral (uncorrelated approximation for the band).
    """
    prediction = PREDICTIONS[prediction_key]
    variable = VARIABLES[variable_key]
    if prediction.generator == 'stripper':
        return read_stripper(stripper_path(prediction.order, variable.stripper_observable, normalized))
    if prediction.generator == 'matrix':
        if variable.matrix_file is None:
            return None
        histogram = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))
        if normalized:
            total = integral(histogram)
            for key in ('central', 'band_down', 'band_up', 'mc_err'):
                histogram[key] = histogram[key] / total
        return histogram
    raise ValueError(f'Unknown generator {prediction.generator}')


def rebinnable_bins(source_edges: np.ndarray, target_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask over target bins that are an exact union of source bins, and the source index of each target edge.

    The index is -1 for target edges that are not source edges.
    """
    source_index = np.full(len(target_edges), -1)
    for target_position, edge in enumerate(target_edges):
        matches = np.flatnonzero(np.isclose(source_edges, edge, rtol=0, atol=EDGE_TOLERANCE))
        if len(matches):
            source_index[target_position] = matches[0]
    compatible = (source_index[:-1] >= 0) & (source_index[1:] >= 0)
    return compatible, source_index


def rebin(histogram: dict, target_edges: np.ndarray) -> dict:
    """Rebin a differential histogram onto target_edges; target bins not made of whole source bins are NaN.

    Central values and bands are merged as width-weighted sums (bands treated as fully correlated across the
    merged bins), `mc_err` in quadrature. The result carries the boolean mask `compatible` over target bins.
    """
    compatible, source_index = rebinnable_bins(histogram['edges'], target_edges)
    source_widths = np.diff(histogram['edges'])
    target_widths = np.diff(target_edges)
    result = {'edges': np.asarray(target_edges), 'compatible': compatible}
    for key in ('central', 'band_down', 'band_up', 'mc_err', 'pdf_down', 'pdf_up'):
        if key not in histogram:
            continue
        values = np.full(len(target_widths), np.nan)
        for target_bin in np.flatnonzero(compatible):
            merged = slice(source_index[target_bin], source_index[target_bin + 1])
            content = histogram[key][merged] * source_widths[merged]
            total = np.sqrt(np.sum(content ** 2)) if key == 'mc_err' else np.sum(content)
            values[target_bin] = total / target_widths[target_bin]
        result[key] = values
    return result


def binnings(variable_key: str) -> dict:
    """Bin edges per binning name ('stripper', 'matrix'), read from the prediction files."""
    variable = VARIABLES[variable_key]
    result = {'stripper': read_stripper(stripper_path('NNLO', variable.stripper_observable, False))['edges']}
    if variable.matrix_file is not None:
        result['matrix'] = read_matrix(os.path.join(MATRIX_DIR, variable.matrix_file))['edges']
    return result
