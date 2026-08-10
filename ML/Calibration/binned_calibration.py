#!/usr/bin/env python
"""Application of binned BIT calibration factors.

Holds the dependency-free pieces shared between the derivation side
(``calibration_runner.py`` / ``calibration_plots.py``) and the consumption side
(``fit/Likelihood.py``). Deliberately imports nothing beyond numpy so that the
likelihood can use it without pulling in matplotlib or the plot syncer.

The calibration itself is a dict with two sub-dicts, ``bins`` and
``calib_factors``, both keyed by ``sanitize_label(format_derivative(derivative))``
-- see ``calibration_plots.py`` for how it is derived and written.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def format_derivative(der) -> str:
    """Pretty label for coefficient combinations, e.g. ('c0','c0','c1') -> c0^2 * c1"""
    if len(der) == 0:
        return "()"
    counts = Counter(der)
    parts = [(v if counts[v] == 1 else f"{v}^{counts[v]}") for v in sorted(counts.keys())]
    return " * ".join(parts)


def sanitize_label(label: str) -> str:
    """Turn a pretty derivative label into a filename/dict-key safe string."""
    return label.replace(" * ", "_x_").replace("^", "pow").replace(" ", "")


def calibration_key(der) -> str:
    """Calibration dict key for a BIT derivative tuple."""
    return sanitize_label(format_derivative(der))


def calibrate_prediction_binned(pred, bins, calib_factors):
    """Rescale ``pred`` by the calibration factor of the bin it falls into.

    ``bins`` holds all lower edges plus the upper edge of the last bin, so
    ``len(calib_factors) == len(bins) - 1``. Events outside ``[bins[0], bins[-1]]``
    are clamped to the first/last bin's factor rather than left unscaled, since
    the bins are derived on one partition and applied to another.
    """
    bin_index = np.clip(np.searchsorted(bins, pred, side="left") - 1, 0, len(bins) - 2)
    return pred * calib_factors[bin_index]


def apply_binned_calibration(predictions, derivatives, calibration) -> np.ndarray:
    """Calibrate an (N, K) BIT prediction matrix column by column.

    Column ``j`` of ``predictions`` corresponds to ``derivatives[j]`` (see
    ``MultiBoostedInformationTree.predict``), which fixes the calibration key
    without needing the POI names from the config.
    """
    calibrated = np.array(predictions, dtype=np.float64, copy=True)
    for column, der in enumerate(derivatives):
        key = calibration_key(der)
        calibrated[:, column] = calibrate_prediction_binned(
            calibrated[:, column], calibration["bins"][key], calibration["calib_factors"][key]
        )
    return calibrated


def check_calibration_covers_derivatives(calibration, derivatives, source: str) -> None:
    """Raise if ``calibration`` does not describe every BIT derivative exactly once."""
    for section in ("bins", "calib_factors"):
        if section not in calibration:
            raise RuntimeError(f"{source}: calibration has no '{section}' section.")

    missing = [calibration_key(der) for der in derivatives
               if calibration_key(der) not in calibration["calib_factors"]]
    if missing:
        raise RuntimeError(
            f"{source}: no calibration factors for {missing}. "
            f"Available keys: {sorted(calibration['calib_factors'])}. "
            "The calibration was most likely derived for a different set of parameters."
        )

    for der in derivatives:
        key = calibration_key(der)
        n_factors = len(calibration["calib_factors"][key])
        n_bins = len(calibration["bins"][key]) - 1
        if n_factors != n_bins:
            raise RuntimeError(
                f"{source}: '{key}' has {n_factors} calibration factors but "
                f"{n_bins} bins ({n_bins + 1} edges)."
            )
