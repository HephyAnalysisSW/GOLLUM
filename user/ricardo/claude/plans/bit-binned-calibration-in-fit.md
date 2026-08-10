# Applying binned BIT calibration factors in likelihood fits

> **Status: implemented.** Rejected alternatives, the out-of-range and empty-bin fixes, and a
> known instability in the calibration factors themselves are recorded in
> `bit-binned-calibration-in-fit-design-decisions.md`. Read that before trusting a
> calibrated fit result.

## Context

`ML/Calibration/calibration_plots.py --calibrate` already derives per-derivative binned
calibration factors from a trained BIT (on the `c2st_train` partition) and writes them to
`<model_dir>/<version>/<region>/BIT/<job_id>/calib_factors_<num_bins>_<binning>_bins.pkl`.
Nothing consumes them: `configs/unbinned_v7_eft/unbinned_2016_eft_binned_calib.yaml` already
declares `runtime.binned_calib_factors`, but no code reads that key.

Goal: make a BIT job's declared calibration factors be applied to its predictions when the
config is used in a likelihood fit, so a calibrated and an uncalibrated fit can be compared.

Two facts from tracing the code settle the design questions in the original note:

1. **No signature threading is needed.** `predict_bit_ratio` ([fit/Likelihood.py:524](../../../../fit/Likelihood.py#L524))
   is called from exactly two places -- `build_cache` ([:925](../../../../fit/Likelihood.py#L925))
   and `_eval_region_surrogates` ([:1468](../../../../fit/Likelihood.py#L1468)) -- both of which run
   *before* the fit. `N2LL.__call__` only contracts the cached `R` matrix; `model.predict` is
   never called inside the minimizer.
2. **The BIT already identifies its own output nodes.** `bit.derivatives` is a list of
   coefficient tuples, one per `predict()` column ([ML/BIT/NumbaBIT.py:161-223](../../../../ML/BIT/NumbaBIT.py#L161)).
   The pkl keys are exactly `sanitize_label(_format_derivative(der))`. So column -> pkl key is
   determined by the BIT object alone -- no POI list, no ordering assumption, no config coupling.

Calibration is therefore applied in `predict_bit_ratio`, not in `BIT.predict()`: the derivation
path (`calibration_runner.py`, `calibration_plots.py`) itself calls `bit.predict()` to compute
residuals, so calibrating inside `predict()` would make derivation circular and require a
bypass flag.

## Bugs in the existing calibration code that must be fixed

These are harmless-ish in a diagnostic plot and dangerous in a fit:

- `calibrate_prediction_binned` ([ML/Calibration/calibration_plots.py:194](../../../../ML/Calibration/calibration_plots.py#L194))
  starts from `np.ones_like(pred)`, so any event outside `[bins[0], bins[-1]]` is silently
  **set to 1.0** rather than rescaled. Bins are derived on `c2st_train`; a fit runs the full
  sample, so out-of-range events are guaranteed.
- `get_binned_calib_factors` ([:183](../../../../ML/Calibration/calibration_plots.py#L183)) uses
  `np.divide(..., where=...)` with no `out=`, so **empty bins receive uninitialized memory**
  as their factor. They must be exactly 1.0.

## Changes

### 1. New module `ML/Calibration/binned_calibration.py`

`fit/Likelihood.py` cannot import from `calibration_plots.py` -- that module imports
`matplotlib`/`mplhep` at module scope ([calibration_plots.py:35-40](../../../../ML/Calibration/calibration_plots.py#L35)).
So move the pure, importable pieces into a new dependency-free module holding:

- `sanitize_label(label)` -- moved verbatim from `calibration_plots.py:111`.
- `format_derivative(der)` -- moved from `calibration_runner.py:148` (`_format_derivative`).
- `calibrate_prediction_binned(pred, bins, calib_factors)` -- moved from
  `calibration_plots.py:190` and **rewritten** to fix the out-of-range bug:

  ```python
  bin_index = np.clip(np.searchsorted(bins, pred, side="left") - 1, 0, len(bins) - 2)
  return pred * calib_factors[bin_index]
  ```

  `searchsorted(side="left") - 1` reproduces the original `(pred > bins[:-1]) & (pred <= bins[1:])`
  masking exactly for in-range events; `clip` extends the first/last bin's factor to events
  below/above the range (decision: clamp to edge factor). It also replaces an `O(num_bins * N)`
  boolean mask with an `O(N log num_bins)` lookup.
- `apply_binned_calibration(predictions, derivatives, calibration)` -- new. Takes the raw
  `(N, K)` `predict()` output, the BIT's `derivatives` list, and the loaded pkl dict; returns a
  calibrated copy, applying `calibrate_prediction_binned` column by column with the key
  `sanitize_label(format_derivative(derivatives[j]))`.

Update `calibration_plots.py` and `calibration_runner.py` to import these instead of defining
them locally (no duplicated definitions anywhere).

Separately, fix `get_binned_calib_factors` in `calibration_plots.py` to pass
`out=np.zeros(len(bins) - 1)` to `np.divide` so empty bins end up at exactly 1.0.

### 2. `common/yaml_loader.py` -- load and attach the pkl

In the `elif jtyp == "bit":` block ([common/yaml_loader.py:704-716](../../../../common/yaml_loader.py#L704)),
right after `predictor.feature_names` is grafted on:

- Read `job.get("runtime", {}).get("binned_calib_factors")`. If absent, set
  `predictor.binned_calibration = None` and continue.
- If present, resolve relative to the same `outdir`, `pickle.load` it, and **hard-crash** if
  the file is missing (a named-but-absent calibration must not be silently ignored).
- Validate that every `sanitize_label(format_derivative(der))` for `der in predictor.derivatives`
  is a key in the loaded dict, and that `len(calib_factors[key]) == len(bins[key]) - 1`.
  Raise on mismatch -- this catches a pkl derived for a different POI set, and mirrors the
  existing ICH/ICPH binning consistency check at [yaml_loader.py:718](../../../../common/yaml_loader.py#L718).
- Attach as `predictor.binned_calibration = loaded_dict` and log that calibration is active.

Note `try_load_bit` ([:584](../../../../common/yaml_loader.py#L584)) swallows all exceptions;
the calibration load must sit *outside* it so failures surface.

### 3. `fit/Likelihood.py` -- apply in `predict_bit_ratio`

In `predict_bit_ratio` ([:524](../../../../fit/Likelihood.py#L524)), after `Y` is shaped and cast:

```python
calibration = getattr(model, "binned_calibration", None)
if calibration is not None:
    Y = apply_binned_calibration(Y, model.derivatives, calibration)
```

Both call sites are covered with no signature change. Import
`apply_binned_calibration` from `ML.Calibration.binned_calibration`.

**Cache invalidation caveat:** `build_cache` writes `R` to
`<cache_root>/<rid>/<cid>.h5` and skips rebuilding when the file exists and `--overwrite` is
not set ([fit/Likelihood.py:806-818](../../../../fit/Likelihood.py#L806)). The cache key does not
include the calibration, so a calibrated run over a stale uncalibrated cache would silently use
raw `R`. Use a distinct cache directory (or `--overwrite`) for the calibrated config, and say so
in the log line from step 2.

### 4. `configs/unbinned_v7_eft/unbinned_2016_eft_rescale.yaml`

A separate config carrying the **same** job id `bit_TT01j2l_EFT_2016_ctG` with
`runtime.binned_calib_factors` set -- a one-line diff against `unbinned_2016_eft.yaml`.
Fit both configs to compare. (Superseded an earlier two-jobs-in-one-config approach, which
would have needed a second, non-existent model directory -- see the design decisions record.)

## Verification

1. **Exact-equality guard on the rewrite (hard stop if it fails).** Before touching
   `calibrate_prediction_binned`, capture reference output from the current masking
   implementation on a real pkl
   (`.../SR_2016/BIT/bit_TT01j2l_EFT_2016_ctG/calib_factors_25_equal_bins.pkl`) over the saved
   `calib_pred_c2st_train.npy`, restricted to in-range events. Assert the searchsorted version
   is bit-for-bit identical there. Out-of-range events are expected to differ -- that is the fix.
2. **Derivation path unchanged.** Re-run
   `python ML/Calibration/calibration_plots.py configs/unbinned_v7_eft/unbinned_2016_eft_binned_calib.yaml --job bit_TT01j2l_EFT_2016_ctG --partition c2st_train --calibrate`
   and confirm the regenerated pkl matches the existing one except for previously-empty bins
   (which should now be exactly 1.0 instead of garbage).
3. **Loader wiring.** `python common/yaml_loader.py configs/unbinned_v7_eft/unbinned_2016_eft_binned_calib.yaml`
   -- `bit_..._ctG_calib_binned` reports calibration active, `bit_..._ctG` does not.
   Then point `binned_calib_factors` at a nonexistent file and confirm it crashes rather than
   loading uncalibrated.
4. **End-to-end fit.** Run `python fit/Likelihood.py` on the config for both jobs, into
   *separate* cache directories. Confirm the cached `R` matrices differ, that the calibrated
   `R` equals the uncalibrated one multiplied by the expected per-bin factors on a spot-checked
   sample of events, and compare the resulting `ctGRe`/`ctGIm` scans.
5. **Sanity check on the factors themselves.** The real pkl contains negative and
   wildly-varying factors in sparse tail bins (`calib_factors['ctGRe'][0] = -0.066`,
   `[2] = 1.38`). With edge clamping these get extrapolated into the tails. Inspect the fitted
   `R` distribution for sign flips before trusting the calibrated scan.
