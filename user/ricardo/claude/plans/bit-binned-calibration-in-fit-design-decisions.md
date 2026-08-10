# Binned BIT calibration in fits -- design decisions

Decision record for the feature planned in `bit-binned-calibration-in-fit.md`. The plan
goes stale once implemented; this does not. It answers the three questions posed in
`bit-calibration-binned.md` and records what was rejected and why.

**Scope of validation.** Every decision below is backed by reading the code, and the numerical
claims by running against the real `bit_TT01j2l_EFT_2016_ctG` model and its stored pkl. **No
likelihood fit was ever run.** Nothing here establishes that the calibration improves closure,
or that a calibrated scan differs from an uncalibrated one in any particular direction. The
per-step breakdown is in the plan's "What was actually verified" section.

---

## 1. Where is the rescaling applied?

**Decided: in `predict_bit_ratio` (`fit/Likelihood.py`), with the calibration attached to
the predictor object at load time by `common/yaml_loader.py`.**

### Rejected: inside `MultiBoostedInformationTree.predict()`

Superficially the tidiest -- calibration becomes part of the model. It is wrong because the
calibration is *derived from* raw predictions: `calibration_runner.get_truth_pred_values` and
`calibration_plots.py` both call `bit.predict()` to compute the residuals that define the
factors. Calibrating inside `predict()` makes the derivation feed on its own output, so it
would need a bypass flag on every derivation call site. A flag whose only purpose is to
disable the feature at half its call sites is a sign the feature is in the wrong place.

Secondary: `predict()` is also used by training diagnostics and closure plots, which want raw
output.

### Rejected: threading factors through the `N2LL.__call__` call chain

This was the original concern in `bit-calibration-binned.md` -- that applying calibration in
the likelihood means passing bins and factors through many tangled calls. **The concern turned
out not to apply.** `predict_bit_ratio` has exactly two call sites,
`N2LL.build_cache()` and `N2LL._eval_region_surrogates()`, and both run *before* any fit.
`N2LL.__call__` never touches a BIT: it contracts the already-materialized `R` matrix
(`_compute_T_from_columns` does `R_slice @ cA`). BIT predictions are cached three times over
-- to HDF5 by `build_cache`, into RAM by `prepare_runtime`, and per-event for observed data by
`setObservation`. So calibrating at the single choke point where `R` is produced reaches
everything, with no signature changes anywhere.

### Cache invalidation: a non-issue, given the config layout in section 4

`build_cache` skips rebuilding when `<cache_root>/<rid>/<cid>.h5` exists and `--overwrite`
is not set, and the cache key does not include the calibration. That sounds like a trap --
a calibrated run reusing an uncalibrated cache would silently fit raw `R`.

It is not one, because the cache path is derived from the **config file name**:
`cache_subdir = NN2LCache/<base>/<version>` with `base` the config basename, consistently
across all four fit entry points (`Likelihood.py:2541`, `MultiDimFit.py:105`,
`RunImpactPlots.py:242`, `ToyGenerator.py:888`). Since the calibrated variant lives in its
own file, `unbinned_2016_eft_rescale.yaml`, it gets its own cache tree and regenerates on
first use. This is an additional argument for the separate-config layout over two jobs in
one config, where both variants would have shared a cache tree *and* a job id.

The one way to defeat this is passing `--base` explicitly with the same value for both
configs, which forces them into a shared cache directory. Don't.

Encoding the calibration in the cache key would make this robust independent of file naming,
and was not done -- a larger change to the cache-path scheme than this feature justifies.

---

## 2. How is a calibration matched to the right output node?

**Decided: via `bit.derivatives`, which the BIT already carries.**

The premise of the original question -- "the BIT object does not store information on the POI
names" -- is not quite right. The BIT does not store a POI *list*, but it stores
`derivatives`: a list of coefficient tuples, one per `predict()` column, backfilled on load by
`_sanitize_after_load`. Column `j` is `derivatives[j]`, and the calibration pkl is keyed by
`sanitize_label(format_derivative(der))`. So the mapping is:

    predict() column j  ->  bit.derivatives[j]  ->  calibration_key(...)  ->  pkl key

This is self-describing. It needs no POI list from the config, and is immune to the ordering
trap documented in `N2LL._prepare_structure` at the `poi_names = sorted(poi_names)` line
(`fit/Likelihood.py:686` at time of writing): the BIT alphabetizes its derivative columns, so
config order and column order differ.

### Rejected: matching by position, or by the config's `eft.parameters` / `pdf` block

Both would silently mis-assign calibrations if the pkl were derived for a different or
reordered parameter set. Nothing downstream would catch it -- the existing consistency check
is a width check only (`R_slice.shape[1] != cA.shape[0]`), which a permutation passes. Since
key lookup is by name, a mismatch is instead caught loudly by
`check_calibration_covers_derivatives` at load time.

---

## 3. Events outside the calibration binning range

**Decided: clamp to the first/last bin's factor.**

The original implementation started from `np.ones_like(pred)`, so any event outside
`[bins[0], bins[-1]]` was silently **set to 1.0** rather than left alone -- not a rescaling at
all, a replacement. Harmless-looking in a diagnostic plot, corrupting in a fit. Bins are
derived on `c2st_train` and applied to the full sample, so out-of-range events are guaranteed.

Measured on the real `bit_TT01j2l_EFT_2016_ctG` model: 0--2 events per derivative per
partition, i.e. <0.01%. Clamping and leaving-uncalibrated are therefore near-equivalent in
practice here; clamping was chosen for continuity at the boundary.

Rejected: crashing on out-of-range (not viable -- the fit sample legitimately spans a wider
range than the derivation partition).

The implementation switched from an `O(num_bins x N)` boolean mask to
`np.searchsorted(...) - 1` with a clip, which is equivalent for in-range events (verified
bit-for-bit on real predictions for all 5 derivatives on both partitions) and handles the
clamp for free.

---

## 4. Config layout for the calibrated/uncalibrated comparison

**Decided: a separate config, `unbinned_2016_eft_rescale.yaml`, carrying the same job id
with `runtime.binned_calib_factors` set -- a one-line diff against `unbinned_2016_eft.yaml`.**

### Rejected: two jobs in one config

The first attempt added a second job `bit_TT01j2l_EFT_2016_ctG_calib_binned` alongside
`bit_TT01j2l_EFT_2016_ctG`, both pointing at the same `output.filename`. This fails: model
paths are `<version>/<region>/BIT/<job_id>/`, so the duplicate job looks for its model in a
directory that does not exist and reports `[MISS]`. `make_model_folders.sh` does not help --
it symlinks across config *versions*, not across job ids within a version. It would have
needed a hand-made symlink per calibrated job, and the duplicated 15-line job block would
have to be kept in sync by hand.

Keeping the job id identical and varying the config means both variants resolve to the same
model directory and the same calibration pkl, and the entire difference between the two fits
is one YAML line.

---

## 5. Where the shared code lives

**Decided: a new `ML/Calibration/binned_calibration.py`, importing only numpy.**

`fit/Likelihood.py` cannot import from `calibration_plots.py`: that module imports
`matplotlib`, `mplhep` and `common.syncer` at module scope, and importing syncer registers an
`atexit` hook that uploads to CERN EOS -- unacceptable as a side effect of running a fit, and
it hard-crashes without an interactive Kerberos token.

Rejected: duplicating `calibrate_prediction_binned` into the fit code. Two copies of a
numerical routine that must agree exactly is precisely the failure mode CLAUDE.md warns about.
`calibration_plots.py` and `calibration_runner.py` were changed to import from the new module
so there is exactly one definition of each helper.

---

## Known limitation: the factors are unstable where the prediction crosses zero

**This is the main reason not to trust a calibrated scan without inspecting it first.** It is a
property of the calibration definition, not of this implementation, and it is unfixed.

The factor is `f_i = 1 + <truth - pred>_i / <pred>_i`. Wherever the bin-mean prediction
approaches zero, the denominator collapses and the factor diverges and can change sign.
For `ctGRe` in `bit_TT01j2l_EFT_2016_ctG`, whose predictions span roughly [-0.78, 0.33], the
zero crossing sits in bins 17--18:

| bin | range | events | sum of weights | factor |
|---|---|---|---|---|
| 16 | -0.070 .. -0.026 | 25 | 0.74 | 5.81 |
| 17 | -0.026 .. 0.018 | 30 | 1.44 | **29.87** |
| 18 | 0.018 .. 0.062 | 10 | 0.11 | **-11.16** |
| 19 | 0.062 .. 0.106 | 1 | 0.03 | -3.17 |
| 24 | 0.282 .. 0.326 | 2 | 0.03 | **-4.07** |

These bins are populated, so the empty-bin fix below does not touch them; they *will* be
applied in a fit. A negative factor flips the sign of that event's `R_A` contribution.
Edge clamping additionally extends bin 24's `-4.07` to every event above 0.326.

Pushing 20k random-normal feature vectors through the real model, 5.0% of `ctGIm` and 0.31%
of `ctGRe` predictions change sign under calibration. **Those inputs are synthetic, not
physical events, so treat the numbers as evidence that the effect is reachable -- not as a
rate.** The rate on real events is unmeasured and is the first thing to establish.

Deliberately not mitigated in this change, since the right guard is a physics decision:
options are a minimum sum-of-weights per bin, leaving bins with `|<pred>|` below a threshold
at 1.0, quantile binning so tail bins are populated by construction, or fitting a smooth
calibration curve instead of independent per-bin factors. Check the `R` distribution for sign
flips before reading anything into a calibrated result.

## Fixed along the way: empty bins held uninitialized memory

`get_binned_calib_factors` called `np.divide(mean_res, mean_pred, where=(mean_pred != 0.0))`
with no `out=`. Where the mask is False, numpy leaves the output buffer untouched -- so empty
bins kept whatever was in the freshly allocated memory, plus 1.0. This was live in the stored
pkl: `ctGRe` bins 22 and 23 are empty and held `1.00075` and `1.00109` instead of exactly 1.0.
Small, but non-deterministic across runs. Fixed by passing `out=np.zeros(...)`.

Re-deriving from `calib_values_c2st_train.csv` after the fix: binning identical, every
populated bin bit-for-bit identical, and exactly 12 changed values across the 5 derivatives --
all of them empty bins, all now exactly 1.0.
