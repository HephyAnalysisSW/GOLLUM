# EFT expansion point: rebase from the SM to the generation point

## Context

`eft/EFTWeightInterface.py` expands the event weight around the SM. The samples are
generated at a different point `r` (1.5 for every operator, -0.5 for `ctGRe` and
`ctGIm`). Expansion around the SM means the BIT learns ratios against `w(SM)`, which is
not the density the events were drawn from.

The goal is to expand around `r` instead, while the fit keeps reporting absolute Wilson
coefficients.

The expansion is exactly quadratic, so this is a rebase, not an approximation. The
Hessian is constant. Only two things change:

```
w(c) = w(r) + sum_i D'_i (c-r)_i + 0.5 sum_ij H_ij (c-r)_i (c-r)_j
```

1. `D'_i = D_i + sum_j H_ij r_j`, the first derivatives at `r`. This is the interface change.
2. `(c-r)`, the Taylor variable. This is the likelihood change.

`w(r)` comes from the `EFTWeight_gen` branch. `H_ij` is unchanged.

Rejected alternatives and their rationale live in
`eft-expansion-point-rebase-design-decisions.md` next to this file.

**Verified on data, not inferred.** On 20000 events of
`v3-2_nJ2p_nB2p_2l/2016/TT01j2l_UL16_mtt_0to700_nominal.root`:

- `EFTWeight_gen` equals `SM + sum_i D_i r_i + sum_{j<=k} f_jk der_jk r_j r_k`
  (with `f = 0.5` on the diagonal) to a maximum relative difference of 3.4e-16.
- The rebased polynomial evaluated at `c = 0` returns `EFTWeight_SM` to 5.0e-14.
- `|gen/SM|` quantiles (1/50/99) are 0.88 / 1.22 / 8.7. No event has `SM <= 0` or
  `gen <= 0`, so there is no zero-denominator case to handle.

All nine nominal files, across 2016APV, 2016 and 2018, carry `EFTWeight_gen` and 152
`der_` branches. The systematic-variation files were not checked.

The failure this change guards against is silent. A mismatched reference point produces
a smooth, converging, wrong fit. Every step below that says "assert" is load-bearing.

## Constraint: `r` lives in two repositories

`make_ntuple.py` is in `cmgrdf-GluonPDF`, not in GOLLUM, so `r` cannot be imported
across. It is therefore duplicated by necessity, and the duplication is checked at
runtime rather than prevented.

Inside GOLLUM, `r` is defined exactly once and flows outward:

```
data/samples_eft.py: GENERATION_POINT
  -> eft/EFTWeightInterface.py (derivative shift)
    -> ML/BIT/eft_bit_training.py sets bit.expansion_point on the artifact
      -> fit/Likelihood.py reads it off the predictor
```

The fit never reads `GENERATION_POINT`. It reads the point the BIT was actually trained
with. A training/fit mismatch is then impossible by construction.

## Changes

### 1. `data/samples_eft.py`

- Add `GENERATION_POINT = {w: (-0.5 if "ctG" in w else 1.5) for w in wc_names}`, next to
  `wc_names`. `EFTWeightInterface` already imports from here, so no new module.
- Add `"EFTWeight_gen"` to `eft_derivatives` (or to `observers` directly) so the
  interface can read it.
- In `_eft_loader`, replace `"EFTWeight_SM"` with `"EFTWeight_gen"` in
  `weight_branches`. Keep `EFTWeight_SM` as an observer for the closure test.

After this, `nominal_weight = k * w(r)`, where `k` is the lumi/xs/sum-of-weights factor
times the scale factors.

This is a single edit that reaches every sample. The module now builds one loader,
`_get_base()`, and every systematic variation is a `clone_from_files` of it via
`_make_variation` and the lazy `__getattr__`. `clone_from_files` copies
`weight_branches`, `observer_names` and `_requested_branches`, so nothing else needs
touching.

**Prerequisite.** `_eft_loader` sets `strict_branches=True`, so every variation file must
also carry `EFTWeight_gen` or the loader raises at construction. This was verified only
for the nine nominal files. Check the ~390 variation files before running the PNN and ICP
jobs (verification step 0). The failure is loud, not silent.

`data/samples_eft_gen.py` keeps its own `EFTWeight_SM` in `weight_branches` and is left
alone: no config references it. If a config is ever pointed at it, it would reintroduce
the SM/generation-point mismatch this plan removes.

### 2. `eft/EFTWeightInterface.py`

- Import `GENERATION_POINT`; store it as `self.reference_point`, restricted to nothing
  (the shift sums over all 16 operators regardless of `self.parameters`).
- `required_observers`: `EFTWeight_gen` replaces `EFTWeight_SM`; keep `der_i` for each
  operator in `self.parameters`; extend the pair list to `der_i_j` for every `i` in
  `self.parameters` and every `j` in all of `wc_names`. Every entry of `r` is nonzero,
  so the shift needs the full row of `H`. A 9-operator job goes from 55 to about 150
  observer columns; a 16-operator job is unchanged at 153.
- `make_weight_matrix`: divide by `EFTWeight_gen` instead of `EFTWeight_SM`. Column 0
  stays `nominal_weight`. Linear columns become
  `nominal_weight * (der_i + sum_j der_ij * r_j) / safe_gen`, with no factor on the
  diagonal (`der_i_i` is already `2 *` the raw coefficient, see `make_ntuple.py`).
  Quadratic columns are unchanged.

### 2b. Consequence for jobs with a subset of operators

The shift sums over all 16 operators because every entry of `r` is nonzero. So a
9-operator job reads about 150 of the 152 `der_` columns, and the saving over the full
set is small.

More importantly, a 9-operator job now pins the other 7 operators at `r`, where it
previously pinned them at the SM. The fitted values of the 9 are therefore conditional
on the remaining 7 sitting at their generated values. This is deliberate: expanding
around a point where no events were generated would defeat the purpose of the rebase.
The 16-operator jobs are unaffected.

### 3. `ML/BIT/eft_bit_training.py`

- After constructing the interface, set `bit.expansion_point = eft.reference_point`
  before the first `bit.save()`, so it is stored in the pickle and in `BIT_best.pkl`.
- Fix the plot legend `"yield (SM, scaled)"` at line 512: it is now the generation point.
- No change to `base_points`. The dicts are unchanged, but the variable underneath them
  is now `c - r`, so `{ctj1: 1}` denotes `c_ctj1 = 2.5`. The rank check still holds, and
  the `positive` option now constrains the weight at unit steps from `r`, where the
  events are, rather than at unit steps from the SM.

### 4. `fit/Likelihood.py`

- `expand_pois_linear_quadratic` and `pois_jacobian_linear_quadratic`: add a
  **required** `reference_point: Dict[str, float]` argument and build the monomials from
  `c - r`. Required, not defaulted: a default of zero would silently return offset-space
  results at any call site that forgets it, and that result looks normal.
- Add `self._poi_reference[(rid, cid)]` next to `self._poi_order`, populated at both
  sites where `_poi_order` is filled (the unbinned setup near line 692, the binned setup
  near line 1038) from `getattr(poi_predictor, "expansion_point", {})`.
- Assert that a class whose BIT job carries an `eft` block has a predictor with a
  non-empty `expansion_point`. PDF BITs legitimately have none, and `{}` means zeros.
- `_assemble_cA_per_class` passes the per-class reference through.

### 5. Remaining call sites

Same pattern, each must pass its reference point explicitly:

- `fit/N2LLExtensions.py:50-51` and `:79-80` (both functions)
- `fit/ToyGenerator.py:428`, the truth route, which reuses
  `provider.truth_weight_matrix` and must stay consistent with it
- `plot/postfitsys/unbinned_fromfit.py:161`

`common/derivative_providers.py` needs no change. `EFTDerivativeProvider` is a thin
wrapper and the interface handles `r` internally.

### 6. Retraining

The nominal weight changes for every EFT sample, so every surrogate trained on them is
invalidated. `configs/unbinned_v7_eft/**` currently holds:

| type | count | effect |
|---|---|---|
| `bit` | ~141 | first order; the whole point of this change |
| `scaler` | 3 | first order; `scaler.accumulate(X, w)` is weighted |
| `icp` | 93 | second order; see below |
| `pnn` | 93 | second order; see below |

**The PNN and ICP effects are second order, and worth measuring before committing to 186
retrainings.** The EFT weight is an LHE-level quantity, identical for a given generated
event in the nominal and the varied file, so the extra factor `f = w(r)/w(SM)` cancels
per event. It does not cancel per bin of `x`. The learned ratio goes from `E[s | x]` to
`E[s * f | x] / E[f | x]`, where `s` is the event-level variation factor. Those agree
only when `s` is a deterministic function of the PNN input features. The residual is the
correlation between `s` and `f` for weight-only systematics, plus a migration term
`E[f | x, varied] / E[f | x, nominal]` for the kinematic ones. ICP has the same structure
without the migration, so its ratio goes from `E[s]` to `E[s * f] / E[f]`, which is
likely the larger shift of the two.

`f` runs from 0.88 to 8.7 across the 1st to 99th percentile, so do not assume the
residual is negligible. Retrain one PNN and one ICP first, compare the deltas against the
old artifacts, and let that decide whether all 186 need redoing.

The PNN and ICP jobs depend on the ~390 systematic-variation files, so do verification
step 0 first.

Bump `version:` in those configs so the new artifacts land in a new model directory and
an old one cannot be picked up silently.

## Verification

Run in this order. Steps 1 and 2 are cheap and catch the whole class of reference-point
bugs.

0. **Branch presence.** Open all `*/*.root` under `BASE_DIRECTORY_EFT` and require
   `EFTWeight_gen` in every one. The nine nominal files pass; the ~390 variation files
   are unchecked. `strict_branches=True` turns any absence into a `KeyError` at loader
   construction, so this only decides whether the PNN and ICP campaign can start.

1. **Branch consistency.** Reconstruct `w(r)` from the `der_` branches using
   `GENERATION_POINT` and require agreement with `EFTWeight_gen` event by event. This is
   the only check that catches a divergence between the GOLLUM constant and the value
   baked into the ntuple by the other repository. Expect ~1e-16 relative. Run it as a
   standalone script, not inside training, because it needs all 153 columns.
2. **Rebase closure.** Evaluate the output of `make_weight_matrix` at `c = 0`, that is
   with monomials built from `-r`, and require `k * EFTWeight_SM` per event. Expect
   ~1e-14 relative. This discriminates every partial application: with no shift, or with
   only the derivative shift, the result is `k * w(r)`; with only the monomial shift it
   is `k * (w0 + r^T H r)`.
3. **Toy closure, truth route.** Generate a toy at `ctGRe = 0.5`, others zero, through
   `fit/ToyGenerator.py:428`, which builds the density from the exact polynomial rather
   than from the BIT. Fit it and require `ctGRe = 0.5` back. A toy built through the BIT
   route cannot test this: the same density appears on both sides and the closure passes
   regardless.
4. **Asimov at the SM.** With `setAsimov` at `c = 0`, fit and require `c = 0`. Consider
   starting the minimizer at `c = r`, where the MC has its statistics.
5. **Retrain one BIT** with `--small` and confirm the loss curve is finite and the
   modification plots are sane before launching the full set.
