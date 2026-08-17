# EFT expansion point rebase: design decisions

Decision record for `eft-expansion-point-rebase.md`. The plan goes stale once
implemented; this does not. Written 2026-08-17.

## Facts established during the design

These were verified against the code and against 20000 events of
`v3-2_nJ2p_nB2p_2l/2016/TT01j2l_UL16_mtt_0to700_nominal.root`. They are not assumptions.

- **The `der_` branches are true partial derivatives, not Taylor coefficients.**
  `make_ntuple.py` writes `der_j_j = 2 * EFTfitCoefficients[...]` and
  `der_j_k = EFTfitCoefficients[...]` for `j != k`. The factorial convention lives
  downstream: `base_point_const` divides the diagonal by 2
  (`ML/BIT/NumbaMultiNode.py:303-306`), as do `ResultNode.prefac`,
  `ICH._combo_weight` and `expand_pois_linear_quadratic`. So the shift
  `D'_i = D_i + sum_j H_ij r_j` carries no factor on the diagonal.
- **`EFTWeight_gen` uses the raw `EFTfitCoefficients`, so it needs no `0.5` anywhere.**
  The raw coefficients are the polynomial coefficients. The `0.5` only appears when
  working through the `der_` branches. Both forms were checked to agree to 3.4e-16.
- **The BIT denominator is column 0 of the weight matrix, for the whole training.**
  Boosting updates only `root.derivatives[1:]` and reads `w0 = boost_weights[:, 0]`
  (`ML/BIT/eft_bit_training.py:841-849`); leaves predict `leaf_value[c+1]/leaf_value[0]`.
- **No event has `EFTWeight_SM <= 0` or `EFTWeight_gen <= 0`** in the sample checked, so
  no zero-denominator handling is needed.
- **All nine nominal EFT files carry `EFTWeight_gen` and 152 `der_` branches**, across
  2016APV, 2016 and 2018. The ~390 systematic-variation files were not checked.
- **One loader swap reaches every sample.** After the systematics merge (401c40a),
  `data/samples_eft.py` builds a single loader in `_get_base()`, and every variation is a
  `clone_from_files` of it through `_make_variation` and the lazy `__getattr__`.
  `clone_from_files` copies `weight_branches`, `observer_names` and
  `_requested_branches`, and since that merge also `weight_rescale`.

## Decisions

### Rebase inside the interface only, leaving the loader at the SM — rejected

The likelihood forms `sum_i W_i (log1p(T_i) - T_i)` (`fit/Likelihood.py:1918-1919`), so
the model density is `W * (1 + T)` and `W` is the loader weight. If `W = k * w(SM)` while
`1 + T = w(r+t)/w(r)`, the product carries a stray factor `w(SM)/w(r)`.

That factor is constant in the POI but not in `x`, so it is a shape distortion, not a
normalization. Measured spread of `|gen/SM|`: 0.88 / 1.22 / 8.7 at the 1/50/99
percentiles. Concretely, fitting the SM sample as pseudo-data would return `c = r`
instead of `c = 0`.

**Decision: the loader weight and the BIT denominator must be the same function of `x`.**
Both become `w(r)`.

### Correcting by `w(r)/w(SM)` inside the likelihood — rejected

Mathematically equivalent to changing the loader weight, but it computes the same
quantity in a second place and forces the fit to materialize all 152 derivative
branches, which it currently does not load at all.

### A computed-weight hook in `RDataLoader` instead of a branch — rejected for now

`weight_branches` is a strict product of existing branch names
(`data/RDataLoader.py:500`) and `w(r)` is a sum, so a product cannot express it. The two
routes were a new branch from `make_ntuple.py`, or a callable weight hook in the loader.

The branch freezes `r` at ntuple-production time, so only the SM and the generation point
are reachable without reprocessing. Accepted: those are the only two points needed now.

**Revisit the loader hook if an arbitrary reference point is ever required.** The branch
route then stops scaling.

### POIs as offsets `t = c - r` — rejected in favour of absolute `c`

Both are the same model with different parameter labels, and neither affects the BIT.
Absolute coefficients were chosen so that scans, limits and plots need no post-hoc
conversion, and so that `setAsimov` at `c = 0` means the SM.

The consequence is that the shift splits across two places, which is easy to get half
right: the derivative shift in `EFTWeightInterface`, and the `(c-r)` monomials in
`expand_pois_linear_quadratic`. Verification step 2 of the plan exists specifically to
catch a partial application, and it discriminates all three wrong combinations.

### Reference point in the YAML config — rejected

It duplicates `r` in every EFT job block, and a typo there is undetectable.

### Reference point read from `GENERATION_POINT` by the fit — rejected

It couples the generic likelihood to the EFT sample module, and it does not prevent the
real hazard, which is a fit using a different point than the training used.

**Decision: store the point on the BIT artifact** (`bit.expansion_point`), set at
training time from the interface. The fit reads what the training actually used, so a
mismatch is impossible rather than merely unlikely. PDF BITs carry no attribute, and the
absent case means zeros, which is the honest default for a BIT trained around zero. An
explicit assertion covers the one dangerous case: an `eft` job whose predictor lost the
attribute.

### `reference_point` defaulting to zeros — rejected

`expand_pois_linear_quadratic` and `pois_jacobian_linear_quadratic` have five call sites
across `Likelihood`, `N2LLExtensions`, `ToyGenerator` and `plot/postfitsys`. A default
would let any forgotten site silently return offset-space numbers that look normal.
The argument is required.

### Reference at the generation point in the fitted directions and the SM elsewhere — rejected

For a job training a subset of the 16 operators, the alternative was to pin the unfitted
operators at the SM rather than at their generated values. Rejected because no events
were generated at that hybrid point, which defeats the numerical purpose of the rebase,
and because the constant term would then need all 153 columns evaluated in Python
instead of the `EFTWeight_gen` branch.

**Accepted consequence:** a 9-operator fit reports values conditional on the other 7
operators sitting at 1.5, or -0.5 for the `ctG` pair, not at the SM.

### Parallel `_gen` loaders alongside the existing ones — rejected

A hard swap of `weight_branches` was chosen instead, per the no-backward-compat rule in
`CLAUDE.md`. Every surrogate trained on EFT samples must be retrained, and the config
`version:` is bumped so a stale artifact cannot be picked up silently.

### Treating the PNN and ICP retraining as mandatory — softened to a measurement

The first reading was that a changed nominal weight invalidates every surrogate equally.
That is right for the BIT and the scaler, and too strong for the PNN and ICP.

`EFTfitCoefficients` are LHE-level, so the EFT weight of a given generated event is
identical in the nominal and the varied file. The extra factor `f = w(r)/w(SM)` therefore
cancels **per event**. It does not cancel per bin of `x`, because the PNN compares
densities: the learned ratio moves from `E[s | x]` to `E[s * f | x] / E[f | x]`, with `s`
the event-level variation factor.

Exact cancellation holds only when `s` is a deterministic function of the PNN input
features. Two residuals survive otherwise:

- weight-only systematics (b-tag, pileup, lepton SF): the correlation of `s` with `f` at
  fixed `x`. The b-tag SF depends on jet flavour, which is not among the features.
- kinematic systematics (JES, JER, unclustered): additionally
  `E[f | x, varied] / E[f | x, nominal]`, from the migration in `x`.

ICP has the same structure with no migration term, so its ratio goes from `E[s]` to
`E[s * f] / E[f]`. Both `f` and the acceptance systematics depend on the same kinematics,
so that correlation is real and probably larger than the PNN residual.

**Decision: measure instead of assume.** `f` spans 0.88 to 8.7 across the 1st to 99th
percentile, so the residual is not obviously negligible, and 186 retrainings are not
obviously necessary. Retrain one PNN and one ICP, compare against the old artifacts, and
decide from the deltas.

### Reading `--value` in the modification plotter as an absolute coefficient — rejected

`plot/bit/modification_plotter.py` scales each term by `args.value ** len(der)`, the
monomial in the expansion variable. That variable is now `c - r`, so `--value` is an
offset, not a Wilson coefficient. Making it absolute would mean a per-operator factor
`(value - r[op0]) * (value - r[op1])`.

Rejected for two reasons. A single scalar cannot express an absolute point, because `r`
differs between the `ctG` pair and the rest. And at `value = 1` the truth curve would
stop being the quantity the BIT learns, which is exactly what makes `--with-bit` a
closure test.

**Decision: the flag stays an offset, and is renamed `--delta-c` to say so.** The plot is
a BIT diagnostic in the derivative basis; absolute coefficients belong in the fit, where
`expand_pois_linear_quadratic` already applies the shift. The accepted cost is that
reading `ctGRe = 0.5` off the plot means passing `1.0`, which is why the expansion point
is printed on the figure.

### Labelling column 0 "gen" everywhere in the plotter — rejected

`plot/bit/modification_plotter.py` and `common/derivative_providers.py` serve both the
EFT and the PDF providers. For PDF, column 0 is the central PDF member, so "gen" would be
wrong there, just as "SM" already was.

**Decision: internal names use `nominal`, printed labels derive from the provider** via
`weight_label = "gen" if provider.expansion_point else "nominal"`. EFT plots read "gen"
as intended, PDF plots stay correct, and it costs one ternary.

### `expansion_point` on the derivative providers — accepted

An extra attribute, so justified by a concrete failure: without it the plotter cannot
state what `--delta-c` is an offset from, and a reader takes `delta c = 1` for the SM
when for `ctGRe` it means `c = 0.5`. A mislabelled physics plot is the kind of silent
error this whole change is guarding against.

`ML/Calibration/calibration_runner.py` does not need it. It works in the ratio basis and
never scales by a coefficient value.

### Giving `data/samples_eft_gen.py` the same treatment — rejected

It carries its own `_derivative_branches` and its own `EFTWeight_SM` in
`weight_branches`, so it stays on the SM. No config references it, only a comment in
`data/samples_eft.py` does. Leaving it alone keeps the change small.

**The hazard, if it is ever revived:** a config pointed at `samples_eft_gen` with a BIT
trained through the rebased `EFTWeightInterface` reproduces exactly the SM-weight against
generation-point-denominator mismatch this plan removes, and it fails silently.

### Closure testing through the BIT toy route — rejected as a test

`_compute_T_from_columns` serves Asimov, observation and toy generation alike
(`fit/Likelihood.py:1369-1372`). A toy drawn from `W * (1 + T')` and fitted with
`W * (1 + T)` returns its input regardless of whether `W` is right, because the same
density sits on both sides. This test passes even when the reference point is wrong.

**Use the truth route** at `fit/ToyGenerator.py:428`, which builds the density from the
exact polynomial, or fit the SM sample and require `c = 0`.

### Rescaling `base_points` — rejected

The dicts hold values in whatever variable the polynomial uses. That variable is now
`c - r`, so the unchanged `{ctj1: 1}` denotes `c_ctj1 = 2.5` instead of `c_ctj1 = 1`.
Unit steps from `r` are the natural probe for an expansion around `r`, so the dicts stay
as they are.

Base points enter only `base_point_const`, which feeds `get_split_vectorized` and the
positivity mask. They decide which splits are chosen. They appear nowhere in
`vectorized_predict`, `MultiBoostedInformationTree.predict` or `fit/Likelihood.py`, so
they cannot corrupt an output column. They can only make the trees better or worse.

Note that the loss metric does change: two BITs with an identical `base_points` config,
one expanded at the SM and one at `r`, do not optimize the same quantity.

### Adding the SM as an extra base point — deferred, not rejected

Considered so that the loss also probes the yield at the SM, where limits get reported.

Mechanically it works. With the SM appended, `base_point_const` becomes square and stays
full rank for 2, 9 and 16 operators, so the assertion at `ML/BIT/NumbaMultiNode.py:308`
still passes. The magnitudes are unremarkable too: the SM row entries are 1.5 and 2.25
against 1 and 2 for the existing rows, and the yield at the SM is about 0.8 of the yield
at `r`. The SM is not a distant point in this metric.

It was deferred for one reason only: it changes which trees get built, and running it at
the same time as the rebase would give a failed closure test two candidate causes.

**Revisit as an A/B once the rebase passes verification steps 1 to 4.** Train with and
without, compare the `ML/Calibration` output near `c = 0`. The change is one line in
`EFTWeightInterface.__init__`, appending `{op: -r[op] for op in self.parameters}`. For a
subset job that point is the SM only in the fitted directions, with the rest still at
`r`.
