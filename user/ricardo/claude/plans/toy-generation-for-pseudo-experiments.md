# Toy dataset generation for pseudo-experiments

## Context

GOLLUM currently supports exactly two data modes in `N2LL`: Asimov
(`setAsimov`, [Likelihood.py:1384](fit/Likelihood.py#L1384)) and observed data
(`setObservation`, [Likelihood.py:1448](fit/Likelihood.py#L1448)). There is no
toy/pseudo-experiment generation anywhere in `fit/` — the only `np.random` call
is the unseeded feature shuffle used by the `--shuffle` diagnostic
([Likelihood.py:872](fit/Likelihood.py#L872)).

Without toys we cannot (a) measure bias of the extracted PDF coefficients,
(b) measure coverage of the reported intervals, or (c) test whether the
`-2dlnL` test statistic follows the asymptotic chi-square distribution, which
is not guaranteed for a POD-basis fit with many weakly-constrained directions.

Two **sources** are needed — the spec key is `source`, and "mode" is reserved
throughout for `N2LL`'s existing Asimov/observation distinction. They are **not**
interchangeable:

1. **`source: cache`** — the model is taken as truth. Fast, and answers "given
   that the surrogates are correct, are my intervals calibrated?"
2. **`source: truth`** — pseudo-data is generated from an exact reweighting, a
   per-process rescaling, or a different sample, while the model still fits
   with the trained surrogates. Probes surrogate mismodelling as a bias in the
   fitted coefficients.

Below, "cache mode" and "truth mode" are shorthand for these two `source`
values.

Decision rule: **anything expressible as a parameter value belongs in cache
mode.** A normalization scaling is already a model parameter — the floating
`nu_norm_ttbar` lnN in `configs/unbinned_v6_rate/unbinned_2016_rate.yaml:52-53`, or
a `rate_shift` POI ([Likelihood.py:654](fit/Likelihood.py#L654)) — so
a spec point carrying `hypothesis: {nu_norm_ttbar: 1.0}` generates it exactly,
with no MC re-reading.
Truth mode is for shifts the model has **no parameter to absorb**, e.g. scaling
a background that carries no rate nuisance and measuring the induced bias on
`c0..c5`.

One deliberate exception: a surrogate-route point used as the *control* for a
truth point must run in truth mode too, so both draw from the same events. See
§"The two POI injection routes".

Scope: the **generator** plus the minimum `N2LL` / `Modeling` surface needed to
fit a toy. No toy-fit driver and no Slurm fan-out.

## Statistical content

`rho(x) = dN/dx` denotes the intensity of the unbinned Poisson point process —
the differential expected yield, whose integral is the total expected yield.
It is not a normalized PDF. `rho_0` is the intensity at `c = 0, nu = 0`.

### Unbinned, cache mode

The model factorizes as `rho(x) = rho_0(x) * (1 + T(x; c, nu))`, with `T` from
`_compute_T_chunk` ([Likelihood.py:1342](fit/Likelihood.py#L1342)); `1 + T` is
the differential cross-section ratio the BIT/PNN surrogates learn. Writing out
the extended unbinned NLL,

    -lnL = int rho - sum_obs ln rho(x_i)
         = int rho_0 + int rho_0*T - sum_obs ln rho_0 - sum_obs log1p(T)

the first and third terms are parameter-independent and drop, leaving
`int rho_0*T - sum_obs log1p(T)`. The first is estimated as `sum_i w0_i * T_i`
over the cached MC — literally [Likelihood.py:1742](fit/Likelihood.py#L1742),
with the per-event term at [Likelihood.py:1775](fit/Likelihood.py#L1775).
**`rho_0` is never evaluated pointwise**; the cache is only its MC estimator,
`sum_i w0_i delta(x - x_i)`.

Symbols used throughout: `w0` is the MC event weight; `g` the classifier
probability for a class; `R` the BIT ratio basis; `Delta::<systid>` a PNN group's
basis matrix. All four are the per-event columns the cache stores.

A sum of independent Poissons placed on those MC points is exactly a Poisson
point process with that intensity, so a toy is

    n_i ~ Poisson( w0_i * (1 + T_i(c', nu')) )

Rows with `n_i > 0` are the toy events, carrying observed weight `n_i`. Since
the likelihood consumes `sum_i w_i log1p(T_i)`, multiplicities can be carried
as weights — no `np.repeat`. No surrogate re-evaluation is needed, since `g`,
`R` and `Delta::*` are already cached per event.

### Unbinned, truth mode

Drawn from an externally supplied weight instead of the surrogate prediction:
`n_i ~ Poisson(w_truth_i)`, with `g`, `R`, `Delta::*` then evaluated by the
trained surrogates on the selected events. The expected-yield term is unchanged,
so both the shape and the normalization of any mismodelling propagate into the
fit — which is the point.

### Binned

In cache mode, `_compute_lambda_binned(rid, hyp)`
([Likelihood.py:1236](fit/Likelihood.py#L1236)) already returns the expected
count vector, so `N_obs ~ Poisson(lambda(c', nu'))` into
`_obs_binned_counts[rid]`. Needed because the v6 configs mix an unbinned
`SR_<era>` with a binned `CR_<era>` in one fit.

In truth mode the binned regions must follow the same truth as the unbinned
ones. The substantive choice is **which lambda is thrown on**, not where the
Poisson is applied:

- `_compute_lambda_binned` returns the **model's** expected counts, from the
  ICH/ICPH surrogates.
- Histogramming the truth-weighted events into `_binned_unroll[rid]['edges']`
  (the histogramming `setObservation` already does) gives the **truth's**
  expected counts, `lambda_truth_b = sum_{i in b} w_truth_i`.

These differ by exactly the ICH/ICPH mismodelling, which is what truth mode
exists to measure, so truth mode throws on `lambda_truth`. Using the model's
lambda would leave binned regions model-truth while unbinned regions are
reweighted, making the resulting bias uninterpretable.

Throw **once per bin**, not per event. Per-event `n_i ~ Poisson(w_truth_i)`
followed by histogramming is a sum of independent Poissons, hence exactly
`Poisson(lambda_truth_b)` — distributionally identical, and regions are disjoint
selections so there is no shared-event correlation. Histogram-first is therefore
strictly cheaper: one draw per bin rather than one per MC event, with no
multiplicity array to materialize.

### Nuisance throwing

Each toy throws `nu_obs ~ Normal(nu_true, 1)` per penalized nuisance, with the
constraint becoming `(nu - nu_obs)^2`. `Hypothesis.penalty()`
([Modeling.py:205](fit/Modeling.py#L205)) hardcodes `sum(p.val**2)`, so this
needs the change below. Floating nuisances get `isPenalized=False`
([Likelihood.py:338-341](fit/Likelihood.py#L338-L341)), so free rate parameters
such as `nu_norm_ttbar` are correctly never thrown.

### UID splitting: which events generate toys

`build_cache` applies **no** UID filtering — `_iter_asimov_batches`
([Likelihood.py:717](fit/Likelihood.py#L717)) materializes every shard, and
nothing under `fit/` consumes `defaults.splitting`. Meanwhile the configs
reserve `final_eval: {fraction: 0.20}`, which **no script consumes at all** —
repo-wide it appears in exactly two places, both comments
([pnn_training_closure_mpl.py:320](ML/PNN/pnn_training_closure_mpl.py#L320),
[modification_plotter.py:347](plot/bit/modification_plotter.py#L347)).

Verified that it is genuinely held out: the BIT trains on the PNN's buckets —
`bit_train_key = "pnn_train"`, `bit_val_key = "pnn_val"` at
[pdf_bit_training.py:158-159](ML/BIT/pdf_bit_training.py#L158-L159) and
[eft_bit_training.py:164-165](ML/BIT/eft_bit_training.py#L164-L165) — and the
TFMC does the same at
[tfmc_training.py:184-185](ML/TFMC/tfmc_training.py#L184-L185). No trainer of any
surrogate the likelihood uses touches `final_eval`.

The right choice differs by mode:

**Cache mode — full dataset, no split.** The model *is* the trained surrogate by
definition, and the toy is a draw from the model's own intensity
`w0 * (1 + T)`. Whether an event was in training does not change what the model
is; it only changes which points carry the MC estimator of `rho_0`. Coverage of
the model against itself is therefore unaffected by overfitting, while MC
granularity argues for using everything. Memorized structure in `T` on training
events appears identically in the toy and in the expected-yield term and
cancels — that belongs to the MC-reuse caveat below, not to a bias.

**Truth mode — `final_eval` by default.** Here the events carry the truth
density and the study measures the gap between exact `w(c)` and surrogate
`R(c)`. On training events the surrogates fit better than on fresh events, so
the measured bias is optimistic in precisely the quantity being measured.
Costless to apply, since truth mode re-reads from loaders: add `uid_fields` to
`required_observers` and mask with `mask_from_np`, the pattern at
[calibration_runner.py:168-169](ML/Calibration/calibration_runner.py#L168-L169)
and [calibration_runner.py:201](ML/Calibration/calibration_runner.py#L201).

Bucket intervals are built from the scheme the way `_uid_c2st_intervals`
([calibration_runner.py:96](ML/Calibration/calibration_runner.py#L96)) does,
but that helper hardcodes the `c2st_train`/`c2st_val` keys. Generalize it to
take the split name as an argument and change its one existing call site over,
rather than duplicating the interval arithmetic. The scheme is read from the
merged config's `defaults.splitting`, which is what the fit path already
carries — not from a per-job block, since toy generation is not job-scoped.

**Normalization must be restored.** Generating from 20% of events while the
expected-yield term `sum w0 * T` runs over the full cache would put the toy
yield at a fifth of the expectation and pull the fit violently. Toy weights are
rescaled by `1 / f_weight` with

    f_weight = sum(w over split) / sum(w over all)

the **weight** fraction, not the nominal bucket fraction `0.20` — UID hashing
distributes buckets, not weights, and using `0.20` would inject a normalization
bias masquerading as the rate bias under measurement. This needs no extra pass:
truth mode already streams every event to evaluate the mask, so both sums
accumulate in that pass while non-split events are discarded.

The cost is granularity: rescaling by ~5x means each retained event stands for
five times more data events, so `n_i ~ Poisson(5 * w_i)` yields many more
duplicates — negligible for `TTLep_pow_*`, potentially severe for a small
background. The `n_i > 1` diagnostic must therefore be reported **after**
rescaling.

Spec key `split` (default `final_eval`, `null` for the whole dataset) keeps the
escape hatch for a purely statistical study where leakage is irrelevant and
granularity dominates.

Applying a split in cache mode would require UID columns in the HDF5 cache, a
format change forcing every cache to be rebuilt. Not done here; noted as the
extension path if it is ever wanted.

### Known caveat (documented, not fixed)

Toys are drawn from the same cached MC that supplies the expected-yield term, so
toy and template fluctuations are correlated. Sub-percent on the widths at
current MC statistics. Record in the module docstring; add no machinery.

## Multi-process handling (truth mode)

`build_cache` streams **all** `classifier.asimov` samples into one concatenated
event set and writes per-class `g`/`R`/`Delta::*` columns for the *same rows*,
sharing one `w0` (per-class columns written together in `build_cache`,
[Likelihood.py:786](fit/Likelihood.py#L786)). The cache
therefore represents the **total** nominal intensity, and the process
decomposition lives entirely in the classifier output `g_p(x)`
(`T = sum_p g_p * (...)`, [Likelihood.py:1364](fit/Likelihood.py#L1364)). The
likelihood never sees a process label. Reference multi-class region:
`configs/Eta_unbinned/Eta_unbinned_2016.yaml:24`,
`asimov: [TTLep_pow_2016, EtaS_2016, EtaP_2016]`.

So a truth toy is built **per process** — that is where per-process weights and
truth modifications live — then concatenated and pushed through the surrogates
**once, as a single unlabeled event set**, exactly as real data is treated.

```python
@dataclass
class ProcessSource:
    """One process's contribution to a truth-mode toy in one region."""
    class_id: str                        # must match a region class id
    sample_name: str | None = None       # default: the class's configured `sample`
    coefficients: dict[str, float] | None = None   # PDF/EFT truth injection
    scale_factor: float = 1.0
    weight_branches: list[str] | None = None
    weight_function: Callable | None = None   # f(X, feature_names) -> (N,)
```

Modifiers are optional and multiplicative, so they compose:

    w_truth = w_coefficients * scale_factor * prod(weight_branches) * weight_function(X)

reducing to the nominal weight when `coefficients` is None.

### Exact PDF/EFT weights (`coefficients`)

The BIT is trained only on *derivatives*, so no branch on any sample carries the
full modified weight `w(c)` and no alternative sample provides it. It has to be
reconstructed — but the machinery already exists and must be **reused, not
re-derived**.

`provider.truth_weight_matrix(G, w, observer_names)` in
`common/derivative_providers.py` returns an `(N, M)` matrix aligned to
`provider.combinations = [(), (p,)..., (p,q) for p<=q]`, already in weight
units, with column 0 the nominal weight
([derivative_providers.py:69](common/derivative_providers.py#L69) returns
`deriv * w`; [EFTWeightInterface.py:59](eft/EFTWeightInterface.py#L59) sets
`out[0] = nominal_weight`). Columns `1:` are the true first and symmetrized
second derivatives. The Taylor sum is then exactly `expand_pois_linear_quadratic`
([Likelihood.py:383](fit/Likelihood.py#L383)):

```python
w_coefficients = deriv_w[:, 0] + deriv_w[:, 1:] @ expand_pois_linear_quadratic(
    provider.parameters, coefficients)
```

The orderings match term for term: `expand_pois_linear_quadratic` emits linears
then `i<=j` quadratics in `combinations_with_replacement` order, identical to
`provider.combinations` after dropping column 0, and its `0.5 if i==j else 1`
factor is the `(1/2) sum_{a,b} -> sum_{a<=b}` rewrite explained in the comment at
[Likelihood.py:388-393](fit/Likelihood.py#L388-L393). `PODBasis.derivatives`
returns true second derivatives (`H = outer + outer.T`, and the parametrization
is linear in `c`, so `d2f/dc_a dc_b = 0` individually).

**The convention is inherited, not re-derived** — verified: BIT training builds
its targets from the identical calls, `pdf.derivatives(x1=..., x2=..., id1=...,
id2=..., Q=...)` at [pdf_bit_training.py:251](ML/BIT/pdf_bit_training.py#L251)
and `eft.make_weight_matrix(G, obs_names, w)` at
[eft_bit_training.py:266](ML/BIT/eft_bit_training.py#L266), which are exactly
what `PDFDerivativeProvider` / `EFTDerivativeProvider` wrap. Truth weight and
model ratio `R = 1 + c_A R_A` therefore agree by construction rather than by
getting an algebra factor right independently — and this holds for EFT without
needing to know whether the `der_*` branches carry a factor of one half.

This is the sharpest available surrogate test: generate from the exact `w(c)`,
fit with the BIT-predicted `R(c)`; residual bias in `c0..c5` is purely BIT
mismodelling plus MC statistics.

Four requirements this imposes:

1. **Observers must be requested.** For PDF this is nearly free: the five
   `Generator_*` branches *and* the UID fields are already in
   `observables.OBSERVERS` ([observables.py:103](data/observables.py#L103)),
   which the samples set as `observer_names`
   ([samples_RunII.py:98](data/samples_RunII.py#L98)). EFT needs
   `EFTWeight_SM` + `der_*`, which are not in that default set. Either way the
   generator must `clone()` the loader
   ([RDataLoader.py:638](data/RDataLoader.py#L638)) and call
   `setFeatures(..., observer_names=provider.required_observers + uid_fields)`
   explicitly, following
   [calibration_runner.py:171](ML/Calibration/calibration_runner.py#L171) —
   cloning so the shared factory loader is never mutated, and explicit so a
   sample lacking a branch fails loudly via `strict_branches` rather than
   silently. Materialize with `what="fow"`.
2. **POI order must be asserted.** `self._poi_order[(rid, cid)]` must equal
   `provider.parameters`, or the c-vector contracts against the wrong columns
   silently. Hard assert, no fallback.
3. **Negative truth weights are physical here.** A truncated quadratic goes
   negative at large `|c|` — truncation breaking down, not a bug. Report the
   fraction and the negative yield share rather than clipping silently.
4. **The provider comes from the class's configured BIT job**, so truth and
   model share a parametrization by construction. There is no override — a
   study that wants a deliberately different parametrization can point
   `sample_name` elsewhere or edit the config, and no current study needs it.

Cost note: the PDF provider evaluates the parametrization per event
(`PODBasis.evaluate`), so this is materially slower than cache mode — once per
source per toy.

`truth_sources` is `dict[region_id, list[ProcessSource]]`. The `injection:` block
of a spec point (§5) deserializes into exactly this — region id, then class id,
then the `ProcessSource` fields — so `injection`, `truth_sources` and
`ProcessSource` are three views of one object. **Any class with no
explicit source gets an implicit one at `scale_factor=1.0` from its own
configured sample**, so the common case is one line:

```python
truth_sources = {"SR_2016": [ProcessSource(class_id="DY_2016", scale_factor=1.2)]}
```

Per-region algorithm:

1. Resolve the class list from `n2ll.regions` (unbinned) or `n2ll.binned`
   (binned); merge explicit sources with implicit defaults. Reject unknown
   `class_id`s and duplicates.
   The loaders are given the **union** of every feature list the region's
   surrogates consume — classifier, BIT and every PNN — because
   `_eval_region_surrogates` calls `make_column_mask` for each, and each mask
   must resolve against the toy `X`. Build the union from the loaded
   predictors' `feature_names`, not from `job["features"]`.
2. Per source: `factory.get(sample_name).clone()`, then
   `setFeatures(..., observer_names=provider.required_observers + uid_fields)`.
   Stream shards with `materialize(what="fow")`, accumulating `sum(w)` over all
   events and `sum(w)` over the `split` mask; apply the modifiers to the masked
   events, rescale by `1 / f_weight`, throw `n_i ~ Poisson(w_truth_i)`, keep
   `n_i > 0`, record a per-event **origin label** (diagnostics and plotting
   only, never used in the likelihood).
3. Assert all sources share identical `feature_names` — the same check
   `_iter_asimov_batches` makes at [Likelihood.py:731](fit/Likelihood.py#L731).
4. Concatenate to `X_toy (M,d)`, `n_toy (M,)`, `origin_toy (M,)`.
5. `by_class = n2ll._eval_region_surrogates(rid, X_toy, feature_names)`, one
   call on the whole set.
6. Emit `{'X': X_toy, 'w': n_toy, 'by_class': by_class}` — the `_obs_unbinned`
   layout.

A `sample_name` differing from the class's configured sample is legal but warns:
the surrogates were trained on the configured sample, so this is deliberately a
mismodelling injection.

Per-source diagnostics, printed and stored: `f_weight` and the split used,
nominal yield, truth yield, drawn
yield, negative-weight fraction, and **the count of rows with `n_i > 1` plus
their share of the yield**. The last matters for low-statistics backgrounds — a
DY event with `w = 8` yields ~8 duplicates at one phase-space point, and a lumpy
toy is an MC-stat artifact, not a fluctuation to read as coverage failure.

## Changes

### 1. `fit/Modeling.py` — constraint centers

- `ModelParameter.__init__`: add keyword `constraint_center=0.0`, stored as a
  plain float (keeps `penalty()` differentiable in `run_autograd_fit`).
- `Hypothesis.penalty()` becomes
  `sum((p.val - p.constraint_center)**2 for p in self.parameters if p.isPenalized)`.
  With the default centre this is bit-identical to today.
- `Hypothesis.set_constraint_centers(mapping)`: set by name, raising on an
  unknown name or a non-penalized target.
- `Rotated`: mirrored nuisance parameters
  ([Modeling.py:338-342](fit/Modeling.py#L338-L342)) copy `constraint_center`
  for correct printing. `Rotated.penalty()` already forwards to `_base`
  ([Modeling.py:390](fit/Modeling.py#L390)), so nothing else changes.

### 2. `fit/Likelihood.py` — collapse the two copies of `T`, add an array setter

`T` is implemented twice: `_compute_T_chunk` reading `self._h5`
([Likelihood.py:1342](fit/Likelihood.py#L1342)) and again inline in observation
mode reading `byc` ([Likelihood.py:1748-1773](fit/Likelihood.py#L1748-L1773)).

- Extract `_compute_T_from_columns(rid, columns_by_class, cA_per_class,
  nuA_per_group, ln_bias, rate_shift, start, stop)`; `_compute_T_chunk` calls it
  with the cache, observation mode calls it with `byc`. One implementation for
  Asimov, observation and toy generation, and it makes the surrogate-route
  injection in truth mode nearly free.
  **The merge must keep `_compute_T_chunk`'s tolerance of `R_slice.shape[1] == 0`**
  ([Likelihood.py:1358-1360](fit/Likelihood.py#L1358-L1360)); the inline
  observation copy raises unconditionally on a dim mismatch
  ([Likelihood.py:1752](fit/Likelihood.py#L1752)), which would break
  `rate_shift`-only and POI-less classes.
- Move `_eval_region_surrogates` **up** from the subclass `N2LLExtensions`
  ([N2LLExtensions.py:470](fit/N2LLExtensions.py#L470); `class
  N2LLExtensions(N2LL)` at [N2LLExtensions.py:12](fit/N2LLExtensions.py#L12))
  into the base `N2LL`, which is what
  [Likelihood.py:2556](fit/Likelihood.py#L2556) instantiates — that is why the
  move is necessary at all. Rewrite `setObservation` to call it, deleting the
  duplicated inline block at
  [Likelihood.py:1551-1616](fit/Likelihood.py#L1551-L1616).
  **The merged version must take `setObservation`'s permissive branch for a
  missing BIT predictor** — the `(N,0)` array at
  [Likelihood.py:1598](fit/Likelihood.py#L1598), not the hard raise at
  [N2LLExtensions.py:525](fit/N2LLExtensions.py#L525). Two cases need it: the
  `Eta_unbinned` `TTLep_pow_2016` class has no `POI` block at all, and the
  `rate_shift` POI blocks that *are* present
  (`Eta_unbinned_2016.yaml:160,299`) carry no `predictor`, so
  `poi.get('predictor')` is `None` either way.
  **What this costs:** `evaluate_ratio` currently fails loudly on a missing BIT
  predictor via that raise; after the merge it will silently return
  `c_dot_R == 0` instead. That is correct for the observation and toy paths, but
  it is a real weakening for `evaluate_ratio` — note it in the docstring rather
  than letting a future reader discover it.
- Add `setObservationArrays(unbinned_blocks=None, binned_counts=None)`:
  populates `_obs_unbinned` / `_obs_binned_counts` from prepared arrays and
  flips the mode flags exactly as
  [Likelihood.py:1494-1507](fit/Likelihood.py#L1494-L1507). `setObservation` is
  refactored to end in a call to it, so one place owns the mode switch.
  Validates known region ids and that `len(w)` matches every `by_class` column.
  Named for what it takes — already-evaluated arrays, as against the loaders
  `setObservation` takes — rather than after the `block` local at
  [Likelihood.py:1710](fit/Likelihood.py#L1710), which is real precedent but
  undocumented jargon at a public signature.
- Add `setToy(toy, hypothesis)`: calls `setObservationArrays` and applies the
  toy's thrown constraint centres to `hypothesis`.

### 3. `common/derivative_providers.py` — shared provider factory

`PDFDerivativeProvider(job["pdf"])` / `EFTDerivativeProvider(job["eft"].get("parameters", []))`
is constructed identically at four call sites:
[pdf_calibration.py:36](ML/Calibration/pdf_calibration.py#L36),
[eft_calibration.py:35](ML/Calibration/eft_calibration.py#L35),
[pdf_modification_plot.py:47](plot/bit/pdf_modification_plot.py#L47),
[eft_modification_plot.py:46](plot/bit/eft_modification_plot.py#L46). The toy
generator would be a fifth. Four instances warrants the helper: add
`build_derivative_provider(job)` dispatching on the presence of a `pdf` or `eft`
block, and change all four existing call sites over (no compatibility shim).

### 4. `fit/ToyGenerator.py` (new)

Pure generation — no plotting, no fit driving.

No `ToyDataset` loader class: every generator below emits `_obs_unbinned`-shaped
arrays consumed by `setObservationArrays`, so nothing ever needs the loader
contract. (A loader would also have to carry `n_split`, which
`data/toy_data.py:_ArrayLoader` does not expose, and would hit
`setObservation`'s `ignore_weights=True` default and overwrite the
multiplicities with ones.)

- `class ProcessSource` — the only normative definition is in §Multi-process
  above.
No `Toy` class either: a toy is a plain dict, with `save_toy(path, toy)` and
`load_toy(path, n2ll)` as functions. There is no state outliving a call and no
behaviour beyond serialization, so per CLAUDE.md this is a function until proven
otherwise. The dict holds per-region unbinned blocks, binned counts, thrown
`constraint_centers`, generating hypothesis values, seed, `source` and
per-source diagnostics.

- `generate_unbinned_toy_from_cache(n2ll, region_id, hypothesis, rng)` — reads
  `n2ll._h5[(rid, cid)]`, computes `T` via `_compute_T_from_columns` (reusing
  `_assemble_cA_per_class`, `_assemble_nuA_groups`,
  `_assemble_rate_shift_per_class`, `_lnN_by_class`), throws multiplicities, and
  returns **row indices plus multiplicities** for `n_i > 0` — not sliced
  columns. The columns are verbatim slices of the cache, so copying them would
  write roughly one cache per toy; `load_toy` rehydrates them by indexing
  `n2ll._h5`. No `X`, which the cache does not store and `__call__` never uses.
- `generate_unbinned_toy_from_truth(n2ll, region_id, sources, rng, *, split="final_eval", hypothesis=None)` —
  the multi-process algorithm above. A non-None `hypothesis` multiplies
  `w_truth` by `1 + T`, which is the only injection route for nuisances and the
  surrogate alternative to `coefficients` for POIs.
- `generate_binned_toy(n2ll, region_id, rng, *, hypothesis=None, sources=None, split="final_eval")` —
  with `sources` (truth) it resolves the class list from `n2ll.binned`,
  histograms truth weights into `_binned_unroll[rid]['edges']` and throws on
  `lambda_truth`; without it (cache) it throws on `_compute_lambda_binned`. The
  `sources` argument is required for the truth path — a signature taking only a
  hypothesis could only ever produce the model lambda that §Binned rules out.
- `throw_constraint_centers(hypothesis, rng)` — for every `p.isPenalized`.
- `generate_toy(n2ll, seed, *, source, hypothesis=None, truth_sources=None, split="final_eval", throw_nuisances=False, allow_negative_weights=False)` —
  top-level. Unbinned regions come from `n2ll.regions`; binned regions from
  `n2ll.binned` (**not** `n2ll.regions`, which holds unbinned only —
  [Likelihood.py:569](fit/Likelihood.py#L569) vs
  [Likelihood.py:579](fit/Likelihood.py#L579); `_binned_regions_ids` is a list
  of ids, not configs, so it cannot supply a class list).

RNG: one `np.random.SeedSequence(seed)`, spawned per `(region_id, class_id)`
via a stable hash, so adding a region or process does not perturb another's
draws.

Negative MC weights: Poisson is undefined for a negative mean. Crash by default
reporting the offending fraction; `allow_negative_weights=True` clips to zero
and logs the clipped weight sum, so the size of the approximation is always
visible.

Persistence (HDF5, one file per toy), mirroring the `NN2LCache` layout. Cache
mode stores indices; truth mode has no cache to index into, so it stores the
evaluated columns:

    /meta   attrs: seed, source, hypothesis json, config version
    /unbinned/<rid>/n                    (M,)   multiplicities = observed weights
    /unbinned/<rid>/indices              (M,)   cache mode only — rows of _h5
    /unbinned/<rid>/X                    (M,d)  truth mode only
    /unbinned/<rid>/origin               (M,)   truth mode only, class index
    /unbinned/<rid>/by_class/<cid>/g     (M,)   truth mode only
    /unbinned/<rid>/by_class/<cid>/R     (M,nA) truth mode only
    /unbinned/<rid>/by_class/<cid>/Delta::<sid>   (M,nB)  truth mode only
    /binned/<rid>/counts                 (Nflat,)
    /constraint_centers                  (names + values)

`/meta` carries no ids or dimensions. For cache mode the binding is structural —
you are indexing the live cache, so misalignment cannot occur. For truth mode
`setObservationArrays` already validates region ids and that `len(w)` matches
every `by_class` column (§2); extend that to assert `nA`/`nB` against the loaded
predictors, which checks the real runtime objects rather than a schema that has
to be kept in sync with them.

### 5. The toy spec file

A standalone YAML/JSON passed by path, **deliberately not a block in the
analysis config**. This mirrors the existing `--rotate` precedent
([Likelihood.py:2350](fit/Likelihood.py#L2350), loaded at
[Likelihood.py:2500](fit/Likelihood.py#L2500)): model-adjacent, versioned,
reusable across invocations, but outside the analysis YAML — so
`common/yaml_loader.py`, `combine_configs()` and `print_summary` are untouched
and no schema is locked in that other configs depend on.

Governing split: **the spec defines the ensemble; the CLI selects which member
of it and where the result goes.** Everything that defines the pseudo-experiment
lives in the file; only the seed and output paths are flags.

The spec declares **named points**, so one file covers a whole study and each
point's name lands in the output filename:

```yaml
source: truth                  # or cache
split: final_eval              # truth mode only; null for the whole dataset
throw_nuisances: true
allow_negative_weights: false
points:
  - name: nominal
  - name: c0_0p5
    injection:
      SR_2018:
        TTLep_pow_2018:
          coefficients: {c0: 0.5}       # exact PDF/EFT weights
  - name: c0_0p5_bit
    hypothesis: {c0: 0.5}               # same point via the BIT surrogate
  - name: dy_up
    injection:
      SR_2018:
        DY_2018: {scale_factor: 1.2}
```

Per-point keys map onto `ProcessSource` fields (`sample_name`, `coefficients`,
`scale_factor`, `weight_branches`, `weight_function`) plus a
`hypothesis` block for surrogate-route injection. `weight_function` is a dotted
import path resolved with the `importlib` pattern already used for
`defaults.module_samples` ([Likelihood.py:2481](fit/Likelihood.py#L2481)).

The presence of a `hypothesis` block **is** the switch for multiplying by
`1 + T`, so no separate `apply_model_hypothesis` flag is needed. For nuisances
it is the only injection route; for POIs it is the surrogate alternative to
`coefficients`.

### The two POI injection routes

`c0_0p5` and `c0_0p5_bit` above inject the same value by different routes, and
both are *fitted* with the BIT:

- `c0_0p5` weights each event by the exact parametrization evaluated on its
  **generator-level** info (`Generator_x1/x2/id1/id2/scalePDF`). No network
  makes the data.
- `c0_0p5_bit` weights by `w0 * (1 + T(c0=0.5))`, with `T` evaluated on the
  event's **reconstructed features**.

`c0_0p5_bit` is therefore the control — generated and fitted by the same object,
unbiased by construction — and the difference in mean fitted `c0` is the BIT
mismodelling.

The exact weight depends on generator quantities absent from the feature vector,
so the BIT cannot reproduce it event by event and should not: the most it can
learn is `E[w(c)/w(0) | x_reco]`, which is exactly what the likelihood needs,
since only the feature-space density ratio enters. The truth toy's intensity is
`rho_0(x) * R_true(x)` and the BIT toy's is `rho_0(x) * R_BIT(x)`, so **a perfect
BIT makes the two densities identical** — which is what makes this a clean null
test.

They are not equally noisy, however. The truth toy carries extra variance from
the spread of `w_exact / w_0` at fixed `x`, a real MC fluctuation present even
with a perfect BIT. **The comparison statistic is the mean fitted POI, not the
width**; reading a width difference as mismodelling would be an error.

Both points must live under the same `source: truth` and `split`, so they run
over the same events and the difference is apples-to-apples. A cache-mode
control would use all events and break that.

Injecting POIs through both routes at once (`coefficients` and `hypothesis`
naming the same POI) is rejected — they exist precisely so their results can be
compared, as in verification item 9.

### 6. Entry points

**`fit/ToyGenerator.py.__main__` — standalone generation.** Nothing in
generation needs the fitter; it needs the config, loaded surrogates,
`build_cache()` and `prepare_runtime()`, i.e. everything
`Likelihood.py.__main__` does before it reaches `run_autograd_fit`. Generating
separately is worth more than convenience:

- **Generate once, fit many ways.** Comparing `--no_syst` against full, or
  different `--rotate` bases, or POI truncations, is far less noisy on
  *identical* pseudo-data; regenerating per variant compares two ensembles.
- **Split the resource profile.** Generation is memory-heavy (whole cache in
  RAM) and done once; fitting is CPU-heavy and fans out.
- **Inspect before spending.** Check per-source diagnostics and plot toy
  distributions before committing hundreds of fits.

```
python fit/ToyGenerator.py <configs...> --toySpec PATH --toyPoint NAME \
    --seeds 0-499 --outputDir DIR
```

Files are written as `<outputDir>/<point>_toy<seed>.h5`, the same convention
`--toyFile` expects, so a Slurm array task computes its own path from
`$SLURM_ARRAY_TASK_ID` without an index file.

**`fit/Likelihood.py.__main__` gains exactly one flag: `--toyFile PATH`.** It
loads a toy and never generates one. Generation lives solely in
`ToyGenerator.py`, so the fit entry point needs no spec parsing, no point
selection, no seeding, and — importantly — no `importlib` resolution of a
user-supplied dotted path. Two flags is one too many here: once standalone
generation exists for the reasons above, a second generating entry point is a
duplicate with no failure of its own to justify it, and it would contradict this
plan's own "no toy-fit driver" scope line.

Rejecting `--data` and `--asimov` alongside it is not sufficient: nominal Asimov
is the *fall-through default* (`elif args.asimov is None:` at
[Likelihood.py:2609](fit/Likelihood.py#L2609)), so the toy branch must be
inserted **before** it, not merely guarded against an explicit flag.

Applied via `n2ll.setToy(toy, hyp)` where `--data` calls `setObservation`
([Likelihood.py:2603](fit/Likelihood.py#L2603)). The point name and seed are
read from the toy's `/meta`, and the output suffix gains `_{POINT}_toy{SEED}` so
`serialize_result` writes one JSON per toy without collision.

## Verification

Run from the repo root against `configs/unbinned_v6/unbinned_2018.yaml`, and
`configs/Eta_unbinned/Eta_unbinned_2016.yaml` for the multi-process path.

1. **No regression** — rerun a known Asimov fit; `fval` and covariance must be
   unchanged, since constraint centres default to 0 and the `T` refactor is
   pure extraction:
   `python fit/Likelihood.py configs/unbinned_v6/unbinned_2018.yaml --overwrite fit`
2. **Deterministic injection check** — build a "toy" whose multiplicities are
   the raw `w0` instead of a Poisson draw, inject via `setObservationArrays`,
   and confirm `n2ll(hyp)` **equals** the nominal Asimov value for several
   hypotheses. Equality is exact, not up to an offset: with `w == w0` the
   observation branch `-2(sum w log1p(T) - sum w0 T)`
   ([Likelihood.py:1742](fit/Likelihood.py#L1742),
   [Likelihood.py:1775](fit/Likelihood.py#L1775)) is algebraically the Asimov
   branch `sum w0 (log1p(T) - T)`
   ([Likelihood.py:1829](fit/Likelihood.py#L1829)). Validates the whole cache
   path with no random numbers involved.
3. **`T` refactor equivalence** — capture `_compute_T_chunk` output **before**
   the refactor and assert `_compute_T_from_columns` reproduces it elementwise.
   Comparing the two afterwards is vacuous, since `_compute_T_chunk` will
   delegate to it.
4. **Yield closure** — over ~200 cache-mode toys at nominal, the mean of
   `sum n_i` must equal `sum w0_i * (1 + T)` within its Poisson error per
   region; likewise binned counts against `lambda`.
5. **Pulls and coverage** — ~500 toys at a point carrying
   `hypothesis: {c0: 0.5}` with `throw_nuisances: true`; the pull
   `(c_fit - c_true)/sigma_fit` should centre at 0 with unit width. Departures
   of the width from 1 are the non-asymptotic effect this exists to expose.
6. **Truth mode closure** — `source: truth` with a point carrying no modifiers,
   on the multi-class Eta config, is the surrogate-correct limit, so its pull
   **mean** must agree with cache mode. **Compare means only, not widths.**
   Truth mode defaults to `split: final_eval` (20% of events, weights rescaled
   ~5x) while cache mode uses the full dataset, so the two ensembles have
   different MC granularity by construction and their pull widths will differ
   for reasons unrelated to the surrogates. A mean shift is a mismodelling
   signal; a width difference is not. (Setting `split: null` for this one point
   makes the widths comparable too, if that is wanted.)
7. **Multi-process injection** — a point with `EtaS_2016: {scale_factor: 1.2}`;
   confirm the drawn yield of that source alone moves by 20% in the diagnostics
   while the others are unchanged, and inspect the induced shift in the fitted
   POIs.
8. **Exact-weight reconstruction check** — at `c = 0`, `w_coefficients` must
   equal the nominal weight elementwise. Then verify the reconstruction against
   an independent route: for a single-coefficient injection, compare
   `w_coefficients` to a direct finite-difference of the parametrization, and
   confirm the ratio `w_coefficients / w_nominal` matches the BIT's `1 + c_A R_A`
   to within the BIT's known calibration accuracy (the quantity
   `ML/Calibration/pdf_calibration.py` already measures). This is the check that
   catches a mis-ordered or mis-normalized contraction.
9. **Surrogate bias measurement** — the headline use, and why the two injection
    routes both exist. Fit the `c0_0p5` point (exact weights) and the
    `c0_0p5_bit` point (surrogate weights) from the example spec above. The
    difference in mean fitted `c0` is the BIT mismodelling bias, with MC
    statistics common to both.
10. **Split normalization** — the failure mode here is silent. Generate the same
    truth-mode point with `split: final_eval` and `split: null`; the mean drawn
    yield per region must agree within its Poisson error, and both must match
    `sum w0` over the cache. Confirm `f_weight` is computed from weights and
    differs from the nominal bucket fraction `0.20` by a small but nonzero
    amount, and check that the `n_i > 1` share is reported after rescaling and
    is roughly five times worse on the split.
11. **Negative-weight reporting** — push `|c|` up until the truncated quadratic
    turns negative and confirm the generator reports the negative fraction and
    refuses to proceed unless `allow_negative_weights` is set.
12. **Persistence round-trip and binding** — generate with
    `fit/ToyGenerator.py`, refit from the saved file, confirm identical `fval`.
    Then load the same toy against a config with a different `version` or a
    retrained BIT and confirm `load_toy` crashes rather than misaligning columns
    silently.

## Implementation order

Work in a **git worktree**, one commit per piece of functionality:

| # | Commit | Contents | Gate |
|---|---|---|---|
| 1 | provider factory | Change 3: `build_derivative_provider(job)` + the four call sites | existing calibration/plot scripts still run |
| 2 | `T` and surrogate-eval refactor | Change 2: `_compute_T_from_columns`, `_eval_region_surrogates` moved to base `N2LL`, `setObservationArrays`, `setToy` | **verification 1-3, exact equality** |
| 3 | constraint centers | Change 1: `fit/Modeling.py` | verification 1 again (centres default to 0, so `fval` unchanged) |
| 4 | the generator | Change 4: `fit/ToyGenerator.py`; Change 5: spec file; Change 6: `--toyFile` | verification 4-12 |

Commits 1 and 3 are independent of everything else and can land in any order.
Commit 2 is the risky one and is gated hardest. Commit 4 is the only one that
adds new behaviour rather than moving existing behaviour.


**Do Changes 2 first, as its own commit, and run verification 1-3 before
touching anything else.** The `T` refactor and the `_eval_region_surrogates`
move are the highest-risk, lowest-creativity part of this work: they sit next to
numba-jitted code (`_weighted_sum_log1p_minus_x`,
[Likelihood.py:1829](fit/Likelihood.py#L1829)) and must preserve numerics
exactly. Verification 2 is an exact-equality check, so it either passes or it
does not. Everything else builds on that foundation and is far cheaper to debug
once it is known good.

**If verification 2 fails and the cause is not obvious within a couple of
attempts, stop and escalate rather than patch.** Silent numerical wrongness is
the failure mode throughout this feature — a toy that is subtly wrong still
produces plausible pulls, plausible coverage and plausible plots. There is no
downstream check that will catch it. Treat a failing exact-equality assertion as
a hard stop, not as a tolerance to loosen.

## Rejected alternatives

Recorded separately in `toy-generation-design-decisions.md`, alongside this
file. Those decisions are settled — read that file before reopening any of
them. Rejections that bear on a specific section are also noted inline where
they apply, so implementing from this file alone will not reintroduce one.
