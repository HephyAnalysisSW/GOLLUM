# Toy generation: design decisions

Rejected alternatives for the work planned in
`toy-generation-for-pseudo-experiments.md`. Recorded for auditability. Several are also noted inline where they bear on a
specific section; this consolidates them with the reasoning. Design settled
2026-07-27.

**Statistical / physics choices**

- *Generating toys from an MC subset statistically independent of the one
  supplying the expected-yield term.* Rejected. It would remove the
  toy/template correlation, but halves effective MC statistics and requires
  reweighting the two halves. The correlation is a sub-percent effect on the
  widths at current MC statistics. Revisit only if MC statistics grow enough
  that the cost disappears, or if the widths are ever shown to be biased at the
  percent level.
- *Fixing nuisances at their true values instead of throwing global
  observables.* Rejected. Coverage would be optimistic wherever systematics
  matter, which is the regime of interest. Cost is the `constraint_center`
  addition to `fit/Modeling.py`, which is small and reduces exactly to current
  behaviour at centre zero.
- *Applying a UID split in cache mode.* Rejected on both grounds: it would
  require UID columns in the HDF5 cache, a format change forcing every cache to
  be rebuilt, and it is unnecessary because the model *is* the surrogate by
  definition, so training leakage cannot bias coverage of the model against
  itself. Only MC granularity is affected, which argues for using everything.
- *Rescaling a split by the nominal bucket fraction (0.20) rather than the
  measured weight fraction.* Rejected. UID hashing distributes buckets, not
  weights, so the nominal fraction injects a normalization bias that would
  masquerade as exactly the rate bias under measurement.
- *Per-event Poisson then histogramming, for binned truth toys.* Rejected in
  favour of one draw per bin. The two are distributionally identical — a sum of
  independent Poissons is Poisson with the summed mean, and regions are disjoint
  selections so there is no shared-event correlation — so histogram-first is
  strictly cheaper.
- *Re-reading ROOT and re-evaluating surrogates per toy in cache mode.*
  Rejected. Physically identical to slicing the cached columns, but far slower,
  since `g`, `R` and `Delta::*` are already stored per event. The genuinely
  different capability this seemed to offer became truth mode instead.

**Scope and interface choices**

- *A `toys:` block in the analysis YAML config.* Rejected in favour of a
  standalone `--toySpec` file. The injection is per-invocation, not part of the
  statistical model — hundreds of runs share one model — whereas
  `classifier.asimov` is in YAML precisely because it defines the model's
  template MC. A new top-level block would also need validation in
  `_apply_defaults_and_checks`, a merge rule in `combine_configs()` and a
  `print_summary` entry, all in `common/yaml_loader.py`, which every training
  entry point depends on. Since CLAUDE.md forbids compatibility shims, a later
  schema change would mean editing every config that adopted it. Precedent for
  the standalone file: `--rotate`. Promoting the spec into the config later is
  mechanical if it proves to be model after all.
- *A toy-fit driver, Slurm fan-out, and merge/plot step.* Deliberately out of
  scope. The generator plus `fit/ToyGenerator.py.__main__` is enough to produce
  and refit toys; the fan-out pattern already exists in `fit/slurm_utils.py`
  when wanted.
- *CLI grammar `--toyScale REGION:CLASS=FACTOR`, `--toySample`,
  `--toyCoefficients`, `--toyHypothesis`, `--toySource`, `--throwNuisances`.*
  Rejected in favour of named points inside the spec file. The ad-hoc
  `REGION:CLASS=` parsing was error-prone, and nine flags collapsed to four once
  the spec defines the ensemble and the CLI only selects a member of it. Point
  names also land in output filenames for free.
- *A `ToyDataset` in-memory loader class.* Rejected. Once `setObservationArrays`
  exists nothing consumes the loader contract; `data/toy_data.py:_ArrayLoader`
  does not expose the required `n_split`; and routing through `setObservation`
  would hit its `ignore_weights=True` default and overwrite the multiplicities
  with ones.
- *An `apply_model_hypothesis` flag on truth-mode generation.* Rejected as
  redundant — the presence of a `hypothesis` block on a spec point is the
  switch.
- *The name `setObservationFromBlocks`.* Rejected for `setObservationArrays`.
  "Block" is real precedent ([Likelihood.py:1710](fit/Likelihood.py#L1710)) but
  undocumented jargon at a public signature; the meaningful distinction from
  `setObservation` is that it takes already-evaluated arrays rather than
  loaders.

**Simplicity pass, 2026-07-28**

Added after a cold-read review against the "Code simplicity" section of
CLAUDE.md, which postdates most of the plan.

- *Toy generation in `fit/Likelihood.py.__main__` as well as `ToyGenerator.py`.*
  Rejected. Once standalone generation exists — justified by generate-once /
  fit-many-ways, which avoids comparing fit variants across two different random
  ensembles — a second generating entry point duplicates spec parsing, point
  selection and seeding with no failure of its own to justify it, and
  contradicts the plan's "no toy-fit driver" scope line. The fitter takes
  `--toyFile` only, which also keeps `importlib` resolution of a user-supplied
  dotted path out of the fit entry point.
- *Persisting `g`/`R`/`Delta::*` per cache-mode toy.* Rejected. Those rows are
  verbatim slices of `n2ll._h5`, so 500 toys would write roughly 500 copies of
  the cache. Storing `(indices, multiplicities)` and rehydrating at load is
  strictly smaller and makes column misalignment impossible by construction
  rather than by validation.
- *A `/meta` binding payload of region and class ids, `nA`/`nB` dims and
  delta-group ids, cross-checked on load.* Rejected as mostly redundant once
  cache-mode toys index the live cache. Truth mode asserts shapes against the
  loaded predictors instead — checking the real runtime objects rather than a
  schema that must be kept in sync with them.
- *`Toy` as a class.* Rejected for a plain dict plus `save_toy` / `load_toy`.
  No state outlives a call and there is no behaviour beyond serialization.
- *A `derivative_job` override on `ProcessSource`.* Rejected. Justified only by
  a hypothetical ("e.g. a different truncation") with no study behind it and no
  verification item. The provider comes from the class's configured BIT job.

**Considered in the same pass and deliberately kept**

Recorded because a cold reader will propose cutting these again — they look
unjustified without the context below.

- *`sample_name`, `weight_branches`, `weight_function` on `ProcessSource`.* All
  three are explicit user requirements: the three mechanisms chosen for
  supplying truth density (an alternative sample, extra weight branches, an
  arbitrary callable). "No named use case in the plan" is true and beside the
  point — the requirement predates the plan.
- *`allow_negative_weights`.* Kept. Crash-and-report is the default; the flag
  exists so a study can proceed knowingly at large `|c|` where the truncated
  quadratic goes negative.
- *`split: null`.* Kept. Verification 6 needs it to make pull *widths*
  comparable between truth and cache mode; without it that comparison is
  mean-only.
- *Named `points:` in the spec.* Kept. One file per injection point turns an
  injection scan into a directory of near-duplicate files; point names also land
  in output filenames for free.

**Two-stage framing and the reweighting split, 2026-08-18**

Added after a session that tried to refit the toys under
`output/eval_sm_point/` (generated with `version: unbinned_v7_eft`) against
`configs/unbinned_v7_eft_genpoint/with_selection/unbinned_2016_eft.yaml`. The
refit crashed, and the diagnosis showed that neither document records the two
points below. Both are settled.

*Cache mode and truth mode are stages, not alternatives.* The `source:` key
presents them as two interchangeable options. They are not. Truth mode asks
whether the surrogates are right: it re-reads ROOT, so it is slow, and it
defaults to a 20% split, so it is coarse. You run it once per surrogate
generation. Cache mode asks whether the statistical model is calibrated, given
correct surrogates: it is fast, cheap on disk, and uses every event. You run it
whenever the model changes. Stage 1 licenses stage 2, and verification 9 in the
plan is exactly the bridge between them, though the plan does not frame it that
way. State the ordering in the module docstring.

*A toy should be exactly as portable as real data, and no more.* This is the
test that separates the coupling a toy cannot escape from the coupling it
should not have. Real data is bound to the selection, to the samples and era it
represents, and to the feature definitions. Real data is not bound to which BIT
was trained, or to how many operators the fit floats.

| Coupling | Kind | Verdict |
|---|---|---|
| `default_selection` | defines the dataset | intrinsic: keep, and check it |
| `classifier.asimov` samples | defines the dataset | intrinsic |
| Feature list and order | defines the dataset | intrinsic, but must be stored |
| Frozen `R` column count | one surrogate's view | incidental: remove |
| Cache row indices | a storage trick | incidental, and the tightest of all |

The plan conflates the two kinds, and so guards neither. Measured on the 2016
EFT SR: the old cache holds 1,281,762 rows with `sum(w0) = 79,237.9`, and the
genpoint no-selection cache holds 1,325,594 rows with `sum(w0) = 127,686.1`,
because the selections differ. A cache toy's largest index, 1,281,746, falls
below both row counts, so stale indices raise no `IndexError` and misalign in
silence. Cache files carry no attributes at all, and `save_toy` omits the
feature list, so nothing can detect either mismatch today.

*Decided: `cache -> BIT reweighting`, `truth -> truth reweighting` are the two
named idioms.* BIT reweighting is not a property of cache mode; truth mode also
applies it, when a spec point carries a `hypothesis` block. Presenting the two
POI routes as co-equal, as the plan's section "The two POI injection routes"
does, mixes the two vocabularies and invites a reader to reach for the slower,
noisier path by default. Keep the mechanism and demote it in the documentation.

The plan justifies BIT reweighting inside truth mode as the control for
verification 9, on the grounds that a cache-mode control would use all events
while the truth point uses 20%. That objection dissolves when the truth point
sets `split: null`, which makes both ensembles run over every event and a
cache-mode control valid. The genuine exception is a *compound* point: an exact
truth weight together with a shifted nuisance. Cache mode cannot express it,
because the truth weight has no cached form. Document the compound point as the
single exception rather than as a second route.

*Rejected: storing `X` in cache-mode toys.* Considered because it would make a
cache toy as portable as real data and remove the index binding entirely.
Rejected on the property that cache mode exists for. Measured: 1.2 MB per cache
toy against 22 MB per truth toy, so 1000 toys grow from 1.2 GB to 22 GB. Fast
generation over all events, at low storage cost, is the whole value of cache
mode for bias and coverage work. Cache-mode safety therefore comes from a
fingerprint check on the cache, not from a self-contained file.

*Decided: the fingerprint checks the dataset, and crashes when it is absent.* The
guard splits into two tiers. Tier 1 is compared in both modes: the normalized
selection string, the selection features, and per region the feature union, the
Asimov sample list and the class ids. Tier 2 is compared in cache mode only: the
cache row count and a blake2b digest of its `w0` column. The digest is the
load-bearing half, because a cache rebuilt to the same length in a different order
misaligns a toy's indices without changing any length. Measured cost: 140 ms per
class to read and hash 10.3 MB of `w0`, once per fit.

Two sub-decisions inside it:

- *Rejected: one combined digest over all fields.* A single digest reports only
  that something differs, which is barely better than the downstream shape error it
  replaces. Separate fields let the error name the field and print both values.
- *Rejected: a JSON field in the cache's sidecar meta.* The cache JSON exists and
  holds only `delta_groups`, so it looks like the natural home. But existing caches
  were written without any new field, so a load-time comparison against it would
  fail for every cache already on disk. Both ends compute the digest from the live
  HDF5 instead, which needs no cache format change and no rebuild.
- *`generating_bit` is recorded and never compared.* A refit under a retrained or
  differently-truncated BIT is the surrogate-bias study, not an error. The field
  still matters to a reader: for a *non-nominal* toy the multiplicities were drawn
  through the generating BIT's `T`, so the pseudo-data does encode it. At nominal
  `T = 0`, which is why nominal toys are the benign case.
- *A toy with no `/fingerprint` crashes.* Warning and proceeding would preserve
  exactly the silence the guard removes, and the toys predating the fingerprint also
  froze their BIT columns on disk, so they are not portable anyway. No flag is added
  for this; demoting it is a one-line local edit if a study ever needs it.

*Rejected: keeping both `X` and a stored `R`, with a fast path when the BIT
matches.* It doubles the read paths to save one forward pass per fit, and the fast
path is the one that rots silently, because nobody exercises it after the BIT
changes. Measured: the BIT costs 69 ms on 57,512 events, against a minimization in
minutes. One path -- `X` in, surrogates evaluated at load, always.

**The operator span of a truth-mode injection, 2026-08-18**

Added after a session asked why `coefficients` cannot reach all 16 operators,
given that the ROOT samples carry the full lower-triangular derivative set over
`wc_names` and the EFT weight is exactly quadratic. The question is sound: the
data does determine `w(c)` at any point in the full 16-dimensional space,
without a BIT. A search of both toy documents found the alternative nowhere.
The bullet above rejecting a `derivative_job` override covers a different
axis -- which job supplies the parametrization, not how many operators it spans.

*Decided: an unnamed coefficient stays at the generation point, and the truth
provider keeps coming from the class's BIT job.* Four reasons, the first
decisive.

- The fit already uses this convention. A subset job holds unfitted operators
  at `r`, not at the SM, and that is recorded and accepted in
  `eft-expansion-point-rebase-design-decisions.md:102-111`. Toy and fit
  therefore condition on the same point. Widening the toy alone would break an
  agreement that is currently exact.
- Widening the provider silently changes every spec already written.
  `expand_pois_linear_quadratic` builds `t = c - r` for every name in
  `poi_names`, reading an absent name as `0.0`. Add the other operators to
  `poi_names` and each one moves from `r` to the SM. The file does not change;
  its meaning does.
- The full-16 SM point is a long extrapolation. It sets `t = -1.5` in fourteen
  directions and `t = +0.5` in two, simultaneously. No events were generated
  there, which is the same objection that killed the hybrid reference point for
  the fit. Expect a wide weight spread and a large negative fraction.
- A closure test needs the truth point inside the model's span. Under the
  current default it is, and the only difference left between the two routes is
  BIT approximation error, which is what the test measures. Under a widened
  default a subset job would see linear and cross terms it has no columns for,
  and a *perfect* BIT would still show a bias.

*Deferred, not rejected: a truth provider spanning all of `wc_names`,
independent of the BIT job.* It has a namable study behind it -- displace an
operator the BIT does not fit, refit, and measure the induced bias -- which is
why it is not filed with the hypotheticals above. It stays out for now because
no verification item needs it, and because it must not arrive as a change of
default. It needs its own spelling in the spec, so that a point written today
keeps its meaning.

*Acknowledged tension.* "Real data is not bound to which BIT was trained, or to
how many operators the fit floats", above, is applied in this document only to
what a toy stores and to what the fingerprint checks. It arguably also applies
to what a toy spans, and the current design binds generation to a BIT job's
operator list. The decision above stands on the fit-conditioning argument, not
on a claim that the tension is resolved.

*Decided: naming a coefficient the class's BIT cannot move is a crash.* Two
cases, separated because the causes differ. A known operator outside the job's
parameters cannot be moved at all, so the message names the generation point it
is held at and suggests a job that fits it. Anything else is a spelling
mistake, reported against the known-coefficient list. Neither group was read by
`expand_pois_linear_quadratic`, which iterates `provider.parameters` alone, so
before the guard both were dropped in silence and the toy landed at a different
point than the spec asked for. Implemented in `99c7976`, together with an INFO
line stating the resolved point in three groups: injected, at the SM, held at
the generation point. The middle group is a correct default rather than a
misconfiguration, so it is reported and not warned about -- but since the
rebase it no longer coincides with the generation point, and silence about
which unset operators sit where would mislead.
