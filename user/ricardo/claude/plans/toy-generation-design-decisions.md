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
