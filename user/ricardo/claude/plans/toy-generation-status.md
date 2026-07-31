# Toy generation: implementation status

Living status file for finishing the work in
`toy-generation-for-pseudo-experiments.md` (design decisions in
`toy-generation-design-decisions.md`). Update this as you debug/extend on your
end so the history doesn't only live in chat.

Branch: `toy-generation-for-pseudo-experiments`, worktree at
`/users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM-toygen`.

## Landed (2026-07-29)

- `f265664` -- `--plot` flag on `fit/ToyGenerator.py`: after generating a
  point's toys, writes one figure per (unbinned region, default_feature),
  overlaying each requested seed's weighted distribution (binning from
  `data/plot_options`). New `plot/toys/toy_diagnostic_plots.py`; new
  `fit.ToyGenerator.materialize_cache_region_features` (re-streams a region's
  Asimov samples once, at the union of requested seeds' cache indices, to
  recover raw kinematics for cache-mode toys, which otherwise only store
  surrogate columns + row indices -- costly by design, opt-in only). Truth-mode
  toys already carry X, but only for the region's own surrogate feature union;
  default_features outside that union are skipped with a logged warning.
  **Real-run verified**: `unbinned_2018.yaml` + `toys_cache_example.yaml`,
  seeds 0-2, nominal point -- ran end to end, no crashes, exactly the 16
  expected `TOP_KINEMATICS + LEPTON_KINEMATICS + ASYMMETRY` features plotted
  for `SR_2018` (32 files: png+pdf). One bug found and fixed along the way:
  the plotting module originally imported `data.colors.cmap_petroff10_mpl`
  (matching `plot/bit/modification_plotter.py`'s convention) for the toy-seed
  color palette, but that module imports ROOT, and ROOT's cling JIT segfaulted
  when initialized a second time in-process after `fit/ToyGenerator.py` had
  already streamed samples via PyROOT/RDataLoader. Fixed by inlining the same
  10-color hex list directly instead of importing `data.colors`, keeping this
  purely-matplotlib diagnostic script ROOT-free. (Separately, CERN EOS sync
  via `common.syncer` fails in this environment -- no valid Kerberos ticket --
  unrelated to the plotting code itself.)

- `f336ff1` -- full Asimov-from-cache reference band added to the diagnostic
  plots: `materialize_cache_region_features` generalized into
  `materialize_cache_region_diagnostics`, which accumulates a weighted
  (`w0*(1+T)` at the toys' generation hypothesis, over the WHOLE cached MC)
  histogram in the same re-streaming pass that recovers cache-mode toys' raw
  kinematics -- no extra pass needed. `_compute_cache_intensity` factored out
  of `generate_unbinned_toy_from_cache`, shared with the new diagnostics path.
  Always computed regardless of toy source (cache/truth), since the surrogate
  cache is always built. **Real-run verified** against
  `unbinned_v7/unbinned_2016.yaml` (`SR_2016`, nominal point, seeds 0-2): grey
  Asimov band tracks the toy seeds' mean, with visibly larger Poisson scatter
  in the low-statistics tail (e.g. `tr_ttbar_mass` above ~2500 GeV), as
  expected.
  - **Real environment finding (not fixed, not code-related)**: two
    consecutive real `--plot` runs (2018 retry, then 2016) crashed with the
    same ROOT/cling segfault as the earlier `data.colors` issue -- but this
    time triggered by `common.syncer`'s own module-level `import ROOT`
    (it monkey-patches `TCanvas.Print`), not by anything in this branch's
    code. Confirmed by re-running the plotting step directly against the
    already-saved toy files with `syncer` excluded: completed cleanly, no
    crash, correct-looking plots. You explicitly decided to keep `syncer` as
    invoked (matches house convention; you'll refresh Kerberos manually for
    EOS uploads) and treat the crash as pre-existing environment flakiness in
    this conda env's cling JIT (mismatched compiler/std headers, see the
    repeated `cannot extract standard library include paths` / `assert.h not
    found` warnings that appear on every ROOT touch here, crash or not) --
    not something to work around in this branch. If it becomes a recurring
    annoyance for the real `--plot` output path (`www/toys/...`), the isolated
    fix is dropping `syncer` from `ToyGenerator.py`'s `--plot` branch alone
    (this diagnostic script has no real need for EOS auto-sync), not touching
    `common/syncer.py` itself.
  - Example plots (grey Asimov band + 3 toy seeds, `SR_2016`) are sitting in
    `/users/ricardo.barrue/.claude/jobs/cb7ec444/tmp/toys_plot_test_2016_plots_only/SR_2016/`
    -- job-scratch, not durable; you said you'd copy them yourself.

## Landed (2026-07-28)

Four commits, in plan order:

1. `44705f9` -- `build_derivative_provider(job)` factory in
   `common/derivative_providers.py`, four call sites switched over
   (`pdf_calibration.py`, `eft_calibration.py`, `pdf_modification_plot.py`,
   `eft_modification_plot.py`).
2. `66b67d7` -- the risky one: `_compute_T_from_columns` unifies
   `_compute_T_chunk` (cache) and `setObservation`'s inline by_class loop;
   `_eval_region_surrogates` moved from `N2LLExtensions` to base `N2LL`, now
   permissive on a missing BIT predictor (documented as a real weakening for
   `evaluate_ratio`); `setObservationArrays(unbinned_blocks, binned_counts)`
   added, `setObservation` refactored to end by calling it.
3. `2029f58` -- `ModelParameter.constraint_center`,
   `Hypothesis.penalty()` centred, `Hypothesis.set_constraint_centers(mapping)`,
   `Rotated` mirrors `constraint_center` on nuisance passthrough.
4. `7439354` -- `fit/ToyGenerator.py` (generation), `N2LL.setToy` +
   `--toyFile` in `fit/Likelihood.py`, `_uid_split_interval` generalized in
   `ML/Calibration/calibration_runner.py`, two example specs
   (`configs/unbinned_v6/toys_example.yaml`,
   `configs/Eta_unbinned/toys_example.yaml`).

## What's actually verified vs. not

No trained surrogates or MC caches exist anywhere in this environment
(`common/yaml_loader.py` reported 0/64 ready artifacts even on the unmodified
`sbi-eft-CMS` branch), so nothing here has run against a real config yet.
Everything below was checked with a synthetic in-memory `N2LL` (fake
classifier/BIT/PNN predictors, fabricated cache arrays) instead:

- **Verified, bit-for-bit / exact:**
  - `_compute_T_chunk` / `_compute_T_from_columns` reproduce the pre-refactor
    reference elementwise (verification item 3).
  - Deterministic-injection identity: observation branch == Asimov branch when
    `w == w0` (verification item 2), to float roundoff (~1e-14).
  - `ModelParameter.constraint_center` default 0 leaves `penalty()`/`fval`
    unchanged.
  - `save_toy` -> `load_toy` -> `setToy` round trip gives a bit-identical
    `n2ll(hyp)` for a cache-mode toy.
  - Exact PDF/EFT weight reconstruction formula
    (`deriv_w[:,0] + deriv_w[:,1:] @ expand_pois_linear_quadratic(...)`)
    checked at c=0 (reduces to nominal) and against an independent
    finite-difference derivative (verification item 8's core math).
- **Verified, statistically (Poisson closure over ~300 toys):**
  - Cache-mode unbinned yield closure (verification item 4, unbinned half).
  - Cache-mode binned yield closure, via a monkeypatched `_compute_lambda_binned`
    (verification item 4, binned half) -- real ICH/ICPH wiring not exercised.
  - `throw_constraint_centers` mean/width.
- **Implemented but NOT exercised at all:**
  - The entire truth-mode ROOT-streaming path in
    `_materialize_truth_weights`: `RDataLoader.clone()`/`setFeatures`,
    `UIDSplitter.mask_from_np`, `f_weight` renormalization,
    `ProcessSource.weight_branches`/`weight_function`, negative-weight
    reporting, per-source diagnostics, multi-process concatenation
    (`generate_unbinned_toy_from_truth`, `generate_binned_toy` truth branch).
  - `_class_bit_job` / `n2ll._toy_jobs_by_id` / `n2ll._toy_splitting_defaults`
    wiring -- the pragmatic choice (not literally specified in the plan) of
    attaching these to `n2ll` from `ToyGenerator.py.__main__` rather than
    threading `cfg` through every generator call. Worth a second look once you
    can actually run it.
  - `fit/ToyGenerator.py.__main__` end to end (config loading, `build_cache`,
    `prepare_runtime`, spec parsing, `--seeds` range parsing).
  - `--toyFile` in `fit/Likelihood.py.__main__` (loads a toy, calls `setToy`,
    suffixes the output path with `_{point}_toy{seed}`).
  - Verification items 1, 5, 6, 7, 9, 10, 11, 12 from the plan -- all need a
    real config with trained surrogates.

## Suggested first real-run order

1. **Verification 1** (no-regression Asimov fit) on
   `configs/unbinned_v6/unbinned_2018.yaml` once its surrogates are trained --
   this is the cheapest way to catch anything the synthetic fixture couldn't,
   since it exercises the real `_prepare_structure`/`prepare_runtime` path the
   synthetic tests bypassed.
2. Generate a handful of cache-mode toys (`source: cache`, small `--seeds`
   range) and check verification 4 (yield closure) for real -- fast, no ROOT
   streaming involved, isolates whether the cache-mode generator is fine
   before touching truth mode.
3. Truth mode: start with the `nominal` point in
   `configs/unbinned_v6/toys_example.yaml` (no modifiers) at `--seeds 0-2` and
   just confirm it runs and produces sane diagnostics -- this is the first
   real test of the whole `RDataLoader`/UID-split/`_eval_region_surrogates`
   chain. Watch for `f_weight` being close to the `final_eval` nominal bucket
   fraction (~0.2) but not exactly it (verification item 10's "small but
   nonzero" check).
4. Then `c0_0p5` / `c0_0p5_bit` (verification 8-9) and the multi-process
   config (verification 6-7).

## Log

- 2026-07-28: initial implementation landed (see "Landed" above). Not yet run
  against a real config.
- 2026-07-28: **verification 1 (no-regression Asimov fit), run for real.**
  Config: `configs/unbinned_v7/unbinned_2018.yaml` (multi-process SR_2018:
  TTLep_pow_2018, SingleTop_2018, DrellYan_LO_HTbinned_2018, TTSemi_pow_2018;
  73/75 artifacts ready, only the two TFMC classifier jobs missing -> classifier
  degrades to `g=1` for all classes, harmless for this bit-identity check).
  Method: a temporary detached-HEAD worktree at `ed8532d` (the commit right
  before any of these 4 commits) at `/users/ricardo.barrue/nsbi_gluon_pdf/GOLLUM-ref`,
  pointed at the same `common/user.py`/`common/directories.py` paths as the
  main worktree, ran `python fit/Likelihood.py configs/unbinned_v7/unbinned_2018.yaml
  --overwrite fit --verbosity 0` there (pre-refactor baseline) and in
  `GOLLUM-toygen` (post-refactor), both against the *same* pre-built cache
  (`NN2LCache/unbinned_2018/unbinned_v7`), then diffed the two `_fit.json`
  outputs. Result: **fval, edm, niter, all 42 parameter values, all 42
  parameter errors, and the full 42x42 covariance and correlation matrices are
  bit-for-bit identical (max abs diff = 0.0 everywhere).** `fval=0` both times,
  as expected for a nominal Asimov self-fit.
  Two environment issues found and fixed along the way, unrelated to the toy
  generation code itself: (1) the worktree had briefly been nested inside
  `GOLLUM/`, which broke every `sys.path.insert(0, '..')` in the codebase
  (CWD-relative, not script-relative) -- moved back to being a sibling
  directory; (2) `fit/Likelihood.py.__main__` unconditionally constructs a
  `Data_<era>` loader even in Asimov mode (pre-existing, confirmed via `git
  diff ed8532d HEAD -- fit/Likelihood.py` that this code is untouched) --
  needed `common/directories.py`'s `SAMPLES_RUNII_BASE_DIRECTORY` pointed at
  `v2-3-2_nJ2p_nB2p_2l` (has `Data_nominal.root` for all four eras) rather than
  `v2-3_nJ2p_nB2p_trvalid` (only has a `2016/` subfolder, no Data at all).
  The `GOLLUM-ref` worktree is still on disk in case it's useful for further
  comparisons -- remove with `git worktree remove` (from `GOLLUM` or
  `GOLLUM-toygen`) plus deleting the directory once done with it.
  Remaining from "Suggested first real-run order": cache-mode toy generation
  (step 2) and truth-mode (steps 3-4) still untested against real data.
- 2026-07-28: **verification 4 (cache-mode yield closure) and the
  `--toyFile`/`setToy`/`load_toy` wiring, run for real**, on the same
  `configs/unbinned_v7/unbinned_2018.yaml`.
  - New spec `configs/unbinned_v7/toys_cache_example.yaml` (`source: cache`,
    `throw_nuisances: true`, points `nominal` and `c0_0p5`).
  - Bug found and fixed: `generate_unbinned_toy_from_cache` hard-crashed
    unconditionally on **any** negative intensity, but 0.33% of events in every
    class of this cache have negative `w0` (routine NLO-generator negative
    weights -- confirmed directly from the cache: `TTLep_pow_2018.h5['w0']`
    has 25683/7760846 negative entries, min -0.11), which triggers even at the
    nominal hypothesis where `T` is identically 0 everywhere. Fixed by
    extending the already-designed `allow_negative_weights` escape hatch (so
    far only wired for truth mode) to `generate_unbinned_toy_from_cache` and
    `generate_binned_toy`'s cache branch too, and threading it through
    `generate_toy`'s cache-mode calls. Same crash-by-default /
    clip-when-allowed semantics as truth mode, just applied to the mundane
    "negative MC weight" cause as well as the "T < -1" cause.
  - Also found and fixed: `fit/ToyGenerator.py` imported
    `common.derivative_providers.build_derivative_provider` at module level,
    which eagerly imports `data/samples_eft.py` (constructs RDataLoaders for
    every declared EFT sample at import time) -- so cache-mode generation
    depended on unrelated EFT sample data being reachable just to *import* the
    module. Moved the import inline into `_materialize_truth_weights` (its
    only call site, truth-mode only); `import fit.ToyGenerator` now succeeds
    standalone with no EFT stub needed.
  - Generated 5 cache-mode toys (`nominal` point, seeds 0-4,
    `allow_negative_weights: true`) via
    `python fit/ToyGenerator.py configs/unbinned_v7/unbinned_2018.yaml --toySpec
    configs/unbinned_v7/toys_cache_example.yaml --toyPoint nominal --seeds 0-4
    --outputDir ...`. Metadata round-trips correctly (seed, source, point, 36
    thrown constraint centers). Yield closure: mean drawn yield over 5 toys =
    234660.8 vs expected `sum(clip(w0,0,None))` = 234907.5, diff = 1.14 sigma
    (Poisson error) -- consistent.
  - `--toyFile` end-to-end: `load_toy` + `setToy` + `n2ll(hyp)` all confirmed
    working via a lightweight direct check (no minimizer): finite `-2lnL` at
    the nominal hypothesis (35.78, nonzero as expected for a real thrown
    fluctuation, unlike the trivial Asimov self-fit), rises sensibly away from
    it (1239 at `c0=0.3`, 44.3 at `nu_pu=0.5`), and repeat-evaluation at the
    same hypothesis is bit-identical (no hidden state mutation across
    `setToy` calls).
  - The **actual fit** (`python fit/Likelihood.py ... --toyFile
    nominal_toy0.h5 --overwrite fit`) was launched in the background
    (2018 toy) and ran for ~3 hours burning 1625% CPU with zero visible log
    output (Python fully block-buffers stdout when redirected to a file, so
    `prepare_runtime()`'s prints were sitting in an unflushed buffer the whole
    time -- not evidence it was stuck). **Killed manually** once the 2016 run
    below revealed why it was never going to finish cleanly. See "NaN at first
    optimizer step" below.
  - **NaN at first optimizer step -- real finding, pre-existing code, not a
    toy-generation bug.** Reran the same kind of fit against a fresh 2016 toy
    (`configs/unbinned_v7/unbinned_2016.yaml`, same spec) with `--verbosity 2`
    and `PYTHONUNBUFFERED=1` for visibility. Two bugs surfaced:
    1. `ModelParameter.__repr__` (`fit/Modeling.py:53`) crashed with
       `TypeError: unsupported format string passed to ArrayBox.__format__`
       -- `f"{self.val:.6e}"` fails when `self.val` is an autograd `ArrayBox`
       (during gradient tracing) rather than a plain float. Pre-existing,
       apparently never hit before since `--verbosity 2` (the only level that
       calls `Hypothesis.print()` from inside `grad(fcn)`) had never been used
       against a real fit. **Fixed and committed (`48c81f0`)**: wrap in
       `float(getval(self.val))` before formatting.
    2. After that fix, the fit reran and got further: eval 1 (`f=36.02`, pure
       penalty term -- `T` is identically 0 at `x=0` for every event by
       construction, so `total_unbinned(x=0)=0` exactly and the whole value
       comes from the 36 thrown constraint centers' `sum(offset^2)`), eval 2
       (autograd's internal forward retrace at the same `x=0`, same value,
       confirms it), **eval 3 = `f=nan`, crashing with `RuntimeError: NaN
       likelihood!`**. Diagnosis: eval 3 is L-BFGS-B's first real step (not a
       per-parameter probe -- autograd computes the whole 42-dim gradient in
       one reverse-mode pass), and that step pushes `T < -1` for at least one
       event, making `log1p(T)` NaN. Likely driven by a large gradient
       component from an outlier event (plausibly among the negative-weight /
       extreme-kinematics ones found earlier) combined with L-BFGS-B taking an
       unscaled first step from `x=0` (the `steps` dict `run_autograd_fit`
       computes is never actually passed to `scipy.optimize.minimize` --
       looks like dead code). **This is not toy-generation-specific**: `T(x=0)`
       is identically 0 for *any* observed dataset regardless of content
       (real data included), so this fragility is inherent to
       `run_autograd_fit`'s unscaled first step from the origin, just never
       exercised before because the only fits run so far were nominal-Asimov
       self-fits that never leave `x=0` (`niter=0` in verification 1).
    - Per your instruction: **not fixing this now** (2026-07-28) -- logging it
      here as a known blocker for verification 12 (refit-converges), out of
      scope for the toy-generation plan itself since it lives in
      `run_autograd_fit`'s optimizer setup, not in anything this branch
      touched. The 2018 background fit was killed rather than left running,
      since it's the same code path against the same kind of toy and was very
      likely going to hit the identical wall.
    - To pick this up: the fix is probably either (a) guard `log1p`/clip `T`
      at `-1+eps` before the log (cheap, but masks the real issue), or (b)
      actually use the computed `steps` dict to scale the initial L-BFGS-B
      step (`options={"eps": ...}` or a proper trust-region method), or
      (c) identify and understand the specific outlier event(s) driving the
      huge gradient component first. Reproduce with:
      `PYTHONUNBUFFERED=1 python -u fit/Likelihood.py
      configs/unbinned_v7/unbinned_2016.yaml --toyFile
      /users/ricardo.barrue/.claude/jobs/cb7ec444/tmp/toys_cache_2016/nominal_toy0.h5
      --overwrite fit --verbosity 2` (toy file may need regenerating if the
      job tmp dir was cleaned).
    - **(2026-07-29) Observation from you**: this NaN does *not* happen when
      fitting the same toy with `run_iminuit_fit` (Minuit/MIGRAD) -- only
      `run_autograd_fit` (L-BFGS-B) hits it. This is consistent with the
      diagnosis above and narrows it further: MIGRAD's first step is scaled
      by its own internal error/step estimates rather than taking an
      unscaled step from `x=0`, so it doesn't overshoot into `T < -1`.
      Strengthens candidate fix (b) (use the computed `steps` dict to scale
      L-BFGS-B's first step) over (a) (clip/guard, masks the issue) as the
      more likely real fix, though (c) (identify the outlier event driving
      the huge gradient component) is still worth doing first to confirm.
  - Remaining: verification 12 (refit-converges) blocked on the above; truth-
    mode generation (steps 3-4 of the suggested order) still untested against
    real data.
  - 2026-07-29: generation of toy in truth mode tested for c0 = 0.5.
    - added examples for truth-mode toy generation from an alternative sample
      or additional weights (e.g. QCD scales). The latter required adding the
      additional weight branches to the list of observers in the toy generator.
    - Minuit fits with truth-mode toys are still crashing. Will debug later.
  - 2026-07-30:
    - checking/building cache for truth-mode runs to avoid crash when plotting
      due to missing nominal cache weights


