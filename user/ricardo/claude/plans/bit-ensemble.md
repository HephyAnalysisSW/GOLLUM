# Training and using an ensemble of BITs

## Original context and idea (Ricardo)

### Context

Tree-based algorithms are designed to be robust against fluctuations in a dataset, although this is not 100% effective.

In order to avoid our BITs picking up on specific fluctuations living in the tails of kinematic distributions and leading to biases in our toy studies, I want to implement the usage of BIT ensembles in our framework.

### Implementation

I want to shuffle training and validation datasets with a random seed for each of the BITs in the ensemble.

To wire this to the current framework, there are three parts:
1. config picking up that we're using an ensemble instead of a single BIT
2. training individual members of the ensemble
3. loading and evaluation of the ensemble

This is what I'm thinking in terms of implementation:

For 1., adding a parameter to the `bit` jobs called `n_ensemble` which, when set, tells the code that we are using an ensemble. This is then propagated downstream.

For 2. use `ML/BIT/eft_bit_training.py`, adding an additional argument `n_ensemble`. This argument defines which member of the ensemble we are training. `n_ensemble` is then used for two things: training and validation shuffling and storing uniquely individual members of the ensemble.

For 3. I need to expose `init()` or `load()` methods, such that they can be used in `fit/Likelihood.py` via `yaml_loader`. I also need to expose a `predict()` method that evaluates the individual ensemble and outputs a mean (possibly weighted).
For this, I was thinking of creating a simple class called `MultiBITEnsemble` (living inside `NumbaBIT.py`). In `yaml_loader`, I check if the job `n_ensemble` is set, and if so, I use the ensemble load method instead of the individual BIT load method.

---

# Plan

## Context

BIT trees are deterministic given their training data: the only randomness in the pipeline is the UID train/valid split (`data/UIDSplitter.py`, seed from `defaults.splitting.seed`). A single BIT can therefore lock onto tail fluctuations of one particular split, which shows up as bias in the toy studies. Averaging several BITs trained on different train/valid partitions of the same events suppresses that split-specific component while leaving the physics signal intact.

The working tree already contains a first sketch (`MultiBITEnsemble` in `ML/BIT/NumbaBIT.py`, `n_ensemble` handling in `common/yaml_loader.py`). It does not work yet and there is no training-side support. This plan replaces the sketch.

Decisions taken (see the questions answered in the planning session):
- Member-specific reshuffling happens **inside the union of the `pnn_train` and `pnn_val` buckets only**, so `c2st_*` and `final_eval` stay exactly as they are for every member and never leak into any member's training set.
- Members live in **per-member subdirectories** (`.../BIT/<job_id>/ensemble_<i>/`), which covers all six per-training artifacts with one change.
- Combination is an **unweighted mean** over members.
- Only `ML/BIT/eft_bit_training.py` gains the option; `pdf_bit_training.py` stays single-BIT.

## 1. Config

Add a top-level key to the `bit` job (not under `extras` -- BIT jobs carry no `extras`):

```yaml
  - id: bit_TT01j2l_EFT_2016_ctGRe
    type: bit
    n_ensemble: 6
    ...
```

Absent or unset means single BIT, i.e. today's behaviour, unchanged. No change needed in `_apply_defaults_and_checks`.

## 2. Training: `ML/BIT/eft_bit_training.py`

**New CLI arg** (next to the existing ones around line 24-34):

```python
p.add_argument("--i_ensemble", type=int, default=None, help="Which ensemble member to train (requires n_ensemble in the job).")
```

Right after the job `J` is resolved (~line 69), validate and fail loudly:
- `--i_ensemble` given but `J` has no `n_ensemble` -> `RuntimeError`.
- `J['n_ensemble']` set but `--i_ensemble` not given -> `RuntimeError` naming the required flag.
- `not 0 <= args.i_ensemble < J['n_ensemble']` -> `RuntimeError`.
- `args.i_ensemble is not None and split_type != 'uid'` -> `RuntimeError`. (A `random` split produces no validation set, so `BIT_best.pkl` -- what the ensemble loads -- cannot be selected.)

**Member-specific train/valid split.** Only the `uid` branch of `iterate_all()` (~lines 222-243) changes. Today it computes `m_tr` and `m_va` from the two global bucket intervals. For a member, keep those two masks as the *pool* definition and re-partition the pool with a member-dependent hash:

```python
# built once, next to `splitter` (~line 145)
member_splitter = None
member_val_hi = None
if args.i_ensemble is not None:
    member_splitter = UIDSplitter(uid_fields=tuple(uid_fields),
                                  seed=split_seed + 1000 * (args.i_ensemble + 1),
                                  n_buckets=uid_n_buckets)
    n_train_buckets = train_interval[1] - train_interval[0]
    n_val_buckets   = val_interval[1] - val_interval[0]
    member_val_hi   = int(round(uid_n_buckets * n_val_buckets / (n_train_buckets + n_val_buckets)))
```

and inside the shard loop, after the two global masks are computed:

```python
if member_splitter is not None:
    pool = m_tr | m_va
    m_va = pool & member_splitter.mask_from_np(O_uid, list(uid_fields), 0, member_val_hi)
    m_tr = pool & ~m_va
```

This preserves the global train+valid pool event-for-event, keeps the 5:1 ratio (0.50/0.10 -> `member_val_hi = 1667`), and gives every member an independent partition of it. Nothing else in the materialization code needs to change.

**Per-member output directories.** In the "build & train BIT" block (~lines 531-546), append the member subdirectory to both `model_dir` and `plot_dir` before `os.makedirs`:

```python
model_dir = os.path.join(user.model_directory, cfg_base, "BIT", J["id"])
plot_dir  = os.path.join(user.plot_directory, "BIT", cfg_base, J["id"])
if args.i_ensemble is not None:
    member = f"ensemble_{args.i_ensemble}"
    model_dir = os.path.join(model_dir, member)
    plot_dir  = os.path.join(plot_dir, member)
```

`model_path`, `weights_path`, `loss_history.txt`, `loss_history_all_terms.txt`, `BIT_best.pkl`, `BIT_best.weights.pkl`, `best_checkpoint.txt` and the `done` marker are all derived from `model_dir`, so they become per-member with no further edits. Resume, `--small`, `--max_n_files` and the best-checkpoint logic keep working per member. `_build_plot_context` builds its own `out_dir` from `user.plot_directory` (~line 338) -- give it the same member suffix so training plots do not collide.

**`list_and_exit()`** (~line 52): for a job with `n_ensemble`, emit one command line per member with `--i_ensemble <i>` instead of a single line.

## 3. Loading and evaluation

### `MultiBITEnsemble` in `ML/BIT/NumbaBIT.py`

Rewrite the sketch (lines 19-38). It must expose the attributes the consumers read, and validate that the members are mutually consistent instead of silently averaging incompatible models:

```python
class MultiBITEnsemble:
    """Evaluation-only container for an ensemble of BITs trained on different train/valid splits."""

    def __init__(self, paths):
        self.members = [MultiBoostedInformationTree.load(path) for path in paths]
        first = self.members[0]
        self.derivatives = first.derivatives
        for member, path in zip(self.members[1:], paths[1:]):
            if member.derivatives != self.derivatives:
                raise RuntimeError(f"Ensemble member {path} has derivatives {member.derivatives}, expected {self.derivatives}.")
        self.expansion_point = getattr(first, "expansion_point", None)
        # same check for expansion_point when it is not None
        self.binned_calibration = None

    @property
    def n_trees_trained(self) -> int:
        """Number of trees usable across all members (members can stop at different best trees)."""
        return min(len(member.trees) for member in self.members)

    def predict(self, feature_array, max_n_tree=None, summed=True, last_tree_counts_full=False):
        predictions = np.stack([member.predict(feature_array, max_n_tree, summed, last_tree_counts_full)
                                for member in self.members])
        return np.mean(predictions, axis=0)
```

Fixes relative to the sketch: list instead of generator into the stack, mean over the member axis (`axis=0`, not `axis=2`), and the attributes `fit/Likelihood.py` needs.

Add the matching property to `MultiBoostedInformationTree` so the two types are interchangeable:

```python
    @property
    def n_trees_trained(self) -> int:
        return len(self.trees)
```

`np` and `os` are already imported; drop the commented-out `load` classmethod.

### `fit/Likelihood.py`

One line in `predict_bit_ratio` (lines 546-547): `len(model.trees)` -> `model.n_trees_trained` (both in the check and in the message). Nothing else changes -- `model.predict(X, max_n_tree)`, `model.derivatives`, `model.binned_calibration` and `model.feature_names` all work on the ensemble. `Likelihood.py:124-134` (the `expansion_point` check for EFT BITs) is satisfied by the attribute above.

### `common/yaml_loader.py`

Replace the sketch in the `jtyp == "bit"` branch (lines 745-774):

- Keep the existing `fname` selection (`BIT_best.pkl`, or `job.output.filename` when `use_last`).
- Ensemble path construction uses the subdirectory, so the `removesuffix(".pkl")` munging (and its dropped-suffix bug) disappears:
  ```python
  paths = [os.path.join(outdir, f"ensemble_{i}", fname) for i in range(n_ensemble)]
  ```
- Do **not** swallow exceptions. Delete `try_load_bit_ensemble` (line 623). Instead: if every path exists, construct `MultiBITEnsemble(paths)` directly (a corrupt pickle should crash, not turn into a `[MISS]`); if some are missing, report `[MISS]` and append one training command per missing member, with `--i_ensemble <i>`.
- Binned calibration: if `n_ensemble` is set **and** `job.runtime.binned_calib_factors` is set, raise a `RuntimeError` saying ensemble calibration is not implemented. Silently skipping it (what the sketch does at line 770) hides a configured calibration.

The `use_last` change already in the working tree is orthogonal to this plan; leave it as it is.

## Files touched

- `ML/BIT/NumbaBIT.py` -- rewrite `MultiBITEnsemble`, add `n_trees_trained` to `MultiBoostedInformationTree`.
- `ML/BIT/eft_bit_training.py` -- `--i_ensemble`, validation, member split, per-member `model_dir`/`plot_dir`, `list_and_exit`.
- `common/yaml_loader.py` -- ensemble path construction, no exception swallowing, calibration guard.
- `fit/Likelihood.py` -- `n_trees_trained` in `predict_bit_ratio`.
- one config for testing, e.g. `configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_ctGRe.yaml` (`n_ensemble: 2` on `bit_TT01j2l_EFT_2016_ctGRe`).

Commits: (1) `MultiBITEnsemble` + `n_trees_trained` + `Likelihood.py`, (2) training-side `--i_ensemble`, (3) `yaml_loader` wiring, (4) test config.

## Verification

Comment out the `common/syncer` import/calls in the training script while testing (EOS sync needs an interactive Kerberos token), and run with `--every 0` to skip ROOT training plots.

1. **Single-BIT behaviour is untouched.** Before any change, record `bit.predict(X)` for an existing trained BIT (e.g. `plot/bit/modification_plotter.py`'s model) on a fixed feature array, pickle it, and after the change assert bit-exact equality (`np.array_equal`). A mismatch is a hard stop.
2. **Loader reports missing members.** With `n_ensemble: 2` in the test config and nothing trained:
   `python common/yaml_loader.py configs/unbinned_v7_eft_genpoint/unbinned_2016_eft_genpoint_ctGRe.yaml`
   must print `[MISS]` plus two commands carrying `--i_ensemble 0` and `--i_ensemble 1`.
3. **Train two members small.**
   `python ML/BIT/eft_bit_training.py <config> --job bit_TT01j2l_EFT_2016_ctGRe --i_ensemble 0 --small --every 0` and the same with `--i_ensemble 1`.
   Check `models/unbinned_v7_eft_genpoint/SR_2016/BIT/bit_TT01j2l_EFT_2016_ctGRe/ensemble_{0,1}/` each contain their own `BIT.pkl`, `BIT_best.pkl`, weights, `loss_history.txt`, `best_checkpoint.txt`, and that omitting `--i_ensemble` raises.
4. **Split correctness** (small standalone script in the scratchpad, reusing `UIDSplitter` and the config's `splitting` block on one materialized shard):
   - member 0 and member 1 train masks differ (`not np.array_equal`);
   - for each member, `m_tr | m_va` equals the *global* `pnn_train | pnn_val` mask exactly -- this is the no-leakage assertion for `final_eval` and `c2st_*`;
   - `m_tr & m_va` is empty and `m_va.sum()/(m_tr.sum()+m_va.sum())` is within a percent of 1/6.
5. **Ensemble evaluation.** Load via `load_surrogates` and assert `ens.predict(X)` equals `0.5*(m0.predict(X) + m1.predict(X))` for the two members loaded by hand, and that `ens.derivatives == m0.derivatives` and `ens.n_trees_trained == min(len(m0.trees), len(m1.trees))`.
6. **End to end.** `python fit/Likelihood.py <config>` on the ensemble config, including a `max_n_tree` smaller than `n_trees_trained` in the class `POI` block, to exercise the `predict_bit_ratio` path.

## Verification results (2026-08-31)

Ran on `bit_TT01j2l_EFT_2016_ctGRe` (`n_ensemble: 2`) with `--small`, `--every 0`, and `common/syncer` temporarily commented out (restored afterward -- working tree is clean, `git status` empty).

| # | Check | Result |
|---|---|---|
| 1 | Single-BIT `predict()` bit-exact vs. pre-change `NumbaBIT.py` (commit `862969b`), with and without `max_n_tree` | **PASS** |
| 2 | `python common/yaml_loader.py <config>` reports `[MISS] BIT bit_TT01j2l_EFT_2016_ctGRe -> [ensemble_0/BIT_best.pkl, ensemble_1/BIT_best.pkl]` and lists both `eft_bit_training.py --i_ensemble 0/1` commands | **PASS** |
| 3 | Trained both members; each `ensemble_{0,1}/` directory has its own `BIT_best.pkl`, `BIT_best.weights.pkl`, `bit_..._small.pkl`/`.weights.pkl`, `loss_history.txt`, `best_checkpoint.txt`, `done` | **PASS** |
| 4 | Split correctness (standalone script): member masks differ; `m_tr \| m_va` == global pool exactly for both members (no leakage into `c2st_*`/`final_eval`); `m_tr & m_va` empty; valid fraction 0.1664/0.1665 vs. target 0.1667 | **PASS** |
| 5 | `load_surrogates` ensemble load: `ens.predict(X)` exactly equals `0.5*(m0.predict(X)+m1.predict(X))`, incl. with `max_n_tree`; `ens.derivatives == m0.derivatives`; `ens.n_trees_trained == min(...)` | **PASS** |
| 6 | `fit.Likelihood.predict_bit_ratio(ens, X, max_n_tree=...)`: correct shape at the bound, raises `ValueError` one past it | **PASS** |

Deviations from the plan's verification text, both harmless:
- Step 6 as written calls for running the full `fit/Likelihood.py` CLI. The test config has three unrelated pre-existing missing BIT jobs (`allWC`, `ML4EFTWC`, `nonML4EFTWC`, not part of this feature) that make `load_surrogates` fail before the fit runs. Ran `predict_bit_ratio` directly instead -- the exact function the fit calls -- against the loaded ensemble.
- The two trained members ended up with different `n_trees_trained` (37 vs. 22): each stops boosting at its own best-validation checkpoint, so this is expected, not a bug. `predict(max_n_tree=None)` correctly lets each member use its own full tree count (the "average of each member's best model" design); an explicit `max_n_tree` is bounded by the ensemble minimum via `n_trees_trained`, as intended.

No code changes were required as a result of verification. The two small test models remain on disk under `bit_TT01j2l_EFT_2016_ctGRe/ensemble_{0,1}/` in the model directory (valid `--small` debug artifacts, left in place).
