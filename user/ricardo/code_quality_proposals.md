# Code Quality Proposals
_Surveyed 2026-06-23. Branch: dev-rbarrue_v8_updates._

---

## Fix soon

### 1. Bug: undefined `vk` in `PNN.predict_ratio()` — `ML/PNN/PNN.py:177`
`predict_ratio()` references `vk` which is not defined anywhere in scope. Raises `NameError` if called.
Not currently wired into the likelihood path, but looks like a complete method. Either delete or fix before it gets used.

### 2. Silent exception swallowing in `common/yaml_loader.py:535-592`
All 8 `try_load_*` functions catch bare `Exception`, meaning file corruption, pickle version mismatches, and import errors are silently eaten and returned as `None`. The surrogate then appears "missing" rather than "broken."

Fix: catch only `FileNotFoundError`, let everything else propagate.
```python
def try_load_scaler(path):
    try:
        from ML.Scaler.Scaler import Scaler
        return Scaler.load(path)
    except FileNotFoundError:
        return None
```
One-line change per function. High diagnostic value, low diff cost.

---

## Cleanup pass (not urgent)

### 3. Mixed `print()` / `logger` in `fit/Likelihood.py`
~20 `print()` calls scattered through the file, mixed with `logger.warning/info()` calls. Debug output can't be controlled via logging config. Mechanical find-and-replace but the file is high-stakes.

### 4. Arg parsing duplicated across all training scripts
`list_jobs_and_exit()` / `list_and_exit()` reimplemented in every `ML/*/`*`_training.py` with minor flag variations. Factor into `common/training_utils.py` when adding v8 training scripts — doing it now would mean touching every existing script for no immediate gain.

---

## Defer until v8 infrastructure work

### 5. Model interface inconsistency
`__init__`, `predict`, `save()`/`load()` signatures differ across TFMC/PNN/BIT/IC/ICH/ICP. `Likelihood.py` must know the concrete type it's calling. A minimal shared protocol for `save(path)` / `load(path)` would help. Tackle when adding new surrogate types.

### 6. Hardcoded data paths in `data/samples.py` and `data/samples_RunII.py`
Paths like `/scratch-cbe/users/robert.schoefbeck/...` are hardcoded. `samples_eft.py` already uses `BASE_DIRECTORY` correctly — standardise the others before new collaborators start running. Framework crashes on unknown users (`user.py`) but silently breaks on missing data paths.

---

## Leave alone

- `user.py` directory config per-user (grows linearly, not worth abstracting yet)
- `try_load_*` structural duplication itself (after fixing exception handling above)
- Selection merge type inconsistency in `yaml_loader.py` (real but low impact)
