# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Domain context

GOLLUM is a framework for Neural Simulation-Based Inference (NSBI) analyses, based on arxiv:2205.12976, arxiv:2406.19076, arxiv:2505.05544. It trains ML surrogates for differential cross-section ratios, POI and systematic variations continuously in phase space (unbinned), then performs likelihood fits to extract PDF parameters. A traditional binned statistical model serves as cross-check. 

Goal of this analysis: use GOLLUM to extract the gluon PDF in dileptonic ttbar, as done in arxiv:2604.13157.

Physics process: dileptonic tt̄ events from CMS Run II (2016APV, 2016, 2017, 2018), and other small backgrounds: Drell-Yan (Z+jets), semileptonic tt events and single top events.

## Environment setup

```bash
conda activate /groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-2
```

Key stack: Python 3.10, TensorFlow, ROOT/PyROOT, NumPy, SciPy, iminuit, autograd, mplhep, uproot. See `env/environment.yaml` for full dependency list.

## New user setup

Add yourself to `common/user.py` with your `plot_directory`, `model_directory`, `cache_directory`, and `output_directory`. The code hard-crashes if your username is not registered — this is intentional.

## Running training jobs

All training scripts are run from the repo root and take a YAML config path as first argument plus `--job <job_id>`:

```bash
# List all trainable jobs in a config (run without --job)
python ML/TFMC/tfmc_training.py configs/unbinned_v6/unbinned_2018.yaml
python ML/PNN/pnn_training.py configs/unbinned_v6/unbinned_2018.yaml

# Train a specific job
python ML/TFMC/tfmc_training.py configs/unbinned_v6/unbinned_2018.yaml --job <job_id>
python ML/PNN/pnn_training.py   configs/unbinned_v6/unbinned_2018.yaml --job <job_id>
python ML/BIT/pdf_bit_training.py configs/unbinned_v6/unbinned_2018.yaml --job <job_id>
python ML/Scaler/scaler_training.py configs/unbinned_v6/unbinned_2018.yaml --job <job_id>
python ML/IC/ic_training.py  configs/... --job <job_id>
python ML/ICP/icp_training.py configs/... --job <job_id>

# Closure test (PNN)
python ML/PNN/pnn_training_closure_mpl.py configs/unbinned_v6/unbinned_2018.yaml --job <job_id>
python ML/TFMC/tfmc_training_closure_mpl.py configs/... --job <job_id>
```

Common flags: `--overwrite` (rewrite model dir), `--small` (debug with first shard only), `--epochs N`.

## Checking which surrogates are trained / missing

```bash
python common/yaml_loader.py configs/unbinned_v6/unbinned_2018.yaml
```

Prints a summary of loaded vs. missing artifacts and the exact training commands for the missing ones.

## Running fits (example)

```bash
# Unbinned likelihood fit (from repo root)
python fit/Likelihood.py configs/unbinned_v6/unbinned_2018.yaml

# Binned fit (see binned config)
bash fit/Likelihood.py configs/binned_v6/binned_2018.yaml

# POD basis 
python orth/orthogonalize_Fisher.py configs/... [args]
```

Fit outputs go to `user.output_directory` (user-specific path from `common/user.py`).

## YAML config system

Configs use a custom include mechanism in `common/yaml_loader.py`. A key can be set to `{include: path/to/other.yaml}` to inline another file's content for that key. This allows shared systematics blocks across eras.

Config structure:
- `version`: string identifying the model directory subfolder
- `defaults`: `module_samples`, `default_features`, `splitting`, `early_stopping`
- `jobs`: list of training jobs with `id`, `type`, `region`, `process`, `features`, `extras`
- `likelihood.regions` (unbinned) or `likelihood.binned`: describe the statistical model

Job types: `scaler`, `ic`, `ich`, `icp`, `icph`, `pnn`, `classifier` (with `framework: tfmc`), `bit`.

Job dependency order: scaler → IC/ICP → TFMC/PNN → likelihood. The `extras.use_scaler`, `extras.use_ic`, `extras.use_icp` fields wire dependencies between jobs.

Multiple per-era configs can be merged with `combine_configs()` for a combined Run II fit.

## Directory structure

Persistent paths (symlinked at repo root):
- `models/` → `user.model_directory` — saved ML artifacts, organized as `<version>/<region>/<ModelType>/<job_id>/`
- `output/` → `user.output_directory` — fit results, scan pickle files
- `caches/` → `user.cache_directory`
- `www/` → `user.plot_directory` — plots (synced to CERN EOS on script exit via `common/syncer.py`)

## ML modules

| Module | Role |
|---|---|
| `ML/TFMC/` | TensorFlow multiclass classifier for process discrimination: calculates relative contribution of each class to the differential cross-section |
| `ML/PNN/` | Parametric NN for systematic variation surrogates: learns differential cross-section ratio (DCR) as function of nuisance parameter |
| `ML/BIT/` | Boosted Information Trees: learns POI-dependent variations continuously in input variable phase space via event weights |
| `ML/Scaler/` | Feature mean/variance scaler, shared across TFMC/PNN |
| `ML/IC/` | Inclusive event yield - DEPRECATED/NOT USED | 
| `ML/ICH/` | Binned event yield parametrization as function of POI  |
| `ML/ICP/` | Inclusive yield parametrization as a function of nuisance parameters |
| `ML/ICPH/` | Binned event yield parametrization as function of nuisance parameters |
| `ML/Calibration/` | Post-hoc calibration checks for BIT predictions |
| `ML/c2st/` | Classifier two-sample test (C2ST) for high-dimension closure check of PNNs |

Each module has a model class (`TFMC`, `PNN`, `BIT`, etc.) with `save()`/`load()` methods, and a separate training script. The model class itself has no I/O or plotting logic.

## Fit infrastructure

- `fit/Likelihood.py` — `load_likelihood(cfg)` wires YAML config + loaded surrogates into a callable `−2ΔlnL` function. Supports both unbinned (NSBI) and binned regions simultaneously.
- `fit/Modeling.py` — `ModelParameter`, `Hypothesis`, `Rotated`: lightweight parameter containers for the minimizer interface (autograd-compatible).
- `fit/MultiDimFit.py` — POI scans à la CMS Combine, Slurm-parallelizable via `--pointRange`.
- `fit/N2LLExtensions.py` — additional likelihood terms and extensions.

## PDF parametrizations

`pdf/PDFParametrization.py` is the factory entry point; supports three backends:
- `Chebyshev` / `Bernstein` via `pdf/AnalyticPDFParametrization.py`
- `PODBasis` via `pdf/PODBasis.py`: PDF basis derived from procedure described in paper. This basis is not derived with the GOLLUM framework, it is provided by our NNPDF collaborations.

## Data

- `data/samples.py` / `data/samples_RunII.py` — define `RDataLoader` instances for each era and systematic variation; imported dynamically via `defaults.module_samples` in YAML configs.
- `data/observables.py` — feature name lists (`TOP_KINEMATICS`, `LEPTON_KINEMATICS`, `ASYMMETRY`, `ALL_FEATURES`) referenced in YAML feature tokens.
- `data/RDataLoader.py` — ROOT TTree loader with weight branch product, `clone_from_files()` for systematic variants.
- `data/UIDSplitter.py` — deterministic train/val/test splitting on event UIDs (run, luminosityBlock, event).

## Plotting and syncing

When saving plots to the `www/` directory, always:
```python
import common.syncer as syncer
# Use plt.savefig (not fig.savefig) — syncer wraps plt.savefig
plt.savefig(os.path.join(plot_dir, "figure.png"))
```
`syncer` registers an `atexit` hook that uploads files under `www/` to CERN EOS via `xrdcp`. Requires `$CERN_USER` in the environment.

## Code style

- Type hints where they don't bulk the code; spaces for indentation (no tabs).
- `logger` (from `logging`) instead of `print`. Exception: list_jobs_and_exit (for piping directly to shell script for direct execution).
- Prefer `matplotlib`/`mplhep` over ROOT for new plots. Only touch ROOT plotting code in existing scripts.
- Import reusable functions from other files; if not importing is a deliberate choice, say so explicitly.
- Prefer controlled crashes over `try/except` silencing — misconfigurations should surface immediately. Only catch exceptions at genuine external boundaries (optional imports, network calls).
- Docstrings on all new files and functions (brief, not multi-paragraph).
- Don't add backward-compatibility shims when refactoring — change all call sites directly.
- Don't abstract logic that appears only once. Three similar instances warrant a helper; one or two do not.
- Don't use shorthand naming for variables that are not auxiliary variables. Be verbose but no more than three words.
- Use ascii characters only, and e.g. write lambda explicitly instead of the lambda character.
- Don't give a random name to plan files. The plan file name should be a shorthand for what is being implemented.

## Non-obvious design decisions

These look wrong to a fresh reader but are correct for this codebase — don't "fix" them:

- **Custom YAML includes**: the `{include: path}` syntax in configs is implemented in `common/yaml_loader.py`, not standard PyYAML. Don't replace with YAML anchors (`&`, `*`) — they don't support cross-file includes.
- **`common/syncer.py` monkey-patching**: importing `syncer` is the entire effect. It hooks into `plt.savefig` and ROOT's `TCanvas.Print` at import time via monkey-patching, and registers an `atexit` sync to CERN EOS. No explicit `syncer.sync()` call is needed.
- **`sys.path.insert(0, '..')`**: all scripts use this pattern because there is no package installation. Scripts must always be run from the repo root (`python ML/PNN/pnn_training.py configs/...`), never from within subdirectories.
- **Hard crash on unknown user**: `common/user.py` raises `RuntimeError` if `$USER` is not registered. This is intentional — a new collaborator must configure their own paths before using the framework.

## Known pitfalls

- **ICH/ICPH binning must exactly match the saved artifact**: `load_surrogates()` checks `axis_names` and bin edges against the YAML config and raises on mismatch. Axis order matters. When retraining an ICH/ICPH, regenerate both the YAML binning and the artifact together.
- **PNN and Scaler must share the same feature list**: enforced in `yaml_loader._apply_defaults_and_checks()`. If a PNN job references a scaler via `extras.use_scaler`, both must have identical `features` (easiest to achieve by relying on `defaults.default_features` for both).
- **`combine_configs()` requires identical `version` and `defaults` across all merged configs**: the surrogate loader uses `version` to locate model directories; a mismatch raises immediately.
- **Region IDs and job IDs must be globally unique** within a merged config. Use era-suffixed IDs everywhere (e.g. `SR_2018`, `pnn_TTLep_pow_2018_PU`).

## Physics and statistical context

- **POI space vs. x-space**: the POIs (`c0`, `c1`, ...) are coefficients of the PDF basis functions, not directly physical. Mapping to x-space requires applying the parametrization. Don't conflate POI scans with PDF scans.
- **POD basis ordering**: the rotated POD basis eigenvector indices are ordered by expected sensitivity (most sensitive first). Truncation decisions should be framed in terms of expected sensitivity (proportional to eigenvalue).
- **Asimov vs. data fits**: `classifier.asimov` in the likelihood config sets which sample is used as pseudo-data in Asimov mode (`TTLep_pow_<era>`). Real data fits use `Data_<era>`. The Asimov sample should always match the signal class sample.