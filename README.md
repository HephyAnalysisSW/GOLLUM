# Learning the gluon PDF with SBI

## Setup

Clone the working branch and enter the repository:

```bash
git clone --branch devel-sbi-rs-hepgpu2-merged --single-branch git@github.com:HephyAnalysisSW/GOLLUM.git
cd GOLLUM
```

On CBE, activate the existing environment (verified with Python 3.10.18):

```bash
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate hephy-ml-gpu-claude
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Run scripts from the repository root. Configure model, cache, output, and plot
paths for your username in `common/user.py`, and sample paths in
`common/directories.py`. Trained models, data, caches, and rotation files are
external to Git and must be available at those paths.

`env/environment.yaml` is a starting point for a new environment; it does not
yet fully reproduce the verified CBE installation.

## Paper command record

[docs/archive/paper-workflow.sh](docs/archive/paper-workflow.sh) preserves the
original paper commands verbatim, including their historical paths.
It is a reference notebook: running the whole file launches multiple analyses
and may overwrite results.
