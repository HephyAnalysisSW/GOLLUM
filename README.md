# Learning the gluon PDF with SBI

## Run the paper's binned “8-1” fit on CBE

Log in with `ssh YOUR_CBE_LOGIN@cbe.vbc.ac.at`. You need GitHub repository access and read access to
Robert's shared Conda environment and artifacts below. This recipe uses the
existing trained surrogates; it does not retrain them or require raw ntuples.

### 1. Clone and activate the environment

```bash
git clone --branch sbi-pdf-paper --single-branch git@github.com:HephyAnalysisSW/GOLLUM.git
cd GOLLUM
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate /groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-claude
export PYTHONPATH="$PWD"
unset CERN_USER  # keep plots local; disable automatic CERN upload
```

Run all remaining commands from this directory. The verified environment uses
Python 3.10.18; `env/environment.yaml` does not yet fully reproduce it.

### 2. Copy the external fit inputs

```bash
export GOLLUM_RUN_DIR="$HOME/gollum-8-1"
shared=/groups/hephy/cms/robert.schoefbeck/SBIPDF/models/binned_delphes_RunII_v5D/SR_binned
models="$GOLLUM_RUN_DIR/models/binned_delphes_RunII_v5D/SR_binned"
mkdir -p "$models/ICH" "$models/ICPH" "$GOLLUM_RUN_DIR/output" \
         "$GOLLUM_RUN_DIR/caches" "$GOLLUM_RUN_DIR/plots"
cp "$shared/ICH/ICH_ich_NG_PDF4LHC21_8_tt2l_delphes.pkl" "$models/ICH/"
cp "$shared"/ICPH/ICPH_icph_TTLep_pow_2016_*.pkl "$models/ICPH/"
cp /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_binned_delphes_8_RunII_binned_delphes_RunII_v5D.json \
   "$GOLLUM_RUN_DIR/output/"
```

These external inputs are required: one ICH surrogate, the ICPH surrogates
(23 used by this fit), and the rotation JSON. No pre-existing likelihood cache is needed. The shared
locations above must remain accessible until you have copied the inputs.

### 3. Configure your local paths

Replace `common/user.py` in your clone with the following configuration
(the change is local; do not commit your personal setup):

```bash
cat > common/user.py <<'PY'
import os
from pathlib import Path

user = os.environ.get("USER", "user")
run = Path(os.environ["GOLLUM_RUN_DIR"]).expanduser()
model_directory = str(run / "models")
cache_directory = str(run / "caches")
output_directory = str(run / "output")
plot_directory = str(run / "plots")
training_data_dir = str(run / "data")  # unused for this pretrained binned fit
tmp_mem_directory = "/dev/shm/" + user + "/"
PY
```

In a new shell, repeat environment activation, `PYTHONPATH`, `unset CERN_USER`,
and `export GOLLUM_RUN_DIR="$HOME/gollum-8-1"` before running.

### 4. Fit

```bash
python fit/Likelihood.py configs/binned_v5D/binned_delphes_8_RunII.yaml \
  --rotate "$GOLLUM_RUN_DIR/output/eigen_basis_binned_delphes_8_RunII_binned_delphes_RunII_v5D.json" \
  --minos --freezePOI 0.001
```

Startup should report `24 ready artifact(s), 0 missing` and freeze `eig_7`
(eigenvalue about `9.04e-5`), leaving seven POIs floating.
The fit runs MIGRAD, HESSE, and MINOS. Its result is
`$GOLLUM_RUN_DIR/output/binned_delphes_8_RunII_binned_delphes_RunII_v5D_rotate_freezePOI0.001_fit.json`;
logs and diagnostic plots are under `$GOLLUM_RUN_DIR/plots/likelihood_fit/`.
An existing result is loaded instead of refitted; add `--overwrite fit` to rerun.

Verified on CBE in an independent clone with a private home directory and
user configuration: all 24 surrogates loaded, MIGRAD/HESSE/MINOS completed,
and the result JSON and diagnostic plots were written (2026-10-20).

## Paper command record

[docs/archive/paper-workflow.sh](docs/archive/paper-workflow.sh) preserves the
original paper commands verbatim, including their historical paths.
It is a reference notebook; running the whole file launches multiple analyses
and may overwrite results.
