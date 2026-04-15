#!/usr/bin/env bash

#SBATCH --job-name=retrain1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-01:00:00
#SBATCH --qos=short
#SBATCH --output=retrain1.stdout

echo "Job started at: $(date)"

python -u Calibration.py  /users/daohan.wang/gluon-pdf-sbi3x/configs/unbinned_v5D/unbinned_delphes.yaml   --job bit_NG_PDF4LHC21_6_tt2l_delphes --max-files 20   --write_pandas

echo "Job finished at: $(date)"
