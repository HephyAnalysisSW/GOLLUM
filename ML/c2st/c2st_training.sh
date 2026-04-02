#!/usr/bin/env bash

#SBATCH --job-name=traina
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=0-00:39:00
#SBATCH --qos=short
#SBATCH --output=traina.stdout

echo "Job started at: $(date)"

python -u dnn_training.py configs/unbinned/unbinned_2017.yaml --job c2st_test1_TTLep_pow_2017_CMS_res_j_0_2017 --n_split 5 --train_seed 43                                                                

echo "Job finished at: $(date)"