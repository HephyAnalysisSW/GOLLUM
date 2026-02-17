#!/bin/bash 

# Usage: sbatch submit_to_cpu.sh "python3 autoplotter.py  --sample wjetstolnuht0100_2018  --output /scratch-cbe/users/alikaan.gueven/2018_limits --config configs/calc_limits.yaml --lumi 59800 --json CustomNanoAOD_v3_bkg.json --datalabel CustomNanoAOD"


#SBATCH --job-name=sbi-pdf-toy-fits
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3500M 
#SBATCH --nodes=1-1 
#SBATCH --partition=c 
#SBATCH --qos=rapid
#SBATCH --time=01:00:00 
echo ----------------------------------------------- 
echo "COMMAND: $1"
$1