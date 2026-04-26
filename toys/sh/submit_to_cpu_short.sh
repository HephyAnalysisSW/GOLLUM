#!/bin/bash 

# Usage: sbatch submit_to_cpu_short.sh "your_command_here"


#SBATCH --job-name=sbi-pdf-toy-fits
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --nodes=1-1 
#SBATCH --partition=c 
#SBATCH --qos=short
#SBATCH --time=08:00:00 
echo ----------------------------------------------- 
echo "COMMAND: $1"
$1