#!/usr/bin/env bash
# Create unbinned YAML configs for a given ERA (N=1..9).
#
# Motivation behind choices:
# - Keep it simple and dependency-free: plain Bash + GNU sed.
# - Safety first: do not overwrite existing files unless -f is provided.
# - Correctness: replace the occurrences of "bit_pod_TTLep_pow_<ERA>" 
#   both in POI and job definition in each generated file.
#
#   Replaces the template job text in both the POI and job definition.
#   for the one with the correct N
# 
# Input:
# - Source: unbinned_<ERA>_original.yaml
# - Output dir: <ERA>/ (must already exist)
#
# Output files:
# - <ERA>/unbinned_<ERA>_1.yaml ... <ERA>/unbinned_<ERA>_9.yaml
#
# Usage:
#   ./make_configs_automated.sh [-f] ERA
#   -f   overwrite existing output files

set -euo pipefail

force_overwrite=false

usage() {
  # Print command usage.
  echo "Usage: $(basename "$0") [-f] ERA"
  echo "  -f    Overwrite existing output files"
}

while getopts ":fh" opt; do
  case "$opt" in
    f) force_overwrite=true ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Error: invalid option -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

era="${1:-}"
if [[ -z "$era" ]]; then
  echo "Error: ERA is required." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${script_dir}/unbinned_${era}_original.yaml"
output_dir="${script_dir}/${era}"

if [[ ! -f "$source_file" ]]; then
  echo "Error: source file not found: $source_file" >&2
  exit 1
fi

if [[ ! -d "$output_dir" ]]; then
  echo "Error: output directory not found: $output_dir" >&2
  exit 1
fi

old_job="bit_pod_TTLep_pow_${era}"
old_parameters="\[c0, c1, c2, c3, c4, c5\]"
template_job="bit_NG_PDF4LHC21_1_TTLep_pow_${era}"
template_job_n="\[1\]"
template_job_file="BIT_NG_PDF4LHC21_1_TTLep_pow_${era}.pkl"


for n in {1..9}; do
  new_job="bit_NG_PDF4LHC21_${n}_TTLep_pow_${era}"
  new_bit_file="BIT_NG_PDF4LHC21_${n}_TTLep_pow_${era}.pkl"
  output_file="${output_dir}/unbinned_${era}_${n}.yaml"

  if [[ -e "$output_file" && "$force_overwrite" != true ]]; then
    echo "Skipping existing file: $output_file"
    continue
  fi

  parameter_list="["
  for ((i=0; i<n; i++)); do
    if (( i > 0 )); then
      parameter_list+=", "
    fi
    parameter_list+="c${i}"
  done
  parameter_list+="]"

  new_job_n_list="["
  for ((i=1; i<=n; i++)); do
    if (( i > 1 )); then
      new_job_n_list+=","
    fi
    new_job_n_list+="${i}"
  done
  new_job_n_list+="]"

  sed \
    -e "0,/${old_job}/s//${new_job}/" \
    -e "0,/${old_parameters}/s//${parameter_list}/" \
    -e "0,/${template_job}/s//${new_job}/" \
    -e "0,/${template_job_n}/s//${new_job_n_list}/" \
    -e "0,/${template_job_file}/s//${new_bit_file}/" \
    "$source_file" > "$output_file"

  echo "Created: $output_file"
done