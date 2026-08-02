import argparse
import os
import numpy as np
import common.user as user

import logging
logging.basicConfig(level=logging.INFO)

"""
Script that takes a config and folder with toys as input and prepares n_jobs shell scripts to be executed on SLURM.

This can be piped with the output of the `find` command if there are many folders with toys.
Example:

find output_SBIEFT/ -maxdepth 1 -type d -name "unbinned_2016_eft*toys" -exec python user/ricardo/launch_eft_fits_toys.py -config configs/unbinned_v7_eft/unbinned_2016_eft.yaml -input_dir {} --n_jobs 4 --output_dir user/ricardo/toy_fit_submission_02082026/ \;

Each script can then be submitted with the `submit` command as `submit 'source SHELL_SCRIPT'` (notice the single quotes).

For submitting many folders at once, you can also pipe it with the output of `find`.
Example:

find user/ricardo/toy_fit_submission_02082026/ -name "*.sh" -exec submit --memory 32 'source {}' --output user/ricardo/output_toy_fits_02082026 \;
"""

ap = argparse.ArgumentParser(description="Prepares shell scripts with blocks of toy fits to be executed on SLURM.")

ap.add_argument("-config", help="Which config to use")
ap.add_argument("-input_dir", help="Folder with the toys to fit")
ap.add_argument("--output_dir", help="Folder with the output scripts to be submitted.")
ap.add_argument("--overwrite", action="store_true", help="Overwrite existing fits.")
ap.add_argument("--n_jobs", type=int, default=1, help="Number of jobs per submission")

args = ap.parse_args()

output_dir = args.output_dir
if output_dir is None:
    output_dir = os.path.join(user.output_directory, "toy_fit_submission")

os.makedirs(output_dir, exist_ok=True)

list_files = np.array(os.listdir(args.input_dir))
if args.n_jobs:
    blocks = np.array_split(list_files, args.n_jobs)

output_file_base = os.path.basename(args.input_dir.strip("/"))

for i, block in enumerate(blocks):
    with open(os.path.join(output_dir, f"launch_{output_file_base}_fits_{i}.sh"),"w") as f:
        for file in block:
            full_path = os.path.join(args.input_dir, file)
            if args.overwrite:
                submit_string = f"python fit/Likelihood.py {args.config} --toyFile {full_path} --overwrite \n"
            else:
                submit_string = f"python fit/Likelihood.py {args.config} --toyFile {full_path} \n"
            f.write(submit_string)

logging.info("Done.")

