import shlex, sys

def get_base_command():
    argv = [a for a in sys.argv if a != "--prepareSlurmJobs"]
    return " ".join(shlex.quote(a) for a in argv)

def get_nuisance_names(hyp):
    return [p.name for p in hyp.nuisances]

def write_nuisance_list(names, outdir):
    path = os.path.join(outdir, "/users/sergio.sanchez.cruz/dev/GOLLUM/fit/nuisance_list.txt")
    with open(path, "w") as f:
        for n in names:
            f.write(n + "\n")
    return path

def write_task_runner(outdir, base_cmd):

    path = os.path.join(outdir, "run_task.sh")
    with open(path, "w") as f:
        f.write(f"""#!/bin/bash
source /users/sergio.sanchez.cruz/miniforge3/etc/profile.d/conda.sh
conda activate /groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-2
cd /users/sergio.sanchez.cruz/dev/GOLLUM/fit

NUISANCE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" nuisance_list.txt)

python {base_cmd}  --nuisanceForImpacts "$NUISANCE"
""")
    os.chmod(path, 0o755)
    return path

def write_slurm_script(outdir, n_tasks):
    path = os.path.join(outdir, "submit_impacts.slurm")
    with open(path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=tfmc-impacts
#SBATCH --array=0-{n_tasks-1}
#SBATCH --ntasks=8
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/impact_%a.out
#SBATCH --error=logs/impact_%a.err

mkdir -p logs

bash "run_task.sh"
""")
    return path

import os

def prepare_slurm_jobs(hyp_for_fit, base_cmd, base, version):
    outdir = os.path.join("slurm", base, version)
    os.makedirs(outdir, exist_ok=True)

    nuisances = get_nuisance_names(hyp_for_fit)

    write_nuisance_list(nuisances, outdir)
    write_task_runner(outdir, base_cmd)
    slurm_script = write_slurm_script(outdir, len(nuisances))

    print(f"Prepared {len(nuisances)} Slurm tasks in {outdir}")
    print()
    print("Submit with:")
    print(f"  sbatch {slurm_script}")
