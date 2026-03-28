"""
Usage: python3 user/kaan/submit_toy_fits.py 


"""



from subprocess import run
import numpy as np
import re
import json
import os
import shutil
from typing import Dict, Any, Tuple


def load_toys_npz(npz_path: str) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns:
      toys[itoy][rid] = (indices, weights)
    where
      indices: (Ndraw,) int
      weights: (Ndraw,) float (signed allowed). If absent, filled with +1.
    """
    _TOY_IDX_RE = re.compile(r"^toy(\d{4})_(.*)_indices$")
    _TOY_WGT_RE = re.compile(r"^toy(\d{4})_(.*)_weights$")


    z = np.load(npz_path, allow_pickle=False)
    # temporary structure: toys[itoy][rid] = {"idx":..., "w":...}
    tmp: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for key in z.files:
        m = _TOY_IDX_RE.match(key)
        if not m:
            continue
        itoy = int(m.group(1))
        rid = m.group(2)
        tmp.setdefault(itoy, {}).setdefault(rid, {})["idx"] = np.asarray(z[key], dtype=np.int64)

    for key in z.files:
        m = _TOY_WGT_RE.match(key)
        if not m:
            continue
        itoy = int(m.group(1))
        rid = m.group(2)
        tmp.setdefault(itoy, {}).setdefault(rid, {})["w"] = np.asarray(z[key], dtype=np.float64)

    if not tmp:
        raise RuntimeError(f"No toy keys matched '{_TOY_IDX_RE.pattern}' in {npz_path}")

    toys: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for itoy, by_rid in tmp.items():
        toys[itoy] = {}
        for rid, d in by_rid.items():
            if "idx" not in d:
                continue
            idx = d["idx"]
            w = d.get("w", None)
            if w is None:
                w = np.ones(idx.shape[0], dtype=np.float64)
            if w.shape[0] != idx.shape[0]:
                raise RuntimeError(
                    f"[toys] itoy={itoy} rid={rid}: indices and weights have different lengths "
                    f"({idx.shape[0]} vs {w.shape[0]})"
                )
            toys[itoy][rid] = (idx, w)

    return toys



config = "user/kaan/fit_toys.py configs/unbinned_v5/2016/unbinned_2016_6.yaml"
# toy_file = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_nominal_N200.npz"
# toy_file = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_PDF4LHC21_mc_m0_rotate_N1000.npz"
# toy_file = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_PDF4LHC21_mc_m0_rw_N9900.npz"
toy_file = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/"
print_info = "--print-every 1 --minuit-print-level 2"
rotate = "--rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_unbinned_2016_6_unbinned_2016_v5.json"
out_dir = "/scratch-cbe/users/alikaan.gueven/SBIPDF/output/toys/toys_nominal_N9900"
os.makedirs(out_dir, exist_ok=True)

toys = load_toys_npz(toy_file)

job_dict = {}

for i in range(len(toys)):
    toy_number = f"--toy-number {i}"
    fit_out = "--out " + os.path.join(out_dir, f"toy_{i}")
    command = ' '.join(["python3", '-u', config, toy_file, toy_number, rotate, print_info, fit_out])
    full_command = f'sbatch user/kaan/sh/submit_to_cpu_rapid.sh "{command}"'
    result = run(full_command, shell=True, capture_output = True, text = True)
    job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
    info_dict = {'command': full_command,               # Save command [important for resubmitting]
                'jobid':   job_id}                      # Save job_id  [identify the status with sacct]
    job_dict[f'toy_{i}'] = info_dict                    # Add to dict
    print(result.stdout[:-1])

out_json_path = os.path.join(out_dir, 'fit_jobs.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f, indent=2)


print('\nFinished. Exiting...')