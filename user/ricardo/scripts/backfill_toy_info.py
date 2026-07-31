#!/usr/bin/env python
"""One-off backfill of the 'toy' provenance block into existing fit JSONs.

fit/Likelihood.py now writes a 'toy' block (point, source, seed, injected
hypothesis) into every fit result produced with --toyFile. Fit JSONs written
before that change lack it, so fit/analyze_toy_fits.py cannot compute pulls
from them. This script recovers the missing information from the toy HDF5 the
fit was run on and injects it in place, so the old fits do not have to be
re-run.

Run from the repo root:
    python user/ricardo/scripts/backfill_toy_info.py --inputDir output_SBIEFT/
"""

import argparse
import glob
import json
import logging
import os
import re

import h5py

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The suffix fit/Likelihood.py appends to the output name for a toy fit.
TOY_SUFFIX_PATTERN = re.compile(r"^(?P<point>.+)_(?P<source>cache|truth)_toy(?P<seed>\d+)$")


def toy_file_for_fit(fit_path: str, fit_payload: dict, input_dir: str) -> tuple:
    """Return (toy_h5_path, point, source, seed) for a fit JSON.

    The fit output name is '<config_basename>_<version><suffix>_fit.json' and the
    toy part of the suffix is '_<point>_<source>_toy<seed>', which is exactly the
    toy file's stem. Toys live in '<input_dir>/<source>_toys_<config_basename>/'.
    """
    stem = os.path.basename(fit_path)[: -len("_fit.json")]
    prefix = f"{fit_payload['config_basename']}_{fit_payload['version']}_"
    if not stem.startswith(prefix):
        raise RuntimeError(f"Fit name '{stem}' does not start with expected prefix '{prefix}'")

    match = TOY_SUFFIX_PATTERN.match(stem[len(prefix):])
    if match is None:
        raise RuntimeError(f"Cannot parse toy point/source/seed from '{stem}'")

    point, source, seed = match["point"], match["source"], int(match["seed"])
    toy_path = os.path.join(input_dir,
                            f"{source}_toys_{fit_payload['config_basename']}",
                            f"{point}_{source}_toy{seed}.h5")
    return toy_path, point, source, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputDir", required=True,
                        help="Directory holding the fit JSONs and the <source>_toys_<config>/ toy folders.")
    parser.add_argument("--dryRun", action="store_true",
                        help="Report what would be patched without writing.")
    args = parser.parse_args()

    fit_paths = sorted(glob.glob(os.path.join(args.inputDir, "**", "*_toy*_fit.json"), recursive=True))
    logger.info("Found %d toy fit JSONs under %s", len(fit_paths), args.inputDir)

    n_patched, n_already = 0, 0
    for fit_path in fit_paths:
        with open(fit_path) as fit_file:
            payload = json.load(fit_file)

        if "toy" in payload:
            n_already += 1
            continue

        toy_path, point, source, seed = toy_file_for_fit(fit_path, payload, args.inputDir)
        with h5py.File(toy_path, "r") as toy_file:
            hypothesis = json.loads(str(toy_file["meta"].attrs["hypothesis"]))
            # Cross-check the filename-derived labels against the toy's own metadata.
            if str(toy_file["meta"].attrs["point"]) != point or int(toy_file["meta"].attrs["seed"]) != seed:
                raise RuntimeError(f"Toy metadata in {toy_path} disagrees with fit name {fit_path}")

        payload["toy"] = {"point": point, "source": source, "seed": seed, "hypothesis": hypothesis}
        if not args.dryRun:
            with open(fit_path, "w") as fit_file:
                json.dump(payload, fit_file, indent=2)
        n_patched += 1

    logger.info("%s %d fit JSONs, %d already had a 'toy' block",
                "Would patch" if args.dryRun else "Patched", n_patched, n_already)


if __name__ == "__main__":
    main()
