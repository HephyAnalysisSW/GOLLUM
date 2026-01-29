''''
Script to make likelihood scans, similar to combine's MultiDimFit method.  Can choose> POIs to scan over, their range, and the number of scan points per dimension
For job submission, one can specify a range of points to produce for each job. 

For example, the following code produces 2D scans over the rate and shape_4 POIs with the given ranges
python MultiDimFit.py ../configs/unbinned/unbinned_2018.yaml --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_2018.json --POIs rate,shape_4 --setParameterRanges rate=-50,50:shape_4=-10,10 --pointRange $SLURM_ARRAY_TASK_ID $(($SLURM_ARRAY_TASK_ID+1)) --npoints 25 --name rate_shape4

'''


import os 
import sys
sys.path.insert(0, '..')

import fit.Likelihood as lh 
import pickle  as pck 
import common.user as user 
import json 
from fit.Modeling import Rotated


import numpy as np
import matplotlib.pyplot as plt
import os


def parse_ranges(s: str) -> dict:
    """
    Parse strings like 'p1=1,2:p2=3,4' into:
        {'p1': [1.0, 2.0], 'p2': [3.0, 4.0]}
    Includes validation without regex.
    """
    if not s:
        raise ValueError("Input string is empty.")

    result = {}
    segments = s.split(":")

    for seg in segments:
        # Must contain exactly one "="
        if "=" not in seg:
            raise ValueError(f"Missing '=' in segment: {seg}")
        name, coords = seg.split("=", 1)

        # Validate name
        if not name:
            raise ValueError(f"Empty point name in segment: {seg}")
        if not (name[0].isalpha() or name[0] == "_"):
            raise ValueError(f"Invalid point name '{name}': must start with letter or underscore.")
        if not all(c.isalnum() or c == "_" for c in name):
            raise ValueError(f"Invalid characters in point name '{name}'.")

        # Coordinates must contain exactly one comma
        if "," not in coords:
            raise ValueError(f"Missing comma in coordinates: {coords}")
        x_str, y_str = coords.split(",", 1)

        # Convert numbers safely
        try:
            x = float(x_str)
            y = float(y_str)
        except ValueError:
            raise ValueError(f"Coordinates must be numeric: {coords}")

        result[name] = [x, y]

    return result





if __name__ == "__main__":
    # ---------------- args ----------------
    import argparse
    p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")
    p.add_argument("--algo", default="grid", choices=["grid"], help="Set type of likelihood scans to be done. For the time being, only grid is allowed")
    p.add_argument("--freezeParameters", default="", help="Parameters to freeze. Otherwise they float")
    p.add_argument("--POIs", default="", help="Set of POIs that are considered as signal")
    p.add_argument("--setParameterRanges", default="", help="Set parameter ranges. Only acting on the POIs we are floating")
    p.add_argument("--pointRange", default=None, nargs=2, type=int, help="Range of points to ran on, for batch submission")
    p.add_argument("--name", default="", help="Name to add to the base name")
    p.add_argument("--npoints", default=100, type=int, help="Number of scan points per dimension")
    p.add_argument(
        "--overwrite",
        nargs="?",
        const="all",
        default=None,
        choices=["fit", "all"],
        help="Overwrite results: 'fit' overwrites fit JSON only; 'all' overwrites fit JSON and cache.",
    )

    args = p.parse_args()

    import common.yaml_loader as yaml_loader 

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    like_info = lh.load_likelihood(cfg)

    base    = os.path.splitext(os.path.basename(args.config))[0] + ("_rotate" if args.rotate else "")
    version = str(cfg.get("version", "v0"))
    overwrite_cache = args.overwrite == "all"


    hyp  = lh.build_hypothesis_from_likelihood(like_info, name="SR")
    rotated = bool(args.rotate)
    hyp_for_fit = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp
    step = 1.0 if rotated else 0.1


    if args.freezeParameters != '': 
        for poi in args.freezeParameters.split("," ):
            #hyp_for_fit.set_nuisance_frozen(poi, True)
            getattr(hyp_for_fit, poi).isFrozen = True


    n2ll = lh.N2LL(
        like_info,
        cfg["defaults"]["module_samples"],
        cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
        cache_root=None,
        overwrite=overwrite_cache,
    )
    n2ll.build_cache()
    print("Preparing runtime")
    n2ll.prepare_runtime()
    print("Done with runtime. setting asimov") 
    n2ll.setAsimov(hyp_for_fit)
    print("Done asimov. Al turron")

    parameterRanges = parse_ranges(args.setParameterRanges)
    print(args.algo)
    if args.algo == 'grid':
        print("Check 1") 
        POIs = args.POIs.split(",")
        ranges_arrays = [parameterRanges[poi] for poi in POIs]
        grid_arrays   = [np.linspace(x[0], x[1], args.npoints) for x in ranges_arrays]
        mesh = np.meshgrid(*grid_arrays, indexing="ij")
        mesh = np.stack(mesh, axis=-1).reshape(-1, len(ranges_arrays))

        # printing lengths
        name_w = max(len(str(p)) for p in POIs)
        val_w  = 8  # numeric field width


        # we are going to scan over these, so we freeze them 

        for i, scan_point in enumerate(mesh):
            print(i)
            if args.pointRange is not None:
                if i<args.pointRange[0]: continue
                if i>=args.pointRange[1]: continue

            fields = [
                f"{POIs[j]:<{name_w}} = {float(scan_point[j]):{val_w}f}"
                for j in range(len(POIs))
            ]
            print(f"Point {i} / {len(mesh)}: " + "   ".join(fields), end="  =>  ")

            hyp_point = hyp_for_fit.clone()

            hyp_point.modify(**dict([ (poi,scan_point[j]) for j, poi in enumerate(POIs)]))

            for poi in POIs:
                getattr(hyp_point, poi).isFrozen = True
                print(f"Freezing {poi}")


            m = lh.run_minuit_fit(n2ll, hyp_point, step=step, print_every=-1, do_migrad=True, do_hesse=False, do_minos=False,verbosity=1)
            print(f"-2logL =  {m.fval:{val_w}f}")
            np.save(f'{base}_{version}_{args.name}_scan_{i}', np.rec.fromarrays(list(scan_point) + [m.fval], names=POIs + ['-2logL']))



            
