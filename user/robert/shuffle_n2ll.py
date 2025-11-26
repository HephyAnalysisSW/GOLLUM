#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np

# Import your likelihood machinery
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from fit.Likelihood import (
    load_likelihood,
    build_hypothesis_from_likelihood,
    N2LL, run_minuit_fit, serialize_result
)
import common.yaml_loader as yaml_loader


p = argparse.ArgumentParser(
    description="Compare true -2logL to Fisher quadratic approximation near (0,0) Asimov."
)
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--overwrite", action="store_true", help="Overwrite caches")
p.add_argument("--shuffle", nargs="*", default=[], help="Shuffle these features")
args = p.parse_args()


cfg = yaml_loader.load_yaml(args.config)
yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

like_info = load_likelihood(cfg)

hyp = build_hypothesis_from_likelihood(like_info, name="SR")

hyp.print()

# we're shuffling 
shuffle_suffix = "_".join( ["shuffle"]+sorted(args.shuffle) ) 
n2ll = N2LL( like_info, 'data.samples',  
             os.path.join( "NN2LCache", os.path.splitext(os.path.basename(args.config))[0], cfg['version'] +"_"+ shuffle_suffix), cache_root=None, overwrite=args.overwrite)

n2ll.shuffle_features = args.shuffle
n2ll.build_cache()
n2ll.prepare_runtime()

# compute A-simov
n2ll.setAsimov()

# compute C-simov (POI or nuisance injection)
#n2ll.setAsimov(hyp.cloneModify(c1=1))

## run Minuit; prints the model every 25 evaluations by default
m, adapter = run_minuit_fit(n2ll, hyp, step=0.1, print_every=1, do_migrad=True, do_hesse=True, do_minos=False)

# best-fit -2logL
print("Best -2logL =", m.fval)

print("Correlation")
print(m.covariance.correlation())

# -------- persist fit result + covariance --------
import os, json, numpy as np
import common.user as user

base    = os.path.splitext(os.path.basename(args.config))[0]
version = str(cfg.get("version", "v0"))
os.makedirs(user.output_directory, exist_ok=True)
out_path = os.path.join(user.output_directory, f"{base}_{version}{shuffle_suffix}_fit.json")
serialize_result(m, base, version, args, out_path)
