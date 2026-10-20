"""Check fixed paper inputs, likelihood values and Autograd derivatives.

Run from the repository root in the configured environment. This reads existing
surrogates/caches and never trains models or rebuilds caches. External artifacts
are deliberately not packaged here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
from autograd import grad
import common.user as user
import common.yaml_loader as loader
from fit.Likelihood import N2LL, load_likelihood, build_hypothesis_from_likelihood
from fit.Modeling import Rotated

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fit", choices=["binned", "unbinned"], required=True)
parser.add_argument("--rotation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--skip-gradients", action="store_true")
parser.add_argument("--numerics", choices=["paper", "cms"], default="paper",
                    help="CMS mode reports reference differences without requiring historical numerical equality.")
args = parser.parse_args()

reference = json.loads(Path("validation/paper_reference.json").read_text())
ref = reference["fits"][args.fit]
config = Path(
    "configs/binned_v5D/binned_delphes_8_RunII.yaml" if args.fit == "binned"
    else "configs/unbinned_v5D/unbinned_delphes_7_RunII.yaml"
)
for path, expected in [(config, ref["config_sha256"]), (args.rotation, ref["rotation_sha256"])]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Input differs from paper reference: {path}")

cfg = loader.load_yaml(str(config))
missing = loader.load_surrogates(cfg, str(config), overwrite=False)
if missing:
    raise RuntimeError(f"Missing surrogates: {missing}")
info = load_likelihood(cfg)
hyp = Rotated(build_hypothesis_from_likelihood(info), str(args.rotation))
payload = json.loads(args.rotation.read_text())
threshold = 0.001 if args.fit == "binned" else 0.01
ev_by_name = dict(zip(payload["basis_labels"], payload["eigenvalues"]))
for p in hyp.POIs:
    if ev_by_name[p.name] < threshold:
        p.freeze(value=0.0)
free = [p for p in hyp.POIs + hyp.nuisances if not p.isFrozen and not getattr(p, "isIgnored", False)]
names = [p.name for p in free]
if names != ref["names"]:
    raise RuntimeError(f"Parameter order changed: {names}")

defaults = cfg["defaults"]
features = loader._resolve_features_list(defaults["default_features"]) if defaults.get("default_features") else None
factory = importlib.import_module(defaults["module_samples"]).Factory(
    features=features, selection=defaults.get("default_selection"),
    selection_features=defaults.get("default_selection_features"),
)
cache_subdir = f"NN2LCache/{config.stem}/{cfg['version']}"
nll = N2LL(info, factory=factory, cache_subdir=cache_subdir, overwrite=False, paper_numerics=args.numerics == 'paper')
if args.fit == "unbinned":
    # Refuse to build or silently substitute an incomplete cache.
    for region in info["regions"]:
        for cls in region["classes"]:
            h5, meta = nll._paths_for(region["id"], cls["id"])
            if not Path(h5).is_file() or not Path(meta).is_file():
                raise RuntimeError(f"Missing paper cache: {h5}, {meta}")
nll.prepare_runtime()
nll.setAsimov()

def objective(x):
    return nll(hyp.cloneModify(**dict(zip(names, x))))

checks = []
for row in ref["rows"]:
    x = np.array([row["changes"].get(name, 0.0) for name in names])
    value = float(objective(x))
    delta = value - row["value"]
    if args.numerics == "paper" and not np.isclose(value, row["value"], rtol=1e-12, atol=1e-8):
        raise AssertionError(f"{row['label']}: {value} != {row['value']} (delta={delta})")
    checks.append(dict(label=row["label"], value=value, reference=row["value"], difference=delta))
    print("VALUE", checks[-1], flush=True)

mixed = next(row["changes"] for row in ref["rows"] if row["label"] == "mixed")
x = np.array([mixed.get(name, 0.0) for name in names])
timings = []
for _ in range(3):
    start = time.perf_counter()
    objective(x)
    timings.append(time.perf_counter() - start)
result = dict(fit=args.fit, numerics=args.numerics, names=names, checks=checks, warmed_objective_seconds=timings)
if not args.skip_gradients:
    gradient = grad(objective)
    start = time.perf_counter()
    analytic = np.asarray(gradient(x), dtype=float)
    gradient_seconds = time.perf_counter() - start
    finite_start = time.perf_counter()
    finite = []
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = 1e-5
        finite.append(float((objective(x + dx) - objective(x - dx)) / 2e-5))
    finite_seconds = time.perf_counter() - finite_start
    finite = np.array(finite)
    scaled = np.abs(analytic - finite) / np.maximum(1.0, np.abs(finite))
    if not np.allclose(analytic, finite, rtol=5e-5, atol=1e-4):
        bad = [(names[i], float(analytic[i]), float(finite[i])) for i in range(len(x))
               if not np.isclose(analytic[i], finite[i], rtol=5e-5, atol=1e-4)]
        raise AssertionError(f"Gradient discrepancy: {bad}")
    result.update(gradient_seconds=gradient_seconds, finite_difference_seconds=finite_seconds,
                  gradient=analytic.tolist(), finite_difference_gradient=finite.tolist(),
                  maximum_scaled_gradient_error=float(scaled.max()))
    print("GRADIENT", result["maximum_scaled_gradient_error"], "seconds", gradient_seconds, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print("PASS", args.output, flush=True)
