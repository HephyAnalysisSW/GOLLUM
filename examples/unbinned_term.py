"""Persistent value/gradient example for the cached paper unbinned likelihood.

Run from the repository root after configuring common/user.py. No fitting,
training, or cache rebuilding is performed. The input arrays stay in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import resource
import time
from pathlib import Path

import numpy as np
from autograd import value_and_grad
import common.yaml_loader as loader
from fit.Likelihood import N2LL, build_hypothesis_from_likelihood, load_likelihood
from fit.Modeling import Rotated


class UnbinnedTerm:
    """The paper 7-1 term: six rotated PDF directions and 25 nuisances.

    objective is differentiable and returns -2 log L. value_and_gradient
    returns the scalar and all parameter derivatives from one forward pass.
    The default includes the term's nuisance constraints; pass
    include_penalty=False if those constraints are supplied elsewhere.
    """

    def __init__(self, rotation, *, numerics="paper", include_penalty=True):
        if numerics not in {"paper", "cms"}:
            raise ValueError("numerics must be 'paper' or 'cms'")
        self.numerics = numerics
        self.include_penalty = include_penalty
        config = Path("configs/unbinned_v5D/unbinned_delphes_7_RunII.yaml")
        cfg = loader.load_yaml(str(config))
        missing = loader.load_surrogates(cfg, str(config), overwrite=False)
        if missing:
            raise RuntimeError(f"Missing trained artifacts; no training will be started: {missing}")
        info = load_likelihood(cfg)
        self.hypothesis = Rotated(build_hypothesis_from_likelihood(info), str(rotation))
        basis = json.loads(Path(rotation).read_text())
        eigenvalues = dict(zip(basis["basis_labels"], basis["eigenvalues"]))
        for parameter in self.hypothesis.POIs:
            if eigenvalues[parameter.name] < 0.01:
                parameter.freeze(value=0.0)
        self.names = tuple(
            p.name for p in self.hypothesis.POIs + self.hypothesis.nuisances
            if not p.isFrozen and not getattr(p, "isIgnored", False)
        )
        self.theta0 = np.zeros(len(self.names))
        defaults = cfg["defaults"]
        import importlib
        features = loader._resolve_features_list(defaults["default_features"])
        factory = importlib.import_module(defaults["module_samples"]).Factory(
            features=features, selection=defaults.get("default_selection"),
            selection_features=defaults.get("default_selection_features"),
        )
        self.nll = N2LL(
            info, factory=factory, overwrite=False,
            cache_subdir=f"NN2LCache/{config.stem}/{cfg['version']}",
            paper_numerics=numerics == "paper",
        )
        for region in info["regions"]:
            for cls in region["classes"]:
                paths = self.nll._paths_for(region["id"], cls["id"])
                if not all(Path(path).is_file() for path in paths):
                    raise RuntimeError(f"Missing cache files; rebuilding is disabled: {paths}")
        self.nll.prepare_runtime()
        self.nll.setAsimov()
        self.events = sum(self.nll._N_region.values())
        self._value_and_grad = value_and_grad(self.objective)

    def objective(self, theta):
        if len(theta) != len(self.names):
            raise ValueError(f"Expected {len(self.names)} parameters in term.names order")
        h = self.hypothesis.cloneModify(**dict(zip(self.names, theta)))
        result = self.nll(h)
        return result if self.include_penalty else result - h._base.penalty()

    def value(self, theta):
        return float(self.objective(theta))

    def value_and_gradient(self, theta):
        value, gradient = self._value_and_grad(theta)
        return float(value), np.asarray(gradient, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--numerics", choices=["paper", "cms"], default="paper")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true", help="Exclude nuisance penalties")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    start = time.perf_counter()
    term = UnbinnedTerm(args.rotation, numerics=args.numerics, include_penalty=not args.data_only)
    setup_seconds = time.perf_counter() - start
    reference = json.loads(Path("validation/paper_reference.json").read_text())["fits"]["unbinned"]
    if list(term.names) != reference["names"]:
        raise RuntimeError("Parameter ordering differs from the validated paper example")
    mixed = next(r for r in reference["rows"] if r["label"] == "mixed")
    theta = np.array([mixed["changes"].get(name, 0.0) for name in term.names])
    start = time.perf_counter()
    first_value = term.value(theta)
    first_value_seconds = time.perf_counter() - start
    if args.numerics == "paper" and not args.data_only:
        if not np.isclose(first_value, mixed["value"], rtol=1e-12, atol=1e-8):
            raise AssertionError(f"Loaded inputs do not reproduce the paper reference: {first_value}")
    start = time.perf_counter()
    value, gradient = term.value_and_gradient(theta)
    first_value_gradient_seconds = time.perf_counter() - start
    forward, combined = [], []
    for _ in range(args.repeat):
        start = time.perf_counter()
        value = term.value(theta)
        forward.append(time.perf_counter() - start)
        start = time.perf_counter()
        joint_value, gradient = term.value_and_gradient(theta)
        combined.append(time.perf_counter() - start)
        if not np.isclose(value, joint_value, rtol=1e-12, atol=1e-8):
            raise AssertionError("Value and differentiable value disagree")
    if not np.isfinite(gradient).all():
        raise AssertionError("Non-finite gradient")
    result = dict(
        host=socket.gethostname(), events=term.events, names=term.names,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        numerics=args.numerics, includes_nuisance_penalty=term.include_penalty,
        thread_environment={k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"]},
        setup_seconds=setup_seconds, first_forward_seconds=first_value_seconds,
        first_value_gradient_seconds=first_value_gradient_seconds,
        warmed_forward_seconds=forward, warmed_value_gradient_seconds=combined,
        value=value, theta=theta.tolist(), gradient=gradient.tolist(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"names", "gradient"}}, indent=2))
    print(f"Parameter names and complete timings: {args.output}")


if __name__ == "__main__":
    main()
