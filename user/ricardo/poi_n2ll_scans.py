#!/usr/bin/env python3
"""Scan the Asimov negative log-likelihood over one or several fixed POIs."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from typing import Any, Optional
from common import user, helpers

import numpy as np

import importlib

import pprint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from common import yaml_loader
from fit.Likelihood import N2LL, build_hypothesis_from_likelihood, load_likelihood

import matplotlib.pyplot as plt

import common.syncer as syncer

def scan_likelihood_unprofiled(args:argparse.Namespace) -> Optional[list[dict[str, Any]]]:

    expanded_config_path = os.path.expanduser(os.path.expandvars(args.config))
    base = os.path.splitext(os.path.basename(expanded_config_path))[0]
    config = yaml_loader.load_yaml(expanded_config_path)
    yaml_loader.print_summary(config, expanded_config_path, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(config, expanded_config_path, overwrite=False)

    like_info = load_likelihood(config)

    asimov_hypothesis = build_hypothesis_from_likelihood(
        like_info,
        name="asimov_reference",
        poi_init=0.0,
        nuis_init=0.0,
    )
    
    asimov_hypothesis.print()

    # Make sample loader factory from default config
    samples_mod = importlib.import_module(config["defaults"]["module_samples"])

    from common.yaml_loader import _resolve_features_list
    default_features = config["defaults"].get("default_features", None)
    features = _resolve_features_list( default_features ) if default_features else None
    factory     = samples_mod.Factory( 
        features  = features,
        selection = config["defaults"].get("default_selection", None),
        selection_features = config["defaults"].get("default_selection_features", None),
        )

    n2ll = N2LL(
        like_info,
        factory = factory,
        cache_subdir=os.path.join("NN2LCache", base, config["version"]),
        cache_root=None,
    )

    # if hasattr(n2ll, "build_cache"):
    n2ll.build_cache()
    n2ll.prepare_runtime()
    n2ll.setAsimov()

    if not args.scan:
        available_parameters = list(asimov_hypothesis.parameters)
        print("Available parameters:")
        for parameter_name in available_parameters:
            print(f"  {parameter_name}")
        return None

    # to fix any parameter outside of the Asimov hypothesis
    fixed_parameter_values: dict[str, float] = {}
    for fixed_definition in args.fix:
        parameter_name, parameter_value = fixed_definition.split("=", 1)
        fixed_parameter_values[parameter_name] = float(parameter_value)

    scan_axes: dict[str, np.ndarray] = {}
    for scan_definition in args.scan:
        parameter_name, minimum, maximum, n_points = scan_definition.split(":")
        scan_axes[parameter_name] = np.linspace(
            float(minimum),
            float(maximum),
            int(n_points),
        )

    nominal_n2ll = float(n2ll(asimov_hypothesis))
    print(f"{nominal_n2ll=}")

    scan_parameter_names = list(scan_axes.keys())
    if len(scan_parameter_names) > 2:
        raise ValueError("Visually, only makes sense to plot up to two-dimensional scans.")
    
    scan_parameter_grids = [scan_axes[parameter_name] for parameter_name in scan_parameter_names]

    results: list[dict[str, Any]] = []
    for scan_point_values in itertools.product(*scan_parameter_grids):
        scan_parameter_values = {
            scan_parameter_names[index]: float(scan_point_values[index])
            for index in range(len(scan_parameter_names))
        }

        scan_hypothesis = asimov_hypothesis.cloneModify(**fixed_parameter_values)
        scan_hypothesis.modify(**scan_parameter_values)

        scan_n2ll = float(n2ll(scan_hypothesis))
        results.append(
            {
                "scan_parameters": scan_parameter_values,
                "n2ll": scan_n2ll,
                # "delta_n2ll_to_nominal": scan_n2ll - nominal_n2ll,
            }
        )

    # grid_min_n2ll = min(result["n2ll"] for result in results)
    # for result in results:
        # result["delta_n2ll_to_grid_min"] = result["n2ll"] - grid_min_n2ll

    best_result = min(results, key=lambda result: result["n2ll"])

    print(f"Nominal Asimov N2LL: {nominal_n2ll}")
    print(f"Evaluated {len(results)} scan points")
    print(f"Best grid point: {best_result['scan_parameters']}")
    print(f"Best grid N2LL: {best_result['n2ll']}")

    # if args.output:
    #     output_path = os.path.join(user.output_directory, os.path.expanduser(os.path.expandvars(args.output)))
    #     output_directory = os.path.dirname(output_path)
    #     if output_directory:
    #         os.makedirs(output_directory, exist_ok=True)

    #     output_payload = {
    #         "config": expanded_config_path,
    #         "fixed_parameters": fixed_parameter_values,
    #         "scan_axes": {
    #             parameter_name: scan_axes[parameter_name].tolist()
    #             for parameter_name in scan_parameter_names
    #         },
    #         "nominal_n2ll": nominal_n2ll,
    #         "grid_min_n2ll": grid_min_n2ll,
    #         "results": results,
    #     }
        
    #     with open(output_path, "w") as output_file:
    #         json.dump(output_payload, output_file, indent=2)
    
    return results

def plot_n2ll_scan(results: list[dict[str, Any]], plot_directory: str) -> None:
    """Plot the N2LL scan results for one- or two-dimensional scans and save to disk."""
    scan_parameter_names = list(results[0]["scan_parameters"].keys())
    n2ll_values = np.array([result["n2ll"] for result in results])
    delta_n2ll = n2ll_values - n2ll_values.min()

    plot_filename = "n2ll_scan_" + "_".join(scan_parameter_names) + ".pdf"
    plot_path = os.path.join(plot_directory, plot_filename)
    os.makedirs(plot_directory, exist_ok=True)

    if len(scan_parameter_names) == 1:
        poi_name = scan_parameter_names[0]
        poi_values = np.array([result["scan_parameters"][poi_name] for result in results])

        fig, ax = plt.subplots()
        # reducing large scans to scans with a small range
        delta_n2ll_reduced_range_mask = np.argwhere(delta_n2ll <= 10.0).flatten()
        poi_values_reduced_range = poi_values[delta_n2ll_reduced_range_mask]
        n2ll_values_reduced_range = n2ll_values[delta_n2ll_reduced_range_mask]

        ax.plot(poi_values_reduced_range, n2ll_values_reduced_range, linestyle='dashed', linewidth=2)
        ax.set_xlabel(poi_name)
        ax.set_ylabel(r"$\Delta$ N2LL")

        quadratic_polynomial_scaled = np.polynomial.Polynomial.fit(poi_values_reduced_range,n2ll_values_reduced_range,deg=2)
        quadratic_polynomial = quadratic_polynomial_scaled.convert()

        cl68_polynomial = quadratic_polynomial - np.polynomial.Polynomial([1.0])
        cl95_polynomial = quadratic_polynomial - np.polynomial.Polynomial([3.84])

        cl68_edges = cl68_polynomial.roots()
        cl95_edges = cl95_polynomial.roots()

        ax.axhline(1.0, color="gray", linestyle="--",
                   label=f"68% CL: [{cl68_edges[0]:.3f}, {cl68_edges[-1]:.3f}]")
        ax.axhline(3.84, color="gray", linestyle=":",
                   label=f"95% CL: [{cl95_edges[0]:.3f}, {cl95_edges[-1]:.3f}]")
        ax.legend()            

    elif len(scan_parameter_names) == 2:
        poi_x, poi_y = scan_parameter_names
        x_values = np.unique([result["scan_parameters"][poi_x] for result in results])
        y_values = np.unique([result["scan_parameters"][poi_y] for result in results])
        delta_n2ll_grid = delta_n2ll.reshape(len(x_values), len(y_values))

        delta_n2ll_reduced_range_mask = delta_n2ll_grid < 10
        x_reduced_range_mask = delta_n2ll_reduced_range_mask.any(axis=1)
        y_reduced_range_mask = delta_n2ll_reduced_range_mask.any(axis=0)

        x_values_reduced_range = x_values[x_reduced_range_mask]
        y_values_reduced_range = y_values[y_reduced_range_mask]
        delta_n2ll_grid_reduced_range = delta_n2ll_grid[np.ix_(x_reduced_range_mask, y_reduced_range_mask)]

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x_values_reduced_range, y_values_reduced_range, delta_n2ll_grid_reduced_range.T, shading="auto")
        ax.contour(x_values_reduced_range, y_values_reduced_range, delta_n2ll_grid_reduced_range.T, levels=[2.30, 6.18], colors="white", linestyles=["--", ":"])
        fig.colorbar(mesh, ax=ax, label=r"$\Delta$ N2LL")
        ax.set_xlabel(poi_x)
        ax.set_ylabel(poi_y)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".pdf",".png"))
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    
    """Run a fixed-parameter Asimov likelihood scan."""
    parser = argparse.ArgumentParser(
        description="Scan the Asimov N2LL over one or several POIs without profiling"
    )
    parser.add_argument("config", help="Path to the global YAML config")
    parser.add_argument(
        "--scan",
        nargs="*",
        default=[],
        help="Scan definitions such as c0:-2:2:41. Currently supporting only one- or two-dimensional scans.",
    )
    parser.add_argument(
        "--fix",
        nargs="*",
        default=[],
        help="Fixed parameter values such as c2=0.3 theta_pdf=0.0. If not given, parameters retain their Asimov values.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path, relative to user.output_directory",
    )
    args = parser.parse_args()
    
    results = scan_likelihood_unprofiled(args)

    # pprint.pprint(results)

    if results and args.output:
        plot_directory = os.path.join(user.plot_directory, args.output)
        plot_n2ll_scan(results, plot_directory)
        helpers.copyIndexPHP(plot_directory)
    syncer.sync()