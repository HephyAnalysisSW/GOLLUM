'''
Plot likelihood scans produced by fit/MultiDimFit.py.

Each scan point is saved as a .npy structured array with fields for the
scanned POIs plus '-2logL'. This script collects all scan files matching
a glob pattern, and plots a 1D profile (POI vs delta -2logL) or a 2D
contour (two POIs vs delta -2logL), depending on how many POI fields
are found.

Usage:
    python plot/postfitsys/plot_scan.py "unbinned_2018_v8_rate_scan_*.npy" --out scan
'''

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import common.user as user
import common.syncer as syncer
from typing import List
import common.helpers as helpers

import mplhep as hep
hep.style.use("CMS")
MAKE_PUBLIC_PLOTS=False

# linear interpolation between each point
def find_nll_bounds(poi_values: np.ndarray, delta_nll: np.ndarray, threshold: float):
    
    bounds = []
    for i_val in range(len(poi_values)-1):

        if (delta_nll[i_val] >= threshold) and (delta_nll[i_val+1] >= threshold):
            continue

        if (delta_nll[i_val] < threshold) and (delta_nll[i_val+1] < threshold):
            continue
        
        # solving for y = ax + b, then inverting
        a = (delta_nll[i_val+1] - delta_nll[i_val])/(poi_values[i_val+1] - poi_values[i_val])
        # equivalent to poi_values[i_val+1] - a*delta_nll[i_val+1]
        b = delta_nll[i_val] - a*poi_values[i_val]

        bounds.append((threshold - b)/a)
    
    return bounds


def load_scan_points(pattern: str) -> np.ndarray:
    """Load and concatenate all .npy scan files matching the glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    records = [np.load(f) for f in files]
    return np.array(records, dtype=records[0].dtype)


def plot_1d_scan(points_list: List[np.ndarray], labels, poi_name: str, out_path: str) -> None:
    """Plot a 1D profile likelihood scan: delta -2logL vs POI value."""

    fig, ax = plt.subplots()
    
    for i_p, points in enumerate(points_list):
        order = np.argsort(points[poi_name])
        poi_values = points[poi_name][order]
        nll_values = points["-2logL"][order]
        delta_nll = nll_values - nll_values.min()

        bounds_68cl = find_nll_bounds(poi_values, delta_nll, 1.0)
        label = f"{labels[i_p]}: "

        #print(f"{labels[i_p]=}, {bounds_68cl=}")
        # if len(bounds_68cl) % 2 == 1:
        #     raise NotImplementedError("Intervals open-ended on one side not implemented yet.")
        if len(bounds_68cl) == 0:
            label += "outside range"
        if len(bounds_68cl) == 1:
            label += f"[{bounds_68cl[0]:.3f},+inf]"
        if len(bounds_68cl) == 2:
            label += f"[{bounds_68cl[0]:.3f},{bounds_68cl[1]:.3f}]"
        if len(bounds_68cl) == 4:
            label += f"[{bounds_68cl[0]:.3f},{bounds_68cl[1]:.3f}] && [{bounds_68cl[2]:.3f},{bounds_68cl[3]:.3f}]"
        if len(bounds_68cl) >= 6:
            raise NotImplementedError("More than two 68% CL bounds, not implemented.")

        ax.plot(poi_values, delta_nll, marker="o", label=label)
        
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.axhline(3.84, color="gray", linestyle=":")
    ax.set_xlabel(poi_name)
    ax.set_ylabel(r"$-2\Delta\ln L$")
    ax.set_ylim(0.0, 6.0)
    ax.legend(frameon=True, title="68%CL", framealpha=1.0)
    hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, ax=ax, loc=0, fontsize=14)
    plt.savefig(out_path+".png")
    print(f"Saved {out_path}")


def plot_2d_scan(points: np.ndarray, poi_names: list[str], out_path: str) -> None:
    """Plot a 2D likelihood scan as a filled contour of delta -2logL."""
    poi_x, poi_y = poi_names
    x_values = points[poi_x]
    y_values = points[poi_y]
    nll_values = points["-2logL"]
    delta_nll = nll_values - nll_values.min()

    fig, ax = plt.subplots()
    contour = ax.tricontourf(x_values, y_values, delta_nll, levels=20)
    ax.tricontour(x_values, y_values, delta_nll, levels=[2.30, 5.99], colors="white")
    fig.colorbar(contour, ax=ax, label=r"$-2\Delta\ln L$")
    ax.set_xlabel(poi_x)
    ax.set_ylabel(poi_y)
    plt.savefig(out_path+".png")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a MultiDimFit.py likelihood scan")
    parser.add_argument("patterns", nargs="+", help="Glob pattern matching the scan .npy files")
    parser.add_argument("--labels", nargs="+", type=str, help="labels for each of the points")
    parser.add_argument("--out", default="scan", help="Output plot path (without file extension)")
    args = parser.parse_args()

    assert len(args.patterns) == len(args.labels)

    points_list = []
    for pattern in args.patterns:
        points = load_scan_points(pattern)
        poi_names = [name for name in points.dtype.names if name != "-2logL"]
        points_list.append(points)
    
    if len(poi_names) == 1:
        plot_1d_scan(points_list, args.labels, poi_names[0], os.path.join(user.plot_directory,"lh_scans",args.out))
    elif len(poi_names) == 2:
        plot_2d_scan(points_list, args.labels, poi_names, os.path.join(user.plot_directory,"lh_scans",args.out))
    else:
        raise ValueError(f"Expected 1 or 2 POIs, found {len(poi_names)}: {poi_names}")
