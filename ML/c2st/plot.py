import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.lines import Line2D

import common.user as user
import common.syncer as syncer
from common.helpers import copyIndexPHP

MAKE_PUBLIC_PLOTS = False

parser = argparse.ArgumentParser(description="Plot the AUC distributions from C2ST.")
# passing test1 and test3
parser.add_argument("--test1", required=True, type=float, help="AUC of test 1 - nominal vs. varied")
parser.add_argument("--test2", required=True, help="Path to the file containing AUC values for test 2 - nominal+varied w/ shuffled labels")
parser.add_argument("--test3", required=True, type=float, help="AUC of test 3 - nominal vs. reweighted using surrogate")
parser.add_argument("--label", help="Plot label, i.e. to identify which NP is being plotted")
parser.add_argument("--output-folder", help="Folder added on top of user.plot_directory/c2st. If not given stores in folder derived from test 2 file name")
args = parser.parse_args()

plt.style.use("petroff10")
hep.style.use("CMS")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

input_file_test2 = args.test2
acc_values = []

with open(input_file_test2, "r") as f:
    for line in f:
        match = re.search(r"weighted_auc=([0-9.]+)", line)
        if match:
            acc_values.append(float(match.group(1)))

if args.output_folder:
    output_dir = os.path.join(user.plot_directory, "c2st", args.output_folder)
else:
    output_folder_stem = os.path.basename(input_file_test2).replace("test2_","").removesuffix(".txt")
    output_dir = os.path.join(user.plot_directory, "c2st", output_folder_stem)

os.makedirs(output_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 8))

ax.hist(
    acc_values,
    bins=15,
    histtype="step",
    color="black",
    linewidth=1.5,
    linestyle="-"
)

ax.axvline(
    x=args.test3,
    color="#ffa90e",
    linestyle="-",
    linewidth=1.5
)

ax.axvline(
    x=args.test1,
    color="#ffa90e",
    linestyle="--",
    linewidth=1.5
)

min_x = min(args.test1, args.test3, min(acc_values))*0.999
max_x = max(args.test1, args.test3, max(acc_values))*1.002
ax.set_xlim(min_x, max_x)

# 所有 tick 位置：间隔 0.0002
xticks = np.arange(min_x, max_x, 0.0004)
ax.set_xticks(xticks)

# 只显示 0.5, 0.502, 0.504, 0.506, 0.508
show_x = np.arange(min_x,max_x, 0.002)
# show_x = np.array([0.498, 0.500, 0.502, 0.504, 0.506, 0.508])

xlabels = []
for x in xticks:
    if np.any(np.isclose(x, show_x, atol=1e-10)):
        if np.isclose(x, 0.5, atol=1e-10):
            xlabels.append("0.5")
        else:
            xlabels.append(f"{x:.3f}")
    else:
        xlabels.append("")
ax.set_xticklabels(xlabels)

ax.set_ylim(0, 163)

yticks = np.arange(0, 161, 4)
ax.set_yticks(yticks)

show_y = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
ylabels = [str(int(y)) if y in show_y else "" for y in yticks]
ax.set_yticklabels(ylabels)

ax.set_xlabel("AUC", fontsize=22, loc="right")
ax.set_ylabel("Count", fontsize=22)

ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    top=True,
    right=True,
    labelsize=20
)

fig.canvas.draw() 

for tick, loc in zip(ax.xaxis.get_major_ticks(), xticks):
    if np.any(np.isclose(loc, show_x, atol=1e-10)):
        tick.tick1line.set_markersize(12)
        tick.tick2line.set_markersize(12)
        tick.tick1line.set_markeredgewidth(1.0)
        tick.tick2line.set_markeredgewidth(1.0)
    else:
        tick.tick1line.set_markersize(5)
        tick.tick2line.set_markersize(5)
        tick.tick1line.set_markeredgewidth(0.8)
        tick.tick2line.set_markeredgewidth(0.8)

for tick, loc in zip(ax.yaxis.get_major_ticks(), yticks):
    if loc in show_y:
        tick.tick1line.set_markersize(12)
        tick.tick2line.set_markersize(12)
        tick.tick1line.set_markeredgewidth(1.0)
        tick.tick2line.set_markeredgewidth(1.0)
    else:
        tick.tick1line.set_markersize(5)
        tick.tick2line.set_markersize(5)
        tick.tick1line.set_markeredgewidth(0.8)
        tick.tick2line.set_markeredgewidth(0.8)

handles = [
    Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="randomized"),
    Line2D([0], [0], color="orange", lw=1.5, linestyle="-", label="reweighted"),
    Line2D([0], [0], color="orange", lw=1.5, linestyle="--", label="not reweighted"),
]

legend = ax.legend(
    handles=handles,
    loc="best",
    # bbox_to_anchor=(0.56, 0.95),
    frameon=False,
    fontsize=20,
    labelspacing=1.2,
    title=args.label, # can receive None (no title)
    title_fontsize = 16,
)

lumi_by_era = {
    "2016APV": 19.50,
    "2016": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "Run 2": 137.62
}

lumi_era = "Run 2"
for era in lumi_by_era:
    # relies on input files being labelled with the era
    # which they should be anyway
    if era in input_file_test2:
        lumi_era = era
        break


hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, ax=ax, year=lumi_era, loc=0, fontsize=14)
plt.savefig(os.path.join(output_dir, "C2ST.pdf"), dpi=1000, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "C2ST.png"), dpi=1000, bbox_inches="tight")

copyIndexPHP(output_dir)