import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})
input_file = "2017.txt"
acc_values = []

with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"weighted_auc=([0-9.]+)", line)
        if match:
            acc_values.append(float(match.group(1)))

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
    x=0.5,
    color="orange",
    linestyle="-",
    linewidth=1.5
)

ax.axvline(
    x=0.5075,
    color="orange",
    linestyle="--",
    linewidth=1.5
)

ax.set_xlim(0.4972, 0.5086)

# 所有 tick 位置：间隔 0.0002
xticks = np.arange(0.4972, 0.5086, 0.0004)
ax.set_xticks(xticks)

# 只显示 0.5, 0.502, 0.504, 0.506, 0.508
show_x = np.array([0.498, 0.500, 0.502, 0.504, 0.506, 0.508])

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
    loc="upper center",
    bbox_to_anchor=(0.56, 0.95),
    frameon=False,
    fontsize=20,
    labelspacing=1.2
)

plt.savefig("trainb_hist.pdf", dpi=1000, bbox_inches="tight")
plt.show()
                                                                   