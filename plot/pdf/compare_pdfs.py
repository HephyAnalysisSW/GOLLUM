"""Create a two-panel comparison plot from one or more PDF result CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd

import common.syncer as syncer
import common.user as user
import os


LOGGER = logging.getLogger(__name__)

def parse_series_entry(raw_series: str) -> tuple[str, Path]:
    """Parse one --series entry into csv path and label."""

    pieces = raw_series.split(":")
    if len(pieces) < 2:
        raise ValueError(
            "Each --series entry must be 'label:path'. "
            f"Invalid entry: {raw_series}"
        )

    csv_path = Path(pieces[0]).expanduser()
    label = pieces[1].strip()

    if not label:
        raise ValueError(f"Series label cannot be empty: {raw_series}")

    return label, csv_path


def load_ratio_frame(csv_path: Path) -> pd.DataFrame:
    """Load and validate a CSV file with x, ratio_q16, and ratio_q84 columns."""

    frame = pd.read_csv(csv_path)
    required_columns = {"x", "ratio_q16", "ratio_q84"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path}: {sorted(missing_columns)}. "
            "Expected at least x, ratio_q16, ratio_q84."
        )

    return frame.sort_values("x")


def plot_ratio_band(
    axis: Axes,
    frame: pd.DataFrame,
    color: str,
    line_style: str,
    label: str | None = None,
) -> None:
    """Plot upper and lower ratio quantile curves on a given axis."""

    axis.plot(
        frame["x"],
        frame["ratio_q84"],
        color=color,
        linestyle=line_style,
        linewidth=1.5,
        label=label,
    )
    axis.plot(
        frame["x"],
        frame["ratio_q16"],
        color=color,
        linestyle=line_style,
        linewidth=1.5,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Plot ratio uncertainty bands from multiple CSV result files."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        required=True,
        help=(
            "Series entries in the form path:label. "
            "Colors and line styles are assigned automatically. "
            "Example: 'data/binned.csv:Binned' 'data/unbinned.csv:NSBI unbinned'"
        ),
    )
    parser.add_argument("--output", help="Output image path, wrt user.plot_directory/gluonPDF_comparison/",required=True)
    parser.add_argument("--q_value", default="1.65", help="Value of Q at which the results were derived (for label).")
    parser.add_argument("--zoom-xmin", type=float, default=0.1, help="Zoom panel x minimum.")
    parser.add_argument("--zoom-xmax", type=float, default=0.6, help="Zoom panel x maximum.")
    parser.add_argument("--ymin", type=float, default=0.5, help="Shared y-axis minimum.")
    parser.add_argument("--ymax", type=float, default=1.5, help="Shared y-axis maximum.")
    args = parser.parse_args()
    
    # plt.style.use("seaborn-v0_8-whitegrid")
    # plt.rcParams["axes.unicode_minus"] = False
    figure, (axis_full, axis_zoom) = plt.subplots(
        ncols=2,
        figsize=(11, 3.8),
        sharey=True,
        constrained_layout=True,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    line_styles = ["-", "--", "-.", ":"]

    for series_index, raw_series in enumerate(args.series):
        label, csv_path = parse_series_entry(raw_series)
        color = colors[series_index % len(colors)]
        line_style = line_styles[(series_index // len(colors)) % len(line_styles)]
        LOGGER.info("Loading %s", csv_path)
        frame = load_ratio_frame(csv_path)
        plot_ratio_band(axis_full, frame, color, line_style, label=label)

        zoomed: pd.DataFrame = frame.loc[
            (frame["x"] >= args.zoom_xmin) & (frame["x"] <= args.zoom_xmax)
        ]
        plot_ratio_band(axis_zoom, zoomed, color, line_style)

    axis_full.set_xscale("log")
    axis_full.set_xlim(left=0.005)
    axis_full.set_ylim(args.ymin, args.ymax)
    axis_full.set_xlabel("$x$")
    axis_full.set_ylabel(r"$g/g^{(\mathrm{ref})}(x, Q)$")
    axis_full.grid(True, which="major")
    axis_full.legend(loc="upper center", frameon=True)

    axis_zoom.set_xlim(args.zoom_xmin, args.zoom_xmax)
    axis_zoom.set_xlabel("$x$")
    axis_zoom.grid(True, which="major")
    if args.q_value:
        axis_zoom.text(
            0.03,
            0.93,
            rf"$Q = {args.q_value}$ GeV",
            transform=axis_zoom.transAxes,
            fontsize=16,
            va="top",
            ha="left",
        )

    output_path = os.path.join(user.plot_directory,"gluonPDF_comparison")
    os.makedirs(output_path, exist_ok=True)

    plt.savefig(f"{output_path}/{args.output}.pdf", dpi=200)
    plt.savefig(f"{output_path}/{args.output}.png", dpi=200)
    syncer.sync()