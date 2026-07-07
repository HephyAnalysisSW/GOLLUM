"""Compare correlation matrices from two fit JSON files."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import common.user as user
import common.syncer as syncer
from data.plot_options import get_nice_parameter_name

import mplhep
plt.style.use(mplhep.style.CMS)

# better formatting for parameter names
# unnecessary for everyday analysis 
MAKE_PUBLIC_PLOTS = False

def load_correlation(json_path: str) -> tuple[list[str], np.ndarray]:
    """Load correlation order and matrix from a fit JSON file."""
    with open(json_path) as file_handle:
        data = json.load(file_handle)

    correlation = data.get("correlation", {})
    order = correlation.get("order")
    matrix = correlation.get("matrix")

    if order is None or matrix is None:
        raise ValueError(f"Missing correlation data in {json_path}")

    correlation_matrix = np.array(matrix, dtype=float)
    return order, correlation_matrix

def create_comparison_canvas(
    order: list[str],
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    output_dir: str,
    label_a: str,
    label_b: str,
    diff_matrix: np.ndarray
) -> None:
    
    """Create a single-canvas comparison plot for two correlation matrices and their difference."""
    max_abs_diff = float(np.max(np.abs(diff_matrix)))
    diff_range = max_abs_diff if max_abs_diff > 0 else 1.0

    side = max(6.0, 0.25 * len(order))
    fig, axes = plt.subplots(1, 3, figsize=(side * 3.0, side))

    colormap = "coolwarm"

    images = [
        axes[0].imshow(matrix_a, vmin=-1.0, vmax=1.0, cmap=colormap),
        axes[1].imshow(matrix_b, vmin=-1.0, vmax=1.0, cmap=colormap),
        axes[2].imshow(diff_matrix, vmin=-diff_range, vmax=diff_range, cmap=colormap),
    ]

    titles = [label_a, label_b, "Correlation difference"]
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xticks(range(len(order)))
        axis.set_yticks(range(len(order)))
        axis.set_xticklabels(order, rotation=90, fontsize=6)
        axis.set_yticklabels(order, fontsize=6)

    for axis, image in zip(axes, images):
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "correlation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "correlation_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


def create_difference_heatmap(
    order: list[str],
    diff_matrix: np.ndarray,
    output_dir: str,
    label_a: str,
    label_b: str
) -> None:
    """Create a standalone heatmap of the difference correlation matrix."""
    max_abs_diff = float(np.max(np.abs(diff_matrix)))
    diff_range = max_abs_diff if max_abs_diff > 0 else 1.0

    side = max(6.0, 0.25 * len(order))
    fig, ax = plt.subplots(figsize=(side, side))

    image = ax.imshow(diff_matrix, vmin=-diff_range, vmax=diff_range, cmap="coolwarm")
    ax.set_title("Correlation Difference")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))

    if MAKE_PUBLIC_PLOTS:
        ticklabels = [get_nice_parameter_name(param_name) for param_name in order]
    else:
        ticklabels = [param_name.removeprefix("nu_") for param_name in order]

    ax.set_xticklabels(ticklabels, rotation=90, fontsize=6)
    ax.set_yticklabels(ticklabels, fontsize=6)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "correlation_difference.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "correlation_difference.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the correlation comparison CLI."""
    parser = argparse.ArgumentParser(
        description="Compare correlation matrices across two JSON files"
    )

    parser.add_argument(
        'fit_files',
        nargs='+',
        help='Path(s) to fit result JSON file(s)'
    )

    parser.add_argument(
        "-l","--labels",
        nargs="+",
        type=str,
        help="Optional labels for plots. If not given, uses version name."
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (relative to user.plot_directory)",
    )
    args = parser.parse_args()

    if len(args.fit_files) != 2:
        raise ValueError("Need to pass two fit files.")

    fit_a_path = args.fit_files[0]
    fit_b_path = args.fit_files[1]
    
    label_a = os.path.basename(fit_a_path).replace(".json", "")
    label_b = os.path.basename(fit_b_path).replace(".json", "")

    if args.labels:
        if len(args.labels) != 2:
            raise ValueError("If passing labels, should pass both labels")

        label_a = args.labels[0]
        label_b = args.labels[1]
            

    output_name = args.output
    if output_name is None:
        output_name = f"correlation_comparison_{label_a}_vs_{label_b}"

    output_dir = os.path.join(user.plot_directory, "fit_comparison" ,output_name)

    order_a, matrix_a = load_correlation(fit_a_path)
    order_b, matrix_b = load_correlation(fit_b_path)

    if order_a != order_b:
        raise ValueError("Correlation orders differ; comparison aborted")

    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"Matrix shapes differ: {matrix_a.shape} vs {matrix_b.shape}"
        )

    diff_matrix = matrix_a - matrix_b

    create_comparison_canvas(order_a, matrix_a, matrix_b, output_dir, label_a, label_b, diff_matrix)
    create_difference_heatmap(order_a, diff_matrix, output_dir, label_a, label_b)
    
    diff_filename = f"correlation_only_diff_{label_a}_vs_{label_b}.json"
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "correlation": {
            "order": order_a,
            "matrix": diff_matrix.tolist()
        }
    }
    with open(os.path.join(output_dir, diff_filename), "w") as file_handle:
        json.dump(payload, file_handle, indent=2)    

    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
