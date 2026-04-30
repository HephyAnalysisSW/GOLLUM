import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import os, sys
import mplhep
plt.style.use(mplhep.style.CMS)

import common.user as user
import common.helpers as helpers
import common.syncer as syncer
from data.plot_options import get_nice_parameter_name

MAKE_PUBLIC_PLOTS = True

def load_fit_results(json_path: str) -> Tuple[str, List[Dict]]:
    """Load parameter values and errors from fit JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    version = data['version']
    param_order = data['free_parameter_order']
    params_by_name = {p['name']: p for p in data['parameters']}
    
    ordered_params = [params_by_name[name] for name in param_order]
    return version, ordered_params


def create_comparison_plot(
    fit_files: List[str],
    labels: Optional[List[str]],
    output_dir: str,
    blind_params: Optional[List[str]] = None,
) -> None:
    """Create split comparison plots for CMS and non-CMS fit parameters."""
    
    blind_params = blind_params or []
    
    fit_data = []
    for fit_file in fit_files:
        version, params = load_fit_results(fit_file)
        fit_data.append((version, params))

    os.makedirs(output_dir, exist_ok=True)

    num_fits = len(fit_data)
    all_param_names = [p['name'] for p in fit_data[0][1]]
    cms_param_names = [param_name for param_name in all_param_names if 'CMS' in param_name]
    other_param_names = [param_name for param_name in all_param_names if 'CMS' not in param_name]

    def save_plot_for_parameters(param_names: List[str], output_basename: str) -> None:
        """Create and save a comparison plot for a specific parameter subset."""
        if not param_names:
            return

        num_params = len(param_names)
        fig, ax = plt.subplots(figsize=(12, max(10, num_params * 0.3)))

        y_positions = np.arange(num_params)
        bar_height = 0.8 / num_fits
        # from https://matplotlib.org/stable/gallery/color/color_sequences.html
        colors = plt.color_sequences['tab10']

        # for smart plot range
        x_min, x_max = -1.0, 1.0
        for fit_idx, (version, params) in enumerate(fit_data):
            params_by_name = {param['name']: param for param in params}
            values = []
            errors = []

            for param_name in param_names:
                param = params_by_name[param_name]

                if param_name in blind_params:
                    values.append(0)
                    errors.append(0)
                else:
                    values.append(param['value'])
                    # NB: will crash for asymmetric errors
                    errors.append(param['error'])
                    lo = param['value']-param['error']
                    hi = param['value']+param['error']

                    if lo < x_min:
                        x_min = lo
                    if hi > x_max:
                        x_max = hi

            y_offset = y_positions + (num_fits - 1) * bar_height / 2 - fit_idx * bar_height

            ax.errorbar(
                values,
                y_offset,
                xerr=errors,
                fmt='o',
                markersize=3,
                linestyle='none',
                label=version if labels is None else labels[fit_idx],
                color=colors[fit_idx],
                ecolor=colors[fit_idx],
                alpha=0.9,
                capsize=3
            )

        # Add horizontal lines between parameters
        for i in range(num_params - 1):
            ax.axhline(y=i + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        ax.set_yticks(y_positions)
        cms_label = "Simulation"
        if MAKE_PUBLIC_PLOTS:
            yticklabels = [get_nice_parameter_name(param_name) for param_name in param_names]
            cms_label += " Preliminary"
        else:
            yticklabels = [param_name.removeprefix("nu_") for param_name in param_names]
            cms_label += " Internal"

        ax.set_yticklabels(yticklabels, fontsize=9)
        ax.set_xlabel('Parameter value', fontsize=11)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        x_range = max(abs(x_min),abs(x_max),1.5)
        ax.set_xlim(-x_range*1.2, x_range*1.2)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Single text call with mathtext to combine bold 'CMS' and italic label
        ax.text(0.0, 1.005, rf"$\bf{{CMS}}\ \mathit{{{cms_label}}}$",
            transform=ax.transAxes, fontsize=13, va='bottom', ha='left')

        lumi_by_era = {
            "2016APV": 19.50,
            "2016": 16.81,
            "2017": 41.48,
            "2018": 59.83,
        }

        lumi = 137.62 # default - Run 2
        for era, era_lumi in lumi_by_era.items():
            if era in version:
                lumi = era_lumi
                break

        ax.text(1.0, 1.005, f'{lumi:.1f} fb$^{{-1}}$ (13 TeV)', transform=ax.transAxes, fontsize=11, va='bottom', ha='right')

        plt.savefig(output_dir + f'/{output_basename}.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir + f'/{output_basename}.pdf', bbox_inches='tight')
        plt.close(fig)

    save_plot_for_parameters(all_param_names, 'fit_comparison_all')
    save_plot_for_parameters(cms_param_names, 'fit_comparison_JME_NPs')
    save_plot_for_parameters(other_param_names, 'fit_comparison_rest')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Compare fit results across multiple JSON files'
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
        '-o', '--output',
        help='Output directory (relative to user.plot_directory)',
    )
    parser.add_argument(
        '-b', '--blind',
        nargs='*',
        default=[],
        help='Parameter names to blind (show as "?" with no error bars)'
    )
    
    args = parser.parse_args()

    if args.labels:
        if len(args.fit_files) != len(args.labels):
            raise ValueError("If giving labels, number of fit files should be equal to number of labels")

    output = args.output

    if output is None:
        list_fit_names = [os.path.basename(fit_file).replace(".json","") for fit_file in args.fit_files]
        output_suffix = "_v_".join(list_fit_names)
        output = f"fit_comparison_{output_suffix}"

    output_dir = os.path.join(user.plot_directory, "fit_comparison",output)
    os.makedirs(output_dir, exist_ok=True)
    helpers.copyIndexPHP(output_dir)
    
    create_comparison_plot(args.fit_files, args.labels, output_dir, args.blind)
    print(f"Plots saved to {output_dir}")