import os 
import sys
sys.path.insert(0, '..')


import fit.Likelihood as lh 
import pickle  as pck 
import common.user as user 
import json, importlib
from fit.Modeling import Rotated


import numpy as np
import matplotlib.pyplot as plt
import os

import common.syncer as syncer

def impact_table_plot(
    nuis_names, nuis_vals, nuis_errs,
    impact_pos, impact_neg, param_names,
    top_n=None, figsize=(12, 0.35*20), outpath=None,
    name_col_width=0.23, pull_col_width=0.27, poi_col_width=0.4
):
    """
    Create a multi-column impact table plot.
    Inputs:
      - nuis_names : list of strings, length M
      - nuis_vals  : array-like length M (central values)
      - nuis_errs  : array-like length M (uncertainties)
      - impact_pos : array shape (M, N) shifts of POIs when nuisance = +1 sigma
      - impact_neg : array shape (M, N) shifts of POIs when nuisance = -1 sigma
      - poi_names  : list of strings length N
    Options:
      - top_n : show top_n nuisances by max absolute impact (default: all)
      - figsize, outpath : figure size and optional file path to save (if None, show only)
    Returns: path saved (if outpath given) or None.
    """
    # Convert to arrays and validate
    nuis_names = np.asarray(nuis_names, dtype=object)
    nuis_vals = np.asarray(nuis_vals, dtype=float)
    nuis_errs = np.asarray(nuis_errs, dtype=float)
    impact_pos = np.asarray(impact_pos, dtype=float)
    impact_neg = np.asarray(impact_neg, dtype=float)
    param_names = list(param_names)

    M = len(nuis_names)
    if nuis_vals.shape[0] != M or nuis_errs.shape[0] != M:
        raise ValueError("Length of nuis_names, nuis_vals and nuis_errs must match.")
    if impact_pos.shape[0] != M or impact_neg.shape[0] != M:
        raise ValueError("impact_pos and impact_neg must have same first dimension as nuisances")
    if impact_pos.shape != impact_neg.shape:
        raise ValueError("impact_pos and impact_neg must have the same shape")
    N = impact_pos.shape[1]
    if len(param_names) != N:
        print(len(param_names), N)
        raise ValueError("Number of poi_names must equal number of columns in impact arrays")

    # sort based on mean average impact (also average between positive and negative)  
    mags = np.mean(np.abs(impact_pos)+np.abs(impact_neg), axis=1)
    order = np.argsort(mags)

    # Reverse ordering so largest appears at the top
    order = order[::-1]
    names = nuis_names[order]
    vals = nuis_vals[order]
    errs = nuis_errs[order]
    pos = impact_pos[order, :]
    neg = impact_neg[order, :]
    Msel = len(names)

    # Column widths (relative)
    col_widths = np.array([name_col_width, pull_col_width] + [poi_col_width]*N)
    col_widths = col_widths / np.sum(col_widths)

    # Create figure and axes manually to align columns
    fig = plt.figure(figsize=figsize)
    left = 0.02; right = 0.98; bottom = 0.03; top = 0.97
    full_width = right - left
    full_height = top - bottom

    # compute cumulative left positions for axes
    axes = []
    cum = left
    spacing = 0.005
    for w in col_widths:
        wpix = full_width * w
        ax = fig.add_axes([cum, bottom, wpix - spacing, full_height])  # subtract small spacing
        axes.append(ax)
        cum += wpix

    ax_name, ax_pull, *ax_pois = axes

    # Vertical positions
    y = np.arange(Msel)

    # --- Column 1: names ---
    ax_name.set_xlim(0, 1)
    ax_name.set_ylim(-0.5, Msel - 0.5)
    ax_name.invert_yaxis()
    ax_name.axis('off')
    for i, nm in enumerate(names):
        ax_name.text(0.98, i, str(nm), va='center', ha='right', fontsize=9)
    ax_name.set_title("Nuisance", fontsize=10, pad=8)

    # --- Column 2: pulls (value +- uncertainty) ---
    ax_pull.set_ylim(-0.5, Msel - 0.5)
    ax_pull.invert_yaxis()
    ax_pull.set_yticks([])
    # pick symmetric-ish x-limits based on values+errors
    span = max(np.nanmax(np.abs(vals + errs)), np.nanmax(np.abs(vals - errs)), 1e-6)
    lim = max(1.2*span, 0.5)
    ax_pull.set_xlim(-lim, lim)
    ax_pull.errorbar(vals, y, xerr=errs, fmt='o', markersize=4, linestyle='None', capsize=3)
    ax_pull.axvline(0, linestyle='--', linewidth=0.6)
    ax_pull.set_title("Pull (value ± σ)", fontsize=10, pad=8)
    # numeric labels (right side inside the pull axis)
    x_text = ax_pull.get_xlim()[1] * 0.98
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax_pull.text(0.98, i+0.25, f"{v:.3f} ± {e:.3f}", va='center', ha='right',
                      transform=ax_pull.get_yaxis_transform(), fontsize=8)
    ax_pull.tick_params(axis='x', labelsize=8)

    # --- POI columns ---
    for j, ax in enumerate(ax_pois):
        ax.set_ylim(-0.5, Msel - 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])
        col_pos = pos[:, j]
        col_neg = neg[:, j]
        x_min = np.nanmin(np.abs(col_neg))
        x_max = np.nanmax(np.abs(col_pos))
        print(x_min, x_max)
        span = max(abs(x_min), abs(x_max), 1e-6)
        ax.set_xlim(-1.2*span, 1.2*span)
        ax.axvline(0, linestyle='--', linewidth=0.6)
        print(parametersForImpacts[j])
        for i in range(Msel):
            p = col_pos[i]
            n = col_neg[i]

            if p >= 0:
                ax.hlines(i, 0.0, p, linewidth=6, alpha=0.8, colors ='r')
            else:
                ax.hlines(i, p, 0.0, linewidth=6, alpha=0.8, colors ='r')

            if n <= 0:
                ax.hlines(i, n, 0.0, linewidth=6, alpha=0.8)
            else:
                ax.hlines(i, 0.0, n, linewidth=6, alpha=0.8)

        ax.set_title(parametersForImpacts[j], fontsize=10, pad=8)
        ax.xaxis.grid(True, linestyle=':', linewidth=0.4, alpha=0.6)
        ax.tick_params(axis='x', labelsize=8)


    if outpath is not None:
        # ensure directory exists
        dirname = os.path.dirname(outpath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return outpath
    else:
        plt.show()
        plt.close(fig)
        return None

def fit_result_to_dict( result ):
    ret={}
    for param in result['parameters']:
        ret[param['name']]=param
    return ret 


if __name__ == "__main__":
    # ---------------- args ----------------
    import argparse
    p = argparse.ArgumentParser(description="Calculates impacts of individual nuisance parameters and plots.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("configs", nargs="+", help="Path to one or more global YAML configs")
    p.add_argument("--base", help="Base name for fit result and cache directories")
    p.add_argument("--step", default="step0", help="step0: run Asimov fit; step1: run fits fixing each nuisance at ±1 sigma; step2: plot impacts")
    p.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")
    p.add_argument("--freezePOIs", default="")
    p.add_argument("--nuisanceForImpacts", default="", help="Which nuisances to vary for impact plot.")
    p.add_argument("--no_syst", action="store_true", help="Disable all nuisances (freeze to 0).")
    p.add_argument("--syst_only", action="store_true", help="Disable all POIs (freeze to 0).")
    p.add_argument("--verbosity", type=int, default=1, help="Verbosity passed to the fitter")
    p.add_argument("--parametersForImpacts", default="",help="Which non-POI parameters to plot impacts for (step1 and step2).")
    p.add_argument(
        "--minuit",
        action="store_true",
        default=False,
        help="Use the original iminuit/MIGRAD backend instead of the autograd+SciPy backend.")
    p.add_argument(
        "--overwrite",
        nargs="?",
        const="all",
        default=None,
        choices=["fit", "all"],
        help="Overwrite results: 'fit' overwrites fit JSON only; 'all' overwrites fit JSON and cache.",
    )
    p.add_argument(
        "--prepareSlurmJobs",
        action="store_true",
        help="Prepare Slurm scripts for per-nuisance step1+impact jobs and exit",
    )
    p.add_argument(
        '--conda',
        type=str,
        help='location of conda.sh path',
        default='/software/f2022/software/anaconda3/2023.03/etc/profile.d/conda.sh')
    
    args = p.parse_args()

    import common.yaml_loader as yaml_loader 

    list_configs = []
    for config_path in args.configs:
        aux_cfg = yaml_loader.load_yaml(config_path)
        yaml_loader.print_summary(aux_cfg, config_path, yaml_loader._INCLUDE_TRACE)
        yaml_loader.load_surrogates(aux_cfg, config_path, overwrite=False)
        list_configs.append(aux_cfg)

    cfg = yaml_loader.combine_configs(list_configs)

    like_info = lh.load_likelihood(cfg)

    version = str(cfg.get("version", "v0"))
    overwrite_fit = (args.overwrite == "fit") or (args.overwrite == "all")
    overwrite_cache = args.overwrite == "all"
    conda_path = args.conda

    hyp  = lh.build_hypothesis_from_likelihood(like_info, name="SR")

    rotated = bool(args.rotate)
    hyp_for_fit = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

    base_list = []
    for config_path in args.configs:
        base_list.append(os.path.splitext(os.path.basename(config_path))[0])
    base  = "_".join(base_list)

    if args.base:
        base = args.base

    suffix = ("_nosyst" if args.no_syst else "") + ("_rotate" if rotated else "")

    if args.freezePOIs != "":
        suffix += f"_freezePOIs_{args.freezePOIs.replace(',','_')}" 
    if args.syst_only:
        suffix = "_systonly"

    if args.no_syst and args.syst_only:
        raise ValueError("You cannot ask for a fit with --no_syst and --syst_only.")

    if args.no_syst:
        for p_ in hyp.nuisances + hyp_for_fit.nuisances:
            p_.val = 0.0
            p_.isFrozen = True
        print("[opts] --no_syst: all nuisances set to 0 and frozen.")
    elif args.syst_only:
        for p_ in hyp.POIs + hyp_for_fit.POIs:
            # p_.val = 0.0 # do I need this ? I don't think I do.
            p_.isFrozen = True
        print("[opts] --syst_only: all POIs frozen.")

    if args.prepareSlurmJobs:
        if args.step != "step1":
            raise ValueError(f"It only makes sense to set up SLURM jobs for step1. Currently {args.step=}")
        from slurm_utils import prepare_slurm_jobs, get_base_command
        base_cmd = get_base_command()
        prepare_slurm_jobs(
            hyp_for_fit=hyp_for_fit,
            base_cmd=base_cmd,
            base=base,
            version=version,
            conda_path=conda_path
        )
        print("Slurm job files prepared. Exiting.")
        sys.exit(0)

    step = 1.0 if rotated else 0.1

    if args.freezePOIs != "": 
        for poi in args.freezePOIs.split("," ):
            hyp_for_fit.set_nuisance_frozen(poi, True)

    # Make sample loader factory from default cfg
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])

    from common.yaml_loader import _resolve_features_list
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list( default_features ) if default_features else None
    factory     = samples_mod.Factory( 
        features  = features,
        selection = cfg["defaults"].get("default_selection", None),
        selection_features = cfg["defaults"].get("default_selection_features", None),
        )

    n2ll = lh.N2LL(
        like_info,
        factory = factory,
        cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
        cache_root=None,
        overwrite=overwrite_cache,
    )

    n2ll.build_cache()
    n2ll.prepare_runtime()

    output_directory = os.path.join(user.output_directory, f"impacts_{base}_{version}{suffix}")

    if args.step == "step0":
        out_path = os.path.join(output_directory, "impacts_initial_fit.json")
        if os.path.exists(out_path) and (not overwrite_fit):
            print(f"Fit result in {out_path} already exists and did not ask to overwrite, skipping.")
            sys.exit(0)

        n2ll.setAsimov(hyp_for_fit)
        with open(os.path.join(output_directory, "asimov.pck"), 'wb') as outf: # store asimov to pick it in the next steps
            pck.dump(hyp_for_fit, outf)

        fitter = lh.run_iminuit_fit if args.minuit else lh.run_autograd_fit
        m = fitter(n2ll, hyp_for_fit, step=step, print_every=100, do_migrad=True, do_hesse=True,
                   do_minos=False, verbosity=args.verbosity)

        lh.serialize_result(m, base, version, args, out_path)

    elif args.step == "step1": 
        # First, we want the exact same asimov dataset as before
        with open(os.path.join(output_directory, "asimov.pck"), 'rb') as inf: 
            n2ll.setAsimov( pck.load( inf )  ) 
        
        # We now get the results from the previous fit 
        in_path = os.path.join(output_directory, "impacts_initial_fit.json")
        with open(in_path) as inf: 
            initial_fit=json.load( inf )
            initial_fit_dict = fit_result_to_dict(initial_fit)
            
        for param_name in initial_fit_dict:

            # skipping parameters which are not defined as nuisances
            if param_name not in [p.name for p in hyp_for_fit.nuisances]: 
                continue

            # skipping nuisances that user wants to check impacts for
            if args.parametersForImpacts != "" and param_name in args.parametersForImpacts.split(","):
                print(f"skipping parameter {param_name} as it is a parameter for which we want to calculate impacts")
                continue

            if args.nuisanceForImpacts != "" and args.nuisanceForImpacts != param_name: 
                continue

            for direction in [ 'down', 'up']:
                out_path = os.path.join(output_directory, f"{param_name}_{direction}_fit.json")
                if os.path.exists(out_path) and (not overwrite_fit):
                    print(f"Fit result in {out_path} already exists and did not ask to overwrite, skipping.")
                    continue
                
                hyp_var = hyp_for_fit.clone()
                value = initial_fit_dict[param_name]["value"] + initial_fit_dict[param_name]["error"] * (1. if direction == "up" else -1.) 
                print(f"Setting parameter {param_name} to {value}")
                hyp_var.modify(**{param_name: value})
                hyp_var.set_nuisance_frozen(param_name, True)
                fitter = lh.run_iminuit_fit if args.minuit else lh.run_autograd_fit
                m = fitter(n2ll, hyp_var, step=0.01, print_every=100, do_migrad=True, do_hesse=False,
                           do_minos=False, verbosity=args.verbosity)

                lh.serialize_result(m, base, version, args, out_path)
            
    elif args.step == "step2": 
        # Results from the first fit
        in_path = os.path.join(output_directory, "impacts_initial_fit.json")

        with open(in_path) as inf: 
            initial_fit=json.load( inf )
            initial_fit_dict = fit_result_to_dict(initial_fit)

        nuisances_names  = []
        nuisances_values = []
        nuisances_errors = []
        impacts_up=[]
        impacts_dn=[]
        POIs= hyp_for_fit.POIs
        nuisances = hyp_for_fit.nuisances

        parametersForImpacts = [p for p in POIs if not p.isFrozen]

        if args.parametersForImpacts != "":
            parametersForImpacts.extend(
                [p for p in hyp_for_fit.nuisances if p.name in args.parametersForImpacts.split("," ) and not p.isFrozen])

        for p in hyp_for_fit.nuisances:
            param_name = p.name
            param = initial_fit_dict[param_name]

            if args.parametersForImpacts != "" and (param_name in args.parametersForImpacts.split("," )):
                continue
            
            nuisances_names.append( param_name )
            nuisances_values.append( param['value'] )
            nuisances_errors.append( param['error'] )

            def get_impacts_for_nuisance( varied_fit_result ):
                ret=[]
                with open(varied_fit_result) as inf:
                    fit_var=json.load( inf )
                    fit_var_dict = fit_result_to_dict(fit_var)

                for param in parametersForImpacts:
                    if param.name in args.freezePOIs.split("," ): 
                        continue
                    if param.isFrozen:
                        continue

                    ret.append( fit_var_dict[param.name]['value'] - initial_fit_dict[param.name]['value'])
                return ret 
                
            try: 
                impacts_up.append( get_impacts_for_nuisance( os.path.join(output_directory, f"impacts_{param_name}_up.json")))
            except FileNotFoundError:
                print(f"Warning: up variation for {param_name} not found")
                impacts_up.append( [0. for _ in parametersForImpacts])
            try:
                impacts_dn.append( get_impacts_for_nuisance( os.path.join(output_directory, f"impacts_{param_name}_down.json")))
            except FileNotFoundError:
                print(f"Warning: down variation for {param_name} not found")
                impacts_dn.append( [0. for _ in parametersForImpacts])

        plot_outpath = os.path.join(user.plot_directory, "likelihood_fit", base, f"{version}{suffix}", "impacts.png")
        impact_table_plot( nuisances_names, nuisances_values, nuisances_errors,
                           impacts_up, impacts_dn, [p.name for p in parametersForImpacts if p.name not in args.freezePOIs.split("," )], outpath=plot_outpath)


        syncer.sync()
                
                    




            
