import os 
import sys
sys.path.insert(0, '..')

import fit.Likelihood as lh 
import pickle  as pck 
import common.user as user 
import json 
from fit.Modeling import Rotated


import numpy as np
import matplotlib.pyplot as plt
import os

def impact_table_plot(
    nuis_names, nuis_vals, nuis_errs,
    impact_pos, impact_neg, poi_names,
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
    poi_names = list(poi_names)

    M = len(nuis_names)
    if nuis_vals.shape[0] != M or nuis_errs.shape[0] != M:
        raise ValueError("Length of nuis_names, nuis_vals and nuis_errs must match.")
    if impact_pos.shape[0] != M or impact_neg.shape[0] != M:
        raise ValueError("impact_pos and impact_neg must have same first dimension as nuisances")
    if impact_pos.shape != impact_neg.shape:
        raise ValueError("impact_pos and impact_neg must have the same shape")
    N = impact_pos.shape[1]
    if len(poi_names) != N:
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
        print(poi_names[j])
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

        ax.set_title(poi_names[j], fontsize=10, pad=8)
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
    p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
    p.add_argument("--step", default="step0")
    p.add_argument("--rotate", action="store_true", help="Rotate?")

    
    args = p.parse_args()

    import common.yaml_loader as yaml_loader 

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    like_info = lh.load_likelihood(cfg)

    base    = os.path.splitext(os.path.basename(args.config))[0] + ("_rotate" if args.rotate else "")
    version = str(cfg.get("version", "v0"))

    hyp  = lh.build_hypothesis_from_likelihood(like_info, name="SR")
    if args.rotate:
        cfg_base_name = os.path.splitext(os.path.basename(args.config))[0]
        hyp_rot = Rotated(hyp, f"/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_merged.json", name="Fisher-basis")
        hyp_rot.print()
        hyp_for_fit = hyp_rot
        step = 1 
    else:
        hyp_for_fit = hyp
        step = 0.1 

    n2ll = lh.N2LL( like_info, 'data.samples',  os.path.join( "NN2LCache",  os.path.splitext(os.path.basename(args.config))[0], cfg['version']), cache_root=None, overwrite=args.overwrite)
    n2ll.build_cache()
    n2ll.prepare_runtime()

    if args.step == "step0": 
        n2ll.setAsimov(hyp_for_fit) 
        with open(f"{base}_{version}_asimov.pck", 'wb') as outf: # store asimov to pick it in the next steps
            pck.dump(hyp_for_fit, outf)
        m = lh.run_minuit_fit(n2ll, hyp_for_fit, step=step, print_every=1, do_migrad=True, do_hesse=True, do_minos=False)

        out_path = os.path.join(user.output_directory, f"{base}_{version}_impacts_initialfit.json")
        args.no_syst = False # to do better. I dont like the no syst option passed to the impact tool
        lh.serialize_result(m, base, version, args, out_path)

    elif args.step == "step1": 
        # First, we want the exact same asimov dataset as before
        with open(f"{base}_{version}_asimov.pck", 'rb') as inf: 
            n2ll.setAsimov( pck.load( inf )  ) 
        
        # We now get the results from the previous fit 
        in_path = os.path.join(user.output_directory, f"{base}_{version}_impacts_initialfit.json")
        with open(in_path) as inf: 
            initial_fit=json.load( inf )
            initial_fit_dict = fit_result_to_dict(initial_fit)
            
        for param_name in initial_fit_dict:
            if param_name not in [p.name for p in hyp_for_fit.nuisances]: 
                continue
            for direction in ['up', 'down']:
                hyp_var = hyp_for_fit.clone()
                value = initial_fit_dict[param_name]["value"] + initial_fit_dict[param_name]["error"] * (1. if direction == "up" else -1.) 
                print(f"Setting parameter {param_name} to {value}")
                hyp_var.modify(**{param_name: value})
                hyp_var.set_nuisance_frozen(param_name, True)
                m = lh.run_minuit_fit(n2ll, hyp_var, step=0.01, print_every=100, do_migrad=True, do_hesse=True, do_minos=False)

                out_path = os.path.join(user.output_directory, f"{base}_{version}_impacts_{param_name}_{direction}.json")
                args.no_syst = False # to do better. I dont like the no syst option passed to the impact tool
                lh.serialize_result(m, base, version, args, out_path)
            
    elif args.step == "step2": 
        # Results from the first fit
        in_path = os.path.join(user.output_directory, f"{base}_{version}_impacts_initialfit.json")

        with open(in_path) as inf: 
            initial_fit=json.load( inf )
            initial_fit_dict = fit_result_to_dict(initial_fit)

        nuisances_names  = []
        nuisances_values = []
        nuisances_errors = []
        impacts_up=[]
        impacts_dn=[]
        POIs= hyp_for_fit.POIs

        for p in hyp_for_fit.nuisances:
            param_name = p.name 
            param = initial_fit_dict[param_name]

            nuisances_names .append( param_name )
            nuisances_values.append( param['value'] )
            nuisances_errors.append( param['error'] )

            def get_impacts_for_nuisance( varied_fit_result ):
                ret=[]
                with open(varied_fit_result) as inf:
                    fit_var=json.load( inf )
                    fit_var_dict = fit_result_to_dict(fit_var)

                for poi in POIs:
                    ret.append( fit_var_dict[poi.name]['value'] - initial_fit_dict[poi.name]['value'])
                return ret 
                
            impacts_up.append( get_impacts_for_nuisance( os.path.join(user.output_directory, f"{base}_{version}_impacts_{param_name}_up.json")))
            impacts_dn.append( get_impacts_for_nuisance( os.path.join(user.output_directory, f"{base}_{version}_impacts_{param_name}_down.json")))
            print( param_name, impacts_up[-1])
                   
            
        impact_table_plot( nuisances_names, nuisances_values, nuisances_errors,
                           impacts_up, impacts_dn, [x.name for x in POIs], outpath=f'{base}_{version}_impacts.png')



                
                    




            
