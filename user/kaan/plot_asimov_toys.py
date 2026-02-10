#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import common.yaml_loader as yaml_loader
import common.user as user

import fit.Likelihood as Likelihood


# =========================
# HARD-CODED SETTINGS
# =========================
VERSION      = None              # or e.g. "v1"
OVERWRITE    = False

REGION_ID    = "SR"              # unbinned region id
FEATURE_NAME = "tr_ttbar_pt"          # change to any feature in loader.feature_names

from pathlib import Path
import re

directory = Path(os.path.join(user.output_directory, 'toys'))
TOY_FILES = sorted(f for f in directory.iterdir() if re.match(r"toy_\d+_\.npz$", f.name))
TOY_NAME = Path(TOY_FILES[0]).stem

density_plots = False


def get_asimov_fw(n2ll: Likelihood.N2LL, rid: str):
    """
    Load Asimov features and weights for a given region.
    Concatenates all Asimov samples and shards.

    Returns
    -------
    feat_names : list[str]
    X_all      : np.ndarray, shape (N_region, n_features)
    w_all      : np.ndarray, shape (N_region,)
    """
    region = None
    for R in n2ll.regions:
        if R["id"] == rid:
            region = R
            break
    if region is None:
        raise RuntimeError(f"Region '{rid}' not found in n2ll.regions.")

    X_list = []
    w_list = []
    feat_names_ref = None

    for sname in region["_asimov_samples"]:
        L = getattr(n2ll.samples_mod, sname)
        feat_names = list(getattr(L, "feature_names", []) or [])
        if feat_names_ref is None:
            feat_names_ref = feat_names
        elif feat_names != feat_names_ref: # check feature name consistency across samples
            raise RuntimeError(
                f"Feature mismatch across Asimov samples in region '{rid}'."
            )

        n_shards = len(getattr(L, "base", L))
        for shard in range(n_shards):
            X, w = L.materialize(shard=shard, what="fw", n=None)
            w_list.append(np.asarray(w, dtype=np.float64))
            X_list.append(np.asarray(X, dtype=np.float64))

    if not X_list or not w_list:
        raise RuntimeError(f"No Asimov events found for region '{rid}'.")

    X_all = np.concatenate(X_list, axis=0)
    w_all = np.concatenate(w_list, axis=0)
    return feat_names_ref, X_all, w_all


def main():
    # -------- parse only the CONFIG path --------
    ap = argparse.ArgumentParser(
        description="Plot feature histograms using toy indices (hardcoded settings)."
    )
    ap.add_argument("config", help="Path to global YAML config.")
    args = ap.parse_args()

    CONFIG = args.config

    # --- load config + surrogates ---
    cfg = yaml_loader.load_yaml(CONFIG)
    yaml_loader.print_summary(cfg, CONFIG, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(
        cfg,
        CONFIG,
        overwrite=False,
        prefer_numba=False,
    )

    Likelihood.cfg = cfg
    like_info = Likelihood.load_likelihood(cfg)
    hyp = Likelihood.build_hypothesis_from_likelihood(like_info, name="SR")
    print("\n[Hypothesis] Initial parameters:")
    hyp.print()

    # --- set up N2LL (needed only to access regions and samples) ---
    base = os.path.splitext(os.path.basename(CONFIG))[0]
    version = VERSION or str(cfg.get("version", "v0"))
    cache_dir = os.path.join("NN2LCache", base, version)

    n2ll = Likelihood.N2LL(
        likelihood=like_info,
        module_samples="data.samples",
        cache_subdir=cache_dir,
        cache_root=None,
        overwrite=OVERWRITE,
    )
    n2ll.build_cache()
    n2ll.prepare_runtime()

    rid = REGION_ID
    print(f"\n[info] Using region: {rid}")

    # --- rebuild Asimov features for this region ---
    feat_names, X_all, w_all = get_asimov_fw(n2ll, rid)
    print('sum(w_all): ', np.sum(w_all))
    print(f"[info] Asimov X_all.shape = {X_all.shape}")
    print(f"[info] Available features: {feat_names}")


    for FEATURE_NAME in feat_names:
        if FEATURE_NAME in ['tr_top_eta', 'tr_topBar_eta', 'tr_ttbar_dEta', 'tr_ttbar_dAbsEta', 'recoLep_dEta', 'recoLep_dAbsEta']:
            BINS = np.linspace(-5, 5, 101)
        elif FEATURE_NAME in ['tr_ttbar_pt', 'tr_top_pt', 'tr_topBar_pt', 'recoLep0_pt', 'recoLep1_pt', 'recoLepPos_pt', 'recoLepNeg_pt', 'recoLep01_pt']:
            BINS = np.linspace(0, 800, 101)
        elif FEATURE_NAME in ['tr_ttbar_mass',]:
            BINS = np.linspace(200, 1000, 200)
        else:
            BINS = np.linspace(0, 400, 100)
        

        feat_idx = feat_names.index(FEATURE_NAME)
        x_asimov = X_all[:, feat_idx]

        x_toy_list = []
        for TOY_FILE in TOY_FILES:
            # --- load toy indices from npz ---
            key = f"{rid}_indices"
            with np.load(TOY_FILE) as data:
                if key not in data:
                    raise RuntimeError(f"Toy file '{TOY_FILE}' does not contain key '{key}'.")
                indices = np.asarray(data[key], dtype=np.int64)

            x_toy = x_asimov[indices]
            x_toy_list.append(x_toy)


        # --- plot ---
        plt.figure(figsize=(7, 5))
        plt.hist(
            x_asimov,
            weights=w_all,
            bins=BINS,
            density=density_plots,
            # color="#01153e",
            color='#022651',
            histtype="stepfilled",
            alpha=.9,
            label="Asimov (all events)",
        )
        plt.hist(
            x_toy,
            weights=np.ones_like(x_toy),
            bins=BINS,
            density=density_plots,
            color='#94cfff',
            histtype="step",
            linewidth=1.5,
            label=f"Toy (N={len(indices)})",
        )

        plt.xlabel(FEATURE_NAME)
        plt.ylabel("Events")
        plt.title(f"Region {rid}")
        plt.legend()
        plt.tight_layout()

        plot_save_dir = os.path.join(user.output_directory, TOY_NAME)
        os.makedirs(plot_save_dir, exist_ok=True)
        

        out_path = os.path.join(
            plot_save_dir,
            f"toy_{rid}_{FEATURE_NAME}.png",
        )

        plt.savefig(out_path)
        plt.close()
        print(f"[plot] Saved {out_path}")


if __name__ == "__main__":
    main()
