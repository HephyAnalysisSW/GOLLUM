#!/usr/bin/env python3
"""
Generate toys by directly reweighting the nominal MC events to a target PDF set.

This follows the event-wise PDF reweighting equation from the paper draft:

    w_target = w_nominal * [f_target(x1,id1,Q) * f_target(x2,id2,Q)]
                         / [f_gen(x1,id1,Q)    * f_gen(x2,id2,Q)]

"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import lhapdf
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import common.user as user
import common.yaml_loader as yaml_loader
from common.yaml_loader import _resolve_features_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate toys by direct PDF reweighting to a target LHAPDF set."
    )
    parser.add_argument("config", help="Path to the YAML config.")
    parser.add_argument("target_pdf_set", help="Target LHAPDF set, for example PDF4LHC21_mc.")
    parser.add_argument("--target-member", type=int, default=0, help="Target LHAPDF member. Default: 0.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed. Default: 123.")
    parser.add_argument("--n-toys", type=int, default=1, help="Number of toys to generate. Default: 1.")
    return parser.parse_args()


def build_factory(cfg):
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    return samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )


def find_pdf_job(cfg):
    for region in cfg.get("likelihood", {}).get("regions", []) or []:
        for cls in region.get("classes", []) or []:
            poi = cls.get("POI", {}) or {}
            if poi.get("type") != "bit":
                continue
            job_id = poi.get("job")
            for job in cfg.get("jobs", []) or []:
                if job.get("id") == job_id:
                    return job
    raise RuntimeError("Could not find a BIT job in the config.")


def get_pdf_setup(cfg):
    from pdf.PDFParametrization import PDFParametrization

    job = find_pdf_job(cfg)
    pdf_cfg = job.get("pdf", {}) or {}
    pdf = PDFParametrization(
        n=list(pdf_cfg.get("pdf_n", []) or []),
        typ=pdf_cfg.get("pdf_type"),
        basis=pdf_cfg.get("pdf_basis"),
        active_pids="all",
    )
    return pdf


def iter_region_observer_weight_batches(region, factory):
    """
    Iterate through the region events in the same sample/shard order used by the cache code.

    This is important because the toy indices must line up with the cache order used later
    in fit_toys.py.
    """
    for sample_name in (region.get("classifier", {}) or {}).get("asimov", []):
        loader = factory.get(sample_name)

        old_split = None
        if hasattr(loader, "n_split"):
            old_split = loader.n_split
            loader.n_split = 100

        n_shards = len(getattr(loader, "base", loader))
        for shard in range(n_shards):
            observers, weights = loader.materialize(shard=shard, what="ow", n=None)
            yield loader.observer_names, np.asarray(observers, dtype=np.float64), np.asarray(weights, dtype=np.float64)

        if old_split is not None:
            loader.n_split = old_split


def pdf_values(pdf, x, ids, Q):
    return np.array(
        [entry.get(pid) for entry, pid in zip(pdf.xfxQ(tuple(x), tuple(Q)), ids)],
        dtype=np.float64,
    )


def compute_pdf_reweight_ratio(target_pdf, gen_pdf, x1, x2, id1, id2, Q):
    """
    Direct PDF reweighting:

      ratio = [f_target(x1,id1,Q) * f_target(x2,id2,Q)]
            / [f_ref   (x1,id1,Q) * f_ref   (x2,id2,Q)]

    We use Generator_scalePDF directly, which matches the convention used in the
    existing training code in this repo.
    """
    target_1 = pdf_values(target_pdf, x1, id1, Q)
    target_2 = pdf_values(target_pdf, x2, id2, Q)
    gen_1 = pdf_values(gen_pdf, x1, id1, Q)
    gen_2 = pdf_values(gen_pdf, x2, id2, Q)

    numerator = target_1 * target_2
    denominator = gen_1 * gen_2

    # if the denominator is zero, the result is 0.
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=(denominator != 0.0),
    )


def compute_lambda_for_region(region, factory, target_pdf, reference_pdf):
    print("In compute_lambda_for_region(...)")
    batches = []

    for observer_names, observers, weights in iter_region_observer_weight_batches(region, factory):
        idx = {name: i for i, name in enumerate(observer_names)}

        x1  = observers[:, idx["Generator_x1"]]
        x2  = observers[:, idx["Generator_x2"]]
        id1 = observers[:, idx["Generator_id1"]].astype(np.int32, copy=False)
        id2 = observers[:, idx["Generator_id2"]].astype(np.int32, copy=False)
        Q   = observers[:, idx["Generator_scalePDF"]]

        ratio = compute_pdf_reweight_ratio(target_pdf, reference_pdf, x1, x2, id1, id2, Q)
        # if np.any(ratio > 10):
        #     print("Warning: a very high ratio is detected.")
        #     print(ratio[ratio > 10])
        ratio_max = 100
        # sometimes ratios are as large as 1e9.
        # we ignore such contributions.
        mask = np.isfinite(ratio) & (np.abs(ratio) < ratio_max)
        lam_batch = weights * ratio
        lam_batch[~mask] = 0.0
        batches.append(lam_batch)


    if not batches:
        return np.empty(0, dtype=np.float64)

    print("Done.\n")
    return np.concatenate(batches, axis=0)


def sample_toy_indices_from_lambda_signed(lam, rng):
    lam = np.asarray(lam, np.float64)
    lam_pos = np.clip(lam, 0.0, None)
    lam_neg = np.clip(-lam, 0.0, None)

    tot_pos = float(lam_pos.sum())
    tot_neg = float(lam_neg.sum())
    if (tot_pos + tot_neg) <= 0.0:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    n_pos = int(rng.poisson(tot_pos)) if tot_pos > 0.0 else 0
    n_neg = int(rng.poisson(tot_neg)) if tot_neg > 0.0 else 0

    idxs, ws = [], []
    if n_pos:
        idxs.append(rng.choice(lam.size, size=n_pos, replace=True, p=lam_pos / tot_pos))
        ws.append(np.ones(n_pos, dtype=np.float64))
    if n_neg:
        idxs.append(rng.choice(lam.size, size=n_neg, replace=True, p=lam_neg / tot_neg))
        ws.append(-np.ones(n_neg, dtype=np.float64))

    if not idxs:
        return np.empty(0, np.int64), np.empty(0, np.float64)

    idx = np.concatenate(idxs).astype(np.int64, copy=False)
    w = np.concatenate(ws).astype(np.float64, copy=False)
    perm = rng.permutation(idx.size)
    return idx[perm], w[perm]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)

    pdf_model = get_pdf_setup(cfg)
    target_pdf = lhapdf.mkPDF(args.target_pdf_set, args.target_member)
    reference_pdf = pdf_model.reference_pdf

    print("\n[pdf reweighting]")
    print(f"  target PDF = {args.target_pdf_set}, member {args.target_member}")
    print(f"  source PDF = {pdf_model.reference_pdf_name}, member 0")
    print(f"  basis PDF  = {pdf_model.var_set}")

    factory = build_factory(cfg)
    regions = list((cfg.get("likelihood", {}) or {}).get("regions", []) or [])

    store = {}
    lambda_sums = {}

    for region in regions:
        rid = region["id"]
        lam = compute_lambda_for_region(region, factory, target_pdf, reference_pdf)
        lam_pos = np.clip(lam, 0.0, None)
        lam_neg = np.clip(-lam, 0.0, None)
        lambda_sums[rid] = {
            "sum_pos": float(lam_pos.sum()),
            "sum_neg": float(lam_neg.sum()),
            "n_events": int(lam.size),
        }

        print(f"[toys] rid={rid}  N={lam.size}  sum(lam+)={lam_pos.sum():.6g}  sum(lam-)={lam_neg.sum():.6g}")

        for itoy in range(args.n_toys):
            if itoy % 100 == 0: print('itoy: ', itoy)
            idx, w = sample_toy_indices_from_lambda_signed(lam, rng=rng)
            store[f"toy{itoy:04d}_{rid}_indices"] = idx
            store[f"toy{itoy:04d}_{rid}_weights"] = w

    out_dir = os.path.join(user.output_directory, "toys")
    os.makedirs(out_dir, exist_ok=True)

    toy_stem = f"toys_{args.target_pdf_set}_m{args.target_member}_rw_N{args.n_toys}"
    toy_path = os.path.join(out_dir, f"{toy_stem}.npz")
    meta_path = os.path.join(out_dir, f"{toy_stem}.json")

    np.savez(toy_path, **store)

    meta = {
        "config": os.path.abspath(args.config),
        "toy_generation_mode": "direct_pdf_reweight_eq_3_9",
        "target_pdf_set": args.target_pdf_set,
        "target_member": int(args.target_member),
        "source_pdf_set": pdf_model.reference_pdf_name,
        "basis_pdf_set": pdf_model.var_set,
        "basis_members": list(pdf_model.original_variations),
        "n_toys": int(args.n_toys),
        "seed": int(args.seed),
        "lambda_sums": lambda_sums,
        "toy_file": toy_path,
    }

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=False)

    print(f"\n[toys] Saved {args.n_toys} toys to {toy_path}")
    print(f"[toys] Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
