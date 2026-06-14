#!/usr/bin/env python3

import argparse
import math
import os
import sys

import awkward as ak
import lhapdf
import numpy as np
import ROOT
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
from data.RDataLoader import RDataLoader
from pdf.PODBasis import PODBasis
import eft_reweighting
import samples_postprocessed


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

GIACOMO_MLL_BINS = np.array(
    [
        50,
        64,
        76,
        82,
        86,
        90,
        98,
        103,
        121,
        127,
        130,
        133,
        148,
        151,
        154,
        157,
        163,
        166,
        172,
        178,
        184,
        205,
        210,
        220,
        235,
        240,
        260,
        265,
        325,
        345,
        500,
        530,
        570,
        618,
        654,
        708,
        3000,
    ],
    dtype=np.float64,
)
GIACOMO_ABSY_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.5], dtype=np.float64)
LOW_MASS_MLL_BINS = np.array([60, 70, 80, 86, 91, 96, 106, 120, 133], dtype=np.float64)
LOW_MASS_ABSY_BINS = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.7, 3.0, 3.4], dtype=np.float64)
MLL_BINS = GIACOMO_MLL_BINS
ABSY_BINS = GIACOMO_ABSY_BINS
COSTHETA_BINS = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=np.float64)

SCALAR_BRANCHES = [
    "dy_born_has_candidate",
    "dy_born_mll",
    "dy_born_yll",
    "dy_born_abs_yll",
    "cs_born_costheta",
    "xsec_weight",
    "gen_id1",
    "gen_id2",
    "gen_x1",
    "gen_x2",
    "gen_scalePDF",
]
VECTOR_BRANCHES = ["LHEReweightingWeight"]
WEIGHT_BRANCHES = ["xsec_weight"]
DEFAULT_SELECTION = "(dy_born_has_candidate > 0)"

QPM_BASIS = ["uplus", "dplus", "splus", "cplus", "uminus", "dminus", "sminus"]
QPM_TEX = {
    "uplus": "u^{+}=u+#bar{u}",
    "dplus": "d^{+}=d+#bar{d}",
    "splus": "s^{+}=s+#bar{s}",
    "cplus": "c^{+}=c+#bar{c}",
    "uminus": "u^{-}=u-#bar{u}",
    "dminus": "d^{-}=d-#bar{d}",
    "sminus": "s^{-}=s-#bar{s}",
    "g": "g",
}
QPM_COLORS = {
    "uplus": ROOT.kBlue + 1,
    "uminus": ROOT.kBlue + 1,
    "dplus": ROOT.kRed + 1,
    "dminus": ROOT.kRed + 1,
    "splus": ROOT.kGreen + 2,
    "sminus": ROOT.kGreen + 2,
    "cplus": ROOT.kOrange + 7,
    "g": ROOT.kBlack,
}
QPM_STYLES = {
    "uplus": 1,
    "dplus": 1,
    "splus": 1,
    "cplus": 1,
    "uminus": 2,
    "dminus": 2,
    "sminus": 2,
    "g": 3,
}
QPM_MARKERS = {
    "uplus": 20,
    "dplus": 21,
    "splus": 22,
    "cplus": 33,
    "uminus": 24,
    "dminus": 25,
    "sminus": 26,
    "g": 27,
}

POD_COLOR_CYCLE = [
    ROOT.kBlack,
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kCyan + 2,
    ROOT.kOrange + 7,
    ROOT.kViolet + 1,
    ROOT.kAzure + 7,
    ROOT.kPink + 7,
]
POD_MARKER_CYCLE = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34]


def sanitize(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def epsilon_filename(epsilon):
    return f"eps{epsilon:.3f}".replace(".", "p").replace("-", "m")


def parse_indices(text):
    indices = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            step = 1 if hi >= lo else -1
            indices.extend(range(lo, hi + step, step))
        else:
            indices.append(int(item))
    if not indices:
        raise ValueError("--pod-indices must contain at least one integer")
    if any(i <= 0 for i in indices):
        raise ValueError("POD indices are 1-based and must be positive")
    return indices


def indices_label(indices):
    if not indices:
        return "idxNone"
    contiguous = indices == list(range(indices[0], indices[-1] + 1))
    if contiguous and len(indices) > 2:
        return f"idx{indices[0]}to{indices[-1]}"
    if len(indices) <= 12:
        return "idx" + "_".join(str(i) for i in indices)
    return f"idx{len(indices)}indices"


def selected_files(component, small=None, max_files=None):
    files = component.files
    if small:
        files = files[::small]
    if max_files is not None:
        files = files[:max_files]
    return files


def make_loader(name, files, selection, files_per_chunk):
    required = list(dict.fromkeys(SCALAR_BRANCHES + VECTOR_BRANCHES))
    if not files:
        raise RuntimeError(f"No complete input files for {name}")
    n_split = max(1, int(math.ceil(len(files) / float(files_per_chunk))))
    loader = RDataLoader(
        input_paths=files,
        tree_name="Events",
        branches=required,
        selection=None,
        n_split=n_split,
        splitting_strategy="files",
        strict_branches=True,
        weight_branches=WEIGHT_BRANCHES,
        feature_names=SCALAR_BRANCHES,
        observer_names=[],
    )
    loader.addSelection(selection, required_branches=SCALAR_BRANCHES)
    loader.name = name
    return loader


def pdf_f(pdf, pid, x, q):
    if not np.isfinite(x) or not np.isfinite(q) or x <= 0.0 or x >= 1.0 or q <= 0.0:
        return np.nan
    return pdf.xfxQ(int(pid), float(x), float(q)) / float(x)


def nominal_physical_pdfs(pdf, x, q):
    return {
        21: pdf_f(pdf, 21, x, q),
        2: pdf_f(pdf, 2, x, q),
        -2: pdf_f(pdf, -2, x, q),
        1: pdf_f(pdf, 1, x, q),
        -1: pdf_f(pdf, -1, x, q),
        3: pdf_f(pdf, 3, x, q),
        -3: pdf_f(pdf, -3, x, q),
        4: pdf_f(pdf, 4, x, q),
        -4: pdf_f(pdf, -4, x, q),
        5: pdf_f(pdf, 5, x, q),
        -5: pdf_f(pdf, -5, x, q),
    }


def varied_pdf_value_qpm(phys, pid, basis_name, sign, epsilon):
    pid = int(pid)
    if pid not in phys:
        return np.nan, np.nan
    if pid in [5, -5] or (pid == 21 and basis_name != "g"):
        return phys[pid], phys[pid]

    u, ubar = phys[2], phys[-2]
    d, dbar = phys[1], phys[-1]
    s, sbar = phys[3], phys[-3]
    c, cbar = phys[4], phys[-4]
    b, bbar = phys[5], phys[-5]
    gluon = phys[21]
    if not all(np.isfinite(x) for x in [u, ubar, d, dbar, s, sbar, c, cbar, b, bbar, gluon]):
        return np.nan, np.nan

    qpm = {
        "uplus": u + ubar,
        "dplus": d + dbar,
        "splus": s + sbar,
        "cplus": c + cbar,
        "uminus": u - ubar,
        "dminus": d - dbar,
        "sminus": s - sbar,
        "cminus": c - cbar,
        "g": gluon,
    }
    qpm[basis_name] *= 1.0 + sign * epsilon
    varied = {
        21: qpm["g"],
        2: 0.5 * (qpm["uplus"] + qpm["uminus"]),
        -2: 0.5 * (qpm["uplus"] - qpm["uminus"]),
        1: 0.5 * (qpm["dplus"] + qpm["dminus"]),
        -1: 0.5 * (qpm["dplus"] - qpm["dminus"]),
        3: 0.5 * (qpm["splus"] + qpm["sminus"]),
        -3: 0.5 * (qpm["splus"] - qpm["sminus"]),
        4: 0.5 * (qpm["cplus"] + qpm["cminus"]),
        -4: 0.5 * (qpm["cplus"] - qpm["cminus"]),
        5: b,
        -5: bbar,
    }
    return phys[pid], varied[pid]


def pdf_ratios(pdf, pid1, pid2, x1, x2, q, basis_name, epsilon):
    phys1 = nominal_physical_pdfs(pdf, x1, q)
    phys2 = nominal_physical_pdfs(pdf, x2, q)
    f1_nom_p, f1_plus = varied_pdf_value_qpm(phys1, pid1, basis_name, +1, epsilon)
    f2_nom_p, f2_plus = varied_pdf_value_qpm(phys2, pid2, basis_name, +1, epsilon)
    f1_nom_m, f1_minus = varied_pdf_value_qpm(phys1, pid1, basis_name, -1, epsilon)
    f2_nom_m, f2_minus = varied_pdf_value_qpm(phys2, pid2, basis_name, -1, epsilon)
    valid = (
        np.isfinite(f1_nom_p)
        and np.isfinite(f2_nom_p)
        and np.isfinite(f1_nom_m)
        and np.isfinite(f2_nom_m)
        and np.isfinite(f1_plus)
        and np.isfinite(f2_plus)
        and np.isfinite(f1_minus)
        and np.isfinite(f2_minus)
        and f1_nom_p > 0.0
        and f2_nom_p > 0.0
        and f1_nom_m > 0.0
        and f2_nom_m > 0.0
    )
    if not valid:
        return np.nan, np.nan
    return (f1_plus / f1_nom_p) * (f2_plus / f2_nom_p), (f1_minus / f1_nom_m) * (f2_minus / f2_nom_m)


def pod_name(index):
    return f"pod{index}"


def pod_index_from_name(name):
    if not name.startswith("pod"):
        raise ValueError(f"Not a POD basis label: {name}")
    return int(name[3:])


def pod_ratios(pod, pid1, pid2, x1, x2, q, epsilon):
    coeff0 = np.zeros(pod.nvariations, dtype=np.float64)
    denom = pod(x1, x2, pid1, pid2, coeff0, q)

    coeff_plus = np.zeros(pod.nvariations, dtype=np.float64)
    coeff_minus = np.zeros(pod.nvariations, dtype=np.float64)
    coeff_plus[0] = epsilon
    coeff_minus[0] = -epsilon

    plus = pod(x1, x2, pid1, pid2, coeff_plus, q)
    minus = pod(x1, x2, pid1, pid2, coeff_minus, q)
    valid = np.isfinite(denom) & np.isfinite(plus) & np.isfinite(minus) & (denom != 0.0)
    r_plus = np.full_like(denom, np.nan, dtype=np.float64)
    r_minus = np.full_like(denom, np.nan, dtype=np.float64)
    r_plus[valid] = plus[valid] / denom[valid]
    r_minus[valid] = minus[valid] / denom[valid]
    return r_plus, r_minus


def unroll_triple_index(mass_bin, abs_y_bin, costheta_bin):
    n_m = len(MLL_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    return (abs_y_bin * n_c + costheta_bin) * n_m + mass_bin


def unroll_yield_index(mass_bin, abs_y_bin, costheta_bin, yield_mode):
    if yield_mode == "mll_y_costheta":
        return unroll_triple_index(mass_bin, abs_y_bin, costheta_bin)
    if yield_mode == "mll_y":
        return unroll_afb_index(mass_bin, abs_y_bin)
    if yield_mode == "mll":
        return mass_bin
    raise RuntimeError(f"Unknown yield mode '{yield_mode}'")


def unroll_afb_index(mass_bin, abs_y_bin):
    n_m = len(MLL_BINS) - 1
    return abs_y_bin * n_m + mass_bin


def make_accumulators(basis_names, yield_mode):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    if yield_mode == "mll_y_costheta":
        n_yield = n_y * n_c * n_m
    elif yield_mode == "mll_y":
        n_yield = n_y * n_m
    elif yield_mode == "mll":
        n_yield = n_m
    else:
        raise RuntimeError(f"Unknown yield mode '{yield_mode}'")
    n_afb = n_y * n_m
    accum = {
        "nom_yield": np.zeros(n_yield, dtype=np.float64),
        "nom_afb_sumw": np.zeros(n_afb, dtype=np.float64),
        "nom_afb_sum_sign": np.zeros(n_afb, dtype=np.float64),
        "yield_plus": {name: np.zeros(n_yield, dtype=np.float64) for name in basis_names},
        "yield_minus": {name: np.zeros(n_yield, dtype=np.float64) for name in basis_names},
        "afb_sumw_plus": {name: np.zeros(n_afb, dtype=np.float64) for name in basis_names},
        "afb_sumw_minus": {name: np.zeros(n_afb, dtype=np.float64) for name in basis_names},
        "afb_sum_sign_plus": {name: np.zeros(n_afb, dtype=np.float64) for name in basis_names},
        "afb_sum_sign_minus": {name: np.zeros(n_afb, dtype=np.float64) for name in basis_names},
    }
    return accum


def accumulate_qpm(loader, pdf, basis_names, epsilon, accum, yield_mode, max_events=None):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    total_selected = 0
    total_binned = 0
    total_processed = 0
    total_empty_lhe_weights = 0
    invalid = {name: 0 for name in basis_names}
    used = {name: 0 for name in basis_names}
    supported_pids = {21, 2, -2, 1, -1, 3, -3, 4, -4, 5, -5}

    for ishard in tqdm(range(len(loader)), desc=f"PDF response chunks {loader.name}", unit="chunk"):
        ar = loader.load_selection_shard(ishard)
        if len(ar) == 0:
            continue
        mll = ak.to_numpy(ar["dy_born_mll"])
        yll = ak.to_numpy(ar["dy_born_yll"])
        abs_yll = ak.to_numpy(ar["dy_born_abs_yll"])
        costheta = ak.to_numpy(ar["cs_born_costheta"])
        nominal_xsec = ak.to_numpy(ar["xsec_weight"])
        gen_id1 = ak.to_numpy(ar["gen_id1"])
        gen_id2 = ak.to_numpy(ar["gen_id2"])
        gen_x1 = ak.to_numpy(ar["gen_x1"])
        gen_x2 = ak.to_numpy(ar["gen_x2"])
        gen_q = ak.to_numpy(ar["gen_scalePDF"])
        lhe_weights = loader.vector_branch(ar, "LHEReweightingWeight")
        n_lhe_weights = ak.to_numpy(ak.num(lhe_weights, axis=1))

        mass_bin = np.searchsorted(MLL_BINS, mll, side="right") - 1
        abs_y_bin = np.searchsorted(ABSY_BINS, abs_yll, side="right") - 1
        costheta_bin = np.searchsorted(COSTHETA_BINS, costheta, side="right") - 1
        signed_costheta = np.sign(yll) * costheta

        valid_common = (
            np.isfinite(mll)
            & np.isfinite(nominal_xsec)
            & np.isfinite(gen_id1)
            & np.isfinite(gen_id2)
            & np.isfinite(gen_x1)
            & np.isfinite(gen_x2)
            & np.isfinite(gen_q)
            & (mll > -998)
            & (mass_bin >= 0)
            & (mass_bin < n_m)
        )
        valid_yield = valid_common.copy()
        if yield_mode in ("mll_y", "mll_y_costheta"):
            valid_yield = valid_yield & np.isfinite(abs_yll) & (abs_yll > -998) & (abs_y_bin >= 0) & (abs_y_bin < n_y)
        if yield_mode == "mll_y_costheta":
            valid_yield = valid_yield & np.isfinite(costheta) & (costheta > -998) & (costheta_bin >= 0) & (costheta_bin < n_c)

        valid_afb = (
            valid_common
            & np.isfinite(yll)
            & np.isfinite(abs_yll)
            & np.isfinite(costheta)
            & (abs_yll > -998)
            & (costheta > -998)
            & (np.sign(yll) != 0)
            & (abs_y_bin >= 0)
            & (abs_y_bin < n_y)
        )
        valid = valid_yield | valid_afb
        empty_lhe_weights = valid & (n_lhe_weights <= 0)
        total_empty_lhe_weights += int(np.count_nonzero(empty_lhe_weights))
        valid = valid & (n_lhe_weights > 0)
        valid_yield = valid_yield & (n_lhe_weights > 0)
        valid_afb = valid_afb & (n_lhe_weights > 0)
        total_selected += len(mll)
        total_binned += int(np.count_nonzero(valid_yield))
        if not np.any(valid):
            continue

        sm_eft_weight = eft_reweighting.eft_weight(lhe_weights[valid], config="auto")
        nominal = nominal_xsec[valid] * sm_eft_weight
        finite_nominal = np.isfinite(nominal)
        if not np.any(finite_nominal):
            continue

        valid_indices = np.flatnonzero(valid)[finite_nominal]
        nominal = nominal[finite_nominal]
        if max_events is not None:
            remaining = max_events - total_processed
            if remaining <= 0:
                break
            valid_indices = valid_indices[:remaining]
            nominal = nominal[:remaining]

        yield_mask = valid_yield[valid_indices]
        afb_mask = valid_afb[valid_indices]
        yield_idx = unroll_yield_index(
            mass_bin[valid_indices[yield_mask]],
            abs_y_bin[valid_indices[yield_mask]],
            costheta_bin[valid_indices[yield_mask]],
            yield_mode,
        )
        afb_idx = unroll_afb_index(mass_bin[valid_indices[afb_mask]], abs_y_bin[valid_indices[afb_mask]])
        a4_basis_valid = signed_costheta[valid_indices[afb_mask]]

        np.add.at(accum["nom_yield"], yield_idx, nominal[yield_mask])
        np.add.at(accum["nom_afb_sumw"], afb_idx, nominal[afb_mask])
        np.add.at(accum["nom_afb_sum_sign"], afb_idx, nominal[afb_mask] * a4_basis_valid)

        for iev, idx in enumerate(valid_indices):
            pid1 = int(gen_id1[idx])
            pid2 = int(gen_id2[idx])
            if pid1 not in supported_pids or pid2 not in supported_pids:
                for basis_name in basis_names:
                    invalid[basis_name] += 1
                continue
            for basis_name in basis_names:
                r_plus, r_minus = pdf_ratios(
                    pdf,
                    pid1,
                    pid2,
                    gen_x1[idx],
                    gen_x2[idx],
                    gen_q[idx],
                    basis_name,
                    epsilon,
                )
                if not np.isfinite(r_plus) or not np.isfinite(r_minus):
                    invalid[basis_name] += 1
                    continue
                wp = nominal[iev] * r_plus
                wm = nominal[iev] * r_minus
                if yield_mask[iev]:
                    iyield = unroll_yield_index(mass_bin[idx], abs_y_bin[idx], costheta_bin[idx], yield_mode)
                    accum["yield_plus"][basis_name][iyield] += wp
                    accum["yield_minus"][basis_name][iyield] += wm
                if afb_mask[iev]:
                    iafb = unroll_afb_index(mass_bin[idx], abs_y_bin[idx])
                    a4_basis = signed_costheta[idx]
                    accum["afb_sumw_plus"][basis_name][iafb] += wp
                    accum["afb_sumw_minus"][basis_name][iafb] += wm
                    accum["afb_sum_sign_plus"][basis_name][iafb] += wp * a4_basis
                    accum["afb_sum_sign_minus"][basis_name][iafb] += wm * a4_basis
                used[basis_name] += 1

        total_processed += len(valid_indices)
        if max_events is not None and total_processed >= max_events:
            break

    print(f"[pdf response] selected events after loader selection: {total_selected}")
    print(f"[pdf response] events inside hard-coded unrolling bins: {total_binned}")
    print(f"[pdf response] processed events: {total_processed}")
    if total_empty_lhe_weights:
        print(f"[pdf response] skipped events with empty LHEReweightingWeight: {total_empty_lhe_weights}")
    for basis_name in basis_names:
        total = used[basis_name] + invalid[basis_name]
        frac = invalid[basis_name] / total if total else 0.0
        print(f"[pdf response] {loader.name} {basis_name}: used={used[basis_name]} invalid={invalid[basis_name]} invalid_fraction={frac:.4g}")
    return total_selected, total_binned, total_processed


def accumulate_pod(loader, pods, basis_names, epsilon, accum, yield_mode, max_events=None):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_c = len(COSTHETA_BINS) - 1
    total_selected = 0
    total_binned = 0
    total_processed = 0
    total_empty_lhe_weights = 0
    invalid = {name: 0 for name in basis_names}
    used = {name: 0 for name in basis_names}
    supported_pids = set(PODBasis.all_pdg_ids)

    for ishard in tqdm(range(len(loader)), desc=f"POD response chunks {loader.name}", unit="chunk"):
        ar = loader.load_selection_shard(ishard)
        if len(ar) == 0:
            continue
        mll = ak.to_numpy(ar["dy_born_mll"])
        yll = ak.to_numpy(ar["dy_born_yll"])
        abs_yll = ak.to_numpy(ar["dy_born_abs_yll"])
        costheta = ak.to_numpy(ar["cs_born_costheta"])
        nominal_xsec = ak.to_numpy(ar["xsec_weight"])
        gen_id1 = ak.to_numpy(ar["gen_id1"]).astype(np.int32)
        gen_id2 = ak.to_numpy(ar["gen_id2"]).astype(np.int32)
        gen_x1 = ak.to_numpy(ar["gen_x1"]).astype(np.float64)
        gen_x2 = ak.to_numpy(ar["gen_x2"]).astype(np.float64)
        gen_q = ak.to_numpy(ar["gen_scalePDF"]).astype(np.float64)
        lhe_weights = loader.vector_branch(ar, "LHEReweightingWeight")
        n_lhe_weights = ak.to_numpy(ak.num(lhe_weights, axis=1))

        mass_bin = np.searchsorted(MLL_BINS, mll, side="right") - 1
        abs_y_bin = np.searchsorted(ABSY_BINS, abs_yll, side="right") - 1
        costheta_bin = np.searchsorted(COSTHETA_BINS, costheta, side="right") - 1
        signed_costheta = np.sign(yll) * costheta

        valid_common = (
            np.isfinite(mll)
            & np.isfinite(nominal_xsec)
            & np.isfinite(gen_id1)
            & np.isfinite(gen_id2)
            & np.isfinite(gen_x1)
            & np.isfinite(gen_x2)
            & np.isfinite(gen_q)
            & (mll > -998)
            & (mass_bin >= 0)
            & (mass_bin < n_m)
        )
        valid_yield = valid_common.copy()
        if yield_mode in ("mll_y", "mll_y_costheta"):
            valid_yield = valid_yield & np.isfinite(abs_yll) & (abs_yll > -998) & (abs_y_bin >= 0) & (abs_y_bin < n_y)
        if yield_mode == "mll_y_costheta":
            valid_yield = valid_yield & np.isfinite(costheta) & (costheta > -998) & (costheta_bin >= 0) & (costheta_bin < n_c)

        valid_afb = (
            valid_common
            & np.isfinite(yll)
            & np.isfinite(abs_yll)
            & np.isfinite(costheta)
            & (abs_yll > -998)
            & (costheta > -998)
            & (np.sign(yll) != 0)
            & (abs_y_bin >= 0)
            & (abs_y_bin < n_y)
        )
        valid = valid_yield | valid_afb
        empty_lhe_weights = valid & (n_lhe_weights <= 0)
        total_empty_lhe_weights += int(np.count_nonzero(empty_lhe_weights))
        valid = valid & (n_lhe_weights > 0)
        valid_yield = valid_yield & (n_lhe_weights > 0)
        valid_afb = valid_afb & (n_lhe_weights > 0)
        total_selected += len(mll)
        total_binned += int(np.count_nonzero(valid_yield))
        if not np.any(valid):
            continue

        sm_eft_weight = eft_reweighting.eft_weight(lhe_weights[valid], config="auto")
        nominal = nominal_xsec[valid] * sm_eft_weight
        finite_nominal = np.isfinite(nominal)
        if not np.any(finite_nominal):
            continue

        valid_indices = np.flatnonzero(valid)[finite_nominal]
        nominal = nominal[finite_nominal]
        if max_events is not None:
            remaining = max_events - total_processed
            if remaining <= 0:
                break
            valid_indices = valid_indices[:remaining]
            nominal = nominal[:remaining]

        yield_mask = valid_yield[valid_indices]
        afb_mask = valid_afb[valid_indices]
        yield_idx = unroll_yield_index(
            mass_bin[valid_indices[yield_mask]],
            abs_y_bin[valid_indices[yield_mask]],
            costheta_bin[valid_indices[yield_mask]],
            yield_mode,
        )
        afb_idx = unroll_afb_index(mass_bin[valid_indices[afb_mask]], abs_y_bin[valid_indices[afb_mask]])
        a4_basis_valid = signed_costheta[valid_indices[afb_mask]]

        np.add.at(accum["nom_yield"], yield_idx, nominal[yield_mask])
        np.add.at(accum["nom_afb_sumw"], afb_idx, nominal[afb_mask])
        np.add.at(accum["nom_afb_sum_sign"], afb_idx, nominal[afb_mask] * a4_basis_valid)

        pid1 = gen_id1[valid_indices]
        pid2 = gen_id2[valid_indices]
        supported = np.isin(pid1, list(supported_pids)) & np.isin(pid2, list(supported_pids))
        for basis_name in basis_names:
            r_plus = np.full(len(valid_indices), np.nan, dtype=np.float64)
            r_minus = np.full(len(valid_indices), np.nan, dtype=np.float64)
            if np.any(supported):
                rp, rm = pod_ratios(
                    pods[basis_name],
                    pid1[supported],
                    pid2[supported],
                    gen_x1[valid_indices][supported],
                    gen_x2[valid_indices][supported],
                    gen_q[valid_indices][supported],
                    epsilon,
                )
                r_plus[supported] = rp
                r_minus[supported] = rm
            good = np.isfinite(r_plus) & np.isfinite(r_minus)
            invalid[basis_name] += int(np.count_nonzero(~good))
            used[basis_name] += int(np.count_nonzero(good))
            if not np.any(good):
                continue
            yield_good = good & yield_mask
            if np.any(yield_good):
                wp_yield = nominal[yield_good] * r_plus[yield_good]
                wm_yield = nominal[yield_good] * r_minus[yield_good]
                yield_idx_good = unroll_yield_index(
                    mass_bin[valid_indices[yield_good]],
                    abs_y_bin[valid_indices[yield_good]],
                    costheta_bin[valid_indices[yield_good]],
                    yield_mode,
                )
                np.add.at(accum["yield_plus"][basis_name], yield_idx_good, wp_yield)
                np.add.at(accum["yield_minus"][basis_name], yield_idx_good, wm_yield)
            afb_good = good & afb_mask
            if np.any(afb_good):
                wp_afb = nominal[afb_good] * r_plus[afb_good]
                wm_afb = nominal[afb_good] * r_minus[afb_good]
                afb_idx_good = unroll_afb_index(mass_bin[valid_indices[afb_good]], abs_y_bin[valid_indices[afb_good]])
                a4_basis_good = signed_costheta[valid_indices[afb_good]]
                np.add.at(accum["afb_sumw_plus"][basis_name], afb_idx_good, wp_afb)
                np.add.at(accum["afb_sumw_minus"][basis_name], afb_idx_good, wm_afb)
                np.add.at(accum["afb_sum_sign_plus"][basis_name], afb_idx_good, wp_afb * a4_basis_good)
                np.add.at(accum["afb_sum_sign_minus"][basis_name], afb_idx_good, wm_afb * a4_basis_good)

        total_processed += len(valid_indices)
        if max_events is not None and total_processed >= max_events:
            break

    print(f"[pod response] selected events after loader selection: {total_selected}")
    print(f"[pod response] events inside hard-coded unrolling bins: {total_binned}")
    print(f"[pod response] processed events: {total_processed}")
    if total_empty_lhe_weights:
        print(f"[pod response] skipped events with empty LHEReweightingWeight: {total_empty_lhe_weights}")
    for basis_name in basis_names:
        total = used[basis_name] + invalid[basis_name]
        frac = invalid[basis_name] / total if total else 0.0
        print(f"[pod response] {loader.name} {basis_name}: used={used[basis_name]} invalid={invalid[basis_name]} invalid_fraction={frac:.4g}")
    return total_selected, total_binned, total_processed


def finalize_variations(basis_names, accum):
    yield_sum = {"SM": accum["nom_yield"]}
    a4_values = {}
    delta_a4 = {}

    sm_a4 = np.full_like(accum["nom_afb_sumw"], np.nan, dtype=np.float64)
    sm_ok = accum["nom_afb_sumw"] != 0.0
    sm_a4[sm_ok] = 4.0 * accum["nom_afb_sum_sign"][sm_ok] / accum["nom_afb_sumw"][sm_ok]
    a4_values["SM"] = sm_a4

    labels = ["SM"]
    for basis_name in basis_names:
        a4_by_sign = {}
        for sign_label, sign_key in [("+", "plus"), ("-", "minus")]:
            label = f"{basis_name}{sign_label}"
            labels.append(label)
            yield_sum[label] = accum[f"yield_{sign_key}"][basis_name]
            values = np.full_like(accum["nom_afb_sumw"], np.nan, dtype=np.float64)
            sumw = accum[f"afb_sumw_{sign_key}"][basis_name]
            sum_sign = accum[f"afb_sum_sign_{sign_key}"][basis_name]
            ok = sumw != 0.0
            values[ok] = 4.0 * sum_sign[ok] / sumw[ok]
            a4_by_sign[sign_key] = values
            a4_values[label] = values
        delta_a4[basis_name] = 0.5 * (a4_by_sign["plus"] - a4_by_sign["minus"])
    return labels, yield_sum, a4_values, delta_a4


def label_basis_and_sign(label):
    if label == "SM":
        return None, None
    if label.endswith("+") or label.endswith("-"):
        return label[:-1], label[-1]
    return label, None


def basis_tex(basis_name):
    if basis_name.startswith("pod"):
        return f"POD {pod_index_from_name(basis_name)}"
    return QPM_TEX[basis_name]


def basis_color(basis_name):
    if basis_name.startswith("pod"):
        return POD_COLOR_CYCLE[(pod_index_from_name(basis_name) - 1) % len(POD_COLOR_CYCLE)]
    return QPM_COLORS[basis_name]


def basis_line_style(basis_name):
    if basis_name.startswith("pod"):
        return 1
    return QPM_STYLES[basis_name]


def basis_marker(basis_name):
    if basis_name.startswith("pod"):
        return POD_MARKER_CYCLE[(pod_index_from_name(basis_name) - 1) % len(POD_MARKER_CYCLE)]
    return QPM_MARKERS[basis_name]


def label_tex(label):
    basis_name, sign = label_basis_and_sign(label)
    if basis_name is None:
        return "SM"
    text = basis_tex(basis_name)
    return f"{text} {sign}" if sign else text


def label_color(label):
    basis_name, _ = label_basis_and_sign(label)
    if basis_name is None:
        return ROOT.kBlack
    return basis_color(basis_name)


def label_line_style(label):
    basis_name, sign = label_basis_and_sign(label)
    if basis_name is None:
        return 1
    base_style = basis_line_style(basis_name)
    if sign == "-":
        return 7 if base_style == 1 else base_style
    return base_style


def label_marker(label):
    basis_name, sign = label_basis_and_sign(label)
    if basis_name is None:
        return 20
    marker = basis_marker(basis_name)
    if sign == "-":
        return marker + 4 if marker < 30 else 27
    return marker


def label_key(label):
    return sanitize(label.replace("+", "_plus").replace("-", "_minus"))


def make_block_graph(values, name, block, n_m, label, positive_only=False):
    graph = ROOT.TGraph()
    graph.SetName(f"{name}_{label_key(label)}_block{block}")
    graph.SetLineColor(label_color(label))
    graph.SetMarkerColor(label_color(label))
    graph.SetLineStyle(label_line_style(label))
    graph.SetLineWidth(2)
    graph.SetMarkerStyle(label_marker(label))
    graph.SetMarkerSize(0.50)
    start = block * n_m
    stop = start + n_m
    ip = 0
    for ibin in range(start, stop):
        value = values[ibin]
        if not np.isfinite(value) or (positive_only and value <= 0):
            continue
        graph.SetPoint(ip, ibin + 0.5, value)
        ip += 1
    return graph if ip else None


def set_pad_ticks(pad):
    pad.SetTickx(1)
    pad.SetTicky(1)


def shorten_ticks(frame, x_length=0.020, y_length=0.010):
    frame.GetXaxis().SetTickLength(x_length)
    frame.GetYaxis().SetTickLength(y_length)


def draw_block_lines(n_blocks, n_m, ymin, ymax):
    stuff = []
    for iblock in range(1, n_blocks):
        line = ROOT.TLine(iblock * n_m, ymin, iblock * n_m, ymax)
        line.SetLineStyle(3)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
        stuff.append(line)
    return stuff


def interval_label(var, lo, hi):
    return f"{lo:g}#leq {var}<{hi:g}"


def draw_triple_labels(n_m, ymin, ymax):
    stuff = []
    n_c = len(COSTHETA_BINS) - 1
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.014)
    y_text = ymax - 0.08 * (ymax - ymin)
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        for ic, (clo, chi) in enumerate(zip(COSTHETA_BINS[:-1], COSTHETA_BINS[1:])):
            block = iy * n_c + ic
            text = (
                f"#splitline{{{interval_label('|y_{#mu#mu}|', ylo, yhi)}}}"
                f"{{{interval_label('cos#theta^{*}', clo, chi)}}}"
            )
            obj = latex.DrawLatex((block + 0.5) * n_m, y_text, text)
            stuff.append(obj)
    stuff.append(latex)
    return stuff


def draw_yield_labels(n_m, ymin, ymax, yield_mode):
    if yield_mode == "mll_y_costheta":
        return draw_triple_labels(n_m, ymin, ymax)
    if yield_mode == "mll":
        return []

    stuff = []
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.030)
    y_text = ymax - 0.08 * (ymax - ymin)
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        obj = latex.DrawLatex((iy + 0.5) * n_m, y_text, interval_label("|y_{#mu#mu}|", ylo, yhi))
        stuff.append(obj)
    stuff.append(latex)
    return stuff


def draw_afb_labels(n_m, ymin, ymax):
    stuff = []
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.032)
    for iy, (ylo, yhi) in enumerate(zip(ABSY_BINS[:-1], ABSY_BINS[1:])):
        obj = latex.DrawLatex(
            (iy + 0.5) * n_m,
            ymax - 0.10 * (ymax - ymin),
            interval_label("|y_{#mu#mu}|", ylo, yhi),
        )
        stuff.append(obj)
    stuff.append(latex)
    return stuff


def configure_mll_axis(axis, n_blocks, n_m):
    axis.SetTitle("m_{#mu#mu} bin [GeV], repeated in |y_{#mu#mu}| blocks")
    axis.SetLabelSize(0.050)
    for block in range(n_blocks):
        for im in range(n_m):
            ibin = block * n_m + im + 1
            if im == n_m - 1:
                label = f"{MLL_BINS[im]:g}-{MLL_BINS[im + 1]:g}"
            else:
                label = f"{MLL_BINS[im]:g}"
            axis.SetBinLabel(ibin, label)
    axis.LabelsOption("v")


def plot_yield(plot_dir, labels, yield_sum, yield_mode):
    n_m = len(MLL_BINS) - 1
    if yield_mode == "mll_y_costheta":
        n_blocks = (len(ABSY_BINS) - 1) * (len(COSTHETA_BINS) - 1)
        suffix = "mll_y_costheta"
        x_title = "Triple diff bin"
    elif yield_mode == "mll_y":
        n_blocks = len(ABSY_BINS) - 1
        suffix = "mll_y"
        x_title = "m_{#mu#mu} unrolled in |y_{#mu#mu}|"
    elif yield_mode == "mll":
        n_blocks = 1
        suffix = "mll"
        x_title = "m_{#mu#mu}"
    else:
        raise RuntimeError(f"Unknown yield mode '{yield_mode}'")

    canvas = ROOT.TCanvas(f"c_giacomo_unrolled_{suffix}", f"Giacomo unrolled {suffix} PDF response", 1800, 760)
    top = ROOT.TPad("top", "top", 0.0, 0.30, 1.0, 1.0)
    bot = ROOT.TPad("bot", "bot", 0.0, 0.0, 1.0, 0.30)
    top.SetBottomMargin(0.02)
    top.SetLogy(True)
    top.SetRightMargin(0.07)
    bot.SetTopMargin(0.03)
    bot.SetBottomMargin(0.34)
    bot.SetRightMargin(0.07)
    set_pad_ticks(top)
    set_pad_ticks(bot)
    top.Draw()
    bot.Draw()
    stuff = [canvas, top, bot]

    positive = [yield_sum[label][i] for label in labels for i in range(len(yield_sum[label])) if yield_sum[label][i] > 0]
    ymax = 2.0 * max(positive) if positive else 1.0
    ymin = max(1e-6, 0.5 * min(positive)) if positive else 1e-6

    top.cd()
    frame_top = ROOT.TH2F("frame_yield", f";{x_title};Events", len(yield_sum["SM"]), 0, len(yield_sum["SM"]), 100, ymin, ymax)
    frame_top.GetXaxis().SetLabelSize(0)
    frame_top.GetYaxis().SetTitleSize(0.055)
    frame_top.GetYaxis().SetLabelSize(0.052)
    frame_top.GetYaxis().SetTitleOffset(0.72)
    shorten_ticks(frame_top, x_length=0.010, y_length=0.006)
    frame_top.Draw()
    stuff.append(frame_top)
    stuff += draw_block_lines(n_blocks, n_m, ymin, ymax)
    stuff += draw_yield_labels(n_m, ymin, ymax, yield_mode)

    legend = ROOT.TLegend(0.10, 0.10, 0.62, 0.30)
    legend.SetNColumns(4)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.030)
    stuff.append(legend)

    for label in labels:
        legend_graph = None
        for block in range(n_blocks):
            graph = make_block_graph(yield_sum[label], "g_yield", block, n_m, label, positive_only=True)
            if graph is None:
                continue
            graph.Draw("L SAME")
            stuff.append(graph)
            if legend_graph is None:
                legend_graph = graph
        if legend_graph is not None:
            legend.AddEntry(legend_graph, label_tex(label), "l")
    legend.Draw()

    bot.cd()
    sm = yield_sum["SM"]
    frame_ratio = ROOT.TH2F("frame_yield_ratio", f";{x_title};Ratio to SM", len(sm), 0, len(sm), 100, 0.97, 1.03)
    frame_ratio.GetXaxis().SetTitleSize(0.13)
    frame_ratio.GetXaxis().SetLabelSize(0.10)
    frame_ratio.GetYaxis().SetTitleSize(0.12)
    frame_ratio.GetYaxis().SetLabelSize(0.10)
    frame_ratio.GetYaxis().SetTitleOffset(0.32)
    frame_ratio.GetYaxis().SetNdivisions(505)
    shorten_ticks(frame_ratio, x_length=0.020, y_length=0.006)
    frame_ratio.Draw()
    stuff.append(frame_ratio)
    for label in labels[1:]:
        ratio = np.full_like(sm, np.nan, dtype=np.float64)
        ok = sm != 0.0
        ratio[ok] = yield_sum[label][ok] / sm[ok]
        for block in range(n_blocks):
            graph = make_block_graph(ratio, "g_yield_ratio", block, n_m, label)
            if graph is None:
                continue
            graph.Draw("L SAME")
            stuff.append(graph)
    line = ROOT.TLine(0, 1.0, len(sm), 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()
    stuff.append(line)
    stuff += draw_block_lines(n_blocks, n_m, 0.97, 1.03)

    base = os.path.join(plot_dir, f"giacomo_unrolled_{suffix}")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")
    print(f"[pdf response] output: {base}.{{png,pdf,root}}")


def plot_a4_response(plot_dir, labels, basis_names, a4_values, delta_a4):
    n_m = len(MLL_BINS) - 1
    n_y = len(ABSY_BINS) - 1
    n_bins = n_y * n_m

    finite_a4_chunks = [values[np.isfinite(values)] for values in a4_values.values() if np.any(np.isfinite(values))]
    finite_a4 = np.concatenate(finite_a4_chunks) if finite_a4_chunks else np.array([], dtype=np.float64)
    if len(finite_a4):
        a4_min = float(np.min(finite_a4))
        a4_max = float(np.max(finite_a4))
        a4_pad = max(0.05 * (a4_max - a4_min), 0.02)
        a4_ymin = a4_min - a4_pad
        a4_ymax = a4_max + a4_pad
    else:
        a4_ymin, a4_ymax = -0.1, 0.1

    delta_ymin = -0.012
    delta_ymax = 0.012

    canvas = ROOT.TCanvas("c_A4_deltaA4_mll_y", "A4 and Delta A4 mll y unrolled PDF response", 1700, 860)
    top = ROOT.TPad("top_A4", "top_A4", 0.0, 0.34, 1.0, 1.0)
    bot = ROOT.TPad("bot_deltaA4", "bot_deltaA4", 0.0, 0.0, 1.0, 0.34)
    top.SetBottomMargin(0.02)
    top.SetTopMargin(0.11)
    top.SetLeftMargin(0.10)
    top.SetRightMargin(0.04)
    bot.SetTopMargin(0.03)
    bot.SetBottomMargin(0.42)
    bot.SetLeftMargin(0.10)
    bot.SetRightMargin(0.04)
    set_pad_ticks(top)
    set_pad_ticks(bot)
    top.Draw()
    bot.Draw()
    stuff = [canvas, top, bot]

    top.cd()
    frame_a4 = ROOT.TH2F(
        "frame_A4",
        ";m_{#mu#mu} unrolled in |y_{#mu#mu}|;A_{4}",
        n_y * n_m,
        0,
        n_y * n_m,
        100,
        a4_ymin,
        a4_ymax,
    )
    frame_a4.GetXaxis().SetLabelSize(0)
    frame_a4.GetYaxis().SetTitleSize(0.060)
    frame_a4.GetYaxis().SetLabelSize(0.052)
    frame_a4.GetYaxis().SetTitleOffset(0.70)
    shorten_ticks(frame_a4, x_length=0.010, y_length=0.007)
    frame_a4.Draw()
    stuff.append(frame_a4)
    stuff += draw_block_lines(n_y, n_m, a4_ymin, a4_ymax)
    stuff += draw_afb_labels(n_m, a4_ymin, a4_ymax)

    legend = ROOT.TLegend(0.10, 0.89, 0.96, 0.97)
    legend.SetNColumns(min(len(labels), 8))
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)
    stuff.append(legend)

    for label in labels:
        legend_graph = None
        for iy in range(n_y):
            graph = make_block_graph(a4_values[label], "g_A4", iy, n_m, label)
            if graph is None:
                continue
            graph.Draw("LP SAME")
            stuff.append(graph)
            if legend_graph is None:
                legend_graph = graph
        if legend_graph is not None:
            legend.AddEntry(legend_graph, label_tex(label), "lp")
    legend.Draw()

    bot.cd()
    frame_delta = ROOT.TH2F(
        "frame_deltaA4",
        ";m_{#mu#mu} bin [GeV], repeated in |y_{#mu#mu}| blocks;#Delta A_{4}",
        n_bins,
        0,
        n_bins,
        100,
        delta_ymin,
        delta_ymax,
    )
    configure_mll_axis(frame_delta.GetXaxis(), n_y, n_m)
    frame_delta.GetXaxis().SetTitleSize(0.095)
    frame_delta.GetXaxis().SetTitleOffset(2.30)
    frame_delta.GetYaxis().SetTitleSize(0.105)
    frame_delta.GetYaxis().SetLabelSize(0.085)
    frame_delta.GetYaxis().SetTitleOffset(0.40)
    frame_delta.GetYaxis().SetNdivisions(505)
    shorten_ticks(frame_delta, x_length=0.014, y_length=0.007)
    frame_delta.Draw()
    stuff.append(frame_delta)

    zero = ROOT.TLine(0, 0, n_bins, 0)
    zero.SetLineStyle(2)
    zero.SetLineColor(ROOT.kGray + 2)
    zero.Draw()
    stuff.append(zero)
    stuff += draw_block_lines(n_y, n_m, delta_ymin, delta_ymax)

    for basis_name in basis_names:
        for iy in range(n_y):
            graph = make_block_graph(delta_a4[basis_name], "g_deltaA4", iy, n_m, basis_name)
            if graph is None:
                continue
            graph.Draw("LP SAME")
            stuff.append(graph)

    base = os.path.join(plot_dir, "A4_DeltaA4_unrolled_mll_y")
    fout = ROOT.TFile.Open(base + ".root", "RECREATE")
    for obj in stuff:
        try:
            obj.Write()
        except Exception:
            pass
    canvas.Write("canvas")
    fout.Close()
    canvas.Print(base + ".png")
    canvas.Print(base + ".pdf")
    print(f"[pdf response] output: {base}.{{png,pdf,root}}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    nargs="+",
    default=None,
    help="Samples from samples_postprocessed.py. Defaults to the two EFT weight-config groups.",
)
parser.add_argument(
    "--low-mass",
    action="store_true",
    help="Use the reduced A4/qpm low-mass binning and default to the low-mass EFT sample group.",
)
parser.add_argument("--selection", default=DEFAULT_SELECTION, help="RDataLoader selection string; default has no mass cut")
parser.add_argument("--small", nargs="?", const=10, type=int, default=None, help="Use every Nth file, e.g. --small 10")
parser.add_argument("--max-files", type=int, default=None)
parser.add_argument("--files-per-chunk", type=int, default=200)
parser.add_argument("--response-mode", choices=["qpm", "pod"], default="qpm")
parser.add_argument("--pdf-set", default="NNPDF40_nnlo_as_01180")
parser.add_argument("--pdf-member", type=int, default=0)
parser.add_argument("--basis-epsilon", type=float, default=0.01)
parser.add_argument("--max-events", type=int, default=None, help="Optional cap after bin selection, useful for tests")
parser.add_argument("--include-gluon-response", action="store_true")
parser.add_argument("--pod-basis", default="250503_pod_basis_40k")
parser.add_argument("--pod-indices", default=None, help="Comma-separated 1-based POD indices, e.g. 1,2,3 or 1-100")
parser.add_argument(
    "--yield-mode",
    choices=["mll_y_costheta", "mll_y", "mll"],
    default="mll_y",
    help="Yield plot unrolling. Default mll_y makes four spectra; mll makes one inclusive spectrum; mll_y_costheta keeps the old triple-differential plot.",
)
args = parser.parse_args()

default_samples = ["DYMuMu_NLO_EFT_SMEFTatNLO_shortEFT", "DYMuMu_NLO_EFT_SMEFTatNLO_fullEFT"]
low_mass_samples = ["DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT"]
if args.samples is None:
    args.samples = low_mass_samples if args.low_mass else default_samples
if args.low_mass:
    MLL_BINS = LOW_MASS_MLL_BINS
    ABSY_BINS = LOW_MASS_ABSY_BINS

if args.response_mode == "pod":
    if args.pod_indices is None:
        raise RuntimeError("--response-mode pod requires --pod-indices, e.g. --pod-indices 1,2,3")
    pod_indices = parse_indices(args.pod_indices)
    basis_names = [pod_name(index) for index in pod_indices]
else:
    pod_indices = []
    basis_names = list(QPM_BASIS)
    if args.include_gluon_response:
        basis_names = ["g"] + basis_names

helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY"))
helpers.copyIndexPHP(os.path.join(user.plot_directory, "DY", "giacomo_unrolled_pdf_response"))

if args.samples == default_samples:
    label = "DYMuMu_NLO_EFT_SMEFTatNLO_allEFT"
elif args.samples == low_mass_samples:
    label = "DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT"
else:
    label = "_".join(args.samples)
if args.small:
    label += f"_small{args.small}"
if args.low_mass:
    label += "_lowMassBins"
if args.yield_mode != "mll_y_costheta":
    label += f"_{args.yield_mode}"
if args.response_mode == "pod":
    label += f"_pod_{sanitize(args.pod_basis)}_{indices_label(pod_indices)}_{epsilon_filename(args.basis_epsilon)}"
else:
    label += f"_qpm_{epsilon_filename(args.basis_epsilon)}"

plot_dir = os.path.join(user.plot_directory, "DY", "giacomo_unrolled_pdf_response", label)
os.makedirs(plot_dir, exist_ok=True)
helpers.copyIndexPHP(plot_dir)
print(f"[pdf response] output directory: {plot_dir}")
print(f"[pdf response] selection: {args.selection}")
print(f"[pdf response] response mode: {args.response_mode}")
print(f"[pdf response] yield mode: {args.yield_mode}")
print(f"[pdf response] binning: {'low-mass A4/qpm' if args.low_mass else 'Giacomo'}")

if args.response_mode == "pod":
    print(f"[pdf response] POD basis: {args.pod_basis}")
    print(f"[pdf response] POD indices: {','.join(str(i) for i in pod_indices)}")
    response_object = {
        pod_name(index): PODBasis(
            variations=[index],
            var_set=args.pod_basis,
            active_pids="all",
            gen_pdf=None,
        )
        for index in pod_indices
    }
else:
    print(f"[pdf response] pdf: {args.pdf_set} member {args.pdf_member}")
    response_object = lhapdf.mkPDF(args.pdf_set, args.pdf_member)
accum = make_accumulators(basis_names, args.yield_mode)
grand_selected = 0
grand_binned = 0
grand_processed = 0

for sample_name in args.samples:
    if sample_name not in samples_postprocessed.samples_by_name:
        raise RuntimeError(f"Unknown sample '{sample_name}'. Known: {', '.join(sorted(samples_postprocessed.samples_by_name))}")
    component = samples_postprocessed.samples_by_name[sample_name]
    files = selected_files(component, small=args.small, max_files=args.max_files)
    print(f"[pdf response] sample: {component.name}")
    print(f"[pdf response] files: {len(files)}")
    print(f"[pdf response] files per shard: {args.files_per_chunk}")
    loader = make_loader(component.name, files, args.selection, args.files_per_chunk)
    remaining_events = None if args.max_events is None else max(0, args.max_events - grand_processed)
    if args.response_mode == "pod":
        selected, binned, processed = accumulate_pod(
            loader,
            response_object,
            basis_names,
            args.basis_epsilon,
            accum,
            args.yield_mode,
            max_events=remaining_events,
        )
    else:
        selected, binned, processed = accumulate_qpm(
            loader,
            response_object,
            basis_names,
            args.basis_epsilon,
            accum,
            args.yield_mode,
            max_events=remaining_events,
        )
    grand_selected += selected
    grand_binned += binned
    grand_processed += processed
    if args.max_events is not None and grand_processed >= args.max_events:
        break

print(f"[pdf response] total selected events after loader selection: {grand_selected}")
print(f"[pdf response] total events inside hard-coded unrolling bins: {grand_binned}")
print(f"[pdf response] total processed events: {grand_processed}")
labels, yield_sum, a4_values, delta_a4 = finalize_variations(basis_names, accum)
plot_yield(plot_dir, labels, yield_sum, args.yield_mode)
plot_a4_response(plot_dir, labels, basis_names, a4_values, delta_a4)
syncer.sync()
