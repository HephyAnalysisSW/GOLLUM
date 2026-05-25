#!/usr/bin/env python3
"""
make_gen_DY_ntuple.py

Modes:

A) Print per-file commands from DY sample names:
   ./make_gen_DY_ntuple.py --samples DYJetsToLL_M50_LO_UL17 DYJetsToLL_M50_LO_ext_UL17

B) Process one NanoAOD file:
   ./make_gen_DY_ntuple.py --file <filename> --xsec <pb> --sumw <combined_sumw>

The --samples mode treats all provided sample names as one process. This is
important for extension samples: their files are listed together and one shared
normalization denominator is computed from the merged Runs.genEventSumw.
"""

import argparse
import math
import os
import re
import sys
import subprocess
import time
import uuid

import ROOT
import awkward as ak
import numpy as np
from tqdm import tqdm
import uproot

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory, tmp_mem_directory
from samples import get_sample, list_sample_keys


CMS_REDIRECTOR_CERN = "root://cms-xrd-global.cern.ch/"

MZ = 91.1876
FLOAT_DEFAULT = -999.0
INT_DEFAULT = -999

FLAG_PROMPT = 0
FLAG_TAU_DECAY_PRODUCT = 2
FLAG_PROMPT_TAU_DECAY_PRODUCT = 3
FLAG_HARD_PROCESS = 7
FLAG_FROM_HARD_PROCESS = 8
FLAG_FROM_HARD_PROCESS_BEFORE_FSR = 11
FLAG_FIRST_COPY = 12
FLAG_LAST_COPY_BEFORE_FSR = 14

SCALAR_INPUT_BRANCHES = [
    "run",
    "luminosityBlock",
    "event",
    "genWeight",
    "Generator_weight",
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_xpdf1",
    "Generator_xpdf2",
    "Generator_id1",
    "Generator_id2",
    "Generator_binvar",
    "LHEWeight_originalXWGTUP",
    "GenMET_pt",
    "GenMET_phi",
]

JAGGED_INPUT_BRANCHES = [
    "LHEPdfWeight",
    "LHEScaleWeight",
    "LHEReweightingWeight",
    "PSWeight",
    "LHEPart_pt",
    "LHEPart_eta",
    "LHEPart_phi",
    "LHEPart_mass",
    "LHEPart_incomingpz",
    "LHEPart_pdgId",
    "LHEPart_status",
    "LHEPart_spin",
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_status",
    "GenPart_statusFlags",
    "GenPart_genPartIdxMother",
    "GenDressedLepton_pt",
    "GenDressedLepton_eta",
    "GenDressedLepton_phi",
    "GenDressedLepton_mass",
    "GenDressedLepton_pdgId",
    "GenDressedLepton_hasTauAnc",
]

READ_BRANCHES = SCALAR_INPUT_BRANCHES + JAGGED_INPUT_BRANCHES

VECTOR_BRANCH_TYPES = {
    "lhe_pdf_weights": "var * float32",
    "lhe_scale_weights": "var * float32",
    "lhe_reweighting_weights": "var * float32",
    "ps_weights": "var * float32",
}

FLOAT_BRANCHES = [
    "event_genWeight",
    "event_Generator_weight",
    "event_LHEWeight_originalXWGTUP",
    "gen_x1",
    "gen_x2",
    "gen_scalePDF",
    "gen_xpdf1",
    "gen_xpdf2",
    "gen_binvar",
    "GenMET_pt",
    "GenMET_phi",
    "xsec",
    "sumw",
    "xsec_weight",
    "lhe_in_incomingpz_0",
    "lhe_in_incomingpz_1",
    "lhe_mll",
    "lhe_yll",
    "lhe_ptll",
    "lhe_phill",
    "dy_lepminus_pt",
    "dy_lepminus_eta",
    "dy_lepminus_phi",
    "dy_lepminus_mass",
    "dy_lepplus_pt",
    "dy_lepplus_eta",
    "dy_lepplus_phi",
    "dy_lepplus_mass",
    "dy_mll",
    "dy_yll",
    "dy_abs_yll",
    "dy_ptll",
    "dy_phill",
    "dy_etall",
    "dy_qzll",
    "dy_mTll",
    "dy_qt_over_m",
    "dy_x1_mll",
    "dy_x2_mll",
    "dy_x1_mT",
    "dy_x2_mT",
    "dy_leading_lep_pt",
    "dy_subleading_lep_pt",
    "dy_max_abs_lep_eta",
    "dy_delta_eta_ll",
    "dy_delta_phi_ll",
    "dy_deltaR_ll",
    "dy_born_mll",
    "dy_born_yll",
    "dy_born_abs_yll",
    "dy_born_ptll",
    "dy_born_qt_over_m",
    "dy_born_lepminus_pt",
    "dy_born_lepplus_pt",
    "cs_costheta",
    "cs_theta",
    "cs_phi",
    "cs_sintheta",
    "cs_cosphi",
    "cs_sinphi",
    "cs_cos2phi",
    "cs_sin2phi",
    "cs_costheta_signed_y",
    "cs_costheta_signed_qz",
    "cs_costheta_trueq",
    "cs_costheta_analytic",
    "cs_costheta_diff",
    "cs_costheta_analytic_signed_y",
    "cs_born_costheta",
    "cs_born_phi",
    "cs_born_costheta_analytic",
    "cs_born_costheta_diff",
    "ang_1_plus_cos2theta",
    "ang_A0_basis",
    "ang_A1_basis",
    "ang_A2_basis",
    "ang_A3_basis",
    "ang_A4_basis",
    "ang_A5_basis",
    "ang_A6_basis",
    "ang_A7_basis",
    "ang_A4_basis_signed_y",
    "ang_A4_basis_trueq",
    "w_lep_pt",
    "w_lep_eta",
    "w_lep_phi",
    "w_nu_pt",
    "w_nu_eta",
    "w_nu_phi",
    "w_pt",
    "w_y",
    "w_abs_y",
    "w_phi",
    "w_mass",
    "w_mt",
    "w_lep_abs_eta",
    "w_x1_mW",
    "w_x2_mW",
    "w_x1_mT",
    "w_x2_mT",
]

INT_BRANCHES = [
    "event_run",
    "event_luminosityBlock",
    "event_event",
    "gen_id1",
    "gen_id2",
    "truth_abs_id1",
    "truth_abs_id2",
    "truth_is_qqbar",
    "truth_is_qg",
    "truth_is_gq",
    "truth_is_gg",
    "truth_is_unknown_initial_state",
    "truth_is_uubar",
    "truth_is_ddbar",
    "truth_is_ssbar",
    "truth_is_ccbar",
    "truth_is_bbbar",
    "truth_is_up_type",
    "truth_is_down_type",
    "truth_flavour_label",
    "truth_quark_direction",
    "truth_antiquark_direction",
    "truth_quark_dir_matches_y_sign",
    "truth_quark_dir_matches_qz_sign",
    "lhe_n_incoming",
    "lhe_n_outgoing",
    "lhe_in_pdgId_0",
    "lhe_in_pdgId_1",
    "lhe_has_dilepton",
    "lhe_dilepton_pdgId_minus",
    "lhe_dilepton_pdgId_plus",
    "dy_has_candidate",
    "dy_channel",
    "dy_lepminus_idx",
    "dy_lepplus_idx",
    "dy_lepminus_pdgId",
    "dy_lepplus_pdgId",
    "dy_lepminus_hasTauAnc",
    "dy_lepplus_hasTauAnc",
    "dy_mass_region",
    "dy_born_has_candidate",
    "dy_born_channel",
    "cs_is_forward_signed_y",
    "cs_is_backward_signed_y",
    "cs_is_forward_trueq",
    "cs_is_backward_trueq",
    "afb_sign_signed_y",
    "afb_sign_trueq",
    "pass_truth_dressed_ee",
    "pass_truth_dressed_mumu",
    "pass_truth_zpole",
    "pass_truth_lowmass",
    "pass_truth_highmass",
    "pass_truth_central_leptons",
    "pass_truth_forward_electron_like",
    "w_has_candidate",
    "w_channel",
    "w_charge",
    "w_lep_pdgId",
    "w_nu_pdgId",
    "truth_is_udbar_to_Wplus",
    "truth_is_dubar_to_Wminus",
    "truth_is_csbar_to_Wplus",
    "truth_is_scbar_to_Wminus",
    "truth_w_initial_flavour_label",
    "has_lhe_pdf_weights",
    "has_lhe_scale_weights",
    "has_lhe_reweighting_weights",
    "has_ps_weights",
]

BRANCH_TYPES = {name: "float64" if name == "sumw" else "float32" for name in FLOAT_BRANCHES}
BRANCH_TYPES.update({name: "int64" if name == "event_event" else "int32" for name in INT_BRANCHES})
BRANCH_TYPES.update(VECTOR_BRANCH_TYPES)

FLAVOUR_CODES = {
    "unknown": 0,
    "uubar": 1,
    "ddbar": 2,
    "ssbar": 3,
    "ccbar": 4,
    "bbbar": 5,
    "qg": 6,
    "gq": 7,
    "gg": 8,
}

W_FLAVOUR_CODES = {
    "unknown": 0,
    "udbar_to_Wplus": 1,
    "dubar_to_Wminus": 2,
    "csbar_to_Wplus": 3,
    "scbar_to_Wminus": 4,
}


def das_list_files(dataset):
    cmd = ["dasgoclient", f"--query=file dataset={dataset}"]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("dasgoclient not found in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dasgoclient failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def is_xrootd_url(path):
    return path.startswith("root://")


def make_xrootd_url(redirector, lfn):
    if is_xrootd_url(lfn):
        return lfn
    if not lfn.startswith("/"):
        raise ValueError(f"LFN must start with '/': got {lfn}")
    if not redirector.endswith("/"):
        raise ValueError(f"Redirector must end with '/': got {redirector}")
    return redirector + lfn


def normalize_input_path(path, redirector_default):
    if is_xrootd_url(path):
        return path
    if path.startswith("/store/"):
        return make_xrootd_url(redirector_default, path)
    return path


def _sanitize(s):
    s = s.strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def sample_id_from_store_path(infile):
    if "/store/" in infile:
        sp = infile[infile.index("/store/"):]
    else:
        sp = infile

    parts = sp.split("/")
    if len(parts) >= 8 and parts[1] == "store":
        campaign = parts[3]
        primary = parts[4]
        processing = parts[6]
        return _sanitize(f"{campaign}__{primary}__{processing}")
    return _sanitize("unknownSample")


def output_path_for_input(infile):
    sid = sample_id_from_store_path(infile)
    outdir = os.path.join(output_directory, "DY-gen-ntuples", sid)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, os.path.basename(infile))


def output_has_events(path):
    if not os.path.exists(path):
        return False
    try:
        with uproot.open(path, object_cache=None, array_cache=None) as fin:
            return "Events" in fin and fin["Events"].num_entries > 0
    except Exception:
        return False


def stage_in_xrootd_file(infile):
    if not is_xrootd_url(infile):
        return infile, None

    os.makedirs(tmp_mem_directory, exist_ok=True)
    base = os.path.basename(infile)
    tag = uuid.uuid4().hex[:10]
    local_path = os.path.join(tmp_mem_directory, f"{tag}__{base}")

    cmd = ["xrdcp", "-f", "-s", infile, local_path]
    t0 = time.time()
    print(f"[make_gen_DY_ntuple] staging input with xrdcp: {infile} -> {local_path}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("xrdcp not found in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xrdcp failed for {infile} -> {local_path}") from e
    dt = time.time() - t0
    size_mb = os.path.getsize(local_path) / 1024.0 / 1024.0
    rate = size_mb / dt if dt > 0 else 0.0
    print(f"[make_gen_DY_ntuple] staged {size_mb:.1f} MB in {dt:.1f} s ({rate:.1f} MB/s)", file=sys.stderr)

    return local_path, local_path


def cleanup_staged_file(path):
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def has_status_flag(flags, bit):
    return (flags & (1 << bit)) != 0


def sum_gen_event_sumw(infile):
    with uproot.open(infile) as fin:
        if "Runs" not in fin:
            raise RuntimeError(f"No Runs tree in {infile}")
        runs = fin["Runs"]
        if "genEventSumw" not in runs:
            raise RuntimeError(f"No Runs.genEventSumw branch in {infile}")
        return float(np.sum(runs["genEventSumw"].array(library="np")))


def combined_sumw(files, redirector):
    total = 0.0
    for i, lfn in enumerate(files, start=1):
        infile = normalize_input_path(lfn, redirector)
        total += sum_gen_event_sumw(infile)
        print(f"[make_gen_DY_ntuple] sumw {i}/{len(files)}: {total:.12g}", file=sys.stderr)
    return total


def vec_from_pt_eta_phi_m(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    E = math.sqrt(max((pt * math.cosh(eta)) ** 2 + mass ** 2, 0.0))
    return (E, px, py, pz)


def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def vec_pt(v):
    return math.hypot(v[1], v[2])


def vec_phi(v):
    return math.atan2(v[2], v[1])


def vec_mass(v):
    return math.sqrt(max(v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2, 0.0))


def vec_eta(v):
    pt = vec_pt(v)
    if pt <= 0:
        return math.copysign(99.0, v[3])
    return math.asinh(v[3] / pt)


def vec_y(v):
    denom = v[0] - v[3]
    numer = v[0] + v[3]
    if denom <= 0 or numer <= 0:
        return FLOAT_DEFAULT
    return 0.5 * math.log(numer / denom)


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def unit3(v):
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if norm <= 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def boost(v, beta):
    b2 = dot3(beta, beta)
    if b2 <= 0:
        return v
    if b2 >= 1:
        b2 = 1 - 1e-12
    gamma = 1.0 / math.sqrt(1.0 - b2)
    p = (v[1], v[2], v[3])
    bp = dot3(beta, p)
    Eprime = gamma * (v[0] - bp)
    factor = ((gamma - 1.0) * bp / b2) - gamma * v[0]
    pprime = (
        p[0] + factor * beta[0],
        p[1] + factor * beta[1],
        p[2] + factor * beta[2],
    )
    return (Eprime, pprime[0], pprime[1], pprime[2])


def collins_soper(lepminus, lepplus, sqrts):
    q = add_vec(lepminus, lepplus)
    if q[0] <= 0:
        return None
    beta = (q[1] / q[0], q[2] / q[0], q[3] / q[0])
    ebeam = sqrts / 2.0
    p1 = (ebeam, 0.0, 0.0, ebeam)
    p2 = (ebeam, 0.0, 0.0, -ebeam)

    p1r = boost(p1, beta)
    p2r = boost(p2, beta)
    lmr = boost(lepminus, beta)

    u1 = unit3((p1r[1], p1r[2], p1r[3]))
    u2 = unit3((p2r[1], p2r[2], p2r[3]))
    zaxis = unit3((u1[0] - u2[0], u1[1] - u2[1], u1[2] - u2[2]))
    xaxis = unit3((u1[0] + u2[0], u1[1] + u2[1], u1[2] + u2[2]))
    yaxis = unit3(cross3(zaxis, xaxis))

    lmhat = unit3((lmr[1], lmr[2], lmr[3]))
    costheta = max(-1.0, min(1.0, dot3(lmhat, zaxis)))
    phi = math.atan2(dot3(lmhat, yaxis), dot3(lmhat, xaxis))
    return costheta, phi


def cs_costheta_analytic(lepminus, lepplus):
    q = add_vec(lepminus, lepplus)
    m = vec_mass(q)
    pt = vec_pt(q)
    denom = m * math.sqrt(m ** 2 + pt ** 2)
    if denom <= 0:
        return FLOAT_DEFAULT
    lm_plus = (lepminus[0] + lepminus[3]) / math.sqrt(2.0)
    lm_minus = (lepminus[0] - lepminus[3]) / math.sqrt(2.0)
    lp_plus = (lepplus[0] + lepplus[3]) / math.sqrt(2.0)
    lp_minus = (lepplus[0] - lepplus[3]) / math.sqrt(2.0)
    return 2.0 * (lm_plus * lp_minus - lm_minus * lp_plus) / denom


def fill_dilepton(prefix, out, lepminus, lepplus, sqrts):
    q = add_vec(lepminus, lepplus)
    m = vec_mass(q)
    pt = vec_pt(q)
    y = vec_y(q)
    phi = vec_phi(q)
    eta = vec_eta(q)
    mt = math.sqrt(max(m ** 2 + pt ** 2, 0.0))
    qz = q[3]
    out[f"{prefix}_mll"] = m
    out[f"{prefix}_yll"] = y
    out[f"{prefix}_abs_yll"] = abs(y) if y != FLOAT_DEFAULT else FLOAT_DEFAULT
    out[f"{prefix}_ptll"] = pt
    if f"{prefix}_phill" in out:
        out[f"{prefix}_phill"] = phi
    if f"{prefix}_etall" in out:
        out[f"{prefix}_etall"] = eta
    if f"{prefix}_qzll" in out:
        out[f"{prefix}_qzll"] = qz
    if f"{prefix}_mTll" in out:
        out[f"{prefix}_mTll"] = mt
    out[f"{prefix}_qt_over_m"] = pt / m if m > 0 else FLOAT_DEFAULT
    if f"{prefix}_x1_mll" in out:
        out[f"{prefix}_x1_mll"] = (m / sqrts) * math.exp(+y) if y != FLOAT_DEFAULT else FLOAT_DEFAULT
        out[f"{prefix}_x2_mll"] = (m / sqrts) * math.exp(-y) if y != FLOAT_DEFAULT else FLOAT_DEFAULT
        out[f"{prefix}_x1_mT"] = (mt / sqrts) * math.exp(+y) if y != FLOAT_DEFAULT else FLOAT_DEFAULT
        out[f"{prefix}_x2_mT"] = (mt / sqrts) * math.exp(-y) if y != FLOAT_DEFAULT else FLOAT_DEFAULT


def init_event_record():
    rec = {}
    for name in FLOAT_BRANCHES:
        rec[name] = FLOAT_DEFAULT
    for name in INT_BRANCHES:
        rec[name] = 0 if name.startswith("has_") or name.startswith("pass_") or name.startswith("truth_is_") or name.startswith("dy_has") or name.startswith("w_has") or name.startswith("cs_is") else INT_DEFAULT
    rec["sumw"] = FLOAT_DEFAULT
    return rec


def scalar_value(arrays, name, i, default):
    if name not in arrays:
        return default
    return arrays[name][i]


def jagged_list(arrays, name, i):
    if name not in arrays:
        return []
    return ak.to_list(arrays[name][i])


def classify_initial_state(id1, id2):
    abs1 = abs(id1)
    abs2 = abs(id2)
    is_q1 = 1 <= abs1 <= 6
    is_q2 = 1 <= abs2 <= 6
    is_g1 = id1 == 21
    is_g2 = id2 == 21
    is_qqbar = is_q1 and is_q2 and abs1 == abs2 and id1 * id2 < 0
    is_qg = is_q1 and is_g2
    is_gq = is_g1 and is_q2
    is_gg = is_g1 and is_g2
    unknown = not (is_qqbar or is_qg or is_gq or is_gg)

    if is_qqbar:
        if id1 > 0 and id2 < 0:
            qdir = 1
        elif id1 < 0 and id2 > 0:
            qdir = -1
        else:
            qdir = 0
    else:
        qdir = 0

    label = "unknown"
    if is_qqbar:
        label = {1: "ddbar", 2: "uubar", 3: "ssbar", 4: "ccbar", 5: "bbbar"}.get(abs1, "unknown")
    elif is_qg:
        label = "qg"
    elif is_gq:
        label = "gq"
    elif is_gg:
        label = "gg"

    return {
        "truth_abs_id1": abs1,
        "truth_abs_id2": abs2,
        "truth_is_qqbar": int(is_qqbar),
        "truth_is_qg": int(is_qg),
        "truth_is_gq": int(is_gq),
        "truth_is_gg": int(is_gg),
        "truth_is_unknown_initial_state": int(unknown),
        "truth_is_uubar": int(label == "uubar"),
        "truth_is_ddbar": int(label == "ddbar"),
        "truth_is_ssbar": int(label == "ssbar"),
        "truth_is_ccbar": int(label == "ccbar"),
        "truth_is_bbbar": int(label == "bbbar"),
        "truth_is_up_type": int(label in {"uubar", "ccbar"}),
        "truth_is_down_type": int(label in {"ddbar", "ssbar", "bbbar"}),
        "truth_flavour_label": FLAVOUR_CODES[label],
        "truth_quark_direction": qdir,
        "truth_antiquark_direction": -qdir,
    }


def classify_w_initial_state(id1, id2):
    pair = (id1, id2)
    label = "unknown"
    if pair in {(2, -1), (-1, 2)}:
        label = "udbar_to_Wplus"
    elif pair in {(1, -2), (-2, 1)}:
        label = "dubar_to_Wminus"
    elif pair in {(4, -3), (-3, 4)}:
        label = "csbar_to_Wplus"
    elif pair in {(3, -4), (-4, 3)}:
        label = "scbar_to_Wminus"
    return {
        "truth_is_udbar_to_Wplus": int(label == "udbar_to_Wplus"),
        "truth_is_dubar_to_Wminus": int(label == "dubar_to_Wminus"),
        "truth_is_csbar_to_Wplus": int(label == "csbar_to_Wplus"),
        "truth_is_scbar_to_Wminus": int(label == "scbar_to_Wminus"),
        "truth_w_initial_flavour_label": W_FLAVOUR_CODES[label],
    }


def build_dressed_dy(rec, arrays, i, sqrts):
    pdg = jagged_list(arrays, "GenDressedLepton_pdgId", i)
    pt = jagged_list(arrays, "GenDressedLepton_pt", i)
    eta = jagged_list(arrays, "GenDressedLepton_eta", i)
    phi = jagged_list(arrays, "GenDressedLepton_phi", i)
    mass = jagged_list(arrays, "GenDressedLepton_mass", i)
    has_tau = jagged_list(arrays, "GenDressedLepton_hasTauAnc", i)

    best = None
    for a in range(len(pdg)):
        if abs(pdg[a]) not in (11, 13) or bool(has_tau[a]):
            continue
        for b in range(a + 1, len(pdg)):
            if abs(pdg[b]) not in (11, 13) or bool(has_tau[b]):
                continue
            if pdg[a] + pdg[b] != 0 or abs(pdg[a]) != abs(pdg[b]):
                continue
            score = pt[a] + pt[b]
            if best is None or score > best[0]:
                best = (score, a, b)

    if best is None:
        return None

    _, a, b = best
    iminus, iplus = (a, b) if pdg[a] > 0 else (b, a)
    lm = vec_from_pt_eta_phi_m(pt[iminus], eta[iminus], phi[iminus], mass[iminus])
    lp = vec_from_pt_eta_phi_m(pt[iplus], eta[iplus], phi[iplus], mass[iplus])

    rec["dy_has_candidate"] = 1
    rec["dy_channel"] = abs(int(pdg[iminus]))
    rec["dy_lepminus_idx"] = iminus
    rec["dy_lepplus_idx"] = iplus
    rec["dy_lepminus_pdgId"] = int(pdg[iminus])
    rec["dy_lepplus_pdgId"] = int(pdg[iplus])
    rec["dy_lepminus_hasTauAnc"] = int(bool(has_tau[iminus]))
    rec["dy_lepplus_hasTauAnc"] = int(bool(has_tau[iplus]))
    rec["dy_lepminus_pt"] = pt[iminus]
    rec["dy_lepminus_eta"] = eta[iminus]
    rec["dy_lepminus_phi"] = phi[iminus]
    rec["dy_lepminus_mass"] = mass[iminus]
    rec["dy_lepplus_pt"] = pt[iplus]
    rec["dy_lepplus_eta"] = eta[iplus]
    rec["dy_lepplus_phi"] = phi[iplus]
    rec["dy_lepplus_mass"] = mass[iplus]

    fill_dilepton("dy", rec, lm, lp, sqrts)

    rec["dy_leading_lep_pt"] = max(pt[iminus], pt[iplus])
    rec["dy_subleading_lep_pt"] = min(pt[iminus], pt[iplus])
    rec["dy_max_abs_lep_eta"] = max(abs(eta[iminus]), abs(eta[iplus]))
    rec["dy_delta_eta_ll"] = eta[iminus] - eta[iplus]
    rec["dy_delta_phi_ll"] = delta_phi(phi[iminus], phi[iplus])
    rec["dy_deltaR_ll"] = math.hypot(rec["dy_delta_eta_ll"], rec["dy_delta_phi_ll"])

    mll = rec["dy_mll"]
    rec["dy_mass_region"] = 1 if 15 < mll < 60 else 2 if 60 < mll < 120 else 3 if mll >= 120 else 0
    rec["pass_truth_dressed_ee"] = int(rec["dy_channel"] == 11)
    rec["pass_truth_dressed_mumu"] = int(rec["dy_channel"] == 13)
    rec["pass_truth_zpole"] = int(60 < mll < 120)
    rec["pass_truth_lowmass"] = int(15 < mll < 60)
    rec["pass_truth_highmass"] = int(mll >= 120)
    rec["pass_truth_central_leptons"] = int(max(abs(eta[iminus]), abs(eta[iplus])) < 2.5 and rec["dy_leading_lep_pt"] > 25 and rec["dy_subleading_lep_pt"] > 15)
    rec["pass_truth_forward_electron_like"] = int(rec["dy_channel"] == 11 and max(abs(eta[iminus]), abs(eta[iplus])) > 2.5)

    return lm, lp


def build_lhe(rec, arrays, i):
    pdg = jagged_list(arrays, "LHEPart_pdgId", i)
    status = jagged_list(arrays, "LHEPart_status", i)
    pt = jagged_list(arrays, "LHEPart_pt", i)
    eta = jagged_list(arrays, "LHEPart_eta", i)
    phi = jagged_list(arrays, "LHEPart_phi", i)
    mass = jagged_list(arrays, "LHEPart_mass", i)
    incomingpz = jagged_list(arrays, "LHEPart_incomingpz", i)

    incoming = [j for j, st in enumerate(status) if st == -1]
    outgoing = [j for j, st in enumerate(status) if st == 1]
    rec["lhe_n_incoming"] = len(incoming)
    rec["lhe_n_outgoing"] = len(outgoing)
    if len(incoming) > 0:
        rec["lhe_in_pdgId_0"] = int(pdg[incoming[0]])
        rec["lhe_in_incomingpz_0"] = incomingpz[incoming[0]]
    if len(incoming) > 1:
        rec["lhe_in_pdgId_1"] = int(pdg[incoming[1]])
        rec["lhe_in_incomingpz_1"] = incomingpz[incoming[1]]

    best = None
    leps = [j for j in outgoing if abs(pdg[j]) in (11, 13)]
    for a_pos, a in enumerate(leps):
        for b in leps[a_pos + 1:]:
            if pdg[a] + pdg[b] != 0 or abs(pdg[a]) != abs(pdg[b]):
                continue
            score = pt[a] + pt[b]
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return

    _, a, b = best
    iminus, iplus = (a, b) if pdg[a] > 0 else (b, a)
    lm = vec_from_pt_eta_phi_m(pt[iminus], eta[iminus], phi[iminus], mass[iminus])
    lp = vec_from_pt_eta_phi_m(pt[iplus], eta[iplus], phi[iplus], mass[iplus])
    q = add_vec(lm, lp)
    rec["lhe_has_dilepton"] = 1
    rec["lhe_dilepton_pdgId_minus"] = int(pdg[iminus])
    rec["lhe_dilepton_pdgId_plus"] = int(pdg[iplus])
    rec["lhe_mll"] = vec_mass(q)
    rec["lhe_yll"] = vec_y(q)
    rec["lhe_ptll"] = vec_pt(q)
    rec["lhe_phill"] = vec_phi(q)


def build_born_dy(rec, arrays, i, sqrts):
    pdg = jagged_list(arrays, "GenPart_pdgId", i)
    flags = jagged_list(arrays, "GenPart_statusFlags", i)
    pt = jagged_list(arrays, "GenPart_pt", i)
    eta = jagged_list(arrays, "GenPart_eta", i)
    phi = jagged_list(arrays, "GenPart_phi", i)
    mass = jagged_list(arrays, "GenPart_mass", i)

    candidates = []
    for j, pid in enumerate(pdg):
        if abs(pid) not in (11, 13):
            continue
        fl = int(flags[j])
        if has_status_flag(fl, FLAG_TAU_DECAY_PRODUCT) or has_status_flag(fl, FLAG_PROMPT_TAU_DECAY_PRODUCT):
            continue
        strict = has_status_flag(fl, FLAG_FROM_HARD_PROCESS_BEFORE_FSR) and has_status_flag(fl, FLAG_LAST_COPY_BEFORE_FSR)
        fallback = has_status_flag(fl, FLAG_FROM_HARD_PROCESS_BEFORE_FSR) or (has_status_flag(fl, FLAG_FROM_HARD_PROCESS) and has_status_flag(fl, FLAG_FIRST_COPY))
        if strict or fallback:
            candidates.append(j)

    best = None
    for a_pos, a in enumerate(candidates):
        for b in candidates[a_pos + 1:]:
            if pdg[a] + pdg[b] != 0 or abs(pdg[a]) != abs(pdg[b]):
                continue
            score = pt[a] + pt[b]
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None

    _, a, b = best
    iminus, iplus = (a, b) if pdg[a] > 0 else (b, a)
    lm = vec_from_pt_eta_phi_m(pt[iminus], eta[iminus], phi[iminus], mass[iminus])
    lp = vec_from_pt_eta_phi_m(pt[iplus], eta[iplus], phi[iplus], mass[iplus])
    rec["dy_born_has_candidate"] = 1
    rec["dy_born_channel"] = abs(int(pdg[iminus]))
    rec["dy_born_lepminus_pt"] = pt[iminus]
    rec["dy_born_lepplus_pt"] = pt[iplus]
    fill_dilepton("dy_born", rec, lm, lp, sqrts)
    return lm, lp


def fill_cs_and_angles(rec, lepminus, lepplus, sqrts, prefix="cs"):
    cs = collins_soper(lepminus, lepplus, sqrts)
    if cs is None:
        return
    costheta, phi = cs
    analytic = cs_costheta_analytic(lepminus, lepplus)
    diff = costheta - analytic if analytic != FLOAT_DEFAULT else FLOAT_DEFAULT
    theta = math.acos(max(-1.0, min(1.0, costheta)))
    sintheta = math.sqrt(max(0.0, 1.0 - costheta ** 2))

    if prefix == "cs":
        rec["cs_costheta"] = costheta
        rec["cs_theta"] = theta
        rec["cs_phi"] = phi
        rec["cs_sintheta"] = sintheta
        rec["cs_cosphi"] = math.cos(phi)
        rec["cs_sinphi"] = math.sin(phi)
        rec["cs_cos2phi"] = math.cos(2 * phi)
        rec["cs_sin2phi"] = math.sin(2 * phi)
        rec["cs_costheta_analytic"] = analytic
        rec["cs_costheta_diff"] = diff
        sign_y = 1 if rec["dy_yll"] > 0 else -1 if rec["dy_yll"] < 0 else 0
        sign_qz = 1 if rec["dy_qzll"] > 0 else -1 if rec["dy_qzll"] < 0 else 0
        rec["cs_costheta_signed_y"] = sign_y * costheta
        rec["cs_costheta_signed_qz"] = sign_qz * costheta
        rec["cs_costheta_analytic_signed_y"] = sign_y * analytic if analytic != FLOAT_DEFAULT else FLOAT_DEFAULT
        rec["cs_costheta_trueq"] = rec["truth_quark_direction"] * costheta
        rec["cs_is_forward_signed_y"] = int(rec["cs_costheta_signed_y"] > 0)
        rec["cs_is_backward_signed_y"] = int(rec["cs_costheta_signed_y"] < 0)
        rec["cs_is_forward_trueq"] = int(rec["cs_costheta_trueq"] > 0)
        rec["cs_is_backward_trueq"] = int(rec["cs_costheta_trueq"] < 0)
        rec["truth_quark_dir_matches_y_sign"] = int(rec["truth_quark_direction"] == sign_y) if sign_y else 0
        rec["truth_quark_dir_matches_qz_sign"] = int(rec["truth_quark_direction"] == sign_qz) if sign_qz else 0

        c = costheta
        s = sintheta
        rec["ang_1_plus_cos2theta"] = 1.0 + c * c
        rec["ang_A0_basis"] = 0.5 * (1.0 - 3.0 * c * c)
        rec["ang_A1_basis"] = 2.0 * s * c * math.cos(phi)
        rec["ang_A2_basis"] = 0.5 * s * s * math.cos(2 * phi)
        rec["ang_A3_basis"] = s * math.cos(phi)
        rec["ang_A4_basis"] = c
        rec["ang_A5_basis"] = s * s * math.sin(2 * phi)
        rec["ang_A6_basis"] = 2.0 * s * c * math.sin(phi)
        rec["ang_A7_basis"] = s * math.sin(phi)
        rec["ang_A4_basis_signed_y"] = rec["cs_costheta_signed_y"]
        rec["ang_A4_basis_trueq"] = rec["cs_costheta_trueq"]
        rec["afb_sign_signed_y"] = 1 if rec["cs_costheta_signed_y"] > 0 else -1 if rec["cs_costheta_signed_y"] < 0 else 0
        rec["afb_sign_trueq"] = 1 if rec["cs_costheta_trueq"] > 0 else -1 if rec["cs_costheta_trueq"] < 0 else 0
    else:
        rec["cs_born_costheta"] = costheta
        rec["cs_born_phi"] = phi
        rec["cs_born_costheta_analytic"] = analytic
        rec["cs_born_costheta_diff"] = diff


def build_w(rec, arrays, i, sqrts):
    pdg = jagged_list(arrays, "GenPart_pdgId", i)
    flags = jagged_list(arrays, "GenPart_statusFlags", i)
    pt = jagged_list(arrays, "GenPart_pt", i)
    eta = jagged_list(arrays, "GenPart_eta", i)
    phi = jagged_list(arrays, "GenPart_phi", i)
    mass = jagged_list(arrays, "GenPart_mass", i)

    leptons = []
    neutrinos = []
    for j, pid in enumerate(pdg):
        apid = abs(pid)
        fl = int(flags[j])
        prompt_or_hp = has_status_flag(fl, FLAG_PROMPT) or has_status_flag(fl, FLAG_FROM_HARD_PROCESS) or has_status_flag(fl, FLAG_FROM_HARD_PROCESS_BEFORE_FSR)
        tau = has_status_flag(fl, FLAG_TAU_DECAY_PRODUCT) or has_status_flag(fl, FLAG_PROMPT_TAU_DECAY_PRODUCT)
        if apid in (11, 13) and prompt_or_hp and not tau:
            leptons.append(j)
        if apid in (12, 14, 16) and prompt_or_hp and not tau:
            neutrinos.append(j)

    def matches(lep_pid, nu_pid):
        if lep_pid == 11 and nu_pid == -12:
            return -1, 11
        if lep_pid == -11 and nu_pid == 12:
            return +1, 11
        if lep_pid == 13 and nu_pid == -14:
            return -1, 13
        if lep_pid == -13 and nu_pid == 14:
            return +1, 13
        return 0, 0

    best = None
    for l in leptons:
        for n in neutrinos:
            charge, channel = matches(pdg[l], pdg[n])
            if charge == 0:
                continue
            score = pt[l] + pt[n]
            if best is None or score > best[0]:
                best = (score, l, n, charge, channel)
    if best is None:
        return

    _, l, n, charge, channel = best
    lv = vec_from_pt_eta_phi_m(pt[l], eta[l], phi[l], mass[l])
    nv = vec_from_pt_eta_phi_m(pt[n], eta[n], phi[n], mass[n])
    wv = add_vec(lv, nv)
    wmass = vec_mass(wv)
    wy = vec_y(wv)
    wmt = math.sqrt(max(2 * pt[l] * pt[n] * (1 - math.cos(delta_phi(phi[l], phi[n]))), 0.0))

    rec["w_has_candidate"] = 1
    rec["w_channel"] = channel
    rec["w_charge"] = charge
    rec["w_lep_pdgId"] = int(pdg[l])
    rec["w_nu_pdgId"] = int(pdg[n])
    rec["w_lep_pt"] = pt[l]
    rec["w_lep_eta"] = eta[l]
    rec["w_lep_phi"] = phi[l]
    rec["w_nu_pt"] = pt[n]
    rec["w_nu_eta"] = eta[n]
    rec["w_nu_phi"] = phi[n]
    rec["w_pt"] = vec_pt(wv)
    rec["w_y"] = wy
    rec["w_abs_y"] = abs(wy) if wy != FLOAT_DEFAULT else FLOAT_DEFAULT
    rec["w_phi"] = vec_phi(wv)
    rec["w_mass"] = wmass
    rec["w_mt"] = wmt
    rec["w_lep_abs_eta"] = abs(eta[l])
    if wy != FLOAT_DEFAULT:
        rec["w_x1_mW"] = (wmass / sqrts) * math.exp(+wy)
        rec["w_x2_mW"] = (wmass / sqrts) * math.exp(-wy)
        rec["w_x1_mT"] = (wmt / sqrts) * math.exp(+wy)
        rec["w_x2_mT"] = (wmt / sqrts) * math.exp(-wy)


def build_chunk(arrays, n, xsec, sumw, sqrts):
    records = []
    vectors = {name: [] for name in VECTOR_BRANCH_TYPES}
    summary = {"dy": 0, "born": 0, "w": 0, "cs": 0, "max_cs_diff": 0.0}

    for i in range(n):
        rec = init_event_record()

        rec["event_run"] = int(scalar_value(arrays, "run", i, INT_DEFAULT))
        rec["event_luminosityBlock"] = int(scalar_value(arrays, "luminosityBlock", i, INT_DEFAULT))
        rec["event_event"] = int(scalar_value(arrays, "event", i, INT_DEFAULT))
        gen_weight = float(scalar_value(arrays, "genWeight", i, scalar_value(arrays, "Generator_weight", i, 1.0)))
        generator_weight = float(scalar_value(arrays, "Generator_weight", i, gen_weight))
        rec["event_genWeight"] = gen_weight
        rec["event_Generator_weight"] = generator_weight
        rec["event_LHEWeight_originalXWGTUP"] = float(scalar_value(arrays, "LHEWeight_originalXWGTUP", i, FLOAT_DEFAULT))
        rec["xsec"] = float(xsec)
        rec["sumw"] = float(sumw)
        rec["xsec_weight"] = gen_weight * float(xsec) / float(sumw)

        rec["gen_id1"] = int(scalar_value(arrays, "Generator_id1", i, INT_DEFAULT))
        rec["gen_id2"] = int(scalar_value(arrays, "Generator_id2", i, INT_DEFAULT))
        rec["gen_x1"] = float(scalar_value(arrays, "Generator_x1", i, FLOAT_DEFAULT))
        rec["gen_x2"] = float(scalar_value(arrays, "Generator_x2", i, FLOAT_DEFAULT))
        rec["gen_scalePDF"] = float(scalar_value(arrays, "Generator_scalePDF", i, FLOAT_DEFAULT))
        rec["gen_xpdf1"] = float(scalar_value(arrays, "Generator_xpdf1", i, FLOAT_DEFAULT))
        rec["gen_xpdf2"] = float(scalar_value(arrays, "Generator_xpdf2", i, FLOAT_DEFAULT))
        rec["gen_binvar"] = float(scalar_value(arrays, "Generator_binvar", i, FLOAT_DEFAULT))
        rec["GenMET_pt"] = float(scalar_value(arrays, "GenMET_pt", i, FLOAT_DEFAULT))
        rec["GenMET_phi"] = float(scalar_value(arrays, "GenMET_phi", i, FLOAT_DEFAULT))

        rec.update(classify_initial_state(rec["gen_id1"], rec["gen_id2"]))
        rec.update(classify_w_initial_state(rec["gen_id1"], rec["gen_id2"]))

        pdf_weights = jagged_list(arrays, "LHEPdfWeight", i)
        scale_weights = jagged_list(arrays, "LHEScaleWeight", i)
        reweighting_weights = jagged_list(arrays, "LHEReweightingWeight", i)
        ps_weights = jagged_list(arrays, "PSWeight", i)
        vectors["lhe_pdf_weights"].append(pdf_weights)
        vectors["lhe_scale_weights"].append(scale_weights)
        vectors["lhe_reweighting_weights"].append(reweighting_weights)
        vectors["ps_weights"].append(ps_weights)
        rec["has_lhe_pdf_weights"] = int(len(pdf_weights) > 0)
        rec["has_lhe_scale_weights"] = int(len(scale_weights) > 0)
        rec["has_lhe_reweighting_weights"] = int(len(reweighting_weights) > 0)
        rec["has_ps_weights"] = int(len(ps_weights) > 0)

        build_lhe(rec, arrays, i)
        dressed = build_dressed_dy(rec, arrays, i, sqrts)
        born = build_born_dy(rec, arrays, i, sqrts)
        build_w(rec, arrays, i, sqrts)

        if dressed is not None:
            summary["dy"] += 1
            fill_cs_and_angles(rec, dressed[0], dressed[1], sqrts, "cs")
            if rec["cs_costheta"] != FLOAT_DEFAULT:
                summary["cs"] += 1
                if rec["cs_costheta_diff"] != FLOAT_DEFAULT:
                    summary["max_cs_diff"] = max(summary["max_cs_diff"], abs(rec["cs_costheta_diff"]))
        if born is not None:
            summary["born"] += 1
            fill_cs_and_angles(rec, born[0], born[1], sqrts, "cs_born")
        if rec["w_has_candidate"]:
            summary["w"] += 1

        records.append(rec)

    out = {}
    for name in FLOAT_BRANCHES:
        dtype = np.float64 if name == "sumw" else np.float32
        out[name] = np.asarray([rec[name] for rec in records], dtype=dtype)
    for name in INT_BRANCHES:
        dtype = np.int64 if name == "event_event" else np.int32
        out[name] = np.asarray([rec[name] for rec in records], dtype=dtype)
    for name, values in vectors.items():
        out[name] = ak.Array(values)
    return out, summary


def available_expressions(local_infile):
    with uproot.open(local_infile) as fin:
        keys = set(fin["Events"].keys())
    missing = [name for name in READ_BRANCHES if name not in keys]
    expressions = [name for name in READ_BRANCHES if name in keys]
    if missing:
        print(f"[make_gen_DY_ntuple] missing optional branches: {', '.join(missing)}", file=sys.stderr)
    return expressions


def num_events(local_infile):
    with uproot.open(local_infile) as fin:
        return int(fin["Events"].num_entries)


def process_file(infile, step_size, max_events, xsec, sumw, sqrts, overwrite):
    if sumw == 0:
        raise ValueError("sumw must be nonzero for cross-section normalization.")

    outfile = output_path_for_input(infile)
    if output_has_events(outfile) and not overwrite:
        print(f"[make_gen_DY_ntuple] output exists, skip: {outfile}", file=sys.stderr)
        return outfile
    if os.path.exists(outfile) and not output_has_events(outfile):
        print(f"[make_gen_DY_ntuple] replacing incomplete output: {outfile}", file=sys.stderr)
    elif os.path.exists(outfile) and overwrite:
        print(f"[make_gen_DY_ntuple] overwriting output: {outfile}", file=sys.stderr)

    local_infile, staged_path = stage_in_xrootd_file(infile)

    try:
        tmp_outfile = f"{outfile}.tmp.{uuid.uuid4().hex[:10]}"
        expressions = available_expressions(local_infile)
        n_total = num_events(local_infile)
        if max_events >= 0:
            n_total = min(n_total, max_events)
        total_written = 0
        total_summary = {"dy": 0, "born": 0, "w": 0, "cs": 0, "max_cs_diff": 0.0}
        tree = None
        process_t0 = time.time()

        try:
            with uproot.recreate(tmp_outfile) as fout:
                chunks = uproot.iterate(
                    {local_infile: "Events"},
                    expressions=expressions,
                    step_size=step_size,
                    library="ak",
                    how=dict,
                )
                for arrays in tqdm(chunks, total=math.ceil(n_total / step_size) if step_size > 0 else None, desc="DY ntuple chunks", unit="chunk"):
                    chunk_n = len(next(iter(arrays.values()))) if arrays else 0
                    if max_events >= 0:
                        keep = max_events - total_written
                        if keep <= 0:
                            break
                        chunk_n = min(chunk_n, keep)
                        arrays = {key: value[:chunk_n] for key, value in arrays.items()}
                    if chunk_n == 0:
                        continue

                    out, summary = build_chunk(arrays, chunk_n, xsec, sumw, sqrts)

                    if tree is None:
                        tree = fout.mktree("Events", BRANCH_TYPES)
                    tree.extend(out)

                    total_written += chunk_n
                    for key in ("dy", "born", "w", "cs"):
                        total_summary[key] += summary[key]
                    total_summary["max_cs_diff"] = max(total_summary["max_cs_diff"], summary["max_cs_diff"])
        except Exception:
            if os.path.exists(tmp_outfile):
                os.remove(tmp_outfile)
            raise

        if total_written <= 0 or tree is None:
            if os.path.exists(tmp_outfile):
                os.remove(tmp_outfile)
            raise RuntimeError(f"No events were written for {infile}; refusing to create empty output {outfile}.")

        os.replace(tmp_outfile, outfile)
        process_dt = time.time() - process_t0

        print(
            "[make_gen_DY_ntuple] sanity: "
            f"events={total_written} "
            f"dy={total_summary['dy']} "
            f"born={total_summary['born']} "
            f"w={total_summary['w']} "
            f"cs={total_summary['cs']} "
            f"max_abs_cs_diff={total_summary['max_cs_diff']:.3g} "
            f"processing_time={process_dt:.1f}s "
            f"events_per_s={total_written / process_dt if process_dt > 0 else 0.0:.1f}",
            file=sys.stderr,
        )

        rf = ROOT.TFile.Open(outfile, "UPDATE")
        if rf and not rf.IsZombie():
            t = rf.Get("Events")
            if t:
                titles = {
                    "xsec_weight": "event_genWeight * xsec / sumw",
                    "truth_flavour_label": "0 unknown, 1 uubar, 2 ddbar, 3 ssbar, 4 ccbar, 5 bbbar, 6 qg, 7 gq, 8 gg",
                    "truth_w_initial_flavour_label": "0 unknown, 1 udbar->W+, 2 dubar->W-, 3 csbar->W+, 4 scbar->W-",
                    "cs_costheta": "Collins-Soper cos(theta) from dressed dilepton",
                    "cs_costheta_analytic": "Analytic Collins-Soper cos(theta) cross-check",
                    "cs_costheta_diff": "cs_costheta - cs_costheta_analytic",
                }
                for bname, title in titles.items():
                    br = t.GetBranch(bname)
                    if br:
                        br.SetTitle(title)
                t.Write("", ROOT.TObject.kOverwrite)
            rf.Close()

        return outfile

    finally:
        cleanup_staged_file(staged_path)


def expand_samples(sample_names, redirector, small):
    samples = [get_sample(name) for name in sample_names]
    xsecs = {round(sample.xsec, 12) for sample in samples}
    if len(xsecs) != 1:
        detail = ", ".join(f"{sample.key}: {sample.xsec}" for sample in samples)
        raise RuntimeError(f"All --samples entries must have the same xsec for merged normalization. Got {detail}")

    files = []
    for sample in samples:
        sample_files = das_list_files(sample.dataset)
        if not sample_files:
            raise RuntimeError(f"No files found for {sample.key}: {sample.dataset}")
        files.extend(sample_files)

    if small:
        files = files[:3]

    sumw = combined_sumw(files, redirector)
    return samples, files, samples[0].xsec, sumw


def main():
    parser = argparse.ArgumentParser(prog="make_gen_DY_ntuple.py")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--samples", nargs="+", help="DY sample keys from samples.py, merged for normalization")
    g.add_argument("--file", type=str, help="Input NanoAOD file (local, /store/..., or root://...)")

    parser.add_argument("--list-samples", nargs="?", const="", default=None, help="List known sample keys, optionally filtered by regex")
    parser.add_argument("--xsec", type=float, default=None, help="Cross section for --file mode")
    parser.add_argument("--sumw", type=float, default=None, help="Merged Runs.genEventSumw denominator for --file mode")
    parser.add_argument("--redirector", type=str, default=CMS_REDIRECTOR_CERN, help="XRootD redirector")
    parser.add_argument("--step-size", type=int, default=20_000, help="Events per chunk")
    parser.add_argument("--max-events", type=int, default=-1, help="Max events to write (-1 = all)")
    parser.add_argument("--sqrts", type=float, default=13000.0, help="Collider sqrt(s) in GeV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite complete output files instead of skipping them")
    parser.add_argument("--small", action="store_true", help="Debug mode: print/process fewer files/events")

    args = parser.parse_args()

    if args.list_samples is not None:
        for key in list_sample_keys(args.list_samples or None):
            sample = get_sample(key)
            print(f"{sample.key:35s}  xsec={sample.xsec:.12g}  {sample.dataset}")
        return

    if bool(args.samples) == bool(args.file):
        parser.error("Specify exactly one of --samples or --file, unless using --list-samples.")

    if args.small:
        if args.max_events < 0:
            args.max_events = 10_000
        args.step_size = min(args.step_size, 20_000)

    if args.samples:
        samples, files, xsec, sumw = expand_samples(args.samples, args.redirector, args.small)
        keys = ",".join(sample.key for sample in samples)
        print(f"# samples: {keys}", file=sys.stderr)
        print(f"# xsec: {xsec:.12g}", file=sys.stderr)
        print(f"# sumw: {sumw:.12g}", file=sys.stderr)

        for f in files:
            cmd = f"./make_gen_DY_ntuple.py --file {make_xrootd_url(args.redirector, f)} --xsec {xsec:.12g} --sumw {sumw:.12g} --sqrts {args.sqrts:.12g}"
            if args.overwrite:
                cmd += " --overwrite"
            if args.small:
                cmd += " --small"
            print(cmd)
        return

    if args.xsec is None or args.sumw is None:
        parser.error("--file mode requires --xsec and --sumw. Use --samples to print normalized per-file jobs.")

    infile = normalize_input_path(args.file, args.redirector)
    out = process_file(
        infile=infile,
        step_size=args.step_size,
        max_events=args.max_events,
        xsec=args.xsec,
        sumw=args.sumw,
        sqrts=args.sqrts,
        overwrite=args.overwrite,
    )
    print(f"[make_gen_DY_ntuple] output: {out}")


if __name__ == "__main__":
    main()
