#!/usr/bin/env python3
"""
make_centralgen_ntuple.py

Modes:

A) Process one job slice from a sample JSON:
   ./make_centralgen_ntuple.py --sample-json signal_samples/UL18_TT01j2l_mtt_0to700.json --nJobs 10 --job 0

B) Process one NanoAOD file:
   ./make_centralgen_ntuple.py --file <filename>

Notes:
- Remote xrootd inputs are staged locally with xrdcp first.
- Output is a flat scalar ntuple.
- Feature definition follows TopEFT/ttbarEFT analysis/centralGen.py.
- Object selection follows TopEFT/ttbarEFT ttbarEFT/modules/analysis_tools.py:
  * GenDressedLepton: e/mu, pt > 20 GeV, |eta| < 2.5
  * GenJet: pt > 30 GeV, |eta| < 2.5, DR(jet, lepton) > 0.4
  * Event: opposite-sign leptons and at least 2 jets
- EFT coefficient interpretation:
  * WC decoding follows:
    https://github.com/TopEFT/topcoffea/blob/285a46f3dddae15035f5c21fe3324443253143d8/topcoffea/modules/utils.py#L158-L179
  * Event-weight reconstruction follows the exchange quoted by the user and is consistent with:
    https://github.com/TopEFT/ttbarEFT/blob/main/analysis/centralGen.py
  * The stored EFTfitCoefficients form an upper-triangular quadratic polynomial basis in [1, wc_0, wc_1, ...].
    For ML we write:
      - linear terms as der_<wc>, equal to the first derivative at the SM point
      - quadratic terms as der_<wc0>_<wc1>, equal to the second derivative coefficient
    Diagonal quadratic entries get a factor of 2 because d^2/dwc_i^2 (a * wc_i^2) = 2a,
    while mixed terms d^2/(dwc_i dwc_j) (a * wc_i * wc_j) = a for i != j.
"""

import argparse
import json
import os
import re
import sys
import subprocess
import uuid

import ROOT
import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory, tmp_mem_directory


CMS_REDIRECTOR_CERN = "root://cms-xrd-global.cern.ch/"
CMS_REDIRECTOR_FNAL = "root://cmsxrootd.fnal.gov/"

READ_BRANCHES = [
    "GenDressedLepton_pt",
    "GenDressedLepton_eta",
    "GenDressedLepton_phi",
    "GenDressedLepton_mass",
    "GenDressedLepton_pdgId",
    "GenJet_pt",
    "GenJet_eta",
    "GenJet_phi",
    "GenJet_mass",
    "Generator_weight",
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "LHEWeight_originalXWGTUP",
    "nEFTfitCoefficients",
    "EFTfitCoefficients",
]


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


def output_path_for_input(infile, small=False, sample_tag=None):
    sid = sample_id_from_store_path(infile)
    base_dir = "TTbarEFT-centralGen-ntuples_small" if small else "TTbarEFT-centralGen-ntuples"
    outdir = os.path.join(output_directory, base_dir, sid)
    if sample_tag:
        outdir = os.path.join(outdir, sample_tag)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, os.path.basename(infile))


def stage_in_xrootd_file(infile, stage_remote=True):
    if not stage_remote or not is_xrootd_url(infile):
        return infile, None

    os.makedirs(tmp_mem_directory, exist_ok=True)
    base = os.path.basename(infile)
    tag = uuid.uuid4().hex[:10]
    local_path = os.path.join(tmp_mem_directory, f"{tag}__{base}")

    cmd = ["xrdcp", "-f", "-s", infile, local_path]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("xrdcp not found in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xrdcp failed for {infile} -> {local_path}") from e

    return local_path, local_path


def cleanup_staged_file(path):
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def load_sample_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_list_of_wc_names(fname):
    """Decode WCnames from NanoAOD using the same logic as topcoffea.modules.utils.get_list_of_wc_names."""
    wc_names_lst = []
    tree = uproot.open(f"{fname}:Events")
    if "WCnames" not in tree.keys():
        return wc_names_lst

    wc_info = tree["WCnames"].array(entry_stop=1)[0]
    for item in wc_info:
        h = hex(int(item))[2:]
        wc_fragment = bytes.fromhex(h).decode("utf-8")
        if not wc_fragment.startswith("-"):
            wc_names_lst.append(wc_fragment)
        else:
            wc_names_lst[-1] = wc_names_lst[-1] + wc_fragment[1:]
    return wc_names_lst


def sanitize_wc_token(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def make_eft_derivative_branch_names(wc_names):
    names = ["EFTWeight_SM"]
    for j, wc_j in enumerate(wc_names):
        tok_j = sanitize_wc_token(wc_j)
        names.append(f"der_{tok_j}")
        for k in range(j + 1):
            tok_k = sanitize_wc_token(wc_names[k])
            names.append(f"der_{tok_j}_{tok_k}")
    return names


def get_job_file_slice(files, n_jobs, job):
    if n_jobs <= 0:
        raise ValueError(f"nJobs must be positive, got {n_jobs}")
    if job < 0 or job >= n_jobs:
        raise ValueError(f"job must satisfy 0 <= job < nJobs, got job={job}, nJobs={n_jobs}")

    n_files = len(files)
    i_start = (job * n_files) // n_jobs
    i_stop = ((job + 1) * n_files) // n_jobs
    return files[i_start:i_stop], i_start, i_stop


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def to_cartesian(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy2 = px * px + py * py + pz * pz + mass * mass
    energy = np.sqrt(np.maximum(energy2, 0.0))
    return px, py, pz, energy


def rapidity(energy, pz):
    out = np.full_like(energy, np.nan, dtype=np.float32)
    mask = energy > np.abs(pz)
    out[mask] = 0.5 * np.log((energy[mask] + pz[mask]) / (energy[mask] - pz[mask]))
    return out


def invariant_mass(px, py, pz, energy):
    mass2 = energy * energy - px * px - py * py - pz * pz
    return np.sqrt(np.maximum(mass2, 0.0)).astype(np.float32, copy=False)


def select_objects(arrays):
    lep = {
        "pt": arrays["GenDressedLepton_pt"],
        "eta": arrays["GenDressedLepton_eta"],
        "phi": arrays["GenDressedLepton_phi"],
        "mass": arrays["GenDressedLepton_mass"],
        "pdgId": arrays["GenDressedLepton_pdgId"],
    }
    jet = {
        "pt": arrays["GenJet_pt"],
        "eta": arrays["GenJet_eta"],
        "phi": arrays["GenJet_phi"],
        "mass": arrays["GenJet_mass"],
    }

    is_emu = (abs(lep["pdgId"]) == 11) | (abs(lep["pdgId"]) == 13)
    lep_mask = is_emu & (lep["pt"] > 20.0) & (abs(lep["eta"]) < 2.5)
    for key in lep:
        lep[key] = lep[key][lep_mask]

    lep_order = ak.argsort(lep["pt"], axis=1, ascending=False)
    for key in lep:
        lep[key] = lep[key][lep_order]

    jet_mask = (jet["pt"] > 30.0) & (abs(jet["eta"]) < 2.5)
    for key in jet:
        jet[key] = jet[key][jet_mask]

    deta = jet["eta"][:, :, None] - lep["eta"][:, None, :]
    dphi = delta_phi(jet["phi"][:, :, None], lep["phi"][:, None, :])
    dr2 = deta * deta + dphi * dphi
    jet_clean = ak.fill_none(ak.all(dr2 > 0.16, axis=2), True)
    for key in jet:
        jet[key] = jet[key][jet_clean]

    jet_order = ak.argsort(jet["pt"], axis=1, ascending=False)
    for key in jet:
        jet[key] = jet[key][jet_order]

    return lep, jet


def process_file(infile, step_size, max_events, stage_remote=True, weight1fb=None, small=False, wc_names=None, sample_tag=None):
    local_infile, staged_path = stage_in_xrootd_file(infile, stage_remote=stage_remote)

    try:
        outfile = output_path_for_input(infile, small=small, sample_tag=sample_tag)
        total_written = 0
        tree = None
        n_eft_coeff = None
        eft_branch_names = None
        if wc_names is None:
            wc_names = get_list_of_wc_names(local_infile)

        branch_types = {
            "l0_pt": "float32",
            "l0_eta": "float32",
            "l0_phi": "float32",
            "l1_pt": "float32",
            "l1_eta": "float32",
            "l1_phi": "float32",
            "lminus_pt": "float32",
            "lminus_eta": "float32",
            "lminus_phi": "float32",
            "lplus_pt": "float32",
            "lplus_eta": "float32",
            "lplus_phi": "float32",
            "dilep_pt": "float32",
            "dilep_mass": "float32",
            "dilep_dphi": "float32",
            "dilep_deta": "float32",
            "j0_pt": "float32",
            "j0_eta": "float32",
            "j0_phi": "float32",
            "j1_pt": "float32",
            "j1_eta": "float32",
            "j1_phi": "float32",
            "dijet_pt": "float32",
            "dijet_mass": "float32",
            "dijet_dphi": "float32",
            "dijet_deta": "float32",
            "pseudo_mtt": "float32",
            "nJets": "int32",
            "max_obj_pair_pt": "float32",
            "Generator_weight": "float32",
            "Generator_scalePDF": "float32",
            "Generator_x1": "float32",
            "Generator_x2": "float32",
            "Generator_id1": "int32",
            "Generator_id2": "int32",
            "LHEWeight_originalXWGTUP": "float32",
            "weight1fb": "float32",
            "nEFTfitCoefficients": "int32",
        }

        with uproot.recreate(outfile) as fout:
            for arrays in uproot.iterate(
                {local_infile: "Events"},
                expressions=READ_BRANCHES,
                step_size=step_size,
                library="ak",
                how=dict,
            ):
                if max_events >= 0 and total_written >= max_events:
                    break

                leps, jets = select_objects(arrays)

                has_neg = ak.any(leps["pdgId"] > 0, axis=1)
                has_pos = ak.any(leps["pdgId"] < 0, axis=1)
                njets = ak.num(jets["pt"], axis=1)
                event_mask = ak.fill_none(has_neg & has_pos & (njets >= 2), False)
                if int(np.sum(ak.to_numpy(event_mask))) == 0:
                    continue

                for key in leps:
                    leps[key] = leps[key][event_mask]
                for key in jets:
                    jets[key] = jets[key][event_mask]

                njet_np = ak.to_numpy(ak.num(jets["pt"], axis=1)).astype(np.int32, copy=False)

                l0_pt = ak.to_numpy(leps["pt"][:, 0]).astype(np.float32, copy=False)
                l0_eta = ak.to_numpy(leps["eta"][:, 0]).astype(np.float32, copy=False)
                l0_phi = ak.to_numpy(leps["phi"][:, 0]).astype(np.float32, copy=False)
                l0_mass = ak.to_numpy(leps["mass"][:, 0]).astype(np.float32, copy=False)

                l1_pt = ak.to_numpy(leps["pt"][:, 1]).astype(np.float32, copy=False)
                l1_eta = ak.to_numpy(leps["eta"][:, 1]).astype(np.float32, copy=False)
                l1_phi = ak.to_numpy(leps["phi"][:, 1]).astype(np.float32, copy=False)
                l1_mass = ak.to_numpy(leps["mass"][:, 1]).astype(np.float32, copy=False)

                minus_mask = leps["pdgId"] > 0
                plus_mask = leps["pdgId"] < 0

                lminus_pt = ak.to_numpy(ak.firsts(leps["pt"][minus_mask], axis=1)).astype(np.float32, copy=False)
                lminus_eta = ak.to_numpy(ak.firsts(leps["eta"][minus_mask], axis=1)).astype(np.float32, copy=False)
                lminus_phi = ak.to_numpy(ak.firsts(leps["phi"][minus_mask], axis=1)).astype(np.float32, copy=False)

                lplus_pt = ak.to_numpy(ak.firsts(leps["pt"][plus_mask], axis=1)).astype(np.float32, copy=False)
                lplus_eta = ak.to_numpy(ak.firsts(leps["eta"][plus_mask], axis=1)).astype(np.float32, copy=False)
                lplus_phi = ak.to_numpy(ak.firsts(leps["phi"][plus_mask], axis=1)).astype(np.float32, copy=False)

                j0_pt = ak.to_numpy(jets["pt"][:, 0]).astype(np.float32, copy=False)
                j0_eta = ak.to_numpy(jets["eta"][:, 0]).astype(np.float32, copy=False)
                j0_phi = ak.to_numpy(jets["phi"][:, 0]).astype(np.float32, copy=False)
                j0_mass = ak.to_numpy(jets["mass"][:, 0]).astype(np.float32, copy=False)

                j1_pt = ak.to_numpy(jets["pt"][:, 1]).astype(np.float32, copy=False)
                j1_eta = ak.to_numpy(jets["eta"][:, 1]).astype(np.float32, copy=False)
                j1_phi = ak.to_numpy(jets["phi"][:, 1]).astype(np.float32, copy=False)
                j1_mass = ak.to_numpy(jets["mass"][:, 1]).astype(np.float32, copy=False)

                l0_px, l0_py, l0_pz, l0_e = to_cartesian(l0_pt, l0_eta, l0_phi, l0_mass)
                l1_px, l1_py, l1_pz, l1_e = to_cartesian(l1_pt, l1_eta, l1_phi, l1_mass)
                j0_px, j0_py, j0_pz, j0_e = to_cartesian(j0_pt, j0_eta, j0_phi, j0_mass)
                j1_px, j1_py, j1_pz, j1_e = to_cartesian(j1_pt, j1_eta, j1_phi, j1_mass)

                dilep_px = l0_px + l1_px
                dilep_py = l0_py + l1_py
                dilep_pz = l0_pz + l1_pz
                dilep_e = l0_e + l1_e
                dilep_pt = np.sqrt(dilep_px * dilep_px + dilep_py * dilep_py).astype(np.float32, copy=False)
                dilep_mass = invariant_mass(dilep_px, dilep_py, dilep_pz, dilep_e)

                dijet_px = j0_px + j1_px
                dijet_py = j0_py + j1_py
                dijet_pz = j0_pz + j1_pz
                dijet_e = j0_e + j1_e
                dijet_pt = np.sqrt(dijet_px * dijet_px + dijet_py * dijet_py).astype(np.float32, copy=False)
                dijet_mass = invariant_mass(dijet_px, dijet_py, dijet_pz, dijet_e)

                pseudo_mtt = invariant_mass(
                    dilep_px + dijet_px,
                    dilep_py + dijet_py,
                    dilep_pz + dijet_pz,
                    dilep_e + dijet_e,
                )

                lj0_pt = np.sqrt((l0_px + j0_px) * (l0_px + j0_px) + (l0_py + j0_py) * (l0_py + j0_py))
                max_obj_pair_pt = np.maximum.reduce([dijet_pt, dilep_pt, lj0_pt.astype(np.float32, copy=False)]).astype(np.float32, copy=False)

                out = {
                    "l0_pt": l0_pt,
                    "l0_eta": l0_eta,
                    "l0_phi": l0_phi,
                    "l1_pt": l1_pt,
                    "l1_eta": l1_eta,
                    "l1_phi": l1_phi,
                    "lminus_pt": lminus_pt,
                    "lminus_eta": lminus_eta,
                    "lminus_phi": lminus_phi,
                    "lplus_pt": lplus_pt,
                    "lplus_eta": lplus_eta,
                    "lplus_phi": lplus_phi,
                    "dilep_pt": dilep_pt,
                    "dilep_mass": dilep_mass,
                    "dilep_dphi": np.abs(delta_phi(lminus_phi, lplus_phi)).astype(np.float32, copy=False),
                    "dilep_deta": np.abs(lminus_eta - lplus_eta).astype(np.float32, copy=False),
                    "j0_pt": j0_pt,
                    "j0_eta": j0_eta,
                    "j0_phi": j0_phi,
                    "j1_pt": j1_pt,
                    "j1_eta": j1_eta,
                    "j1_phi": j1_phi,
                    "dijet_pt": dijet_pt,
                    "dijet_mass": dijet_mass,
                    "dijet_dphi": np.abs(delta_phi(j0_phi, j1_phi)).astype(np.float32, copy=False),
                    "dijet_deta": np.abs(j0_eta - j1_eta).astype(np.float32, copy=False),
                    "pseudo_mtt": pseudo_mtt,
                    "nJets": njet_np,
                    "max_obj_pair_pt": max_obj_pair_pt,
                    "Generator_weight": ak.to_numpy(arrays["Generator_weight"])[event_mask].astype(np.float32, copy=False),
                    "Generator_scalePDF": ak.to_numpy(arrays["Generator_scalePDF"])[event_mask].astype(np.float32, copy=False),
                    "Generator_x1": ak.to_numpy(arrays["Generator_x1"])[event_mask].astype(np.float32, copy=False),
                    "Generator_x2": ak.to_numpy(arrays["Generator_x2"])[event_mask].astype(np.float32, copy=False),
                    "Generator_id1": ak.to_numpy(arrays["Generator_id1"])[event_mask].astype(np.int32, copy=False),
                    "Generator_id2": ak.to_numpy(arrays["Generator_id2"])[event_mask].astype(np.int32, copy=False),
                    "LHEWeight_originalXWGTUP": ak.to_numpy(arrays["LHEWeight_originalXWGTUP"])[event_mask].astype(np.float32, copy=False),
                    "weight1fb": np.full(np.count_nonzero(ak.to_numpy(event_mask)), np.nan if weight1fb is None else weight1fb, dtype=np.float32),
                    "nEFTfitCoefficients": ak.to_numpy(arrays["nEFTfitCoefficients"])[event_mask].astype(np.int32, copy=False),
                }

                eft = ak.to_numpy(arrays["EFTfitCoefficients"][event_mask]).astype(np.float32, copy=False)
                if eft.ndim != 2:
                    raise RuntimeError(f"Expected EFTfitCoefficients to be 2D after selection, got shape {eft.shape}")

                if n_eft_coeff is None:
                    n_eft_coeff = eft.shape[1]
                    eft_branch_names = make_eft_derivative_branch_names(wc_names)
                    if len(eft_branch_names) != n_eft_coeff:
                        raise RuntimeError(
                            f"WCnames imply {len(eft_branch_names)} EFT basis terms, but EFTfitCoefficients has length {n_eft_coeff}"
                        )
                    for name in eft_branch_names:
                        branch_types[name] = "float32"
                elif eft.shape[1] != n_eft_coeff:
                    raise RuntimeError(
                        f"Inconsistent EFTfitCoefficients size: expected {n_eft_coeff}, got {eft.shape[1]}"
                    )

                # Translate the packed quadratic-polynomial coefficients into derivative-style branches:
                # - der_wc         = coefficient of wc, i.e. first derivative at the SM point
                # - der_wc_wc      = 2 * coefficient of wc^2
                # - der_wc1_wc0    = coefficient of wc1*wc0 for wc1 != wc0
                out["EFTWeight_SM"] = eft[:, 0]
                idx = 1
                for j, wc_j in enumerate(wc_names):
                    tok_j = sanitize_wc_token(wc_j)
                    out[f"der_{tok_j}"] = eft[:, idx]
                    idx += 1
                    for k in range(j + 1):
                        tok_k = sanitize_wc_token(wc_names[k])
                        val = eft[:, idx]
                        if j == k:
                            val = 2.0 * val
                        out[f"der_{tok_j}_{tok_k}"] = val.astype(np.float32, copy=False)
                        idx += 1

                finite_mass = np.isfinite(out["dilep_mass"])
                if not np.all(finite_mass):
                    for key in out:
                        out[key] = out[key][finite_mass]

                if max_events >= 0:
                    remaining = max_events - total_written
                    if remaining <= 0:
                        break
                    if len(out["l0_pt"]) > remaining:
                        for key in out:
                            out[key] = out[key][:remaining]

                if len(out["l0_pt"]) == 0:
                    continue

                if tree is None:
                    tree = fout.mktree("Events", branch_types)
                tree.extend(out)
                total_written += len(out["l0_pt"])

        branch_doc = {
            "l0_pt": "Leading selected GenDressedLepton transverse momentum [GeV]",
            "l0_eta": "Leading selected GenDressedLepton pseudorapidity",
            "l0_phi": "Leading selected GenDressedLepton azimuthal angle",
            "l1_pt": "Subleading selected GenDressedLepton transverse momentum [GeV]",
            "l1_eta": "Subleading selected GenDressedLepton pseudorapidity",
            "l1_phi": "Subleading selected GenDressedLepton azimuthal angle",
            "lminus_pt": "Leading negatively charged selected lepton transverse momentum [GeV]",
            "lminus_eta": "Leading negatively charged selected lepton pseudorapidity",
            "lminus_phi": "Leading negatively charged selected lepton azimuthal angle",
            "lplus_pt": "Leading positively charged selected lepton transverse momentum [GeV]",
            "lplus_eta": "Leading positively charged selected lepton pseudorapidity",
            "lplus_phi": "Leading positively charged selected lepton azimuthal angle",
            "dilep_pt": "Transverse momentum of the leading-lepton pair [GeV]",
            "dilep_mass": "Invariant mass of the leading-lepton pair [GeV]",
            "dilep_dphi": "Absolute delta phi between selected opposite-sign leptons",
            "dilep_deta": "Absolute delta eta between selected opposite-sign leptons",
            "j0_pt": "Leading selected GenJet transverse momentum [GeV]",
            "j0_eta": "Leading selected GenJet pseudorapidity",
            "j0_phi": "Leading selected GenJet azimuthal angle",
            "j1_pt": "Subleading selected GenJet transverse momentum [GeV]",
            "j1_eta": "Subleading selected GenJet pseudorapidity",
            "j1_phi": "Subleading selected GenJet azimuthal angle",
            "dijet_pt": "Transverse momentum of the two leading selected jets [GeV]",
            "dijet_mass": "Invariant mass of the two leading selected jets [GeV]",
            "dijet_dphi": "Absolute delta phi between the two leading selected jets",
            "dijet_deta": "Absolute delta eta between the two leading selected jets",
            "pseudo_mtt": "Invariant mass of the dilepton+dijet system [GeV]",
            "nJets": "Number of selected GenJets",
            "max_obj_pair_pt": "Maximum of {pT(dijet), pT(dilep), pT(l0+j0)} [GeV]",
            "Generator_weight": "Generator event weight",
            "Generator_scalePDF": "Generator PDF scale Q used for incoming partons [GeV]",
            "Generator_x1": "Generator-level incoming parton momentum fraction x1",
            "Generator_x2": "Generator-level incoming parton momentum fraction x2",
            "Generator_id1": "PDG id of incoming parton 1",
            "Generator_id2": "PDG id of incoming parton 2",
            "LHEWeight_originalXWGTUP": "Original LHE event weight",
            "weight1fb": "Sample normalization weight 1000*xsec/nSumOfWeights [fb]",
            "nEFTfitCoefficients": "Number of EFT fit coefficients stored for this event",
        }

        if n_eft_coeff is not None:
            branch_doc["EFTWeight_SM"] = "SM constant term in the quadratic EFT expansion"
            for j, wc_j in enumerate(wc_names):
                tok_j = sanitize_wc_token(wc_j)
                branch_doc[f"der_{tok_j}"] = f"First derivative at the SM point with respect to {wc_j}"
                for k in range(j + 1):
                    tok_k = sanitize_wc_token(wc_names[k])
                    diag_note = " Includes the factor of 2 for the diagonal second derivative." if j == k else ""
                    branch_doc[f"der_{tok_j}_{tok_k}"] = (
                        f"Second-derivative coefficient for {wc_j} and {wc_names[k]}.{diag_note}"
                    )

        rf = ROOT.TFile.Open(outfile, "UPDATE")
        if rf and not rf.IsZombie():
            t = rf.Get("Events")
            if t:
                for bname, title in branch_doc.items():
                    br = t.GetBranch(bname)
                    if br:
                        br.SetTitle(title)
                t.Write("", ROOT.TObject.kOverwrite)
            rf.Close()

        return outfile

    finally:
        cleanup_staged_file(staged_path)


def main():
    parser = argparse.ArgumentParser(prog="make_centralgen_ntuple.py")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-json", type=str, help="Sample JSON with a 'files' list")
    g.add_argument("--file", type=str, help="Input NanoAOD file (local, /store/..., or root://...)")

    parser.add_argument("--redirector", type=str, default=CMS_REDIRECTOR_CERN, help="XRootD redirector")
    parser.add_argument("--step-size", type=int, default=200_000, help="Events per chunk")
    parser.add_argument("--max-events", type=int, default=-1, help="Max selected events to write (-1 = all)")
    parser.add_argument("--nJobs", type=int, default=1, help="Total number of file-splitting jobs for --sample-json mode")
    parser.add_argument("--job", type=int, default=0, help="Job index in [0, nJobs-1] for --sample-json mode")
    parser.add_argument("--no-stage", action="store_true", help="Read remote ROOT files directly instead of staging with xrdcp")
    parser.add_argument("--small", action="store_true", help="Debug mode: smaller chunking/event cap and write to *_small output directory")

    args = parser.parse_args()

    if args.small:
        if args.max_events < 0:
            args.max_events = 10_000
        args.step_size = min(args.step_size, 20_000)

    if args.sample_json:
        sample = load_sample_json(args.sample_json)
        sample_tag = os.path.splitext(os.path.basename(args.sample_json))[0]
        files = sample.get("files", [])
        if not files:
            print(f"[make_centralgen_ntuple] No files found in sample JSON: {args.sample_json}", file=sys.stderr)
            sys.exit(2)

        try:
            files, i_start, i_stop = get_job_file_slice(files, args.nJobs, args.job)
        except ValueError as e:
            print(f"[make_centralgen_ntuple] {e}", file=sys.stderr)
            sys.exit(2)

        if args.small:
            files = files[:3]

        if not files:
            print(
                f"[make_centralgen_ntuple] No files assigned to job {args.job}/{args.nJobs} "
                f"for sample JSON: {args.sample_json} (slice {i_start}:{i_stop})",
                file=sys.stderr,
            )
            return

        xsec = sample.get("xsec")
        n_sum_of_weights = sample.get("nSumOfWeights")
        if xsec is None or n_sum_of_weights in (None, 0):
            print(
                f"[make_centralgen_ntuple] Sample JSON must contain valid xsec and nSumOfWeights for weight1fb: {args.sample_json}",
                file=sys.stderr,
            )
            sys.exit(2)
        weight1fb = np.float32(1000.0 * float(xsec) / float(n_sum_of_weights))

        print(
            f"[make_centralgen_ntuple] sample={args.sample_json} job={args.job}/{args.nJobs} "
            f"files={len(files)} slice={i_start}:{i_stop} weight1fb={weight1fb:.8g}"
        )

        for f in tqdm(files, desc=f"job {args.job}/{args.nJobs}", unit="file"):
            infile = normalize_input_path(f, args.redirector)
            out = process_file(
                infile=infile,
                step_size=args.step_size,
                max_events=args.max_events,
                stage_remote=not args.no_stage,
                weight1fb=weight1fb,
                small=args.small,
                wc_names=sample.get("WCnames"),
                sample_tag=sample_tag,
            )
            print(f"[make_centralgen_ntuple] output: {out}")
        return

    infile = normalize_input_path(args.file, args.redirector)
    out = process_file(
        infile=infile,
        step_size=args.step_size,
        max_events=args.max_events,
        stage_remote=not args.no_stage,
        weight1fb=None,
        small=args.small,
        wc_names=None,
        sample_tag=None,
    )
    print(f"[make_centralgen_ntuple] output: {out}")


if __name__ == "__main__":
    main()
