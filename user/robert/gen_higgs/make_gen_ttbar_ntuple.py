#!/usr/bin/env python3
"""
make_gen_ttbar_ntuple.py

Modes:

A) Print per-file commands from DAS:
   ./make_gen_ttbar_ntuple.py --sample <dataset>

B) Process one NanoAOD file:
   ./make_gen_ttbar_ntuple.py --file <filename>

Notes:
- Remote xrootd inputs are staged locally with xrdcp first.
- Output is a flat scalar ntuple.
- ttbar selection is generator-based: require last-copy top and anti-top.
- Added latent-PDF kinematics for the ttbar system:
    ttbar_mT, ttbar_abs_y, ttbar_x1, ttbar_x2, ttbar_xmin, ttbar_xmax
- Added decay info:
    t_decay_flavor, tbar_decay_flavor, ttbar_decay_class, ttbar_ne, ttbar_nmu, ttbar_ntau
- W may be absent in off-shell decays, so decay classification falls back to direct top daughters.
"""

import argparse
import os
import re
import sys
import subprocess
import uuid

import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory, tmp_mem_directory


CMS_REDIRECTOR_CERN = "root://cms-xrd-global.cern.ch/"
CMS_REDIRECTOR_FNAL = "root://cmsxrootd.fnal.gov/"

IS_LAST_COPY_BIT = 13
IS_LAST_COPY_MASK = (1 << IS_LAST_COPY_BIT)

READ_BRANCHES = [
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_statusFlags",
    "GenPart_genPartIdxMother",
    "Generator_weight",
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    "LHEWeight_originalXWGTUP",
]


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
    outdir = os.path.join(output_directory, "TTbar-gen-ntuples", sid)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, os.path.basename(infile))


def stage_in_xrootd_file(infile):
    if not is_xrootd_url(infile):
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


def take_per_event(jagged, idx):
    loc = ak.local_index(jagged, axis=1)
    picked = jagged[loc == idx]
    return ak.firsts(picked, axis=1)


def to_np_float(x, fill=np.nan):
    return ak.to_numpy(ak.fill_none(x, fill)).astype(np.float32, copy=False)


def to_np_int(x, fill=-999):
    return ak.to_numpy(ak.fill_none(x, fill)).astype(np.int32, copy=False)


def process_file(infile, step_size, max_events, sqrts):
    local_infile, staged_path = stage_in_xrootd_file(infile)

    try:
        outfile = output_path_for_input(infile)
        total_written = 0
        tree = None

        branch_types = {
            "t_pt": "float32",
            "t_eta": "float32",
            "t_phi": "float32",
            "t_mass": "float32",
            "t_y": "float32",
            "t_abs_y": "float32",
            "tbar_pt": "float32",
            "tbar_eta": "float32",
            "tbar_phi": "float32",
            "tbar_mass": "float32",
            "tbar_y": "float32",
            "tbar_abs_y": "float32",
            "ttbar_mass": "float32",
            "ttbar_pt": "float32",
            "ttbar_y": "float32",
            "ttbar_abs_y": "float32",
            "ttbar_mT": "float32",
            "ttbar_x1": "float32",
            "ttbar_x2": "float32",
            "ttbar_xmin": "float32",
            "ttbar_xmax": "float32",
            "t_has_W": "int32",
            "tbar_has_W": "int32",
            "t_decay_flavor": "int32",
            "tbar_decay_flavor": "int32",
            "ttbar_decay_class": "int32",
            "ttbar_ne": "int32",
            "ttbar_nmu": "int32",
            "ttbar_ntau": "int32",
            "Generator_weight": "float32",
            "Generator_scalePDF": "float32",
            "Generator_x1": "float32",
            "Generator_x2": "float32",
            "Generator_id1": "int32",
            "Generator_id2": "int32",
            "LHEWeight_originalXWGTUP": "float32",
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

                gen_pt = arrays["GenPart_pt"]
                gen_eta = arrays["GenPart_eta"]
                gen_phi = arrays["GenPart_phi"]
                gen_m = arrays["GenPart_mass"]
                gen_pdgid = arrays["GenPart_pdgId"]
                gen_flags = arrays["GenPart_statusFlags"]
                gen_mom = arrays["GenPart_genPartIdxMother"]

                idx = ak.local_index(gen_pdgid, axis=1)
                is_last = (gen_flags & IS_LAST_COPY_MASK) != 0

                # last-copy top and anti-top, choose highest-pt if multiple
                t_cand = (gen_pdgid == 6) & is_last
                tbar_cand = (gen_pdgid == -6) & is_last

                t_idx = ak.firsts(idx[t_cand][ak.argsort(gen_pt[t_cand], axis=1, ascending=False)], axis=1)
                tbar_idx = ak.firsts(idx[tbar_cand][ak.argsort(gen_pt[tbar_cand], axis=1, ascending=False)], axis=1)

                have_tt = (~ak.is_none(t_idx)) & (~ak.is_none(tbar_idx))
                if int(np.sum(ak.to_numpy(have_tt))) == 0:
                    continue

                # top / antitop kinematics
                t_pt   = take_per_event(gen_pt, t_idx)
                t_eta  = take_per_event(gen_eta, t_idx)
                t_phi  = take_per_event(gen_phi, t_idx)
                t_mass = take_per_event(gen_m, t_idx)

                tbar_pt   = take_per_event(gen_pt, tbar_idx)
                tbar_eta  = take_per_event(gen_eta, tbar_idx)
                tbar_phi  = take_per_event(gen_phi, tbar_idx)
                tbar_mass = take_per_event(gen_m, tbar_idx)

                # direct daughters of top / antitop
                t_idx_filled = ak.fill_none(t_idx, -9999)
                tbar_idx_filled = ak.fill_none(tbar_idx, -9999)

                t_dau_idx = idx[gen_mom == t_idx_filled]
                t_dau_pdg = gen_pdgid[gen_mom == t_idx_filled]

                tbar_dau_idx = idx[gen_mom == tbar_idx_filled]
                tbar_dau_pdg = gen_pdgid[gen_mom == tbar_idx_filled]

                # explicit W if present
                t_W_idx = ak.firsts(t_dau_idx[abs(t_dau_pdg) == 24], axis=1)
                tbar_W_idx = ak.firsts(tbar_dau_idx[abs(tbar_dau_pdg) == 24], axis=1)

                t_has_W = ak.num(t_dau_idx[abs(t_dau_pdg) == 24], axis=1) > 0
                tbar_has_W = ak.num(tbar_dau_idx[abs(tbar_dau_pdg) == 24], axis=1) > 0

                # if W exists: classify from W daughters
                t_W_idx_filled = ak.fill_none(t_W_idx, -9999)
                tbar_W_idx_filled = ak.fill_none(tbar_W_idx, -9999)

                t_W_dau_pdg = gen_pdgid[gen_mom == t_W_idx_filled]
                tbar_W_dau_pdg = gen_pdgid[gen_mom == tbar_W_idx_filled]

                # if no W exists: classify from top daughters excluding b and W
                t_now_pdg = t_dau_pdg[(abs(t_dau_pdg) != 5) & (abs(t_dau_pdg) != 24)]
                tbar_now_pdg = tbar_dau_pdg[(abs(tbar_dau_pdg) != 5) & (abs(tbar_dau_pdg) != 24)]

                t_lep_W = ak.firsts(abs(t_W_dau_pdg[(abs(t_W_dau_pdg) == 11) | (abs(t_W_dau_pdg) == 13) | (abs(t_W_dau_pdg) == 15)]), axis=1)
                tbar_lep_W = ak.firsts(abs(tbar_W_dau_pdg[(abs(tbar_W_dau_pdg) == 11) | (abs(tbar_W_dau_pdg) == 13) | (abs(tbar_W_dau_pdg) == 15)]), axis=1)

                t_lep_now = ak.firsts(abs(t_now_pdg[(abs(t_now_pdg) == 11) | (abs(t_now_pdg) == 13) | (abs(t_now_pdg) == 15)]), axis=1)
                tbar_lep_now = ak.firsts(abs(tbar_now_pdg[(abs(tbar_now_pdg) == 11) | (abs(tbar_now_pdg) == 13) | (abs(tbar_now_pdg) == 15)]), axis=1)

                t_has_W_np = ak.to_numpy(t_has_W)
                tbar_has_W_np = ak.to_numpy(tbar_has_W)

                t_decay_flavor = np.where(t_has_W_np, to_np_int(t_lep_W, fill=0), to_np_int(t_lep_now, fill=0)).astype(np.int32, copy=False)
                tbar_decay_flavor = np.where(tbar_has_W_np, to_np_int(tbar_lep_W, fill=0), to_np_int(tbar_lep_now, fill=0)).astype(np.int32, copy=False)

                ttbar_ne = ((t_decay_flavor == 11).astype(np.int32) + (tbar_decay_flavor == 11).astype(np.int32)).astype(np.int32, copy=False)
                ttbar_nmu = ((t_decay_flavor == 13).astype(np.int32) + (tbar_decay_flavor == 13).astype(np.int32)).astype(np.int32, copy=False)
                ttbar_ntau = ((t_decay_flavor == 15).astype(np.int32) + (tbar_decay_flavor == 15).astype(np.int32)).astype(np.int32, copy=False)

                ttbar_nlep = (t_decay_flavor > 0).astype(np.int32) + (tbar_decay_flavor > 0).astype(np.int32)
                ttbar_decay_class = ttbar_nlep.astype(np.int32, copy=False)  # 0=had, 1=semilep, 2=dilep

                # convert top kinematics to numpy
                t_pt_np   = to_np_float(t_pt)
                t_eta_np  = to_np_float(t_eta)
                t_phi_np  = to_np_float(t_phi)
                t_mass_np = to_np_float(t_mass)

                tbar_pt_np   = to_np_float(tbar_pt)
                tbar_eta_np  = to_np_float(tbar_eta)
                tbar_phi_np  = to_np_float(tbar_phi)
                tbar_mass_np = to_np_float(tbar_mass)

                # top 4-vectors
                t_px = t_pt_np * np.cos(t_phi_np)
                t_py = t_pt_np * np.sin(t_phi_np)
                t_pz = t_pt_np * np.sinh(t_eta_np)
                t_E  = np.sqrt((t_pt_np * np.cosh(t_eta_np)) ** 2 + t_mass_np ** 2)
                t_y  = 0.5 * np.log((t_E + t_pz) / (t_E - t_pz))
                t_abs_y = np.abs(t_y)

                tbar_px = tbar_pt_np * np.cos(tbar_phi_np)
                tbar_py = tbar_pt_np * np.sin(tbar_phi_np)
                tbar_pz = tbar_pt_np * np.sinh(tbar_eta_np)
                tbar_E  = np.sqrt((tbar_pt_np * np.cosh(tbar_eta_np)) ** 2 + tbar_mass_np ** 2)
                tbar_y  = 0.5 * np.log((tbar_E + tbar_pz) / (tbar_E - tbar_pz))
                tbar_abs_y = np.abs(tbar_y)

                # ttbar system
                tt_px = t_px + tbar_px
                tt_py = t_py + tbar_py
                tt_pz = t_pz + tbar_pz
                tt_E  = t_E + tbar_E

                ttbar_pt = np.sqrt(tt_px ** 2 + tt_py ** 2)
                ttbar_mass2 = np.maximum(tt_E ** 2 - tt_px ** 2 - tt_py ** 2 - tt_pz ** 2, 0.0)
                ttbar_mass = np.sqrt(ttbar_mass2)
                ttbar_y = 0.5 * np.log((tt_E + tt_pz) / (tt_E - tt_pz))
                ttbar_abs_y = np.abs(ttbar_y)
                ttbar_mT = np.sqrt(ttbar_mass ** 2 + ttbar_pt ** 2)

                ttbar_x1 = (ttbar_mT / sqrts) * np.exp(+ttbar_y)
                ttbar_x2 = (ttbar_mT / sqrts) * np.exp(-ttbar_y)
                ttbar_xmin = np.minimum(ttbar_x1, ttbar_x2)
                ttbar_xmax = np.maximum(ttbar_x1, ttbar_x2)

                mask = ak.to_numpy(have_tt)
                n_pass = int(mask.sum())
                if n_pass == 0:
                    continue

                out = {
                    "t_pt": t_pt_np[mask],
                    "t_eta": t_eta_np[mask],
                    "t_phi": t_phi_np[mask],
                    "t_mass": t_mass_np[mask],
                    "t_y": t_y[mask].astype(np.float32, copy=False),
                    "t_abs_y": t_abs_y[mask].astype(np.float32, copy=False),

                    "tbar_pt": tbar_pt_np[mask],
                    "tbar_eta": tbar_eta_np[mask],
                    "tbar_phi": tbar_phi_np[mask],
                    "tbar_mass": tbar_mass_np[mask],
                    "tbar_y": tbar_y[mask].astype(np.float32, copy=False),
                    "tbar_abs_y": tbar_abs_y[mask].astype(np.float32, copy=False),

                    "ttbar_mass": ttbar_mass[mask].astype(np.float32, copy=False),
                    "ttbar_pt": ttbar_pt[mask].astype(np.float32, copy=False),
                    "ttbar_y": ttbar_y[mask].astype(np.float32, copy=False),
                    "ttbar_abs_y": ttbar_abs_y[mask].astype(np.float32, copy=False),
                    "ttbar_mT": ttbar_mT[mask].astype(np.float32, copy=False),
                    "ttbar_x1": ttbar_x1[mask].astype(np.float32, copy=False),
                    "ttbar_x2": ttbar_x2[mask].astype(np.float32, copy=False),
                    "ttbar_xmin": ttbar_xmin[mask].astype(np.float32, copy=False),
                    "ttbar_xmax": ttbar_xmax[mask].astype(np.float32, copy=False),

                    "t_has_W": t_has_W_np[mask].astype(np.int32, copy=False),
                    "tbar_has_W": tbar_has_W_np[mask].astype(np.int32, copy=False),
                    "t_decay_flavor": t_decay_flavor[mask].astype(np.int32, copy=False),
                    "tbar_decay_flavor": tbar_decay_flavor[mask].astype(np.int32, copy=False),
                    "ttbar_decay_class": ttbar_decay_class[mask].astype(np.int32, copy=False),
                    "ttbar_ne": ttbar_ne[mask].astype(np.int32, copy=False),
                    "ttbar_nmu": ttbar_nmu[mask].astype(np.int32, copy=False),
                    "ttbar_ntau": ttbar_ntau[mask].astype(np.int32, copy=False),

                    "Generator_weight": ak.to_numpy(arrays["Generator_weight"])[mask].astype(np.float32, copy=False),
                    "Generator_scalePDF": ak.to_numpy(arrays["Generator_scalePDF"])[mask].astype(np.float32, copy=False),
                    "Generator_x1": ak.to_numpy(arrays["Generator_x1"])[mask].astype(np.float32, copy=False),
                    "Generator_x2": ak.to_numpy(arrays["Generator_x2"])[mask].astype(np.float32, copy=False),
                    "Generator_id1": ak.to_numpy(arrays["Generator_id1"])[mask].astype(np.int32, copy=False),
                    "Generator_id2": ak.to_numpy(arrays["Generator_id2"])[mask].astype(np.int32, copy=False),
                    "LHEWeight_originalXWGTUP": ak.to_numpy(arrays["LHEWeight_originalXWGTUP"])[mask].astype(np.float32, copy=False),
                }

                if tree is None:
                    tree = fout.mktree("Events", branch_types)
                tree.extend(out)

                total_written += len(out["t_pt"])

        return outfile

    finally:
        cleanup_staged_file(staged_path)


def main():
    parser = argparse.ArgumentParser(prog="make_gen_ttbar_ntuple.py")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=str, help="DAS dataset name")
    g.add_argument("--file", type=str, help="Input NanoAOD file (local, /store/..., or root://...)")

    parser.add_argument("--redirector", type=str, default=CMS_REDIRECTOR_CERN, help="XRootD redirector")
    parser.add_argument("--step-size", type=int, default=200_000, help="Events per chunk")
    parser.add_argument("--max-events", type=int, default=-1, help="Max selected events to write (-1 = all)")
    parser.add_argument("--sqrts", type=float, default=13000.0, help="Collider sqrt(s) in GeV")
    parser.add_argument("--small", action="store_true", help="Debug mode: smaller chunking and event cap")

    args = parser.parse_args()

    if args.small:
        if args.max_events < 0:
            args.max_events = 10000
        args.step_size = min(args.step_size, 20000)

    if args.sample:
        files = das_list_files(args.sample)
        if not files:
            print(f"[make_gen_ttbar_ntuple] No files found for dataset: {args.sample}", file=sys.stderr)
            sys.exit(2)

        if args.small:
            files = files[:3]

        for f in files:
            cmd = f"./make_gen_ttbar_ntuple.py --file {make_xrootd_url(args.redirector, f)} --sqrts {args.sqrts}"
            if args.small:
                cmd += " --small"
            print(cmd)
        return

    infile = normalize_input_path(args.file, args.redirector)
    out = process_file(
        infile=infile,
        step_size=args.step_size,
        max_events=args.max_events,
        sqrts=args.sqrts,
    )
    print(f"[make_gen_ttbar_ntuple] output: {out}")


if __name__ == "__main__":
    main()
