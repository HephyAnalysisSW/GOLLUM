#!/usr/bin/env python3
"""
make_gen_higgs_ntuple.py

Modes:

A) Print per-file commands from DAS:
   ./make_gen_higgs_ntuple.py --sample <dataset>

B) Process one NanoAOD file:
   ./make_gen_higgs_ntuple.py --file <filename>

Notes:
- Remote xrootd inputs are staged locally with xrdcp first.
- Output is a flat scalar ntuple.
- Higgs selection is generator-based, using H -> gamma gamma.
- Added latent-PDF kinematics:
    H_mT, H_abs_y, H_x1, H_x2, H_xmin, H_xmax
    gg_mT, gg_abs_y, gg_x1, gg_x2, gg_xmin, gg_xmax
"""

import argparse
import os
import re
import sys
import subprocess
import uuid
import ROOT
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
    outdir = os.path.join(output_directory, "Hgg-gen-ntuples", sid)
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
            "H_pt": "float32",
            "H_mass": "float32",
            "H_y": "float32",
            "H_abs_y": "float32",
            "H_mT": "float32",
            "H_x1": "float32",
            "H_x2": "float32",
            "H_xmin": "float32",
            "H_xmax": "float32",
            "mgg": "float32",
            "ptgg": "float32",
            "ygg": "float32",
            "gg_abs_y": "float32",
            "gg_mT": "float32",
            "gg_x1": "float32",
            "gg_x2": "float32",
            "gg_xmin": "float32",
            "gg_xmax": "float32",
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

                # Pick last-copy Higgs, highest pt if multiple
                is_h = gen_pdgid == 25
                is_last = (gen_flags & IS_LAST_COPY_MASK) != 0
                h_cand = is_h & is_last
                h_idx = ak.firsts(idx[h_cand][ak.argsort(gen_pt[h_cand], axis=1, ascending=False)], axis=1)

                h_pt = take_per_event(gen_pt, h_idx)
                h_eta = take_per_event(gen_eta, h_idx)
                h_phi = take_per_event(gen_phi, h_idx)
                h_mass = take_per_event(gen_m, h_idx)

                # Daughter photons
                h_idx_filled = ak.fill_none(h_idx, -9999)
                pho_sel = (gen_pdgid == 22) & (gen_mom == h_idx_filled)

                pho_pt = gen_pt[pho_sel]
                pho_eta = gen_eta[pho_sel]
                pho_phi = gen_phi[pho_sel]

                order = ak.argsort(pho_pt, axis=1, ascending=False)
                pho_pt = ak.pad_none(pho_pt[order], 2, axis=1, clip=True)
                pho_eta = ak.pad_none(pho_eta[order], 2, axis=1, clip=True)
                pho_phi = ak.pad_none(pho_phi[order], 2, axis=1, clip=True)

                g1_pt = pho_pt[:, 0]
                g1_eta = pho_eta[:, 0]
                g1_phi = pho_phi[:, 0]
                g2_pt = pho_pt[:, 1]
                g2_eta = pho_eta[:, 1]
                g2_phi = pho_phi[:, 1]

                # Convert to numpy once
                H_pt = to_np_float(h_pt)
                H_eta = to_np_float(h_eta)
                H_phi = to_np_float(h_phi)
                H_mass = to_np_float(h_mass)

                g1_pt = to_np_float(g1_pt)
                g1_eta = to_np_float(g1_eta)
                g1_phi = to_np_float(g1_phi)
                g2_pt = to_np_float(g2_pt)
                g2_eta = to_np_float(g2_eta)
                g2_phi = to_np_float(g2_phi)

                # Higgs truth kinematics
                H_pz = H_pt * np.sinh(H_eta)
                H_E = np.sqrt((H_pt * np.cosh(H_eta)) ** 2 + H_mass ** 2)
                H_y = 0.5 * np.log((H_E + H_pz) / (H_E - H_pz))
                H_abs_y = np.abs(H_y)
                H_mT = np.sqrt(H_mass ** 2 + H_pt ** 2)
                H_x1 = (H_mT / sqrts) * np.exp(+H_y)
                H_x2 = (H_mT / sqrts) * np.exp(-H_y)
                H_xmin = np.minimum(H_x1, H_x2)
                H_xmax = np.maximum(H_x1, H_x2)

                # Diphoton kinematics
                px1 = g1_pt * np.cos(g1_phi)
                py1 = g1_pt * np.sin(g1_phi)
                pz1 = g1_pt * np.sinh(g1_eta)
                E1 = g1_pt * np.cosh(g1_eta)

                px2 = g2_pt * np.cos(g2_phi)
                py2 = g2_pt * np.sin(g2_phi)
                pz2 = g2_pt * np.sinh(g2_eta)
                E2 = g2_pt * np.cosh(g2_eta)

                pxgg = px1 + px2
                pygg = py1 + py2
                pzgg = pz1 + pz2
                Egg = E1 + E2

                ptgg = np.sqrt(pxgg ** 2 + pygg ** 2)
                mgg2 = np.maximum(Egg ** 2 - pxgg ** 2 - pygg ** 2 - pzgg ** 2, 0.0)
                mgg = np.sqrt(mgg2)
                ygg = 0.5 * np.log((Egg + pzgg) / (Egg - pzgg))
                gg_abs_y = np.abs(ygg)
                gg_mT = np.sqrt(mgg ** 2 + ptgg ** 2)
                gg_x1 = (gg_mT / sqrts) * np.exp(+ygg)
                gg_x2 = (gg_mT / sqrts) * np.exp(-ygg)
                gg_xmin = np.minimum(gg_x1, gg_x2)
                gg_xmax = np.maximum(gg_x1, gg_x2)

                # CMS-like photon acceptance
                g1_aeta = np.abs(g1_eta)
                g2_aeta = np.abs(g2_eta)
                g1_acc = (g1_aeta < 2.5) & ~((g1_aeta > 1.4442) & (g1_aeta < 1.566))
                g2_acc = (g2_aeta < 2.5) & ~((g2_aeta > 1.4442) & (g2_aeta < 1.566))

                have_h = np.isfinite(H_pt)
                have2 = np.isfinite(g1_pt) & np.isfinite(g2_pt)
                finite_mgg = np.isfinite(mgg) & (mgg > 0.0)

                scaled = finite_mgg & (g1_pt / mgg > 1.0 / 3.0) & (g2_pt / mgg > 1.0 / 4.0)
                masswin = finite_mgg & (mgg > 100.0) & (mgg < 180.0)

                mask = have_h & have2 & g1_acc & g2_acc & scaled & masswin

                n_pass = int(mask.sum())
                if n_pass == 0:
                    continue

                out = {
                    "H_pt": H_pt[mask],
                    "H_mass": H_mass[mask],
                    "H_y": H_y[mask].astype(np.float32, copy=False),
                    "H_abs_y": H_abs_y[mask].astype(np.float32, copy=False),
                    "H_mT": H_mT[mask].astype(np.float32, copy=False),
                    "H_x1": H_x1[mask].astype(np.float32, copy=False),
                    "H_x2": H_x2[mask].astype(np.float32, copy=False),
                    "H_xmin": H_xmin[mask].astype(np.float32, copy=False),
                    "H_xmax": H_xmax[mask].astype(np.float32, copy=False),
                    "mgg": mgg[mask].astype(np.float32, copy=False),
                    "ptgg": ptgg[mask].astype(np.float32, copy=False),
                    "ygg": ygg[mask].astype(np.float32, copy=False),
                    "gg_abs_y": gg_abs_y[mask].astype(np.float32, copy=False),
                    "gg_mT": gg_mT[mask].astype(np.float32, copy=False),
                    "gg_x1": gg_x1[mask].astype(np.float32, copy=False),
                    "gg_x2": gg_x2[mask].astype(np.float32, copy=False),
                    "gg_xmin": gg_xmin[mask].astype(np.float32, copy=False),
                    "gg_xmax": gg_xmax[mask].astype(np.float32, copy=False),
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

                total_written += len(out["H_pt"])
        # ------------------------------------------------------------------
        # Add branch descriptions with PyROOT after writing with uproot
        # ------------------------------------------------------------------
        branch_doc = {
            "H_pt": "Generator-level Higgs transverse momentum [GeV]",
            "H_mass": "Generator-level Higgs invariant mass [GeV]",
            "H_y": "Generator-level Higgs rapidity y",
            "H_abs_y": "Absolute value of generator-level Higgs rapidity |y|",
            "H_mT": "Generator-level Higgs transverse mass mT = sqrt(m_H^2 + pT_H^2) [GeV]",
            "H_x1": "System-level incoming momentum fraction proxy x1 = (mT/sqrt(s))*exp(+y) from Higgs kinematics",
            "H_x2": "System-level incoming momentum fraction proxy x2 = (mT/sqrt(s))*exp(-y) from Higgs kinematics",
            "H_xmin": "min(H_x1, H_x2)",
            "H_xmax": "max(H_x1, H_x2)",

            "mgg": "Generator-level diphoton invariant mass [GeV]",
            "ptgg": "Generator-level diphoton transverse momentum [GeV]",
            "ygg": "Generator-level diphoton rapidity y",
            "gg_abs_y": "Absolute value of generator-level diphoton rapidity |y|",
            "gg_mT": "Generator-level diphoton transverse mass mT = sqrt(m_gg^2 + pT_gg^2) [GeV]",
            "gg_x1": "System-level incoming momentum fraction proxy x1 = (mT/sqrt(s))*exp(+y) from diphoton kinematics",
            "gg_x2": "System-level incoming momentum fraction proxy x2 = (mT/sqrt(s))*exp(-y) from diphoton kinematics",
            "gg_xmin": "min(gg_x1, gg_x2)",
            "gg_xmax": "max(gg_x1, gg_x2)",

            "Generator_weight": "Generator event weight",
            "Generator_scalePDF": "Generator PDF scale Q used for incoming partons [GeV]",
            "Generator_x1": "Generator-level incoming parton momentum fraction x1",
            "Generator_x2": "Generator-level incoming parton momentum fraction x2",
            "Generator_id1": "PDG id of incoming parton 1",
            "Generator_id2": "PDG id of incoming parton 2",
            "LHEWeight_originalXWGTUP": "Original LHE event weight",
        }

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
    parser = argparse.ArgumentParser(prog="make_gen_higgs_ntuple.py")
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
            args.max_events = 10_000
        args.step_size = min(args.step_size, 20_000)

    if args.sample:
        files = das_list_files(args.sample)
        if not files:
            print(f"[make_gen_higgs_ntuple] No files found for dataset: {args.sample}", file=sys.stderr)
            sys.exit(2)

        if args.small:
            files = files[:3]

        for f in files:
            cmd = f"./make_gen_higgs_ntuple.py --file {make_xrootd_url(args.redirector, f)} --sqrts {args.sqrts}"
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
    print(f"[make_gen_higgs_ntuple] output: {out}")


if __name__ == "__main__":
    main()
