#!/usr/bin/env python3
"""
make_gen_higgs_ntuple.py

Two modes:

A) Query DAS and print per-file commands, then exit:
   ./make_gen_higgs_ntuple.py --sample <dataset_name>

B) Process one NanoAOD ROOT file and write a small gen-level ntuple:
   ./make_gen_higgs_ntuple.py --file <filename>

This version adds a robust network workaround:
- If input is an xrootd URL (root://...), it is first copied locally with `xrdcp`
  into `common.user.tmp_mem_directory`, processed from the local file, and deleted
  afterwards (even on failures).

Other features:
- Minimal scalar output branches.
- ROOT Lorentz vectors for kinematics (stores Higgs rapidity y).
- uproot+awkward for reading.
- Output path: common.user.output_directory/Hgg-gen-ntuples/<sample_id>/<same_basename>.root
- Redirector handling: LFNs start with '/', redirectors end with '/', so concatenation yields //store/...

Selection (simplified CMS-like, generator-based):
- Higgs: GenPart pdgId==25 and isLastCopy (statusFlags bit 13), choose highest-pt if multiple.
- Daughter photons: GenPart pdgId==22 with mother index == Higgs index; take two leading by pT.
- Photon acceptance: |eta|<2.5 excluding 1.4442<|eta|<1.566.
- Scaled pT cuts: pT1/mgg > 1/3 and pT2/mgg > 1/4.
- Diphoton mass window: 100 < mgg < 180.

Note: Your environment uses Awkward1-style API (no ak.take(axis=...), no ak.isfinite).
"""

import argparse
import os
import re
import sys
import subprocess
import uuid
from typing import Dict, Tuple, List, Optional

import numpy as np
import awkward as ak
import uproot

# Keep your local import style
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory, tmp_mem_directory  # noqa: E402
import ROOT  # noqa: E402


# --------------------------
# Redirectors
# --------------------------
CMS_REDIRECTOR_CERN = "root://cms-xrd-global.cern.ch/"
CMS_REDIRECTOR_FNAL = "root://cmsxrootd.fnal.gov/"


# --------------------------
# ROOT helpers (Lorentz vectors in C++)
# --------------------------
ROOT.gInterpreter.Declare(
    r"""
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <Math/Vector4Dfwd.h>
#include <algorithm>
#include <limits>

using ROOT::VecOps::RVec;
using ROOT::Math::PtEtaPhiMVector;

RVec<float> ComputeRapidityFromPtEtaPhiM(const RVec<float>& pt,
                                        const RVec<float>& eta,
                                        const RVec<float>& phi,
                                        const RVec<float>& mass) {
    const size_t n = std::min({pt.size(), eta.size(), phi.size(), mass.size()});
    RVec<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        PtEtaPhiMVector v(pt[i], eta[i], phi[i], mass[i]);
        out[i] = static_cast<float>(v.Rapidity());
    }
    return out;
}

RVec<float> ComputeDiphotonMass(const RVec<float>& pt1,
                                const RVec<float>& eta1,
                                const RVec<float>& phi1,
                                const RVec<float>& pt2,
                                const RVec<float>& eta2,
                                const RVec<float>& phi2) {
    const size_t n = std::min({pt1.size(), eta1.size(), phi1.size(),
                               pt2.size(), eta2.size(), phi2.size()});
    RVec<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        PtEtaPhiMVector g1(pt1[i], eta1[i], phi1[i], 0.0);
        PtEtaPhiMVector g2(pt2[i], eta2[i], phi2[i], 0.0);
        out[i] = static_cast<float>((g1 + g2).M());
    }
    return out;
}

RVec<float> ComputeDiphotonPt(const RVec<float>& pt1,
                              const RVec<float>& eta1,
                              const RVec<float>& phi1,
                              const RVec<float>& pt2,
                              const RVec<float>& eta2,
                              const RVec<float>& phi2) {
    const size_t n = std::min({pt1.size(), eta1.size(), phi1.size(),
                               pt2.size(), eta2.size(), phi2.size()});
    RVec<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        PtEtaPhiMVector g1(pt1[i], eta1[i], phi1[i], 0.0);
        PtEtaPhiMVector g2(pt2[i], eta2[i], phi2[i], 0.0);
        out[i] = static_cast<float>((g1 + g2).Pt());
    }
    return out;
}

RVec<float> ComputeDiphotonRapidity(const RVec<float>& pt1,
                                    const RVec<float>& eta1,
                                    const RVec<float>& phi1,
                                    const RVec<float>& pt2,
                                    const RVec<float>& eta2,
                                    const RVec<float>& phi2) {
    const size_t n = std::min({pt1.size(), eta1.size(), phi1.size(),
                               pt2.size(), eta2.size(), phi2.size()});
    RVec<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        PtEtaPhiMVector g1(pt1[i], eta1[i], phi1[i], 0.0);
        PtEtaPhiMVector g2(pt2[i], eta2[i], phi2[i], 0.0);
        out[i] = static_cast<float>((g1 + g2).Rapidity());
    }
    return out;
}
"""
)


def _as_rvecf(x: np.ndarray):
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return ROOT.VecOps.AsRVec(x)


def _rvecf_to_numpy(v) -> np.ndarray:
    return np.array(list(v), dtype=np.float32)


# --------------------------
# DAS helpers
# --------------------------
def das_list_files(dataset: str) -> list:
    cmd = ["dasgoclient", f'--query=file dataset={dataset}']
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("dasgoclient not found in PATH. Load CMSSW / dasgoclient first.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"dasgoclient failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def is_xrootd_url(path: str) -> bool:
    return path.startswith("root://")


def make_xrootd_url(redirector: str, lfn: str) -> str:
    """
    redirector ends with '/', lfn starts with '/': concatenation yields //store/...
    """
    if is_xrootd_url(lfn):
        return lfn
    if not lfn.startswith("/"):
        raise ValueError(f"LFN must start with '/': got {lfn}")
    if not redirector.endswith("/"):
        raise ValueError(f"Redirector must end with '/': got {redirector}")
    return redirector + lfn


def normalize_input_path(path: str, redirector_default: str) -> str:
    if is_xrootd_url(path):
        return path
    if path.startswith("/store/"):
        return make_xrootd_url(redirector_default, path)
    return path


# --------------------------
# Output path logic
# --------------------------
def _sanitize(s: str) -> str:
    s = s.strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def sample_id_from_store_path(infile: str) -> str:
    # Extract /store/... from xrootd URL if present
    if "/store/" in infile:
        sp = infile[infile.index("/store/") :]
    else:
        sp = infile

    parts = sp.split("/")
    # /store/mc/<campaign>/<primary>/<datatype>/<processing>/.../<file.root>
    if len(parts) >= 8 and parts[1] == "store":
        campaign = parts[3]
        primary = parts[4]
        processing = parts[6]
        return _sanitize(f"{campaign}__{primary}__{processing}")
    return _sanitize("unknownSample")


def output_path_for_input(infile: str) -> str:
    sid = sample_id_from_store_path(infile)
    outdir = os.path.join(output_directory, "Hgg-gen-ntuples", sid)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, os.path.basename(infile))


# --------------------------
# Local staging of xrootd files (xrdcp)
# --------------------------
def stage_in_xrootd_file(infile: str) -> Tuple[str, Optional[str]]:
    """
    If infile is an xrootd URL, copy it to tmp_mem_directory via xrdcp and return (local_path, local_path).
    If infile is local already, return (infile, None).
    The second return value is the path to remove afterwards (None if not staged).
    """
    if not is_xrootd_url(infile):
        return infile, None

    os.makedirs(tmp_mem_directory, exist_ok=True)

    base = os.path.basename(infile)
    # Unique local name to avoid collisions in parallel jobs
    tag = uuid.uuid4().hex[:10]
    local_path = os.path.join(tmp_mem_directory, f"{tag}__{base}")

    # xrdcp will overwrite only if -f; we avoid that by unique name
    cmd = ["xrdcp", "-f", "-s", infile, local_path]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "xrdcp not found in PATH. Ensure XRootD client is available in your environment."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xrdcp failed for {infile} -> {local_path} (exit {e.returncode})") from e

    return local_path, local_path


def cleanup_staged_file(path: Optional[str]) -> None:
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        # best-effort cleanup
        pass


# --------------------------
# Awkward1-compatible helpers
# --------------------------
IS_LAST_COPY_BIT = 13
IS_LAST_COPY_MASK = (1 << IS_LAST_COPY_BIT)


def take_per_event(jagged: ak.Array, idx: ak.Array) -> ak.Array:
    """
    Per-event gather for jagged arrays: jagged[event, idx[event]].
    Works on Awkward1 (no ak.take(axis=...)).
    """
    loc = ak.local_index(jagged, axis=1)
    picked = jagged[loc == idx]
    return ak.firsts(picked, axis=1)


def cms_photon_acceptance(eta: ak.Array) -> ak.Array:
    aeta = abs(eta)
    in_acc = aeta < 2.5
    in_gap = (aeta > 1.4442) & (aeta < 1.566)
    return in_acc & (~in_gap)


def pick_lastcopy_higgs_idx(gen_pdgid, gen_statusflags, gen_pt) -> ak.Array:
    idx = ak.local_index(gen_pdgid, axis=1)
    is_h = (gen_pdgid == 25)
    is_last = (gen_statusflags & IS_LAST_COPY_MASK) != 0
    cand = is_h & is_last
    cand_idx = idx[cand]
    cand_pt = gen_pt[cand]
    order = ak.argsort(cand_pt, axis=1, ascending=False)
    return ak.firsts(cand_idx[order], axis=1)


def higgs_daughter_photons(gen_pdgid, gen_momidx, gen_pt, gen_eta, gen_phi, higgs_idx):
    higgs_idx_filled = ak.fill_none(higgs_idx, -9999)
    sel = (gen_pdgid == 22) & (gen_momidx == higgs_idx_filled)
    return gen_pt[sel], gen_eta[sel], gen_phi[sel]


def sort_pad_two(pt, eta, phi):
    order = ak.argsort(pt, axis=1, ascending=False)
    pt = ak.pad_none(pt[order], 2, axis=1, clip=True)
    eta = ak.pad_none(eta[order], 2, axis=1, clip=True)
    phi = ak.pad_none(phi[order], 2, axis=1, clip=True)
    return pt, eta, phi


def build_gen_hgg_mask(arrays: Dict[str, ak.Array]) -> Tuple[ak.Array, Dict[str, ak.Array]]:
    gen_pt = arrays["GenPart_pt"]
    gen_eta = arrays["GenPart_eta"]
    gen_phi = arrays["GenPart_phi"]
    gen_m = arrays["GenPart_mass"]
    gen_pdgid = arrays["GenPart_pdgId"]
    gen_flags = arrays["GenPart_statusFlags"]
    gen_mom = arrays["GenPart_genPartIdxMother"]

    higgs_idx = pick_lastcopy_higgs_idx(gen_pdgid, gen_flags, gen_pt)
    have_h = ~ak.is_none(higgs_idx)

    h_pt = take_per_event(gen_pt, higgs_idx)
    h_eta = take_per_event(gen_eta, higgs_idx)   # only needed for rapidity computation interface
    h_phi = take_per_event(gen_phi, higgs_idx)
    h_m = take_per_event(gen_m, higgs_idx)

    pho_pt, pho_eta, pho_phi = higgs_daughter_photons(gen_pdgid, gen_mom, gen_pt, gen_eta, gen_phi, higgs_idx)
    pho_pt2, pho_eta2, pho_phi2 = sort_pad_two(pho_pt, pho_eta, pho_phi)

    g1_pt = pho_pt2[:, 0]
    g1_eta = pho_eta2[:, 0]
    g1_phi = pho_phi2[:, 0]
    g2_pt = pho_pt2[:, 1]
    g2_eta = pho_eta2[:, 1]
    g2_phi = pho_phi2[:, 1]

    have2 = have_h & (~ak.is_none(g1_pt)) & (~ak.is_none(g2_pt))
    acc = have2 & cms_photon_acceptance(g1_eta) & cms_photon_acceptance(g2_eta)

    # mgg via ROOT for events with two photons
    have2_np = ak.to_numpy(have2)
    n = len(have2_np)
    mgg_np = np.full(n, np.nan, dtype=np.float32)

    if np.any(have2_np):
        idx = np.where(have2_np)[0]
        g1_pt_np = ak.to_numpy(ak.fill_none(g1_pt, np.nan)).astype(np.float32)
        g1_eta_np = ak.to_numpy(ak.fill_none(g1_eta, np.nan)).astype(np.float32)
        g1_phi_np = ak.to_numpy(ak.fill_none(g1_phi, np.nan)).astype(np.float32)
        g2_pt_np = ak.to_numpy(ak.fill_none(g2_pt, np.nan)).astype(np.float32)
        g2_eta_np = ak.to_numpy(ak.fill_none(g2_eta, np.nan)).astype(np.float32)
        g2_phi_np = ak.to_numpy(ak.fill_none(g2_phi, np.nan)).astype(np.float32)

        mgg_valid = ROOT.ComputeDiphotonMass(
            _as_rvecf(g1_pt_np[idx]),
            _as_rvecf(g1_eta_np[idx]),
            _as_rvecf(g1_phi_np[idx]),
            _as_rvecf(g2_pt_np[idx]),
            _as_rvecf(g2_eta_np[idx]),
            _as_rvecf(g2_phi_np[idx]),
        )
        mgg_np[idx] = _rvecf_to_numpy(mgg_valid)

    mgg = ak.Array(mgg_np)

    finite_np = np.isfinite(mgg_np) & (mgg_np > 0.0)
    finite_mgg = acc & ak.Array(finite_np)

    scaled = finite_mgg & (g1_pt / mgg > (1.0 / 3.0)) & (g2_pt / mgg > (1.0 / 4.0))
    masswin = finite_mgg & (mgg > 100.0) & (mgg < 180.0)

    final_mask = scaled & masswin

    aux = {
        "H_pt": h_pt,
        "H_eta": h_eta,
        "H_phi": h_phi,
        "H_mass": h_m,
        "g1_pt": g1_pt,
        "g1_eta": g1_eta,
        "g1_phi": g1_phi,
        "g2_pt": g2_pt,
        "g2_eta": g2_eta,
        "g2_phi": g2_phi,
        "mgg": mgg,
    }
    return final_mask, aux


# --------------------------
# Minimal branches to read
# --------------------------
READ_BRANCHES = [
    # GenPart (needed for selection and Higgs/photons)
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_statusFlags",
    "GenPart_genPartIdxMother",
    # Generator (PDF correlation essentials)
    "Generator_weight",
    "Generator_scalePDF",
    "Generator_x1",
    "Generator_x2",
    "Generator_id1",
    "Generator_id2",
    # LHE (optional but usually important for PDF variations)
    "LHEWeight_originalXWGTUP",
    # "LHEPdfWeight",
]


# --------------------------
# Writing helpers
# --------------------------
def _np_float32(x: ak.Array, fill=np.nan) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(x, fill)).astype(np.float32, copy=False)


def _np_int32(x: ak.Array, fill=-999) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(x, fill)).astype(np.int32, copy=False)


def _cast_jagged_to_float32_list(jagged: ak.Array) -> ak.Array:
    """
    Convert jagged array to ak.Array(list-of-float32 lists), filling missing lists with [].
    (Kept for optional LHEPdfWeight; not enabled by default.)
    """
    jagged = ak.fill_none(jagged, [])
    as_list = ak.to_list(jagged)  # list-of-lists
    out_list: List[List[float]] = []
    for row in as_list:
        if row is None:
            out_list.append([])
        else:
            out_list.append(np.asarray(row, dtype=np.float32).tolist())
    return ak.Array(out_list)


# --------------------------
# Processing
# --------------------------
def process_file(infile: str, step_size: int, max_events: int) -> str:
    # Stage-in if remote, and ensure cleanup
    local_infile, staged_path = stage_in_xrootd_file(infile)
    try:
        outfile = output_path_for_input(infile)

        total_written = 0
        tree = None

        branch_types = {
            "H_pt": "float32",
            "H_y": "float32",
            "mgg": "float32",
            "ptgg": "float32",
            "ygg": "float32",
            "Generator_weight": "float32",
            "Generator_scalePDF": "float32",
            "Generator_x1": "float32",
            "Generator_x2": "float32",
            "Generator_id1": "int32",
            "Generator_id2": "int32",
            "LHEWeight_originalXWGTUP": "float32",
            # "LHEPdfWeight": "var * float32",
        }

        with uproot.recreate(outfile) as fout:
            for arrays in uproot.iterate(
                {local_infile: "Events"},
                expressions=READ_BRANCHES,
                step_size=step_size,
                library="ak",
                how=dict,
            ):
                # hard stop on selected events
                if max_events >= 0 and total_written >= max_events:
                    break

                mask, aux = build_gen_hgg_mask(arrays)

                # avoid awkward.sum pitfalls across versions
                n_pass = int(np.sum(ak.to_numpy(mask)))
                if n_pass == 0:
                    continue

                # apply event mask
                sel = {k: v[mask] for k, v in arrays.items()}
                aux_sel = {k: v[mask] for k, v in aux.items()}

                # Higgs rapidity y with ROOT
                H_pt_np = _np_float32(aux_sel["H_pt"])
                H_eta_np = _np_float32(aux_sel["H_eta"])
                H_phi_np = _np_float32(aux_sel["H_phi"])
                H_m_np = _np_float32(aux_sel["H_mass"])
                H_y_np = _rvecf_to_numpy(
                    ROOT.ComputeRapidityFromPtEtaPhiM(
                        _as_rvecf(H_pt_np),
                        _as_rvecf(H_eta_np),
                        _as_rvecf(H_phi_np),
                        _as_rvecf(H_m_np),
                    )
                )

                # diphoton pt, y with ROOT
                g1_pt_np = _np_float32(aux_sel["g1_pt"])
                g1_eta_np = _np_float32(aux_sel["g1_eta"])
                g1_phi_np = _np_float32(aux_sel["g1_phi"])
                g2_pt_np = _np_float32(aux_sel["g2_pt"])
                g2_eta_np = _np_float32(aux_sel["g2_eta"])
                g2_phi_np = _np_float32(aux_sel["g2_phi"])

                mgg_np = ak.to_numpy(aux_sel["mgg"]).astype(np.float32, copy=False)

                ptgg_np = _rvecf_to_numpy(
                    ROOT.ComputeDiphotonPt(
                        _as_rvecf(g1_pt_np),
                        _as_rvecf(g1_eta_np),
                        _as_rvecf(g1_phi_np),
                        _as_rvecf(g2_pt_np),
                        _as_rvecf(g2_eta_np),
                        _as_rvecf(g2_phi_np),
                    )
                )
                ygg_np = _rvecf_to_numpy(
                    ROOT.ComputeDiphotonRapidity(
                        _as_rvecf(g1_pt_np),
                        _as_rvecf(g1_eta_np),
                        _as_rvecf(g1_phi_np),
                        _as_rvecf(g2_pt_np),
                        _as_rvecf(g2_eta_np),
                        _as_rvecf(g2_phi_np),
                    )
                )

                out_w = {
                    "H_pt": H_pt_np,
                    "H_y": H_y_np,
                    "mgg": mgg_np,
                    "ptgg": ptgg_np,
                    "ygg": ygg_np,
                    "Generator_weight": _np_float32(sel["Generator_weight"]),
                    "Generator_scalePDF": _np_float32(sel["Generator_scalePDF"]),
                    "Generator_x1": _np_float32(sel["Generator_x1"]),
                    "Generator_x2": _np_float32(sel["Generator_x2"]),
                    "Generator_id1": _np_int32(sel["Generator_id1"]),
                    "Generator_id2": _np_int32(sel["Generator_id2"]),
                    "LHEWeight_originalXWGTUP": _np_float32(sel["LHEWeight_originalXWGTUP"]),
                    # "LHEPdfWeight": _cast_jagged_to_float32_list(sel["LHEPdfWeight"]),
                }

                if tree is None:
                    tree = fout.mktree("Events", branch_types)

                tree.extend(out_w)
                total_written += len(H_pt_np)

        return outfile
    finally:
        cleanup_staged_file(staged_path)


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(prog="make_gen_higgs_ntuple.py")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=str, help="DAS dataset name (e.g. /GluGluHToGG.../NANOAODSIM)")
    g.add_argument("--file", type=str, help="Input NanoAOD ROOT file (local, /store/..., or root://...)")

    parser.add_argument(
        "--redirector",
        type=str,
        default=CMS_REDIRECTOR_CERN,
        help=f"XRootD redirector prefix (default: {CMS_REDIRECTOR_CERN}; fallback: {CMS_REDIRECTOR_FNAL})",
    )
    # Put back chunk size to 200k as requested
    parser.add_argument("--step-size", type=int, default=200_000, help="Events per chunk for uproot.iterate")
    parser.add_argument("--max-events", type=int, default=-1, help="Max selected events to write (-1 = all)")

    args = parser.parse_args()

    if args.sample:
        files = das_list_files(args.sample)
        if not files:
            print(f"[make_gen_higgs_ntuple] No files found for dataset: {args.sample}", file=sys.stderr)
            sys.exit(2)
        for f in files:
            print(f"./make_gen_higgs_ntuple.py --file {make_xrootd_url(args.redirector, f)}")
        return

    infile = normalize_input_path(args.file, args.redirector)
    out = process_file(infile=infile, step_size=args.step_size, max_events=args.max_events)
    print(f"[make_gen_higgs_ntuple] output: {out}")


if __name__ == "__main__":
    main()

