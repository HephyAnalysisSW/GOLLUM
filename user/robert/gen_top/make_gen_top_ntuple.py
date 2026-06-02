#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import uuid

import ROOT

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory, tmp_mem_directory


CMS_REDIRECTOR_CERN = "root://cms-xrd-global.cern.ch/"


ROOT.gInterpreter.Declare(r"""
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "ROOT/RVec.hxx"
#include "TLorentzVector.h"
#include "TVector3.h"

struct GenTTbarVars {
  double parton_Top_pt, parton_Top_eta, parton_Top_phi, parton_Top_mass, parton_Top_y;
  int parton_Top_pdgId;
  double parton_AntiTop_pt, parton_AntiTop_eta, parton_AntiTop_phi, parton_AntiTop_mass, parton_AntiTop_y;
  int parton_AntiTop_pdgId;
  double parton_ttbar_pt, parton_ttbar_mass, parton_ttbar_eta, parton_ttbar_y;
  double parton_ttbar_dEta, parton_ttbar_dAbsEta, parton_Mtt, parton_cosTheta_t;

  double parton_cosThetaPlus_n, parton_cosThetaMinus_n;
  double parton_cosThetaPlus_r, parton_cosThetaMinus_r;
  double parton_cosThetaPlus_k, parton_cosThetaMinus_k;
  double parton_cosThetaPlus_r_star, parton_cosThetaMinus_r_star;
  double parton_cosThetaPlus_k_star, parton_cosThetaMinus_k_star;

  double parton_xi_nn, parton_xi_rr, parton_xi_kk;
  double parton_xi_nr_plus, parton_xi_nr_minus;
  double parton_xi_rk_plus, parton_xi_rk_minus;
  double parton_xi_nk_plus, parton_xi_nk_minus;
  double parton_xi_r_star_k, parton_xi_k_r_star, parton_xi_kk_star;

  double parton_cos_phi, parton_cos_phi_lab, parton_abs_delta_phi_ll_lab;
  double parton_c_hel, parton_c_han;
  int parton_hasGenTops, parton_hasGenSpin, parton_genSpinCat;
};

inline bool HasStatusFlag(int flags, int bit) { return (flags & (1 << bit)) != 0; }

inline double DeltaPhi(double phi1, double phi2) {
  double dphi = std::fmod(phi1 - phi2, 2.0 * M_PI);
  if (dphi > M_PI) dphi -= 2.0 * M_PI;
  if (dphi <= -M_PI) dphi += 2.0 * M_PI;
  return dphi;
}

GenTTbarVars BuildGenTTbarVars(
  const ROOT::RVecF &GenPart_pt,
  const ROOT::RVecF &GenPart_eta,
  const ROOT::RVecF &GenPart_phi,
  const ROOT::RVecF &GenPart_mass,
  const ROOT::RVecI &GenPart_pdgId,
  const ROOT::RVecI &GenPart_status,
  const ROOT::RVecI &GenPart_genPartIdxMother,
  const ROOT::RVecI &GenPart_statusFlags
) {
  GenTTbarVars out;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  out.parton_Top_pt = out.parton_Top_eta = out.parton_Top_phi = out.parton_Top_mass = out.parton_Top_y = nan;
  out.parton_Top_pdgId = 0;
  out.parton_AntiTop_pt = out.parton_AntiTop_eta = out.parton_AntiTop_phi = out.parton_AntiTop_mass = out.parton_AntiTop_y = nan;
  out.parton_AntiTop_pdgId = 0;
  out.parton_ttbar_pt = out.parton_ttbar_mass = out.parton_ttbar_eta = out.parton_ttbar_y = nan;
  out.parton_ttbar_dEta = out.parton_ttbar_dAbsEta = out.parton_Mtt = out.parton_cosTheta_t = nan;

  out.parton_cosThetaPlus_n = out.parton_cosThetaMinus_n = nan;
  out.parton_cosThetaPlus_r = out.parton_cosThetaMinus_r = nan;
  out.parton_cosThetaPlus_k = out.parton_cosThetaMinus_k = nan;
  out.parton_cosThetaPlus_r_star = out.parton_cosThetaMinus_r_star = nan;
  out.parton_cosThetaPlus_k_star = out.parton_cosThetaMinus_k_star = nan;
  out.parton_xi_nn = out.parton_xi_rr = out.parton_xi_kk = nan;
  out.parton_xi_nr_plus = out.parton_xi_nr_minus = nan;
  out.parton_xi_rk_plus = out.parton_xi_rk_minus = nan;
  out.parton_xi_nk_plus = out.parton_xi_nk_minus = nan;
  out.parton_xi_r_star_k = out.parton_xi_k_r_star = out.parton_xi_kk_star = nan;
  out.parton_cos_phi = out.parton_cos_phi_lab = out.parton_abs_delta_phi_ll_lab = nan;
  out.parton_c_hel = out.parton_c_han = nan;
  out.parton_hasGenTops = 0;
  out.parton_hasGenSpin = 0;
  out.parton_genSpinCat = 0;

  const int n = GenPart_pdgId.size();
  std::vector<int> topIdx;
  for (int i = 0; i < n; ++i) {
    if (std::abs(GenPart_pdgId[i]) != 6) continue;
    if (!HasStatusFlag(GenPart_statusFlags[i], 13)) continue;
    topIdx.push_back(i);
  }
  if (topIdx.size() < 2) return out;

  std::sort(topIdx.begin(), topIdx.end(), [&](int i, int j){ return GenPart_pt[i] > GenPart_pt[j]; });
  topIdx.resize(2);
  if (GenPart_pdgId[topIdx[0]] + GenPart_pdgId[topIdx[1]] != 0) return out;

  const int iTop  = GenPart_pdgId[topIdx[0]] ==  6 ? topIdx[0] : topIdx[1];
  const int iTbar = GenPart_pdgId[topIdx[0]] == -6 ? topIdx[0] : topIdx[1];

  TLorentzVector top, tbar;
  top .SetPtEtaPhiM(GenPart_pt[iTop],  GenPart_eta[iTop],  GenPart_phi[iTop],  GenPart_mass[iTop]);
  tbar.SetPtEtaPhiM(GenPart_pt[iTbar], GenPart_eta[iTbar], GenPart_phi[iTbar], GenPart_mass[iTbar]);
  TLorentzVector tt = top + tbar;

  out.parton_Top_pt = top.Pt();
  out.parton_Top_eta = top.Eta();
  out.parton_Top_phi = top.Phi();
  out.parton_Top_mass = top.M();
  out.parton_Top_y = top.Rapidity();
  out.parton_Top_pdgId = GenPart_pdgId[iTop];
  out.parton_AntiTop_pt = tbar.Pt();
  out.parton_AntiTop_eta = tbar.Eta();
  out.parton_AntiTop_phi = tbar.Phi();
  out.parton_AntiTop_mass = tbar.M();
  out.parton_AntiTop_y = tbar.Rapidity();
  out.parton_AntiTop_pdgId = GenPart_pdgId[iTbar];
  out.parton_ttbar_pt = tt.Pt();
  out.parton_ttbar_mass = tt.M();
  out.parton_ttbar_eta = tt.Eta();
  out.parton_ttbar_y = tt.Rapidity();
  out.parton_ttbar_dEta = tbar.Eta() - top.Eta();
  out.parton_ttbar_dAbsEta = std::abs(tbar.Eta()) - std::abs(top.Eta());
  out.parton_Mtt = tt.M();

  TLorentzVector top_cm = top;
  top_cm.Boost(-tt.BoostVector());
  out.parton_cosTheta_t = top_cm.Vect().Unit().Dot(TVector3(0., 0., 1.));
  out.parton_hasGenTops = 1;

  int iLplus = -1, iLminus = -1;
  bool lplusFromTau = false, lminusFromTau = false;
  double bestPtPlus = -1.0, bestPtMinus = -1.0;

  for (int i = 0; i < n; ++i) {
    const int id = GenPart_pdgId[i];
    const int absId = std::abs(id);
    if ((absId != 11 && absId != 13) || GenPart_status[i] != 1) continue;

    const int flags = GenPart_statusFlags[i];
    const bool isPrompt = HasStatusFlag(flags, 0);
    const bool isPromptTau = HasStatusFlag(flags, 3) || HasStatusFlag(flags, 5);
    const bool isDirectHad = HasStatusFlag(flags, 6);
    if (!(isPrompt || isPromptTau) || isDirectHad) continue;

    int m = GenPart_genPartIdxMother[i];
    int topSign = 0;
    for (int guard = 0; m >= 0 && m < n && guard < 50; ++guard) {
      const int mid = GenPart_pdgId[m];
      if (std::abs(mid) == 6) { topSign = mid > 0 ? +1 : -1; break; }
      m = GenPart_genPartIdxMother[m];
    }
    if (topSign == 0) continue;

    if (id < 0 && topSign == +1 && GenPart_pt[i] > bestPtPlus) {
      bestPtPlus = GenPart_pt[i];
      iLplus = i;
      lplusFromTau = isPromptTau;
    } else if (id > 0 && topSign == -1 && GenPart_pt[i] > bestPtMinus) {
      bestPtMinus = GenPart_pt[i];
      iLminus = i;
      lminusFromTau = isPromptTau;
    }
  }
  if (iLplus < 0 || iLminus < 0) return out;

  TLorentzVector lp, lm;
  lp.SetPtEtaPhiM(GenPart_pt[iLplus], GenPart_eta[iLplus], GenPart_phi[iLplus], 0.);
  lm.SetPtEtaPhiM(GenPart_pt[iLminus], GenPart_eta[iLminus], GenPart_phi[iLminus], 0.);
  out.parton_cos_phi_lab = lm.Vect().Unit().Dot(lp.Vect().Unit());
  out.parton_abs_delta_phi_ll_lab = std::abs(DeltaPhi(lm.Phi(), lp.Phi()));

  const double sign_star = (std::abs(top.Rapidity()) > std::abs(tbar.Rapidity())) ? 1.0 :
                           (std::abs(top.Rapidity()) < std::abs(tbar.Rapidity())) ? -1.0 : 0.0;

  TLorentzVector topcms(top), tbarcms(tbar), lpcms(lp), lmcms(lm);
  const TVector3 boostTT = tt.BoostVector();
  topcms.Boost(-boostTT);
  tbarcms.Boost(-boostTT);
  lpcms.Boost(-boostTT);
  lmcms.Boost(-boostTT);

  const double ctb = topcms.CosTheta();
  if (1.0 - ctb * ctb <= 1e-12) return out;
  const double pz_tt = topcms.Pz();

  lpcms.Boost(-topcms.BoostVector());
  lmcms.Boost(-tbarcms.BoostVector());

  const double topphi = -topcms.Phi();
  const double rottheta = -topcms.Theta();
  topcms.RotateZ(topphi); tbarcms.RotateZ(topphi); lpcms.RotateZ(topphi); lmcms.RotateZ(topphi);
  topcms.RotateY(rottheta); tbarcms.RotateY(rottheta); lpcms.RotateY(rottheta); lmcms.RotateY(rottheta);

  topcms.SetPx(-topcms.Px()); tbarcms.SetPx(-tbarcms.Px()); lpcms.SetPx(-lpcms.Px()); lmcms.SetPx(-lmcms.Px());
  if (pz_tt < 0.) {
    topcms.SetPx(-topcms.Px()); topcms.SetPy(-topcms.Py());
    tbarcms.SetPx(-tbarcms.Px()); tbarcms.SetPy(-tbarcms.Py());
    lpcms.SetPx(-lpcms.Px()); lpcms.SetPy(-lpcms.Py());
    lmcms.SetPx(-lmcms.Px()); lmcms.SetPy(-lmcms.Py());
  }

  if (lpcms.P() <= 0. || lmcms.P() <= 0.) return out;
  const TVector3 A(lpcms.Px()/lpcms.P(), lpcms.Py()/lpcms.P(), lpcms.Pz()/lpcms.P());
  const TVector3 B(lmcms.Px()/lmcms.P(), lmcms.Py()/lmcms.P(), lmcms.Pz()/lmcms.P());

  out.parton_cosThetaPlus_r = A.X();  out.parton_cosThetaMinus_r = B.X();
  out.parton_cosThetaPlus_n = A.Y();  out.parton_cosThetaMinus_n = B.Y();
  out.parton_cosThetaPlus_k = A.Z();  out.parton_cosThetaMinus_k = B.Z();
  out.parton_cosThetaPlus_r_star = sign_star * A.X();
  out.parton_cosThetaMinus_r_star = sign_star * B.X();
  out.parton_cosThetaPlus_k_star = sign_star * A.Z();
  out.parton_cosThetaMinus_k_star = sign_star * B.Z();

  out.parton_xi_nn = out.parton_cosThetaPlus_n * out.parton_cosThetaMinus_n;
  out.parton_xi_rr = out.parton_cosThetaPlus_r * out.parton_cosThetaMinus_r;
  out.parton_xi_kk = out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_k;
  out.parton_xi_nr_plus = out.parton_cosThetaPlus_n * out.parton_cosThetaMinus_r + out.parton_cosThetaPlus_r * out.parton_cosThetaMinus_n;
  out.parton_xi_nr_minus = out.parton_cosThetaPlus_n * out.parton_cosThetaMinus_r - out.parton_cosThetaPlus_r * out.parton_cosThetaMinus_n;
  out.parton_xi_rk_plus = out.parton_cosThetaPlus_r * out.parton_cosThetaMinus_k + out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_r;
  out.parton_xi_rk_minus = out.parton_cosThetaPlus_r * out.parton_cosThetaMinus_k - out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_r;
  out.parton_xi_nk_plus = out.parton_cosThetaPlus_n * out.parton_cosThetaMinus_k + out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_n;
  out.parton_xi_nk_minus = out.parton_cosThetaPlus_n * out.parton_cosThetaMinus_k - out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_n;
  out.parton_xi_r_star_k = out.parton_cosThetaPlus_r_star * out.parton_cosThetaMinus_k;
  out.parton_xi_k_r_star = out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_r_star;
  out.parton_xi_kk_star = out.parton_cosThetaPlus_k * out.parton_cosThetaMinus_k_star;
  out.parton_c_hel = A.Dot(B);
  out.parton_c_han = TVector3(A.X(), A.Y(), -A.Z()).Dot(B);
  out.parton_cos_phi = out.parton_c_hel;
  out.parton_hasGenSpin = 1;
  out.parton_genSpinCat = (lplusFromTau || lminusFromTau) ? 1 : 2;
  return out;
}
""")


DERIVED_BRANCHES = [
    "parton_Top_pt", "parton_Top_eta", "parton_Top_phi", "parton_Top_mass", "parton_Top_y", "parton_Top_pdgId",
    "parton_AntiTop_pt", "parton_AntiTop_eta", "parton_AntiTop_phi", "parton_AntiTop_mass", "parton_AntiTop_y", "parton_AntiTop_pdgId",
    "parton_ttbar_pt", "parton_ttbar_mass", "parton_ttbar_eta", "parton_ttbar_y", "parton_ttbar_dEta", "parton_ttbar_dAbsEta",
    "parton_Mtt", "parton_cosTheta_t",
    "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k",
    "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star",
    "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus",
    "parton_xi_nk_plus", "parton_xi_nk_minus", "parton_xi_r_star_k", "parton_xi_k_r_star", "parton_xi_kk_star",
    "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab", "parton_c_hel", "parton_c_han",
    "parton_hasGenTops", "parton_hasGenSpin", "parton_genSpinCat",
]

GEN_WEIGHT_BRANCHES = [
    "run", "luminosityBlock", "event",
    "genWeight", "Generator_weight", "Generator_scalePDF", "Generator_x1", "Generator_x2",
    "Generator_id1", "Generator_id2", "Generator_xpdf1", "Generator_xpdf2", "Generator_binvar",
    "LHEWeight_originalXWGTUP", "LHEPdfWeight", "LHEScaleWeight", "LHEReweightingWeight", "PSWeight",
]

REQUIRED_INPUT_BRANCHES = [
    "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
    "GenPart_pdgId", "GenPart_status", "GenPart_genPartIdxMother", "GenPart_statusFlags",
]


def das_list_files(dataset):
    res = subprocess.run(["dasgoclient", f"--query=file dataset={dataset}"], check=True, capture_output=True, text=True)
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def is_xrootd_url(path):
    return path.startswith("root://")


def make_xrootd_url(redirector, lfn):
    if is_xrootd_url(lfn):
        return lfn
    if not redirector.endswith("/"):
        redirector += "/"
    return redirector + lfn


def normalize_input_path(path, redirector):
    if is_xrootd_url(path):
        return path
    if path.startswith("/store/"):
        return make_xrootd_url(redirector, path)
    return path


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().strip("/"))


def sample_id_from_store_path(infile):
    sp = infile[infile.index("/store/"):] if "/store/" in infile else infile
    parts = sp.split("/")
    if len(parts) >= 8 and parts[1] == "store":
        return sanitize(f"{parts[3]}__{parts[4]}__{parts[6]}")
    return "unknownSample"


def output_path_for_input(infile):
    outdir = os.path.join(output_directory, "Top-gen-ntuples", sample_id_from_store_path(infile))
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, os.path.basename(infile))


def stage_in_xrootd_file(infile):
    if not is_xrootd_url(infile):
        return infile, None
    os.makedirs(tmp_mem_directory, exist_ok=True)
    local_path = os.path.join(tmp_mem_directory, f"{uuid.uuid4().hex[:10]}__{os.path.basename(infile)}")
    subprocess.run(["xrdcp", "-f", "-s", infile, local_path], check=True)
    return local_path, local_path


def cleanup(path):
    if path and os.path.exists(path):
        os.remove(path)


def process_file(infile, max_events=-1):
    local_infile, staged_path = stage_in_xrootd_file(infile)
    try:
        outfile = output_path_for_input(infile)
        df = ROOT.RDataFrame("Events", local_infile)
        if max_events >= 0:
            df = df.Range(max_events)

        missing = [branch for branch in REQUIRED_INPUT_BRANCHES if not df.HasColumn(branch)]
        if missing:
            raise RuntimeError(f"Missing required generator branches in {infile}: {', '.join(missing)}")

        df = df.Define(
            "gen_ttbar",
            "BuildGenTTbarVars(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, "
            "GenPart_pdgId, GenPart_status, GenPart_genPartIdxMother, GenPart_statusFlags)"
        )

        for branch in DERIVED_BRANCHES:
            df = df.Define(branch, f"gen_ttbar.{branch}")

        cols = ROOT.std.vector("string")()
        for branch in DERIVED_BRANCHES + GEN_WEIGHT_BRANCHES:
            if df.HasColumn(branch):
                cols.push_back(branch)

        print(f"[make_gen_top_ntuple] writing {cols.size()} branches to {outfile}")
        df.Snapshot("Events", outfile, cols)
        return outfile
    finally:
        cleanup(staged_path)


def main():
    parser = argparse.ArgumentParser(prog="make_gen_top_ntuple.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", help="DAS dataset name")
    group.add_argument("--file", help="Input NanoAOD file")
    parser.add_argument("--redirector", default=CMS_REDIRECTOR_CERN)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    if args.small and args.max_events < 0:
        args.max_events = 10_000

    if args.sample:
        files = das_list_files(args.sample)
        if args.small:
            files = files[:3]
        for lfn in files:
            cmd = f"./make_gen_top_ntuple.py --file {make_xrootd_url(args.redirector, lfn)}"
            if args.small:
                cmd += " --small"
            print(cmd)
        return

    infile = normalize_input_path(args.file, args.redirector)
    outfile = process_file(infile, max_events=args.max_events)
    print(f"[make_gen_top_ntuple] output: {outfile}")


if __name__ == "__main__":
    main()
