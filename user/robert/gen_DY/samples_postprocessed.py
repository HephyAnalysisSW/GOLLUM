import glob
import os
import sys
from dataclasses import dataclass

import ROOT
import uproot

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

from common.user import output_directory
from data.RDataLoader import RDataLoader


DY_GEN_FEATURES = [
    "dy_born_mll",
    "dy_born_yll",
    "dy_born_abs_yll",
    "dy_born_ptll",
    "dy_born_qt_over_m",
    "dy_ptll",
    "dy_qt_over_m",
    "cs_costheta",
    "cs_phi",
    "cs_born_costheta",
    "cs_born_phi",
    "truth_quark_direction",
    "truth_flavour_label",
    "gen_id1",
    "gen_id2",
    "gen_x1",
    "gen_x2",
    "gen_scalePDF",
]

DY_GEN_OBSERVERS = [
    "event_run",
    "event_luminosityBlock",
    "event_event",
    "dy_born_has_candidate",
    "xsec_weight",
]

DY_FIDUCIAL_SELECTION_BRANCHES = [
    "dy_born_has_candidate",
    "dy_born_mll",
    "dy_born_abs_yll",
    "dy_has_candidate",
    "dy_channel",
    "dy_lepminus_pt",
    "dy_lepminus_eta",
    "dy_lepplus_pt",
    "dy_lepplus_eta",
    "dy_leading_lep_pt",
    "dy_subleading_lep_pt",
    "dy_max_abs_lep_eta",
]

_em_central_minus = "((np.abs(dy_lepminus_eta) < 1.44) | ((np.abs(dy_lepminus_eta) > 1.57) & (np.abs(dy_lepminus_eta) < 2.50)))"
_em_central_plus = "((np.abs(dy_lepplus_eta) < 1.44) | ((np.abs(dy_lepplus_eta) > 1.57) & (np.abs(dy_lepplus_eta) < 2.50)))"
_em_forward_gamma_minus = "((np.abs(dy_lepminus_eta) > 2.50) & (np.abs(dy_lepminus_eta) < 2.87))"
_em_forward_gamma_plus = "((np.abs(dy_lepplus_eta) > 2.50) & (np.abs(dy_lepplus_eta) < 2.87))"
_em_forward_hcal_minus = "((np.abs(dy_lepminus_eta) > 3.14) & (np.abs(dy_lepminus_eta) < 4.36))"
_em_forward_hcal_plus = "((np.abs(dy_lepplus_eta) > 3.14) & (np.abs(dy_lepplus_eta) < 4.36))"
_em_endcap_central_minus = "((np.abs(dy_lepminus_eta) > 1.57) & (np.abs(dy_lepminus_eta) < 2.50))"
_em_endcap_central_plus = "((np.abs(dy_lepplus_eta) > 1.57) & (np.abs(dy_lepplus_eta) < 2.50))"

DY_GEN_SELECTION = (
    "(dy_born_has_candidate > 0)"
    " & (dy_has_candidate > 0)"
    " & (dy_born_mll >= 54.) & (dy_born_mll < 150.)"
    " & (dy_born_abs_yll < 3.4)"
    " & ("
    "((dy_channel == 13) & (dy_max_abs_lep_eta < 2.4) & (dy_leading_lep_pt > 20.) & (dy_subleading_lep_pt > 10.))"
    " | "
    f"((dy_channel == 11) & {_em_central_minus} & {_em_central_plus} & (dy_leading_lep_pt > 25.) & (dy_subleading_lep_pt > 15.))"
    " | "
    f"((dy_channel == 11) & ((({_em_central_minus}) & ({_em_forward_gamma_plus}) & (dy_lepminus_pt > 30.) & (dy_lepplus_pt > 20.)) | (({_em_central_plus}) & ({_em_forward_gamma_minus}) & (dy_lepplus_pt > 30.) & (dy_lepminus_pt > 20.))))"
    " | "
    f"((dy_channel == 11) & ((({_em_endcap_central_minus}) & ({_em_forward_hcal_plus}) & (dy_lepminus_pt > 30.) & (dy_lepplus_pt > 20.)) | (({_em_endcap_central_plus}) & ({_em_forward_hcal_minus}) & (dy_lepplus_pt > 30.) & (dy_lepminus_pt > 20.))))"
    ")"
)
DY_PARTON_SELECTION = (
    "(dy_born_has_candidate > 0)"
    " & (dy_born_mll >= 54.) & (dy_born_mll < 150.)"
    " & (dy_born_abs_yll < 3.4)"
)
DY_SELECTIONS = {
    "fiducial": (DY_GEN_SELECTION, DY_FIDUCIAL_SELECTION_BRANCHES),
    "parton": (DY_PARTON_SELECTION, ["dy_born_has_candidate", "dy_born_mll", "dy_born_abs_yll"]),
}
DY_WEIGHT_BRANCHES = ["xsec_weight"]


def _files(dirname):
    return sorted(glob.glob(os.path.join(output_directory, "DY-gen-ntuples", dirname, "*.root")))


def _make_loader(name, tex_name, files, color, max_files=None, selection="fiducial"):
    if selection not in DY_SELECTIONS:
        raise ValueError(f"Unknown DY selection '{selection}'. Known: {', '.join(sorted(DY_SELECTIONS))}")
    selection_string, selection_branches = DY_SELECTIONS[selection]
    required_branches = list(dict.fromkeys(DY_GEN_FEATURES + DY_GEN_OBSERVERS + DY_WEIGHT_BRANCHES + selection_branches))
    usable_files = []
    for path in files:
        try:
            with uproot.open(path, object_cache=None, array_cache=None) as fin:
                if "Events" not in fin:
                    print(f"[samples_postprocessed] skip incomplete file without Events tree: {path}")
                    continue
                keys = set(fin["Events"].keys())
                missing = [branch for branch in required_branches if branch not in keys]
                if missing:
                    print(f"[samples_postprocessed] skip file with missing branches: {path} ({', '.join(missing)})")
                    continue
        except Exception as err:
            print(f"[samples_postprocessed] skip unreadable file: {path} ({type(err).__name__}: {err})")
            continue
        usable_files.append(path)
        if max_files is not None and len(usable_files) >= max_files:
            break
    if not usable_files:
        raise FileNotFoundError(f"No complete postprocessed files available for sample {name}.")

    loader = RDataLoader(
        input_paths=usable_files,
        tree_name="Events",
        branches=required_branches,
        selection=None,
        n_split=1,
        splitting_strategy="files",
        strict_branches=True,
        weight_branches=DY_WEIGHT_BRANCHES,
        feature_names=DY_GEN_FEATURES,
        observer_names=DY_GEN_OBSERVERS,
    )
    loader.addSelection(selection_string, required_branches=selection_branches)
    loader.name = name
    loader.tex_name = tex_name
    loader.color = color
    loader.selection_name = selection
    return loader


@dataclass
class PostProcessedSample:
    name: str
    tex_name: str
    files: list[str]
    color: int

    def get_loader(self, max_files=None, selection="fiducial"):
        return _make_loader(self.name, self.tex_name, self.files, self.color, max_files=max_files, selection=selection)


DYJetsToLL_M50_LO_UL17_files = _files(
    "RunIISummer20UL17NanoAODv9__DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8__106X_mc2017_realistic_v9-v1"
)
DYJetsToLL_M50_LO_ext_UL17_files = _files(
    "RunIISummer20UL17NanoAODv9__DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8__106X_mc2017_realistic_v9_ext1-v1"
)

DYJetsToLL_M50_LO_UL17 = PostProcessedSample(
    name="DYJetsToLL_M50_LO_UL17",
    tex_name="DYJetsToLL M-50 LO UL17",
    files=DYJetsToLL_M50_LO_UL17_files,
    color=ROOT.kAzure + 1,
)

DYJetsToLL_M50_LO_ext_UL17 = PostProcessedSample(
    name="DYJetsToLL_M50_LO_ext_UL17",
    tex_name="DYJetsToLL M-50 LO ext UL17",
    files=DYJetsToLL_M50_LO_ext_UL17_files,
    color=ROOT.kOrange + 7,
)

DYJetsToLL_M50_LO_UL17_merged = PostProcessedSample(
    name="DYJetsToLL_M50_LO_UL17_merged",
    tex_name="DYJetsToLL M-50 LO UL17 + ext",
    files=DYJetsToLL_M50_LO_UL17_files + DYJetsToLL_M50_LO_ext_UL17_files,
    color=ROOT.kAzure + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120",
    tex_name="DYMuMu SMEFTatNLO 50 #leq m_{#mu#mu} < 120",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne"),
    color=ROOT.kAzure + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200",
    tex_name="DYMuMu SMEFTatNLO 120 #leq m_{#mu#mu} < 200",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne"),
    color=ROOT.kOrange + 7,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400",
    tex_name="DYMuMu SMEFTatNLO 200 #leq m_{#mu#mu} < 400",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne"),
    color=ROOT.kGreen + 2,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600",
    tex_name="DYMuMu SMEFTatNLO 400 #leq m_{#mu#mu} < 600",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne"),
    color=ROOT.kRed + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800",
    tex_name="DYMuMu SMEFTatNLO 600 #leq m_{#mu#mu} < 800",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne"),
    color=ROOT.kMagenta + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000",
    tex_name="DYMuMu SMEFTatNLO 800 #leq m_{#mu#mu} < 1000",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne"),
    color=ROOT.kCyan + 2,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500 = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500",
    tex_name="DYMuMu SMEFTatNLO 1000 #leq m_{#mu#mu} < 1500",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne"),
    color=ROOT.kViolet + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf",
    tex_name="DYMuMu SMEFTatNLO m_{#mu#mu} #geq 1500",
    files=_files("DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne"),
    color=ROOT.kBlack,
)

DYMuMu_NLO_EFT_SMEFTatNLO_shortEFT = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_shortEFT",
    tex_name="DYMuMu SMEFTatNLO short EFT config",
    files=(
        DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500.files
    ),
    color=ROOT.kAzure + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_fullEFT = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_fullEFT",
    tex_name="DYMuMu SMEFTatNLO full EFT config",
    files=(
        DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf.files
    ),
    color=ROOT.kRed + 1,
)

DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT = PostProcessedSample(
    name="DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT",
    tex_name="DYMuMu SMEFTatNLO low-mass EFT config",
    files=(
        DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120.files
        + DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200.files
    ),
    color=ROOT.kAzure + 1,
)


all_samples = [
    DYJetsToLL_M50_LO_UL17,
    DYJetsToLL_M50_LO_ext_UL17,
    DYJetsToLL_M50_LO_UL17_merged,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500,
    DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf,
    DYMuMu_NLO_EFT_SMEFTatNLO_shortEFT,
    DYMuMu_NLO_EFT_SMEFTatNLO_fullEFT,
    DYMuMu_NLO_EFT_SMEFTatNLO_lowMassEFT,
]

samples_by_name = {sample.name: sample for sample in all_samples}
