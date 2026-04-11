# File: data/samples_RunII.py
from __future__ import annotations
from typing import Dict, List, Optional

import os
from pathlib import Path
import sys
import copy

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

import observables
from systematics_RunII import SYSTEMATICS
import common.user as user

# Use Path so that BASE_DIRECTORY / "2018" / "file.root" works.
BASE_DIRECTORY = Path(
    "/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-3-2_nJ2p_nB2p_2l/"
)
ERAS = ["2016", "2016APV", "2017", "2018", "RunII"]

GROUPS = {
    "SingleTop": ["TBar_tch", "TBar_tWch_noFullyHad", "T_tch", "T_tWch_noFullyHad"],
    "DrellYan":  ["DYJetsToLL_M50", "DYJetsToLL_M10to50_LO"],
    "EtaS": ["EtaT_scalar_m343_w2p8_ll"],
    "EtaP": ["EtaT_m343_w2p8_ll"],
}

process_labels = {
    "SingleTop":  "Single top",
    "TTSemi_pow": "t#bar{t} (1l)",
    "TTLep_pow":  "t#bar{t} (2l)",
    "DrellYan":   "DY",
    "EtaS": "#chi_{t}/H",
    "EtaP": "#eta_{t}/A",
}

# -----------------------------
# Base sample (nominal)
# -----------------------------
_base = None


def _get_base() -> RDataLoader:
    global _base
    if _base is None:
        _base = RDataLoader(
            input_paths=[
                str(BASE_DIRECTORY / "2018" / "TTLep_pow_nominal.root"),
            ],
            tree_name="Events",
            branches=(
                observables.OBSERVERS
                + observables.TOP_KINEMATICS
                + observables.LEPTON_KINEMATICS
                + observables.ASYMMETRY
            ),
            selection=None,
            n_split=1,
            splitting_strategy="events",
            strict_branches=False,
            weight_branches=[
                "weight",
                "L1PreFiringWeight_Nom",
                "JetPUID_SF",
                "Pileup_SF",
                "btagSF_fixedWP_SF",
                "lepEle_SF",
                "lepMu_SF",
            ],
            feature_names=(
                observables.TOP_KINEMATICS
                + observables.LEPTON_KINEMATICS
                + observables.ASYMMETRY
            ),
            observer_names=observables.OBSERVERS,
        )
        # FIXME The RDataloader should allow string based selections already in
        # the constructor. Also, & binds stronger than &&!!! So parenthesis are
        # needed.
        _base.addSelection(
            "(lep1_pt>20) & (tr_isvalid>0) & (isOS>0) & (offZ>0)",
            required_branches=["lep1_pt", "isOS", "offZ", "tr_isvalid"],
        )
    return _base

delphes_OBSERVERS = ["Generator_x1", "Generator_x2", "Generator_id1", "Generator_id2", "Generator_scalePDF", "tr_isvalid", "run", "luminosityBlock", "event"]
tt2l_delphes = RDataLoader( 
        input_paths=[ 
            #"/scratch/rschoefbeck/TTLep_pow_selected/",
            "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/",
            #"/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/TTLep_pow_0.root",
            #"/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/TTLep_pow_1.root",
            #"/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/TTLep_pow_2.root",
            #"/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/TTLep_pow_3.root",
            #"/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/TTLep_pow_4.root",
            ],
    tree_name="Events",
    branches=(
        delphes_OBSERVERS 
        + observables.TOP_KINEMATICS
        + observables.LEPTON_KINEMATICS
        + observables.ASYMMETRY
    ),
    selection=None,
    n_split=1,
    splitting_strategy="events",
    strict_branches=True,
    max_files=20,
    weight_branches=[
        "weight1fb",
    ],
    feature_names=(
        observables.TOP_KINEMATICS
        + observables.LEPTON_KINEMATICS
        + observables.ASYMMETRY
    ),
    observer_names=delphes_OBSERVERS,
    weight_rescale = 24.42,# 2016 lumi scale, based on total number of events
)
#tt2l_delphes.addSelection( "(lep1_pt>20) & (tr_isvalid>0) & (isOS>0) & (offZ>0)", required_branches = ["lep1_pt", "isOS", "offZ", "tr_isvalid"]) 

tt2l_delphes_RunII = copy.deepcopy(tt2l_delphes)
tt2l_delphes_RunII.weight_rescale=24.12*137./16.81 # first scale to 2016 (=16.81/fb), then to RunII

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _parse_name(name: str) -> tuple[str, str, Optional[str]]:
    """
    Parse a name of the form

        <process>_<era>
        <process>_<era>_<variation>

    where <process> may contain underscores and
    <era> is one of ERAS.

    Returns (process, era, tag) where tag can be None (→ nominal).
    """
    found_era: Optional[str] = None
    process: Optional[str] = None
    tag: Optional[str] = None

    # Match longer eras first (so '2016APV' wins over '2016').
    for era in sorted(ERAS, key=len, reverse=True):
        middle = f"_{era}_"
        end = f"_{era}"

        if middle in name:
            idx = name.index(middle)
            process = name[:idx]
            tag = name[idx + len(middle) :]  # everything after "<era>_"
            found_era = era
            break
        elif name.endswith(end):
            idx = name.rfind(end)
            process = name[:idx]
            tag = None  # nominal
            found_era = era
            break

    if not found_era or not process:
        raise ValueError(
            f"Could not decode sample name {name!r}.\n"
            "Expected pattern '<process>_<era>' or '<process>_<era>_<variation>'.\n"
            f"Known eras: {', '.join(ERAS)}\n"
            "Examples:\n"
            "  TTSemi_pow_2018\n"
            "  TTSemi_pow_2018_CMS_res_j_0_2018_down\n"
            "  T_tWch_noFullyHad_2018_CMS_res_j_0_2018_up"
        )

    return process, found_era, tag


def _available_tags_for(process: str, era: str) -> List[str]:
    """
    List all tags actually present on disk for a given process and era,
    based on files matching '<process>_<tag>.root'.
    """
    root_dir = BASE_DIRECTORY / era
    if not root_dir.is_dir():
        return []

    prefix = f"{process}_"
    tags = set()
    for f in root_dir.glob("*.root"):
        stem = f.stem
        if stem.startswith(prefix):
            tags.add(stem[len(prefix) :])

    return sorted(tags)


def _available_files_for_era(era: str) -> List[str]:
    """
    List all ROOT filenames for a given era directory.
    Useful for error messages when the process itself is wrong.
    """
    root_dir = BASE_DIRECTORY / era
    if not root_dir.is_dir():
        return []

    return sorted(f.name for f in root_dir.glob("*.root"))


# ----------------------------------------------------------------------
# Helpers for making variations
# ----------------------------------------------------------------------

def _make_variation(process: str, era: str, tag: str, BASE_DIRECTORY: str = BASE_DIRECTORY) -> RDataLoader:
    """
    Construct a variation from a process, era, and tag, e.g.

        process = "TTSemi_pow"
        era     = "2018"
        tag     = "CMS_res_j_0_2018_down"

    Path on disk:
        <BASE_DIRECTORY>/<era>/<process>_<tag>.root

    The new loader is cloned from the baseline sample `_base`.
    """
    if not tag:
        tag = "nominal"

    eras = [era] if era!="RunII" else ["2016", "2016APV", "2017", "2018"]

    rootfiles = []
    for era in eras:
        root_dir = BASE_DIRECTORY / era
        rootfile = root_dir / f"{process}_{tag}.root"

        if not root_dir.is_dir():
            raise FileNotFoundError(
                f"Era directory '{root_dir}' does not exist "
                f"(process={process!r}, tag={tag!r})."
            )

        if not rootfile.is_file():
            raise FileNotFoundError(
                f"Did not find ROOT file for process={process!r}, era={era!r}, tag={tag!r}.\n"
                f"Expected file: {rootfile}"
            )
        rootfiles.append(str(rootfile))

    return _get_base().clone_from_files(rootfiles)

def _make_group_variation(group: str, era: str, tag: str, BASE_DIRECTORY: str = BASE_DIRECTORY) -> RDataLoader:
    """
    Construct a variation for a group of processes, e.g.

        group = "singleTop"
        era   = "2016"
        tag   = "nominal"

    This will look for all files
        <BASE_DIRECTORY>/<era>/<process>_<tag>.root
    for each process in GROUPS[group].
    """
    if not tag:
        tag = "nominal"

    if group not in GROUPS:
        raise FileNotFoundError(
            f"Unknown group {group!r}. Known groups: {', '.join(GROUPS.keys())}"
        )

    eras = [era] if era!="RunII" else ["2016", "2016APV", "2017", "2018"]

    members = GROUPS[group]
    missing = []
    existing = []

    for era in eras:
        root_dir = BASE_DIRECTORY / era
        if not root_dir.is_dir():
            raise FileNotFoundError(
                f"Era directory '{root_dir}' does not exist for group={group!r}, tag={tag!r}."
            )

        for proc in members:
            f = root_dir / f"{proc}_{tag}.root"
            if f.is_file():
                existing.append(str(f))
            else:
                missing.append(str(f))

        if missing:
            raise FileNotFoundError(
                "Some or all files for group "
                f"{group!r}, era={era!r}, tag={tag!r} are missing.\n"
                "Missing files:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

    # Clone base loader using all member files
    return _get_base().clone_from_files(existing)

# ----------------------------------------------------------------------
# Module-level __getattr__: lazy instantiation for all samples
# ----------------------------------------------------------------------
def __getattr__(name: str):
    """
    Lazily construct RDataLoaders on first access.

    Supported patterns:

      - <process>_<era>
            → uses tag 'nominal'
        e.g.  TTSemi_pow_2018

      - <process>_<era>_<tag>
            → uses tag exactly as given
        e.g.  TTSemi_pow_2018_CMS_res_j_0_2018_down
              TTSemi_pow_2018_Uncl_up

      - <group>_<era>_<tag>
            → group of processes defined in GROUPS
        e.g.  singleTop_2016_nominal
              DrellYan_2018_nominal

    On disk:

        BASE_DIRECTORY / era / f"{process}_{tag}.root"
    """
    # --- NEW: ignore special/dunder attributes like __path__, __all__, etc. ---
    if name.startswith("__") and name.endswith("__"):
        # For internal attributes we MUST behave like a normal module
        # and raise AttributeError, otherwise import/tooling gets upset.
        raise AttributeError(name)

    # Parse name into process, era, tag
    # (you can remove your debug print now)
    try:
        process, era, tag = _parse_name(name)
    except ValueError as e:
        # For module-level __getattr__, ImportError ensures the message is visible
        # in "from samples_RunII import <name>" failures.
        raise ImportError(str(e)) from None

    if not tag:
        tag = "nominal"

    is_group = process in GROUPS

    # Try to build the loader
    try:
        if is_group:
            loader = _make_group_variation(process, era, tag)
        else:
            loader = _make_variation(process, era, tag)
    except FileNotFoundError as e:
        root_dir = BASE_DIRECTORY / era
        msg_lines = [
            f"Could not construct sample {name!r}.",
            str(e),
            "",
        ]

        if is_group:
            msg_lines.append(f"{process!r} is defined as a group with members:")
            msg_lines.append("  " + ", ".join(GROUPS[process]))
            msg_lines.append("")
            msg_lines.append(
                f"Each member is expected to have a file "
                f"'<process>_{tag}.root' in {root_dir}"
            )
            msg_lines.append("")
        elif root_dir.is_dir():
            available_tags = _available_tags_for(process, era)
            if available_tags:
                msg_lines.append(
                    f"Available variations for process={process!r} in era={era!r} "
                    f"(i.e. files '{process}_<tag>.root') are:"
                )
                msg_lines.append(", ".join(available_tags))
                msg_lines.append("")
            else:
                msg_lines.append(
                    f"No ROOT files found for process={process!r} in era={era!r}."
                )
                era_files = _available_files_for_era(era)
                if era_files:
                    msg_lines.append(
                        f"Some ROOT files available in {root_dir} (era={era!r}) are:"
                    )
                    max_show = 30
                    shown = era_files[:max_show]
                    msg_lines.append("\n".join(f"  - {f}" for f in shown))
                    if len(era_files) > max_show:
                        msg_lines.append(
                            f"  ... and {len(era_files) - max_show} more."
                        )
                    msg_lines.append("")

        syst_tags = SYSTEMATICS.get(era, [])
        if syst_tags:
            msg_lines.append(
                f"Recognised systematic tags for era={era!r} from SYSTEMATICS "
                "(may or may not exist on disk for this process/group):"
            )
            msg_lines.append(", ".join(sorted(syst_tags)))
            msg_lines.append("")

        raise ImportError("\n".join(msg_lines)) from None

    globals()[name] = loader
    return loader

# ----------------------------------------------------------------------
#  A factory class that exposes the base directory
# ----------------------------------------------------------------------
class Factory:

    def __init__( self, 
            BASE_DIRECTORY: str = BASE_DIRECTORY,
            features: List[str] = None,
            selection: str = None,
            selection_features: List[str] = None,
            ):
        if type(BASE_DIRECTORY) == str:
            self.BASE_DIRECTORY = Path(BASE_DIRECTORY)
        else:
            self.BASE_DIRECTORY = BASE_DIRECTORY

        self.features = features
        self.selection = selection
        self.selection_features = selection_features

    def get(self, process: str, era: str = None, tag: str = None) -> RDataLoader:

        # Prefer already-defined module globals over lazy parsing
        if era is None and tag is None:
            try:
                return globals()[process]
            except KeyError:
                pass

        if not era and tag:
            raise RuntimeError(f"When you specify a tag (here: {tag}) you must specify the era." )
        
        if not era:
            process, era, tag = _parse_name(process)
            
        if not tag:
            tag = "nominal"

        is_group = process in GROUPS

        # Try to build the loader
        try:
            if is_group:
                loader = _make_group_variation(process, era, tag, BASE_DIRECTORY = self.BASE_DIRECTORY)
            else:
                loader = _make_variation(process, era, tag, BASE_DIRECTORY = self.BASE_DIRECTORY)
        except FileNotFoundError as e:
            root_dir = self.BASE_DIRECTORY / era
            msg_lines = [
                f"Could not construct sample {process} {era!r} {tag}.",
                str(e),
                "",
            ]

            if is_group:
                msg_lines.append(f"{process!r} is defined as a group with members:")
                msg_lines.append("  " + ", ".join(GROUPS[process]))
                msg_lines.append("")
                msg_lines.append(
                    f"Each member is expected to have a file "
                    f"'<process>_{tag}.root' in {root_dir}"
                )
                msg_lines.append("")
            elif root_dir.is_dir():
                available_tags = _available_tags_for(process, era)
                if available_tags:
                    msg_lines.append(
                        f"Available variations for process={process!r} in era={era!r} "
                        f"(i.e. files '{process}_<tag>.root') are:"
                    )
                    msg_lines.append(", ".join(available_tags))
                    msg_lines.append("")
                else:
                    msg_lines.append(
                        f"No ROOT files found for process={process!r} in era={era!r}."
                    )
                    era_files = _available_files_for_era(era)
                    if era_files:
                        msg_lines.append(
                            f"Some ROOT files available in {root_dir} (era={era!r}) are:"
                        )
                        max_show = 30
                        shown = era_files[:max_show]
                        msg_lines.append("\n".join(f"  - {f}" for f in shown))
                        if len(era_files) > max_show:
                            msg_lines.append(
                                f"  ... and {len(era_files) - max_show} more."
                            )
                        msg_lines.append("")

            syst_tags = SYSTEMATICS.get(era, [])
            if syst_tags:
                msg_lines.append(
                    f"Recognised systematic tags for era={era!r} from SYSTEMATICS "
                    "(may or may not exist on disk for this process/group):"
                )
                msg_lines.append(", ".join(sorted(syst_tags)))
                msg_lines.append("")

            raise e 
        if self.features:
            loader.setFeatures( self.features )
        if self.selection:
            loader.addSelection( self.selection, required_branches = self.selection_features )
        return loader

if __name__ == "__main__":
    #print("Base:", _base)
    print("Base:", tt2l_delphes)
    F,O,W = tt2l_delphes.materialize(0,"fow")
    print("Shapes:", F.shape, O.shape, W.shape)
