from __future__ import annotations

import os
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from data.RDataLoader import RDataLoader
import common.user as user
import observables
from data.plot_options import plot_options
from typing import Optional, List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# used in EFT sample factory class
import data.samples_RunII as samples_RunII
from data.samples_RunII import Factory as Factory_RunII
from data.samples_RunII import BASE_DIRECTORY as BASE_DIRECTORY_RUNII
from data.samples_RunII import _parse_name
from systematics_RunII import SYSTEMATICS

BASE_DIRECTORY_EFT = "/groups/hephy/cms/ricardo.barrue/CMGRDF_ntuples_ttbar_EFT/v3-2_nJ2p_nB2p_2l"

ERA_LABELS = {"2016APV": "UL16APV", "2016": "UL16", "2018": "UL18"}
MTT_SLICES = ["mtt_0to700", "mtt_700to900", "mtt_900toInf"]
RUNII_ERAS = ["2016APV", "2016", "2018"]

# no longer needed since lumi normalization is done at cmgrdf level
# LUMI = {
#     "2016APV": 19.52,
#     "2016": 16.81,
#     "2017": 41.48,
#     "2018": 59.83,
# }

wc_names = [
    "cQd1",
    "ctj1",
    "cQj31",
    "ctj8", # ML4EFT
    "ctd1",
    "ctd8", # ML4EFT
    "ctGRe", # ML4EFT
    "ctGIm", # ML4EFT
    "cQj11",
    "cQj18", # ML4EFT
    "ctu8", # ML4EFT
    "cQd8", # ML4EFT
    "ctu1",
    "cQu1",
    "cQj38", # ML4EFT
    "cQu8", # ML4EFT
]

# Point around which the EFT samples were generated (make_ntuple.py, cmgrdf-GluonPDF).
# The expansion in EFTWeightInterface is rebased here instead of at the SM, since this
# is the density the events were actually drawn from.
GENERATION_POINT = {w: (-0.5 if "ctG" in w else 1.5) for w in wc_names}


# lower triangular matrix in coefficients
# same as in postprocessing script (make_ntuple)
def _derivative_branches(wcs):
    out = ["EFTWeight_SM"]
    for j, wc_j in enumerate(wcs):
        out.append(f"der_{wc_j}")
        for k in range(j + 1):
            out.append(f"der_{wc_j}_{wcs[k]}")
    return out


eft_derivatives = _derivative_branches(wc_names)

observers = [
    "weight", # lumi*xs/sumw factor (from CMGRDF)
    "Generator_weight", # 1.0 for all samples
    "LHEWeight_originalXWGTUP",
    "nEFTfitCoefficients",
    "run",
    "luminosityBlock",
    "event",
    "EFTWeight_gen", # weight at GENERATION_POINT; the expansion is rebased here
] + eft_derivatives



def _eft_loader(*relpaths: str) -> RDataLoader:

    loader = RDataLoader(
        input_paths=[os.path.join(BASE_DIRECTORY_EFT, relpath) for relpath in relpaths],
        tree_name="Events",
        branches=observables.ALL_FEATURES + observers,
        selection=None,
        n_split=1,
        splitting_strategy="files",
        strict_branches=True,
        weight_branches=[
            "weight", # weight contains lumi*xs/sumw normalization from CMGRDF
            "EFTWeight_gen", # weight at GENERATION_POINT, the point the events were drawn from
            "L1PreFiringWeight_Nom",
            "JetPUID_SF",
            "Pileup_SF",
            "btagSF_fixedWP_SF",
            "lepEle_SF",
            "lepMu_SF",
        ],
        feature_names=observables.ALL_FEATURES,
        observer_names=observers,
        weight_rescale=1000.0 # forgot to convert the cross-section to fb when processing the samples
    )

    # matching the selection in samples_RunII.py
    loader.addSelection(
            "(dilep_mass > 20) & (lep1_pt>20) & (tr_isvalid>0) & (isOS>0) & (offZ>0)",
            required_branches=["dilep_mass", "lep1_pt", "isOS", "offZ", "tr_isvalid"],
        )
    # selection to truncate outliers (see samples_eft_gen) can be added in the config
    return loader

# ----------------------------------------------------------------------
# Base sample (nominal) and lazy variation builders
# ----------------------------------------------------------------------
_base_eft = None


def _get_base() -> RDataLoader:
    """Return the single nominal EFT loader that every variation clones from."""
    global _base_eft
    if _base_eft is None:
        _base_eft = _eft_loader("2016/TT01j2l_UL16_mtt_0to700_nominal.root")
    return _base_eft


def _split_slice(tag: str) -> tuple[List[str], str]:
    """
    Split a tag into the mtt slice(s) it selects and the remaining tag.

    - tag is exactly a slice (e.g. 'mtt_0to700')      -> that slice,  tag 'nominal'
    - tag starts with '<slice>_' (e.g. 'mtt_0to700_Uncl_up') -> that slice, remainder
    - otherwise (e.g. 'nominal', 'Uncl_up')           -> all three slices, tag unchanged
    """
    if tag in MTT_SLICES:
        return [tag], "nominal"
    for mtt_slice in MTT_SLICES:
        prefix = f"{mtt_slice}_"
        if tag.startswith(prefix):
            return [mtt_slice], tag[len(prefix):]
    return list(MTT_SLICES), tag


def _make_variation(era: str, tag: str) -> RDataLoader:
    """
    Construct a TT01j2l_EFT variation from an era and tag, e.g.

        era = "2016", tag = "CMS_res_j_0_2016_up"

    Path(s) on disk:
        <BASE_DIRECTORY_EFT>/<era>/TT01j2l_<ERA_LABELS[era]>_<slice>_<tag>.root

    for each era (expanded from "RunII") and each mtt slice selected by `tag`.
    The new loader is cloned from the baseline sample `_get_base()`.
    """
    eras = RUNII_ERAS if era == "RunII" else [era]
    slices, file_tag = _split_slice(tag)

    files = []
    missing = []
    for one_era in eras:
        era_dir = os.path.join(BASE_DIRECTORY_EFT, one_era)
        if not os.path.isdir(era_dir) or one_era not in ERA_LABELS:
            missing.append(f"{era_dir} (era directory not available for EFT samples)")
            continue
        for mtt_slice in slices:
            rootfile = os.path.join(
                era_dir, f"TT01j2l_{ERA_LABELS[one_era]}_{mtt_slice}_{file_tag}.root"
            )
            if os.path.isfile(rootfile):
                files.append(rootfile)
            else:
                missing.append(rootfile)

    if missing:
        raise FileNotFoundError(
            "Missing EFT ROOT file(s) for era=" + repr(era) + ", tag=" + repr(tag) + ":\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return _get_base().clone_from_files(files)


def __getattr__(name: str):
    """
    Lazily construct TT01j2l_EFT RDataLoaders on first access.

    Supported patterns (see _parse_name / _split_slice):

      - TT01j2l_EFT_<era>                       -> tag 'nominal', all mtt slices
      - TT01j2l_EFT_<era>_<mtt slice>            -> tag 'nominal', one mtt slice
      - TT01j2l_EFT_<era>_<mtt slice>_<tag>      -> one mtt slice
      - TT01j2l_EFT_<era>_<tag>                  -> all mtt slices

    <era> may be "RunII" (expands to 2016APV, 2016, 2018).

    Any other process name is delegated to data.samples_RunII, which mirrors
    the fallback the EFT Factory already performs.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    try:
        process, era, tag = _parse_name(name)
    except ValueError as e:
        raise ImportError(str(e)) from None

    if not tag:
        tag = "nominal"

    if process != "TT01j2l_EFT":
        try:
            loader = getattr(samples_RunII, name)
        except Exception as e:
            raise type(e)(
                f"data.samples_eft does not recognise process {process!r}; delegated "
                f"lookup of {name!r} to data.samples_RunII, which raised: {e}"
            ) from e
        globals()[name] = loader
        return loader

    try:
        loader = _make_variation(era, tag)
    except FileNotFoundError as e:
        era_dir = Path(BASE_DIRECTORY_EFT) / era
        prefix = f"TT01j2l_{ERA_LABELS[era]}_" if era in ERA_LABELS else None
        available = (
            sorted(f.stem[len(prefix):] for f in era_dir.glob("*.root"))
            if prefix and era_dir.is_dir() else []
        )
        raise ImportError(
            f"Could not construct EFT sample {name!r} (era={era!r}, tag={tag!r}).\n{e}\n"
            f"Tags present on disk for era={era!r}: {', '.join(available) if available else '(none)'}\n"
            f"Known systematic tags for era={era!r}: {', '.join(sorted(SYSTEMATICS.get(era, [])))}"
        ) from None

    globals()[name] = loader
    return loader

# Factory class to allow running fits with EFT samples (closure studies) and mixing EFT samples with Run 2 samples
class Factory:

    def __init__( self, 
            BASE_DIRECTORY: str = BASE_DIRECTORY_EFT,
            features: Optional[List[str]] = None,
            selection: Optional[str] = None,
            selection_features: Optional[List[str]] = None,
            ):
        if type(BASE_DIRECTORY) == str:
            self.BASE_DIRECTORY = Path(BASE_DIRECTORY)
        else:
            self.BASE_DIRECTORY = BASE_DIRECTORY

        self.features = features
        self.selection = selection
        self.selection_features = selection_features

        self.Factory_RunII = Factory_RunII(BASE_DIRECTORY_RUNII,
                                           self.features,
                                           self.selection,
                                           self.selection_features)
    
    def get(self, process: str, era: Optional[str] = None, tag: Optional[str] = None) -> RDataLoader:

        if era is None and tag is None:
            try:
                loader = getattr(sys.modules[__name__], process)
            except (AttributeError, ImportError):
                logger.debug(f"EFT loader with name {process} not found! Reverting to Run 2 sample module.")
                loader = self.Factory_RunII.get(process)
        elif process == "TT01j2l_EFT":
            loader = _make_variation(era, tag if tag else "nominal")
        else:
            logger.debug("era/tag given for a non-EFT process, forwarding to Run 2 sample module.")
            loader = self.Factory_RunII.get(process, era, tag)

        if self.features:
            loader.setFeatures( self.features )
        if self.selection:
            loader.addSelection( self.selection, required_branches = self.selection_features )
        return loader


if __name__ == "__main__":

    # lower triangular matrix of derivatives
    print(eft_derivatives)

    base_loader = _get_base()
    print("Base:_nominal.root", base_loader)
    F, O, W = base_loader.materialize(0, "fow")
    print("Shapes:_nominal.root", F.shape, O.shape, W.shape)

    factory = Factory(
        features=base_loader.feature_names,
        selection=base_loader.selection,
        selection_features=base_loader.feature_names
    )

    L_EFT = factory.get("TT01j2l_EFT_2016")
    F, O, W = L_EFT.materialize(0, "fow")
    print("Shapes:L_EFT.root", F.shape, O.shape, W.shape)
    print("w[:5]",W[:5])

    L_from_samples_RunII = factory.get("TTLep_pow_2016")
    F, O, W = L_from_samples_RunII.materialize(0, "fow")
    print(L_from_samples_RunII)
    print("Shapes:L_from_samples_RunII.root", F.shape, O.shape, W.shape)
    print("w[:5]",W[:5])

    # Real check that the systematic EFT files carry the derivative branches:
    # _eft_loader sets strict_branches=True, so a missing der_* branch crashes here.
    L_syst = getattr(sys.modules[__name__], "TT01j2l_EFT_2016_Uncl_up")
    F, O, W = L_syst.materialize(0, "fow")
    print("Shapes:TT01j2l_EFT_2016_Uncl_up", F.shape, O.shape, W.shape)