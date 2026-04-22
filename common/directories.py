from __future__ import annotations

import socket
from pathlib import Path


_HOSTNAME = socket.gethostname()


# Keep the default paths as the default. Only switch explicitly on hepgpu2.
SAMPLES_RUNII_BASE_DIRECTORY = Path(
    "/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-3-2_nJ2p_nB2p_2l/"
)

DELPHES_TT2L_DIRECTORY = Path(
    "/scratch-cbe/users/robert.schoefbeck/TT2lUnbinned/nanoTuples/delphes/v1/TTLep_pow_selected/"
)

if "hepgpu2" in _HOSTNAME:
    DELPHES_TT2L_DIRECTORY = Path("/scratch/rschoefbeck/TTLep_pow_selected/")
