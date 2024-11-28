import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from FeatureSelector import FeatureSelector

selector = FeatureSelector()

inclusive    = selector.build_selector([])
preselection = selector.build_selector(["DER_mass_transverse_met_lep<80"])
VBF          = selector.build_selector(["DER_mass_transverse_met_lep<80", "PRI_jet_leading_pt>=50", "PRI_jet_subleading_pt>=30", "DER_deltaeta_jet_jet>3.0", "DER_mass_vis>40"])
ptH          = selector.build_selector(["DER_mass_transverse_met_lep<80", "DER_pt_h>=100"])
ggH          = lambda data: np.logical_and( ptH(data), np.logical_not( VBF(data) ) )
