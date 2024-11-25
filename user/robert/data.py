import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from data_loader.data_loader_2 import H5DataLoader
import common.user as user

datasets = ['data', 'weights', 'detailed_labels']
batch_size = None #64**2
n_split    = 10

# Initialize the data loader
def get_data_loader( name = "nominal", n_split=10):
    return H5DataLoader(os.path.join( user.data_directory, name+'.h5') , datasets, batch_size=batch_size, n_split=n_split)

#feature_names = ["PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi","PRI_had_pt", "PRI_had_eta", "PRI_had_phi","PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi","PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi","PRI_n_jets","PRI_jet_all_pt","PRI_met", "PRI_met_phi", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_had_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality", "DER_lep_eta_centrality", ]

if __name__=="__main__":
    # Iterate through the dataset
    for batch in get_data_loader(n_split=1000):
        data = batch['data']
        weights = batch['weights']
        labels = batch['detailed_labels']
        print(data.shape, weights.shape, labels.shape)

        break
