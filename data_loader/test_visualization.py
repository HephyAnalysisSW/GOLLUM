import numpy as np
import pandas as pd
import h5py
from visualization import Dataset_visualise 
import matplotlib.pyplot as plt
# from systematics import systematics

# data_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/data/data.parquet"
# labels_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/labels/data.labels"
# detailed_labels_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/detailed_labels/data.detailed_labels"
# weights_path = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train/weights/data.weights"

columns_list = ["PRI_had_pt", "PRI_had_eta", "PRI_had_phi", "PRI_lep_pt", "PRI_lep_eta","PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_jet_num", "PRI_jet_leading_pt","PRI_jet_leading_eta", "PRI_jet_leading_phi", "PRI_jet_subleading_pt","PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt","DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h","DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet","DER_deltar_had_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau","DER_met_phi_centrality", "DER_lep_eta_centrality"]

visualiser = Dataset_visualise("/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train_dataset/syst_train_set_test0.h5")
with h5py.File("/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train_dataset/syst_train_set_test1.h5", 'r') as hf:
    syst_train_set = {
        "data": pd.DataFrame(hf['data'][:],columns=columns_list),
        "labels": pd.Series(hf['labels'][:]),
        "weights": pd.Series(hf['weights'][:]),
        "detailed_labels": pd.Series(hf['detailed_labels'][:])
    }	
# visualiser.pair_plots_syst(df_syst=syst_train_set["data"], columns = ["PRI_had_pt", "PRI_met", "PRI_lep_pt"], sample_size=100)

# visualiser= Dataset_visualise(data_set, name="Higgs_Dataset")
# visualiser.histogram_syst(df_nominal=visualiser.dfall, weight_nominal=visualiser.weights,df_syst=syst_train_set["data"], weight_syst=syst_train_set["weights"], columns=["PRI_had_pt"], nbin=25)
# visualiser.event_vise_syst(df_syst=syst_train_set["data"], columns=["PRI_had_pt", "PRI_met", "PRI_lep_pt"], sample_size=50)
visualiser.event_vise_syst_arrow(df_syst=syst_train_set["data"], columns=["PRI_had_pt", "PRI_met", "PRI_lep_pt"], sample_size=50)                                                                               

# visualiser.examine_dataset()

# visualiser.histogram_dataset(columns=["PRI_had_pt", "PRI_lep_pt", "PRI_met"]) 
# visualiser.correlation_plots(columns=["PRI_had_pt", "PRI_lep_pt", "PRI_met"])
# visualiser.pair_plots(sample_size=1000, columns=["PRI_had_pt", "PRI_lep_pt", "PRI_met"])
# visualiser.stacked_histogram(field_name="PRI_had_pt", mu_hat=1.0, bins=50, y_scale='log')







