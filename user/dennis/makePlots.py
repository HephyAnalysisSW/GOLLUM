import sys, os
import math
import numpy as np
import ROOT

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from common.FeatureSelector import FeatureSelector
from common.features import feature_names, class_labels
import common.features as features
from data_loader.data_loader_2 import H5DataLoader
from histograms import npHistograms, convertNPtoROOT

# Configure Data loader
NBatches = 100
# processes = class_labels
processes = ["inclusive"]
for process in processes:
    filename = process+'_nominal.h5'
    if process == "inclusive":
        filename = "nominal.h5"
    print("Prosessing %s"%(filename))
    dataLoader = H5DataLoader(
        os.path.join( user.data_directory, filename),
        ['data', 'weights', 'detailed_labels'],
        batch_size         = None,
        n_split            = NBatches,
        selection_function = None
    )

    # Loop over batches and events
    NBatchesDone = 0
    for batch in dataLoader:
        data = batch['data']
        weights = batch['weights']
        rawLabels = batch['detailed_labels']

        for k, feature in enumerate(features.feature_names):
            featureValues = data[:, k]
            binning = npHistograms[process][feature]["binning"]
            npHistograms[process][feature]["hist"] += np.histogram( featureValues, binning, weights=weights )[0]
            npHistograms[process][feature]["sumw2"] += np.histogram( featureValues, binning, weights=weights**2 )[0]

        NBatchesDone += 1
        print("%i/%i batches processed"%(NBatchesDone,NBatches))

# Convert to root hists and put in root file
outfile = ROOT.TFile("output.root","RECREATE")
outfile.cd()
for process in processes:
    for feature in features.feature_names:
        axisLabel = features.plot_options[feature]['tex']
        rootHist = convertNPtoROOT(
            npHist = npHistograms[process][feature]["hist"],
            binedges = npHistograms[process][feature]["binning"],
            sumw2 = npHistograms[process][feature]["sumw2"],
            name = feature+"__"+process,
            axistitle = axisLabel
        )
        rootHist.Write(feature+"__"+process)
outfile.Close()

print("Done.")
