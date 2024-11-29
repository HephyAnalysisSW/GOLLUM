import sys, os
import math
import numpy as np
import ROOT

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import common.user as user
from data_loader.data_loader_2 import H5DataLoader
from histograms import npHistograms, convertNPtoROOT
import common.common as common
# import selections

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--process', action='store', default=None, help='input file')
argParser.add_argument('--batches', action='store', default=10, type=int)
args = argParser.parse_args()

if args.process == "test":
    filename = os.path.join( user.data_directory, "test.h5")
elif args.process == "nominal":
    filename = "/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/nominal.h5"

print("Prosessing %s"%(filename))
dataLoader = H5DataLoader(
    filename,
    batch_size         = None,
    n_split            = args.batches,
    selection_function = None
)

# Loop over batches and events
NBatchesDone = 0
for batch in dataLoader:
    features, weights, labels = dataLoader.split(batch)
    for k, feature in enumerate(common.feature_names):
        featureValues = features[:, k]
        binning = npHistograms[args.process][feature]["binning"]
        npHistograms[args.process][feature]["hist"] += np.histogram( featureValues, binning, weights=weights )[0]
        npHistograms[args.process][feature]["sumw2"] += np.histogram( featureValues, binning, weights=weights**2 )[0]

    NBatchesDone += 1
    print("%i/%i batches processed"%(NBatchesDone,args.batches))

# Convert to root hists and put in root file
outfile = ROOT.TFile(f"hists/{args.process}.root","RECREATE")
outfile.cd()
for feature in common.feature_names:
    axisLabel = common.plot_options[feature]['tex']
    rootHist = convertNPtoROOT(
        npHist = npHistograms[args.process][feature]["hist"],
        binedges = npHistograms[args.process][feature]["binning"],
        sumw2 = npHistograms[args.process][feature]["sumw2"],
        name = feature+"__"+args.process,
        axistitle = axisLabel
    )
    rootHist.Write(feature)
outfile.Close()

print("Done.")
