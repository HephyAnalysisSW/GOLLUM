import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from ML.Scaler.Scaler import Scaler 
scaler = Scaler.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/Scaler/Scaler_lowMT_VBFJet.pkl")

print(scaler.feature_means)
print(scaler.feature_variances)

print(scaler)

