import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from ML.Scaler.Scaler import Scaler 
scaler = Scaler.load("/groups/hephy/cms/robert.schoefbeck/SBIPDF/models/Scaler/Scaler_tt2l_inclusive.pkl")

print(scaler.feature_means)
print(scaler.feature_variances)

print(scaler)

