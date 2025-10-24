import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from ML.IC.IC import InclusiveCrosssection
ic = InclusiveCrosssection.load("/groups/hephy/cms/robert.schoefbeck/Challenge/models/IC/IC_lowMT_VBFJet.pkl")

print(ic.predict('htautau'))
print(ic.predict(0))

