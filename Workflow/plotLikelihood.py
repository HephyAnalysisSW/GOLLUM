import sys
sys.path.insert(0, "..")
import numpy as np
from common.LikelihoodScanPlotter import LikelihoodScanPlotter
import common.user as user
import common.syncer

data = np.load('likelihoodScan.npz')

muPoints = data['mu']
deltaQ = data['deltaQ']
p = LikelihoodScanPlotter(muPoints, deltaQ, "LikelihoodScan")
p.plot_dir = user.plot_directory
p.draw()
