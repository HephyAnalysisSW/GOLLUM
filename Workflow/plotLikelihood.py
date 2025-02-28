import sys
import argparse
sys.path.insert(0, "..")
import numpy as np
from common.LikelihoodScanPlotter import LikelihoodScanPlotter
import common.user as user
import common.syncer

parser = argparse.ArgumentParser(description="Likelihood Scan Plot.")
parser.add_argument("-f","--file", help="Path to the file containing the scan.")
args = parser.parse_args()

# Load arrays
data = np.load(args.file)
muPoints = data['mu']
deltaQ = data['deltaQ']

# make plot
p = LikelihoodScanPlotter(muPoints, deltaQ, args.file.replace(".npz", ""))
p.plot_dir = user.plot_directory
p.draw()

# print interval
# this part can be moved to another file since the plot is only a cross check for us
# but irrelevant for the submission
from common.intervalFinder import intervalFinder
intF = intervalFinder(muPoints, deltaQ, 1.0)
x_thresholds = intF.getInterval()
print(x_thresholds)
