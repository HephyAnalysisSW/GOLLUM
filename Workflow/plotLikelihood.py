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


data = np.load(args.file)

muPoints = data['mu']
deltaQ = data['deltaQ']
p = LikelihoodScanPlotter(muPoints, deltaQ, args.file.replace(".npz", ""))
p.plot_dir = user.plot_directory
p.draw()
