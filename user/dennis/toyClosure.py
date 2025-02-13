import ROOT
import pickle
import sys
import os
import argparse
import numpy as np
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import common.syncer
import common.user as user
from common.muCalibrator import muCalibrator
from common.calibrationPlotter import calibrationPlotter
from helpers import calculateScore
ROOT.gROOT.SetBatch(ROOT.kTRUE)

parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument("--inflate", type=float, default=None)
args = parser.parse_args()

scoreFiles = [
    "output_mu_1p0_20250212_181138_896c164cf7934cdab3b20bd50656d439.npz",
    "output_mu_1p0_20250213_033645_ac060cc2f3474e9a9f26c611ad1da8e4.npz",
    "output_mu_2p0_20250213_034434_731c2aca64744360a8ea7772f37a8bff.npz",
    "output_mu_3p0_20250213_032018_8f033bf3f1394ceea97960506e898da9.npz",
    "output_mu_4p0_20250213_032219_f20b152540d34d96bcc79fc0e5e80a98.npz",
    "output_mu_5p0_20250213_032050_a0b98b55010d4899a29616cf5d950e51.npz",
]

for i,file in enumerate(scoreFiles):
    filename = os.path.join("/users/dennis.schwarz/HEPHY-uncertainty/submission/", file)
    data = np.load(filename, allow_pickle=True)
    if i==0:
        mu_true=data["mu_true"]
        mu_measured=data["mu_measured"]
        mu_measured_up=data["mu_measured_up"]
        mu_measured_down=data["mu_measured_down"]
        toypaths=data["toypaths"]
    else:
        mu_true = np.append(mu_true, data["mu_true"] )
        mu_measured = np.append(mu_measured, data["mu_measured"] )
        mu_measured_up = np.append(mu_measured_up, data["mu_measured_up"] )
        mu_measured_down = np.append(mu_measured_down, data["mu_measured_down"] )
        toypaths = np.append(toypaths, data["toypaths"])

if args.inflate is not None:
    uncert_factor = args.inflate
    for i in range(len(mu_measured)):
        mu_measured_up[i] = mu_measured[i] + uncert_factor * (mu_measured_up[i]-mu_measured[i])
        mu_measured_down[i] = mu_measured[i] - uncert_factor * (mu_measured[i]-mu_measured_down[i])

output_name = os.path.join( user.plot_directory, "ClosureTests", "Toys.pdf" )
p = calibrationPlotter(output_name)
p.setMus(mu_true, mu_measured, mu_measured_down, mu_measured_up)
p.draw()

score, average_width, coverage = calculateScore(mu_true, mu_measured_down, mu_measured_up)
print("SCORE =", score)
print("AVG. WIDTH =", average_width)
print("COVERAGE =", coverage)
