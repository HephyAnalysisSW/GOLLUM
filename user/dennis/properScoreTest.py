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
from common.calibrationPlotter import calibrationPlotter
from helpers import calculateScore,alphaToNu
ROOT.gROOT.SetBatch(ROOT.kTRUE)

def list_files_in_directory(directory_path):
    toyList = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            if ".npz" in file:
                toyList.append(os.path.join(directory_path, file))
    return toyList

def getCorrectGraph(mu_true, mu_measured, mu_measured_up, mu_measured_down):
    mu_t = []
    mu = []
    up = []
    down = []
    for i in range(len(mu_true)):
        if mu_true[i] > mu_measured_down[i] and mu_true[i] < mu_measured_up[i]:
            mu_t.append(mu_true[i])
            mu.append(mu_measured[i])
            up.append(mu_measured_up[i]-mu_measured[i])
            down.append(mu_measured[i]-mu_measured_down[i])
    g = ROOT.TGraphAsymmErrors(len(mu))
    for i in range(len(mu)):
        g.SetPoint(i, mu_t[i], mu[i])
        g.SetPointError(i, 0.0, 0.0, down[i], up[i])
    return g


################################################################################

parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument("--inflate", type=float, default=1.0)
parser.add_argument("--offset", type=float, default=0.0)
args = parser.parse_args()

dir = os.path.join(user.output_directory, "toyFits")
result_files = list_files_in_directory(dir)

mu_true_all = np.array([])
mu_measured_all = np.array([])
mu_measured_up_all = np.array([])
mu_measured_down_all = np.array([])
toypaths_all = np.array([])
for result_file in result_files:
    addGraphs = []
    data = np.load(result_file, allow_pickle=True)
    mu_true = data["mu_true"]
    mu_measured = data["mu_measured"] + args.offset
    mu_measured_up = data["mu_measured"] + args.inflate*(data["mu_measured_up"]-data["mu_measured"]) + args.offset
    mu_measured_down = data["mu_measured"] - args.inflate*(data["mu_measured"]-data["mu_measured_down"]) + args.offset

    mu_true_all = np.append(mu_true_all, mu_true)
    mu_measured_all = np.append(mu_measured_all, mu_measured )
    mu_measured_up_all = np.append(mu_measured_up_all, mu_measured_up )
    mu_measured_down_all = np.append(mu_measured_down_all, mu_measured_down )


addGraphs.append(getCorrectGraph(mu_true_all, mu_measured_all, mu_measured_up_all, mu_measured_down_all))

output_name = os.path.join( user.plot_directory, "ClosureTests", "ProperToyTest.pdf" )
p = calibrationPlotter(output_name)
p.xmax, p.ymax = 3.0, 3.0
p.setMus(mu_true_all, mu_measured_all, mu_measured_down_all, mu_measured_up_all)
for g in addGraphs:
    p.addGraph(g)
p.draw()

score, average_width, coverage = calculateScore(mu_true_all, mu_measured_down_all, mu_measured_up_all, Ntoys_for_penalty=1000)
print("SCORE =", score)
print("AVG. WIDTH =", average_width)
print("COVERAGE =", coverage)
print("-------------------------")
