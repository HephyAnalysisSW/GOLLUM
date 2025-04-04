import ROOT
import pickle
import sys
import os
import argparse
import array
import numpy as np
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import common.syncer
import common.user as user
from common.calibrationPlotter import calibrationPlotter
from common.intervalFinder import intervalFinder
from helpers import calculateScore,alphaToNu

ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)
ROOT.gStyle.SetOptStat(0)

def findCrossings(x, y, threshold):
    crossings = []
    for i in range(len(y) - 1):
        # find the two y values, where one is above and one below y = crossing
        if (y[i] - threshold) * (y[i+1] - threshold) < 0:
            # do linear interpolation to find the x value
            x_cross = x[i] + (threshold - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(x_cross)
    return crossings

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
parser.add_argument("--save", action="store_true", default=False)
parser.add_argument("--postfix", type=str, default=None)
args = parser.parse_args()

mu_true_all = np.array([])
mu_measured_all = np.array([])
mu_measured_up_all = np.array([])
mu_measured_down_all = np.array([])

outname = "data_100k_calib_v2.npz" if args.postfix is None else f"data_100k_calib_v2_{args.postfix}.npz"

if args.save:
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

    np.savez(outname, mu_true=mu_true_all, mu_measured=mu_measured_all, mu_measured_up=mu_measured_up_all, mu_measured_down=mu_measured_down_all)
    print(f"Saved file: {outname}")
    sys.exit()


loaded_data = np.load(outname)
mu_true_original = loaded_data["mu_true"]
mu_measured = loaded_data["mu_measured"]
mu_measured_up_orignial = loaded_data["mu_measured_up"]
mu_measured_down_orignial = loaded_data["mu_measured_down"]


cuts = [
    (0.0, 3.0),
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
]
xmin, xmax = 1.0, 1.2
ymin, ymax = 0.6, 0.75


d_dummy = ROOT.TGraph(2, array.array('d', [xmin,xmax]), array.array('d', [ymin,ymax]) )
d_dummy.SetMarkerStyle(20)
d_dummy.SetTitle("")
d_dummy.GetXaxis().SetTitle("Inflate")
d_dummy.GetYaxis().SetTitle("Coverage")
d_dummy.GetXaxis().SetRangeUser(xmin, xmax)
d_dummy.GetYaxis().SetRangeUser(ymin, ymax)
d_dummy.SetMarkerSize(0.0)

c_coverage = ROOT.TCanvas("cov", "", 600, 600)
ROOT.gPad.SetLeftMargin(0.15)
d_dummy.Draw("AP")

graphs = []

Ntoys = []

for (low_thresh, high_thresh) in cuts:
    inflates = [1.0+i*0.01 for i in range(21)]
    coverages = []
    scores = []
    intervals = []
    errors_x = []
    errors_y = []

    for inf in inflates:
        mu_true = mu_true_original
        mu_measured_up = mu_measured + inf*(mu_measured_up_orignial-mu_measured)
        mu_measured_down = mu_measured - inf*(mu_measured-mu_measured_down_orignial)

        # apply boundary
        mu_measured_up[mu_measured_up > 3.0] = 3.01
        mu_measured_down[mu_measured_down < 0.1] = 0.09

        # select entries
        # print(f"{len(mu_true)} toys before selection")
        mask = (mu_true > low_thresh) & (mu_true < high_thresh)
        mu_true = mu_true[mask]
        mu_measured_up = mu_measured_up[mask]
        mu_measured_down = mu_measured_down[mask]
        if inf == 1.0:
            Ntoys.append(len(mu_true))
        # print(f"{len(mu_true)} toys after selection")

        score, average_width, coverage = calculateScore(mu_true, mu_measured_down, mu_measured_up, Ntoys_for_penalty=1000)
        coverages.append(coverage)
        scores.append(score)
        intervals.append(average_width)
        errors_x.append(0.0)
        errors_y.append(1/np.sqrt(len(mu_true)))

    g_coverage = ROOT.TGraphErrors(len(inflates), array.array('d', inflates), array.array('d', coverages), array.array('d', errors_x), array.array('d', errors_y) )
    g_coverage.SetMarkerStyle(20)
    graphs.append(g_coverage)

    if low_thresh < 0.1 and high_thresh > 2.9:
        target =  0.6827
        boundaries = findCrossings(inflates, coverages, target)
        if len(boundaries) > 0:
            inflate_target = boundaries[0]
            print(boundaries)
            line_x = ROOT.TLine(inflate_target, ymin, inflate_target, ymax)
            line_x.SetLineColor(15)
            line_x.SetLineStyle(15)
            line_x.Draw()
            line_y = ROOT.TLine(xmin, target, xmax, target)
            line_y.SetLineColor(15)
            line_y.SetLineStyle(15)
            line_y.Draw()

leg = ROOT.TLegend(0.5, 0.2, 0.9, 0.5)
for i in range(len(cuts)):
    graphs[i].SetMarkerColor(i+1)
    graphs[i].SetLineColor(i+1)
    graphs[i].Draw("PL SAME")
    lo, hi = cuts[i]
    leg.AddEntry(graphs[i], f"{lo} < #mu < {hi}")
leg.Draw()

plotname = os.path.join(user.plot_directory, "ScoreStudies", "Coverage.pdf") if args.postfix is None else os.path.join(user.plot_directory, "ScoreStudies", f"Coverage_{args.postfix}.pdf")
c_coverage.Print(plotname)

print(cuts)
print(Ntoys)
