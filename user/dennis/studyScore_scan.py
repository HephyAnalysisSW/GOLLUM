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
ROOT.gStyle.SetEndErrorSize(0)

def drawWidth(Ntoys, bin_edges, avg_widths_1sigma, avg_widths_2sigma, name):
    xmin = 0.0
    xmax = 3.0
    ymin = 0.0
    ymax = 1.2
    c = ROOT.TCanvas("","",600,600)
    ROOT.gPad.SetTopMargin(0.05)
    ROOT.gPad.SetRightMargin(0.05)
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetBottomMargin(0.15)
    dummy = getDummyGraph(xmin, xmax, ymin, ymax, "#mu_{true}", "Mean interval width")
    dummy.Draw("AP")
    g_1sigma = ROOT.TGraphErrors(len(avg_widths_1sigma))
    g_2sigma = ROOT.TGraphErrors(len(avg_widths_2sigma))
    for i in range(len(bin_edges)):
        if i == len(bin_edges)-1:
            continue
        binWidth = bin_edges[i+1]-bin_edges[i]
        binCenter = bin_edges[i]+0.5*binWidth
        binError = 0.5*binWidth
        g_1sigma.SetPoint(i, binCenter, avg_widths_1sigma[i])
        g_1sigma.SetPointError(i, binError, 0.0)
        g_2sigma.SetPoint(i, binCenter, avg_widths_2sigma[i])
        g_2sigma.SetPointError(i, binError, 0.0)
    g_1sigma.SetMarkerStyle(20)
    g_1sigma.SetMarkerColor(ROOT.kAzure+7)
    g_1sigma.SetLineColor(ROOT.kAzure+7)
    g_1sigma.Draw("P SAME")
    g_2sigma.SetMarkerStyle(20)
    g_2sigma.SetMarkerColor(ROOT.kRed-2)
    g_2sigma.SetLineColor(ROOT.kRed-2)
    g_2sigma.Draw("P SAME")
    leg = ROOT.TLegend(.65, .175, .92, .4)
    leg.AddEntry(g_1sigma, "1#sigma interval", "pel")
    leg.AddEntry(g_2sigma, "2#sigma interval", "pel")
    leg.SetTextSize(0.03)
    leg.Draw()
    ROOT.gPad.RedrawAxis()
    c.Print(name)

def drawCoverage(Ntoys, bin_edges, coverages_1sigma, coverages_2sigma, name):
    xmin = 0.0
    xmax = 3.0
    ymin = 0.0
    ymax = 1.05
    c = ROOT.TCanvas("","",600,600)
    ROOT.gPad.SetTopMargin(0.05)
    ROOT.gPad.SetRightMargin(0.05)
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetBottomMargin(0.15)
    dummy = getDummyGraph(xmin, xmax, ymin, ymax, "#mu_{true}", "Coverage")
    dummy.Draw("AP")
    # lines
    line_6827 = ROOT.TLine(xmin, 0.6827, xmax, 0.6827)
    line_6827.SetLineColor(15)
    line_6827.SetLineStyle(2)
    line_6827.SetLineWidth(2)
    line_6827.Draw()
    line_9500 = ROOT.TLine(xmin, 0.95, xmax, 0.95)
    line_9500.SetLineColor(15)
    line_9500.SetLineStyle(9)
    line_9500.SetLineWidth(2)
    line_9500.Draw()
    #
    g_1sigma = ROOT.TGraphErrors(len(coverages_1sigma))
    g_2sigma = ROOT.TGraphErrors(len(coverages_2sigma))
    for i in range(len(bin_edges)):
        if i == len(bin_edges)-1:
            continue
        binWidth = bin_edges[i+1]-bin_edges[i]
        binCenter = bin_edges[i]+0.5*binWidth
        binError = 0.5*binWidth
        coverage_err_1sigma = np.sqrt(coverages_1sigma[i]*(1-coverages_1sigma[i])/Ntoys[i])
        g_1sigma.SetPoint(i, binCenter, coverages_1sigma[i])
        g_1sigma.SetPointError(i, binError, coverage_err_1sigma)
        coverage_err_2sigma = np.sqrt(coverages_2sigma[i]*(1-coverages_2sigma[i])/Ntoys[i])
        g_2sigma.SetPoint(i, binCenter, coverages_2sigma[i])
        g_2sigma.SetPointError(i, binError, coverage_err_2sigma)
    g_1sigma.SetMarkerStyle(20)
    g_1sigma.SetMarkerColor(ROOT.kAzure+7)
    g_1sigma.SetLineColor(ROOT.kAzure+7)
    g_1sigma.Draw("P SAME")
    g_2sigma.SetMarkerStyle(20)
    g_2sigma.SetMarkerColor(ROOT.kRed-2)
    g_2sigma.SetLineColor(ROOT.kRed-2)
    g_2sigma.Draw("P SAME")
    leg = ROOT.TLegend(.3, .175, .92, .4)
    leg.AddEntry(g_1sigma, "1#sigma interval", "pel")
    leg.AddEntry(g_2sigma, "2#sigma interval", "pel")
    leg.AddEntry(line_6827, "Coverage = 0.6827", "l")
    leg.AddEntry(line_9500, "Coverage = 0.95", "l")
    leg.SetTextSize(0.03)
    leg.SetNColumns(2)
    leg.Draw()
    ROOT.gPad.RedrawAxis()
    c.Print(name)

def getDummyGraph(xmin, xmax, ymin, ymax, xtitle, ytitle):
    dummy = ROOT.TGraph(2)
    dummy.SetPoint(0, xmin, ymin)
    dummy.SetPoint(1, xmax, ymax)
    dummy.GetXaxis().SetTitle(xtitle)
    dummy.GetYaxis().SetTitle(ytitle)
    dummy.GetXaxis().SetTitleSize(0.05)
    dummy.GetYaxis().SetTitleSize(0.05)
    dummy.GetXaxis().SetRangeUser(xmin, xmax)
    dummy.GetYaxis().SetRangeUser(ymin, ymax)
    dummy.SetTitle("")
    dummy.SetMarkerSize(0.0)
    return dummy

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

outname = "data_100k_scan_calib_v2.npz" if args.postfix is None else f"data_100k_scan_calib_v2_{args.postfix}.npz"

if args.save:
    dir = os.path.join(user.output_directory, "toyFits_scan")
    result_files = list_files_in_directory(dir)

    mu_true_all = np.array([])
    mu_measured_all = np.array([])
    mu_measured_up_all = np.array([])
    mu_measured_down_all = np.array([])
    mu_measured_2up_all = np.array([])
    mu_measured_2down_all = np.array([])
    toypaths_all = np.array([])
    for result_file in result_files:
        addGraphs = []
        data = np.load(result_file, allow_pickle=True)
        mu_true = data["mu_true"]
        mu_measured = data["mu_measured"] + args.offset
        mu_measured_up = data["mu_measured"] + args.inflate*(data["mu_measured_up"]-data["mu_measured"]) + args.offset
        mu_measured_down = data["mu_measured"] - args.inflate*(data["mu_measured"]-data["mu_measured_down"]) + args.offset
        mu_measured_2up = data["mu_measured"] + args.inflate*(data["mu_measured_up_2sigma"]-data["mu_measured"]) + args.offset
        mu_measured_2down = data["mu_measured"] - args.inflate*(data["mu_measured"]-data["mu_measured_down_2sigma"]) + args.offset
        mu_true_all = np.append(mu_true_all, mu_true)
        mu_measured_all = np.append(mu_measured_all, mu_measured )
        mu_measured_up_all = np.append(mu_measured_up_all, mu_measured_up )
        mu_measured_down_all = np.append(mu_measured_down_all, mu_measured_down )
        mu_measured_2up_all = np.append(mu_measured_2up_all, mu_measured_2up )
        mu_measured_2down_all = np.append(mu_measured_2down_all, mu_measured_2down )

    np.savez(
        outname, mu_true=mu_true_all, mu_measured=mu_measured_all,
        mu_measured_up=mu_measured_up_all, mu_measured_down=mu_measured_down_all,
        mu_measured_2up=mu_measured_2up_all, mu_measured_2down=mu_measured_2down_all
        )
    print(f"Saved file: {outname}")
    sys.exit()


loaded_data = np.load(outname)
mu_true = loaded_data["mu_true"]
mu_measured = loaded_data["mu_measured"]
mu_measured_up = loaded_data["mu_measured_up"]
mu_measured_down = loaded_data["mu_measured_down"]
mu_measured_2up = loaded_data["mu_measured_2up"]
mu_measured_2down = loaded_data["mu_measured_2down"]
print(f"Total number of toys: {len(mu_measured)}")


# Make bin edges in mu true
bin_edges = np.linspace(0.0, 3.0, num=7)

# Determine the bin index for each mu_true value
bin_indices = np.searchsorted(bin_edges, mu_true, side='right') - 1

# Now split the other arrays into bins based on bin_indices
mu_true_bins = [mu_true[bin_indices == i] for i in range(len(bin_edges) - 1)]
mu_measured_bins = [mu_measured[bin_indices == i] for i in range(len(bin_edges) - 1)]
mu_measured_up_bins = [mu_measured_up[bin_indices == i] for i in range(len(bin_edges) - 1)]
mu_measured_down_bins = [mu_measured_down[bin_indices == i] for i in range(len(bin_edges) - 1)]
mu_measured_2up_bins = [mu_measured_2up[bin_indices == i] for i in range(len(bin_edges) - 1)]
mu_measured_2down_bins = [mu_measured_2down[bin_indices == i] for i in range(len(bin_edges) - 1)]

avg_widths_1sigma = []
coverages_1sigma = []
avg_widths_2sigma = []
coverages_2sigma = []
Ntoys = []

for i, (mu_true_bin, mu_measured_bin, mu_measured_up_bin, mu_measured_down_bin,
        mu_measured_2up_bin, mu_measured_2down_bin) in enumerate(zip(
            mu_true_bins,
            mu_measured_bins,
            mu_measured_up_bins,
            mu_measured_down_bins,
            mu_measured_2up_bins,
            mu_measured_2down_bins)):
    print("=============================")
    print(f"Bin {bin_edges[i]} - {bin_edges[i+1]}")
    _, average_width_1sigma, coverage_1sigma = calculateScore(mu_true_bin, mu_measured_down_bin, mu_measured_up_bin, Ntoys_for_penalty=1000)
    _, average_width_2sigma, coverage_2sigma = calculateScore(mu_true_bin, mu_measured_2down_bin, mu_measured_2up_bin, Ntoys_for_penalty=1000)

    avg_widths_1sigma.append(average_width_1sigma)
    coverages_1sigma.append(coverage_1sigma)
    avg_widths_2sigma.append(average_width_2sigma)
    coverages_2sigma.append(coverage_2sigma)
    Ntoys.append(len(mu_measured_bin))

    print(f"Coverage of 1 sigma interval = {coverage_1sigma}")
    print(f"Coverage of 2 sigma interval = {coverage_2sigma}")

    output_name = os.path.join( user.plot_directory, "ClosureTests", f"Toys_profiled_1sigma_{bin_edges[i]}_{bin_edges[i+1]}.pdf" )
    p = calibrationPlotter(output_name)
    p.xmax, p.ymax = 3.0, 3.0
    p.colorCoverage = True
    p.setMus(mu_true_bin, mu_measured_bin, mu_measured_down_bin, mu_measured_up_bin)
    p.draw()

    output_name = os.path.join( user.plot_directory, "ClosureTests", f"Toys_profiled_2sigma_{bin_edges[i]}_{bin_edges[i+1]}.pdf" )
    p = calibrationPlotter(output_name)
    p.xmax, p.ymax = 3.0, 3.0
    p.colorCoverage = True
    p.setMus(mu_true_bin, mu_measured_bin, mu_measured_2down_bin, mu_measured_2up_bin)
    p.draw()

width_output_name = os.path.join( user.plot_directory, "ClosureTests", f"AverageWidths.pdf" )
drawWidth(Ntoys, bin_edges, avg_widths_1sigma, avg_widths_2sigma, width_output_name)
coverage_output_name = os.path.join( user.plot_directory, "ClosureTests", f"Coverages.pdf" )
drawCoverage(Ntoys, bin_edges, coverages_1sigma, coverages_2sigma, coverage_output_name)
