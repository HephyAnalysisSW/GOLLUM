import ROOT
import pickle
import sys
import argparse
import numpy as np
sys.path.insert(0, "..")
import common.syncer
import common.user as user
import os

xboundary = 0.7
separation = 0.02
TopMargin = 0.02
LeftMargin = 0.35
RightMargin = 0.01
BottomMargin = 0.1

xmin, xmax = -2.5, 2.5
ymin, ymax = 0, 7.5


def getMatrix(fitResult, nuisanceList):
    parameters = ["mu"]+nuisanceList
    Nparams = len(parameters)
    matrix = np.zeros((Nparams, Nparams))
    for i, p1 in enumerate(parameters):
        for j, p2 in enumerate(parameters):
            matrix[i,j] = fitResult["cov__"+p1+"__"+p2]
    return matrix

def calcImpactsFromInverse(fitResult, nuisanceList):
    cov = getMatrix(fitResult, nuisanceList)
    inverse = np.linalg.inv(cov)
    impacts = {}
    for i,nu in enumerate(nuisanceList):
        idx = i+1 # matrix has mu entries at i = 0
        uncert_nu = np.sqrt(fitResult["cov__"+nu+"__"+nu])
        impacts[nu] = -inverse[0, idx]*uncert_nu/inverse[0, 0]
    return impacts

def calcPosition(x,y):
    axis_length_x = xmax - xmin
    axis_length_canvas_x = 1-RightMargin-LeftMargin
    x_canvas = LeftMargin + (x/axis_length_x) * axis_length_canvas_x

    axis_length_y = ymax - ymin
    axis_length_canvas_y = 1-TopMargin-BottomMargin
    y_canvas = BottomMargin + (y/axis_length_y) * axis_length_canvas_y
    return x_canvas, y_canvas

def addText(x, y, text, font=43, size=16, color=ROOT.kBlack):
    latex = ROOT.TLatex(3.5, 24, text)
    latex.SetNDC()
    latex.SetTextAlign(12)
    latex.SetTextFont(font)
    latex.SetTextSize(size)
    latex.SetTextColor(color)
    latex.SetX(x)
    latex.SetY(y)
    return latex

def getDummy(title, xmin, xmax, ymin, ymax, factor=1.0):
    g = ROOT.TGraph(2)
    g.SetPoint(0, xmin, ymin)
    g.SetPoint(1, xmax, ymax)
    g.SetMarkerSize(0.0)
    g.SetTitle('')
    g.GetXaxis().SetTitle(title)
    g.GetYaxis().SetTitle("")
    g.GetXaxis().SetRangeUser(xmin, xmax)
    g.GetYaxis().SetRangeUser(ymin, ymax)
    g.GetYaxis().SetLabelSize(0.0)
    g.GetYaxis().SetTickLength(0.0)
    g.GetXaxis().SetTitleOffset(0.8/factor)
    g.GetXaxis().SetLabelOffset(0.001/factor)
    g.GetXaxis().SetTitleSize(0.05*factor)
    g.GetXaxis().SetLabelSize(0.03*factor)
    g.GetXaxis().SetNdivisions(505)
    return g

def getCorrelations(fitResult, nuisanceList):
    label_dict = {
        "mu": "#mu",
        "nu_jes": "jes",
        "nu_tes": "tes",
        "nu_met": "met",
        "nu_bkg": "bkg",
        "nu_tt":  "tt",
        "nu_diboson": "Diboson",
    }
    nuisanceList = ["mu"]+nuisanceList
    Npoints = len(nuisanceList)
    hist = ROOT.TH2F("cor", "cor", Npoints, 0, Npoints, Npoints, 0, Npoints)
    for i, n1 in enumerate(nuisanceList):
        for j, n2 in enumerate(nuisanceList):
            covEntry = fitResult["cov__"+n1+"__"+n2]
            uncert1 = np.sqrt(fitResult["cov__"+n1+"__"+n1])
            uncert2 = np.sqrt(fitResult["cov__"+n2+"__"+n2])
            correlation = covEntry/(uncert1*uncert2)
            hist.SetBinContent(i+1, j+1, correlation)

    for i, n in enumerate(nuisanceList):
        hist.GetXaxis().SetBinLabel(i+1, label_dict[n])
        hist.GetYaxis().SetBinLabel(i+1, label_dict[n])
    hist.SetTitle("")
    hist.GetXaxis().SetTitle("")
    hist.GetYaxis().SetTitle("")
    hist.GetZaxis().SetTitle('#rho')
    hist.SetStats(0)
    hist.GetZaxis().SetRangeUser(-1., 1.)
    return hist

def createGraph(dict, nuisanceList):
    label_dict = {
        "nu_jes": "Jet energy scale",
        "nu_tes": "#tau energy scale",
        "nu_met": "Soft MET",
        "nu_bkg": "Background rate",
        "nu_tt":  "t#bar{t} rate",
        "nu_diboson": "Diboson rate",
    }

    Npoints = len(nuisanceList)
    g = ROOT.TGraphAsymmErrors(Npoints)
    g_prefit = ROOT.TGraphAsymmErrors(Npoints)
    labels = []
    for i, n in enumerate(nuisanceList):
        (mle, lower, upper) = dict[n]
        g.SetPoint(i, mle, Npoints-i)
        g.SetPointError(i, mle-lower, upper-mle, 0.0, 0.0)
        g_prefit.SetPoint(i, 0.0, Npoints-i)
        g_prefit.SetPointError(i, 1.0, 1.0, 0.2, 0.2)
        _, y = calcPosition(0, Npoints-i)
        labels.append(addText(0.01, y, label_dict[n]))
    return g, g_prefit, labels

def createGraph_impacts(dict, nuisanceList):
    label_dict = {
        "nu_jes": "Jet energy scale",
        "nu_tes": "#tau energy scale",
        "nu_met": "Soft MET",
        "nu_bkg": "Background rate",
        "nu_tt":  "t#bar{t} rate",
        "nu_diboson": "Diboson rate",
    }

    Npoints = len(nuisanceList)
    g_plus = ROOT.TGraphAsymmErrors(Npoints)
    g_minus = ROOT.TGraphAsymmErrors(Npoints)
    labels = []
    for i, n in enumerate(nuisanceList):
        impact = dict[n]
        g_plus.SetPoint(i, 0, Npoints-i)
        g_minus.SetPoint(i, 0, Npoints-i)
        if impact < 0:
            g_plus.SetPointError(i, abs(impact), 0., 0.2, 0.2)
            g_minus.SetPointError(i, 0., abs(impact), 0.2, 0.2)
        else:
            g_minus.SetPointError(i, abs(impact), 0., 0.2, 0.2)
            g_plus.SetPointError(i, 0., abs(impact), 0.2, 0.2)

    return g_plus, g_minus

def getLines():
    lines = []
    lines.append(ROOT.TLine(0, ymin, 0, ymax-1.))
    lines.append(ROOT.TLine(1., ymin, 1., ymax-1.))
    lines.append(ROOT.TLine(-1., ymin, -1., ymax-1.))
    for i,l in enumerate(lines):
        l.SetLineStyle(2)
        l.SetLineWidth(2)
    return lines

################################################################################
# MAIN
parser = argparse.ArgumentParser(description="PostFit Uncertainty Plot.")
parser.add_argument("-f","--file", help="Path to the file containing the uncertainties.")
args = parser.parse_args()

# Open fit results file
with open(args.file, 'rb') as file:
    fitResult = pickle.load(file)

nuisanceList = ["nu_jes", "nu_tes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]

# Access the impacts on mu
# These can be calculated via
impacts = {}
maxImpact = 0
impacts_inverse = calcImpactsFromInverse(fitResult, nuisanceList)
for nu in nuisanceList:
    cov_mu_nu = fitResult["cov__mu__"+nu]
    sigma_nu = np.sqrt(fitResult["cov__"+nu+"__"+nu])
    # impact = fitResult["cov__mu__"+nu]*sigma_nu/fitResult["cov__"+nu+"__"+nu]
    impact = impacts_inverse[nu]
    impacts[nu] = impact
    if abs(impact) > maxImpact:
        maxImpact = abs(impact)

xmin2 = -1.1*maxImpact
xmax2 =  1.1*maxImpact



# for nu in nuisanceList:
#     relDifference = abs( (impacts[nu]-impacts_inverse[nu])/impacts[nu] )
#     if relDifference > 0.1:
#         print("[Warning] Impacts from matrix inversion give different results:")
#         print(f"         {nu}: {impacts[nu]} vs. {impacts_inverse[nu]} (from inversion)")
################################################################################
# IMPACT PLOT
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(0)
ROOT.gStyle.SetLegendBorderSize(0)
c = ROOT.TCanvas("postFitUncerts", "", 600, 600)
pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.0, xboundary, 1.0)
pad1.SetTopMargin(TopMargin)
pad1.SetLeftMargin(LeftMargin)
pad1.SetRightMargin(separation/2)
pad1.SetBottomMargin(BottomMargin)
pad1.Draw()
pad2 = ROOT.TPad("pad2", "pad2", xboundary, 0.0, 1.0, 1.0)
pad2.SetTopMargin(TopMargin)
pad2.SetLeftMargin(separation/2)
pad2.SetRightMargin(RightMargin)
pad2.SetBottomMargin(BottomMargin)
pad2.Draw()

# Put post fit uncerts in left panel
pad1.cd()
g_dummy = getDummy("#nu", xmin, xmax, ymin, ymax)
g_dummy.Draw("AP")
g, g_prefit,labels = createGraph(fitResult, nuisanceList)
g_prefit.SetFillColor(17)
g_prefit.Draw("E2 SAME")
g.SetMarkerStyle(20)
g.Draw("P SAME")
for l in labels:
    l.Draw()
lines = getLines()
for line in lines:
    line.Draw("SAME")
leg = ROOT.TLegend(LeftMargin+0.1, 1-TopMargin-0.1, 1-0.1-separation, 1-TopMargin-0.01)
leg.SetNColumns(2)
leg.AddEntry(g, "PostFit", "pl")
leg.AddEntry(g_prefit, "PreFit", "f")
leg.Draw()
ROOT.gPad.RedrawAxis()

# Put impacts in right panel
pad2.cd()
g_dummy2 = getDummy("#Delta #mu", xmin2, xmax2, ymin, ymax, factor=2)
g_dummy2.Draw("AP")
g_plus, g_minus = createGraph_impacts(impacts, nuisanceList)
g_plus.SetMarkerSize(0)
g_minus.SetMarkerSize(0)
g_plus.SetFillColor(ROOT.kRed)
g_minus.SetFillColor(ROOT.kBlue)
g_plus.Draw("E2 SAME")
g_minus.Draw("E2 SAME")

line = ROOT.TLine(0, ymin, 0, ymax-1.)
line.SetLineStyle(2)
line.SetLineWidth(2)
line.Draw("SAME")

muresult = addText(0.2, 0.9, "#mu = %.2f #pm %.2f"%(fitResult["mu_mle"], np.sqrt(fitResult["cov__mu__mu"])), font=43, size=16, color=ROOT.kBlack)
muresult.Draw()

c.Print(os.path.join( user.plot_directory, 'impacts', os.path.basename(args.file).replace("fitResult.", "impacts.").replace(".pkl",".pdf")))

################################################################################
# Correlation
c_cor = ROOT.TCanvas("correlations", "", 600, 600)
ROOT.gStyle.SetPalette(ROOT.kSunset)
ROOT.gPad.SetTopMargin(0.03)
ROOT.gPad.SetLeftMargin(0.1)
ROOT.gPad.SetRightMargin(0.15)
ROOT.gPad.SetBottomMargin(0.07)
h_cor = getCorrelations(fitResult, nuisanceList)
h_cor.Draw("COLZ")
c_cor.Print(os.path.join( user.plot_directory, 'impacts', os.path.basename(args.file).replace("fitResult.", "correlations.").replace(".pkl",".pdf")))
