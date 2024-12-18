import ROOT
import pickle
import sys
sys.path.insert(0, "..")
import common.syncer
import common.user as user

TopMargin = 0.02
LeftMargin = 0.2
RightMargin = 0.01
BottomMargin = 0.1

xmin, xmax = -2.5, 2.5
ymin, ymax = 0, 7.5

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

def getDummy():
    g = ROOT.TGraph(2)
    g.SetPoint(0, xmin, ymin)
    g.SetPoint(1, xmax, ymax)
    g.SetMarkerSize(0.0)
    return g

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

def getLines():
    lines = []
    lines.append(ROOT.TLine(0, ymin, 0, ymax-1.))
    lines.append(ROOT.TLine(1., ymin, 1., ymax-1.))
    lines.append(ROOT.TLine(-1., ymin, -1., ymax-1.))
    for i,l in enumerate(lines):
        l.SetLineStyle(2)
        l.SetLineWidth(2)
    return lines

with open('postFitUncerts.pkl', 'rb') as file:
    postFitUncerts = pickle.load(file)


nuisanceList = ["nu_jes", "nu_tes", "nu_met", "nu_bkg", "nu_tt", "nu_diboson"]


ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(0)
ROOT.gStyle.SetLegendBorderSize(0)
c = ROOT.TCanvas("postFitUncerts", "", 600, 600)
ROOT.gPad.SetTopMargin(TopMargin)
ROOT.gPad.SetLeftMargin(LeftMargin)
ROOT.gPad.SetRightMargin(RightMargin)
ROOT.gPad.SetBottomMargin(BottomMargin)
g_dummy = getDummy()
g_dummy.SetTitle('')
g_dummy.GetXaxis().SetTitle("#nu")
g_dummy.GetYaxis().SetTitle("")
g_dummy.GetXaxis().SetRangeUser(xmin, xmax)
g_dummy.GetYaxis().SetRangeUser(ymin, ymax)
g_dummy.GetYaxis().SetLabelSize(0.0)
g_dummy.GetYaxis().SetTickLength(0.0)
g_dummy.Draw("AP")
g, g_prefit,labels = createGraph(postFitUncerts, nuisanceList)
g_prefit.SetFillColor(17)
g_prefit.Draw("E2 SAME")
g.SetMarkerStyle(20)
g.Draw("P SAME")
for l in labels:
    l.Draw()
lines = getLines()
for line in lines:
    line.Draw("SAME")

leg = ROOT.TLegend(LeftMargin+0.1, 1-TopMargin-0.1, 1-RightMargin-0.1, 1-TopMargin-0.01)
leg.SetNColumns(2)
leg.AddEntry(g, "PostFit", "pl")
leg.AddEntry(g_prefit, "PreFit", "f")
leg.Draw()
ROOT.gPad.RedrawAxis()
c.Print(user.plot_directory+"/postFitUncerts.pdf")
