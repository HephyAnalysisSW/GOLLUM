import ROOT
import pickle
import sys
import os
import argparse
import numpy as np
sys.path.insert(0, "..")
import common.syncer
import common.user as user
from common.muCalibrator import muCalibrator

ROOT.gROOT.SetBatch(ROOT.kTRUE)

def getDummy(xtitle, ytitle, xmin, xmax, ymin, ymax, factor=1.0):
    g = ROOT.TGraph(2)
    g.SetPoint(0, xmin, ymin)
    g.SetPoint(1, xmax, ymax)
    g.SetMarkerSize(0.0)
    g.SetTitle('')
    g.GetXaxis().SetTitle(xtitle)
    g.GetYaxis().SetTitle(ytitle)
    g.GetXaxis().SetRangeUser(xmin, xmax)
    g.GetXaxis().SetTitleOffset(0.8/factor)
    g.GetXaxis().SetLabelOffset(0.001/factor)
    g.GetXaxis().SetTitleSize(0.05*factor)
    g.GetXaxis().SetLabelSize(0.03*factor)
    g.GetXaxis().SetNdivisions(505)
    g.GetYaxis().SetRangeUser(ymin, ymax)
    g.GetYaxis().SetTitleOffset(0.8/factor)
    g.GetYaxis().SetLabelOffset(0.001/factor)
    g.GetYaxis().SetTitleSize(0.05*factor)
    g.GetYaxis().SetLabelSize(0.03*factor)
    g.GetYaxis().SetNdivisions(505)
    return g

def list_files_in_directory(directory_path):
    toyList = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            passedAll = True
            for veto in ["nominal", "ttbar", "htautau", "diboson", "ztautau"]:
                if veto in file:
                    passedAll = False
            if passedAll:
                toyList.append(file)
    return toyList

################################################################################
parser = argparse.ArgumentParser(description="ML inference.")
#parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
parser.add_argument("--otherSys", action="store_true")
parser.add_argument('--calibration', action='store', default=None)
args = parser.parse_args()

muAll = [
    0.1,
    0.3,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
]

sysList = ["nominal.h5"]

if args.otherSys:
    toyPath = "/scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/lowMT_VBFJet/"
    sysList += list_files_in_directory(toyPath)

calibrationFile = None
if args.calibration is not None:
    calibrator = muCalibrator(args.calibration)

fitResultDir = "/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference/"

missingFiles = []

isFirstCanvas = True
merged_output_name = os.path.join( user.plot_directory, "ClosureTests", "All.pdf" )
for sys in sysList:
    postfix = "" if sys == "nominal.h5" else "_"+sys.replace(".h5", "")
    plotname = sys.replace(".h5", ".pdf")
    muMeasured = []
    muMeasuredErr = []
    muTruth = []
    muCalibrated = []

    for mu in muAll:
        integer_part = int(mu)
        fractional_part = abs(mu - integer_part)
        fractional_str = f"{fractional_part:.3f}"[2:]
        mustring = f"{integer_part}p{fractional_str}"
        filename = fitResultDir+"fitResult.config_reference_mu_"+mustring+postfix+".pkl"

        if not os.path.exists(filename):
            missingFiles.append(filename)
            continue

        with open(filename, 'rb') as file:
            fitResult = pickle.load(file)

        muMeasured.append(fitResult["mu"])
        muMeasuredErr.append(np.sqrt(fitResult["cov__mu__mu"]))
        muTruth.append(mu)
        if args.calibration is not None:
            mu_corrected = calibrator.getMu(mu=fitResult["mu"], nu_jes=fitResult["nu_jes"], nu_tes=fitResult["nu_tes"], nu_met=fitResult["nu_met"])
            muCalibrated.append(mu_corrected)




    g = ROOT.TGraphAsymmErrors(len(muTruth))
    for i in range(len(muTruth)):
        g.SetPoint(i, muTruth[i], muMeasured[i])
        g.SetPointError(i, 0.0, 0.0, muMeasuredErr[i], muMeasuredErr[i])

    if args.calibration is not None:
        g_calibration = ROOT.TGraphAsymmErrors(len(muTruth))
        for i in range(len(muTruth)):
            g_calibration.SetPoint(i, muTruth[i], muCalibrated[i])
            g_calibration.SetPointError(i, 0.0, 0.0, muMeasuredErr[i], muMeasuredErr[i])
    else:
        g_calibration = ROOT.TGraphAsymmErrors(0)

    TopMargin = 0.02
    LeftMargin = 0.1
    RightMargin = 0.01
    BottomMargin = 0.1

    xmin, xmax = 0., 4.
    ymin, ymax = 0., 4.
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)
    c = ROOT.TCanvas("postFitUncerts"+postfix, "", 600, 600)
    if isFirstCanvas:
        c.Print(merged_output_name+"[")
        isFirstCanvas = False

    ROOT.gPad.SetTopMargin(TopMargin)
    ROOT.gPad.SetLeftMargin(LeftMargin)
    ROOT.gPad.SetRightMargin(RightMargin)
    ROOT.gPad.SetBottomMargin(BottomMargin)

    g_dummy = getDummy("#mu true", "#mu measured", xmin, xmax, ymin, ymax, factor=1.0)
    g_dummy.Draw("AP")

    line = ROOT.TLine(xmin, ymin, xmax, ymax)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    g.SetMarkerStyle(20)
    g.Draw("P SAME")

    g_calibration.SetMarkerStyle(20)
    g_calibration.SetMarkerColor(ROOT.kRed)
    g_calibration.SetLineColor(ROOT.kRed)
    g_calibration.Draw("P SAME")

    latex = ROOT.TLatex(3.5, 24, plotname.replace(".pdf", ""))
    latex.SetNDC()
    latex.SetTextAlign(13)
    latex.SetTextFont(43)
    latex.SetTextSize(12)
    latex.SetX(.6)
    latex.SetY(.2)
    latex.Draw()

    # c.Print(os.path.join( user.plot_directory, "ClosureTests", plotname ))
    c.Print(merged_output_name)

c.Print(merged_output_name+"]")
missingFiles_file = "missingJobs.sh"
print("MISSING FILES LIST SAVED IN %s"%(missingFiles_file))
cmd_fit = 'python runInference.py --config config_reference.yaml --predict --asimov_mu <MU> --postfix <TOYNAME> --modify Toy_name="<TOYNAME>" Save.Toy.filename="<TOYNAME>.h5"'

with open(missingFiles_file, "w") as f:
    for m in missingFiles:
        fname = m.replace("/groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference/fitResult.config_reference_mu_", "").replace(".pkl", "")
        muString = fname.split("_")[0]
        muValue = int(muString.split("p")[0])+0.001*int(muString.split("p")[1])
        toyname = fname.replace(muString+"_", "")
        cmd = cmd_fit.replace("<MU>", "%.2f"%muValue).replace("<TOYNAME>", toyname)
        f.write(cmd)
        f.write("\n")
