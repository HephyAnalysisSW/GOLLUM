import sys, ROOT
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from Plotter import Plotter
import common.syncer
import common.data_structure as data_structure
import common.user as user

ROOT.gROOT.SetBatch(ROOT.kTRUE)

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--selection', action='store', default=None)
argParser.add_argument('--toy', action='store', default=None)
args = argParser.parse_args()


def getObjFromFile(fname, hname):
    gDir = ROOT.gDirectory.GetName()
    f = ROOT.TFile(fname)
    assert not f.IsZombie()
    f.cd()
    htmp = f.Get(hname)
    if not htmp:  return htmp
    ROOT.gDirectory.cd('PyROOT:/')
    res = htmp.Clone()
    f.Close()
    ROOT.gDirectory.cd(gDir+':/')
    return res


processes = ["htautau", "ztautau", "ttbar", "diboson"]
legnames = {
    "htautau": "H#rightarrow#tau#tau",
    "ztautau": "Z#rightarrow#tau#tau",
    "ttbar":   "t#bar{t}",
    "diboson": "Diboson",
}

total_yield = 0
total_yield_toy = 0

for feature in data_structure.feature_names:
    p = Plotter(feature)
    p.plot_dir = user.plot_directory+"/Toy_"+args.toy+"/"+args.selection+"/"
    p.drawRatio = True
    p.ratiorange = 0.9, 1.1
    p.xtitle = data_structure.plot_options[feature]["tex"]
    isFirst = True
    for process in processes:
        filename = "hists/"+args.selection+"__"+process+".root"
        hist = getObjFromFile(filename,feature)
        if feature == "PRI_n_jets":
            total_yield += hist.Integral()
        if process != "htautau":
            p.addBackground(hist, legnames[process], data_structure.plot_styles[process]["fill_color"])
        if isFirst:
            h_bkg_and_signal = hist.Clone()
            isFirst = False
        else:
            h_bkg_and_signal.Add(hist)


    p.addSignal(h_bkg_and_signal, legnames["htautau"], data_structure.plot_styles["htautau"]["fill_color"])
    filename_toy = "hists/"+args.selection+"__toy__"+args.toy+".root"
    # print(f"- {filename_toy}")
    h_toy = getObjFromFile(filename_toy, feature)
    if feature == "PRI_n_jets":
        total_yield_toy = h_toy.Integral()
    p.addData(h_toy, "Toy")
    p.draw()
