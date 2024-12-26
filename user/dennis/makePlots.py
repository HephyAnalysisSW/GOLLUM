import sys, ROOT
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from Plotter import Plotter
import common.syncer
import common.data_structure as data_structure
import common.user as user


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

for feature in data_structure.feature_names:
    p = Plotter(feature)
    p.plot_dir = user.plot_directory
    p.drawRatio = True
    p.ratiorange = 0.0, 0.1
    p.xtitle = data_structure.plot_options[feature]["tex"]
    h_all = ROOT.TH1F()
    for process in processes:
        filename = f"hists/{process}.root"
        hist = getObjFromFile(filename,feature)
        if process == "htautau":
            p.addSignal(hist, legnames[process], data_structure.plot_styles[process]["fill_color"])
        else:
            p.addBackground(hist, legnames[process], data_structure.plot_styles[process]["fill_color"])
    p.draw()
