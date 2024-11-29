import sys, ROOT
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from Plotter import Plotter
import common.syncer
import common.data_structure as data_structure
import common.user as user

file = ROOT.TFile("hists/nominal.root")
for feature in data_structure.feature_names:

    hist = file.Get(feature)

    p = Plotter(feature)
    p.plot_dir = user.plot_directory
    p.drawRatio = True
    p.ratiorange = 0.2, 1.8
    p.xtitle = data_structure.plot_options[feature]["tex"]
    p.addBackground(hist, "test", ROOT.kRed)
    p.draw()
    # # Add Backgrounds, those will be stacked
    # p.addBackground(hist1, "Background 1", ROOT.kAzure+7)
    # p.addBackground(hist2, "Background 2", 15)
    # # Assign a shape systematic by providing up/down variations
    # p.addSystematic(up1, down1, "SYS1", "Background 1")
    # # Add a normalisation uncertainty
    # p.addNormSystematic("Background 2", 0.3)
    # # Add some signals
    # p.addSignal(signal1, "Signal 1", ROOT.kBlue-2)
    # p.addSignal(signal2, "Signal 2", ROOT.kRed, lineStyle=2)
    # # Add data
    # p.addData(data)
    # # Draw
    # p.draw()
