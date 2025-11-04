import ROOT
import array

class LikelihoodScanPlotter:
    def __init__(self, muValues, qValues, name):
        self.name = name
        self.qValues =  array.array('d', qValues)
        self.muValues =  array.array('d', muValues )
        self.xtitle = "#mu"
        self.ytitle = "-2 #Delta ln L"
        self.xmin = muValues[0]
        self.xmax = muValues[-1]
        self.ymin = 0.
        self.ymax = 10
        self.plot_dir = ""

    def __getDummy(self):
        d = ROOT.TGraph(2)
        d.SetPoint(0, self.xmin, self.ymin)
        d.SetPoint(1, self.xmax, self.ymax)
        d.SetMarkerSize(0.0)
        d.SetTitle('')
        d.GetXaxis().SetTitle(self.xtitle)
        d.GetYaxis().SetTitle(self.ytitle)
        return d


    def draw(self):
        c = ROOT.TCanvas(self.name, "", 600, 600)
        ROOT.gPad.SetTopMargin(0.02)
        dummy = self.__getDummy()
        dummy.Draw("AP")
        dummy.GetXaxis().SetRangeUser(self.xmin, self.xmax)
        dummy.GetYaxis().SetRangeUser(self.ymin, self.ymax)
        g = ROOT.TGraph(len(self.muValues), self.muValues, self.qValues)
        g.SetLineWidth(2)
        g.Draw("L SAME")
        l1, l2 = self.getLines()
        l1.Draw("SAME")
        l2.Draw("SAME")
        g.Draw("L SAME")
        ROOT.gPad.RedrawAxis()
        c.Print(self.plot_dir+"/"+self.name+".pdf")

    def getLines(self):
        y_1sigma = 1.00
        y_2sigma = 3.84
        line_1sigma = ROOT.TLine(self.xmin, y_1sigma, self.xmax, y_1sigma)
        line_2sigma = ROOT.TLine(self.xmin, y_2sigma, self.xmax, y_2sigma)
        line_1sigma.SetLineWidth(2)
        line_2sigma.SetLineWidth(2)
        line_1sigma.SetLineStyle(2)
        line_2sigma.SetLineStyle(2)
        return line_1sigma, line_2sigma
