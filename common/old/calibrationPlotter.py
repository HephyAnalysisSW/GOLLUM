import ROOT
import uuid
class calibrationPlotter:
    def __init__(self, name):
        self.name = name
        self.xmin = 0.
        self.xmax = 6.
        self.ymin = 0.
        self.ymax = 6.
        self.addGraphs = []
        self.colorCoverage = False

    def setMus(self, mu_true, mu_measured, mu_measured_down, mu_measured_up):
        if len(mu_true) != len(mu_measured):
            print("[ERROR] true and measured mu have different length!")
        if len(mu_true) != len(mu_measured_down):
            print("[ERROR] true and low boundary have different length!")
        if len(mu_true) != len(mu_measured_up):
            print("[ERROR] true and high boundary have different length!")
        self.mu_true = mu_true
        self.mu_measured = mu_measured
        self.mu_measured_down = mu_measured_down
        self.mu_measured_up = mu_measured_up

    def addGraph(self, graph, markerstyle=20, color=ROOT.kRed):
        g_add = graph.Clone()
        g_add.SetMarkerStyle(markerstyle)
        g_add.SetMarkerColor(color)
        g_add.SetLineColor(color)
        self.addGraphs.append(g_add)




    def getDummy(self, xtitle, ytitle, xmin, xmax, ymin, ymax, factor=1.0):
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

    def draw(self):
        TopMargin = 0.02
        LeftMargin = 0.1
        RightMargin = 0.01
        BottomMargin = 0.1
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        ROOT.gStyle.SetLegendBorderSize(0)
        c = ROOT.TCanvas(f"{uuid.uuid4().hex}", "", 600, 600)
        ROOT.gPad.SetTopMargin(TopMargin)
        ROOT.gPad.SetLeftMargin(LeftMargin)
        ROOT.gPad.SetRightMargin(RightMargin)
        ROOT.gPad.SetBottomMargin(BottomMargin)
        g_dummy = self.getDummy("#mu true", "#mu measured", self.xmin, self.xmax, self.ymin, self.ymax, factor=1.0)
        g_dummy.Draw("AP")
        line = ROOT.TLine(self.xmin, self.ymin, self.xmax, self.ymax)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.Draw("SAME")
        graph = ROOT.TGraphAsymmErrors(len(self.mu_true))
        for i in range(len(self.mu_true)):
            graph.SetPoint(i, self.mu_true[i], self.mu_measured[i])
            graph.SetPointError(i, 0.0, 0.0, self.mu_measured[i]-self.mu_measured_down[i], self.mu_measured_up[i]-self.mu_measured[i])
        graph.SetMarkerStyle(20)
        graph.Draw("P SAME")
        if self.colorCoverage:
            mu_true_cov = []
            mu_measured_cov = []
            mu_measured_up_cov = []
            mu_measured_down_cov = []

            for i in range(len(self.mu_true)):
                if self.mu_true[i] > self.mu_measured_down[i] and self.mu_true[i] < self.mu_measured_up[i]:
                    mu_true_cov.append(self.mu_true[i])
                    mu_measured_cov.append(self.mu_measured[i])
                    mu_measured_up_cov.append(self.mu_measured_up[i])
                    mu_measured_down_cov.append(self.mu_measured_down[i])
            graph_cov = ROOT.TGraphAsymmErrors(len(mu_true_cov))
            for i in range(len(mu_true_cov)):
                graph_cov.SetPoint(i, mu_true_cov[i], mu_measured_cov[i])
                graph_cov.SetPointError(i, 0.0, 0.0, mu_measured_cov[i]-mu_measured_down_cov[i], mu_measured_up_cov[i]-mu_measured_cov[i])
            graph_cov.SetMarkerStyle(20)
            graph_cov.SetMarkerColor(ROOT.kRed)
            graph_cov.SetLineColor(ROOT.kRed)
            graph_cov.Draw("P SAME")

        for g in self.addGraphs:
            g.Draw("P SAME")
        c.Print(self.name)
