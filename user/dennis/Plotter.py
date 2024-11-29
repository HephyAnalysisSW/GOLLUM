"""
This is a plotter class. It can be used to draw CMS publication-ready plots.
You can add three types of histograms to a plot: backgrounds, signals and data.
Backgrounds have a name (that appears in the legend), a color, are stacked and
appear as colored areas.
Signals have a name (that appears in the legend) and a color and appear as
lines in the plot.
Data can be added only once and appears as black markers.
Systematic uncertainties are plugged in as variaitons of single backgrounds and
treated internally in the plotter.

Some parameters can be changed in order to customize the plot.
"""


import ROOT,os,sys
import array as arr
from math                                import sqrt
from itertools                           import count


class Plotter:
    _ids = count(0)
    def __init__(self, name):
        # Keep track of instances of this class to have unique canvas names
        self.id = next(self._ids)
        self.debug = False

        # Check if plotter is run with pyroot or python and ROOT bindings
        if sys.version_info[0] > 2:
            self.isPYROOT = True
        else:
            self.isPYROOT = False

        # A few global drawing options
        ROOT.gStyle.SetLegendBorderSize(0)
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        ROOT.gStyle.SetOptStat(0)

        # Some parameters that can be changed
        self.plotname = name                        # Name of the pdf file
        self.plot_dir = os.getcwd()                 # Directory to save in
        self.NcolumnsLegend = 2                     # Number of columns in legend
        self.ytitle = "Events"                      # Title Y-axis
        self.xtitle = ""                            # Title X-axis
        self.ratiotitle = "#frac{Data}{SM}"         # Title of ratio
        self.drawRatio = False                      # Draw ratio?
        self.drawLogo = False                       # Draw CMS (and subtext) label?
        self.subtext = "Work in progress"           # Subtext of CMS label
        self.simtext = None                         # Here you can set a "Simulation" label
        self.lumi = None                            # lumi value in label
        self.log = False                            # log scale?
        self.yfactor = 1.7                          # scale y-axis
        self.rebin = 1                              # rebin hist?
        self.ratiorange = 0.5,1.5                   # y-range of ratio plot
        self.legshift = (0., 0., 0., 0.)            # shift the Legend coordinates (x1, y1, x2, y2)
        self.legtextsize = 0.035                    # Size of legend text
        self.totalUncText = "Total uncertainty"     # Legend text of the uncertainty area
        self.divideByWidth = False                  # divide bin content by bin width?


        # Internal parameters that are set automatically
        self.__legend = ROOT.TLegend()                # Legend
        self.__bkgtotal = ROOT.TH1F()                 # Hist for sum of backgrounds
        self.__errorhist = ROOT.TGraphAsymmErrors()   # Hist for sys uncert
        self.__hasData = False                        # Keep track if date have been added
        self.__hasSignal = False                      # Keep track if signals have been added
        self.__hasBackground = False                  # Keep track if backgrounds have been added
        self.__doSystematics = False                  # Keep track if systematics have been added
        self.__backgrounds = []                       # Hists and infos of all backgrounds
        self.__sysDeltas = []                         # List of systematic shifts
        self.__sysnames = []                          # List of systematic names
        self.__signals = []                           # Hists and infos of all signals
        self.__data = {}                              # Hist and info of data
        self.__stack = ROOT.THStack()                 # Stack for backgrounds
        self.__autoYrange = True                      # Set Y range automatically?
        self.__ymin = 0                               # minimum of any histogram
        self.__ymax = 0                               # maximum of any histogram
        self.__xmin = -1                              # min of x-axis
        self.__xmax = -1                              # max of x-axis
        self.__xmin_draw = -1                         # custom x axis range
        self.__xmax_draw = -1                         # custom x axis range
        self.__customXrange = False                   # custom x axis range
        self.__NlegEntries = 0                        # number of entries in legend
        self.__latexTexts = []                        # List of all latex text boxes
        self.__binning = []                           # Store binning for this plot
        self.__Nbins = 0                              # Store number of bins
        self.__ratioCounter = 0                       # keep track of number of ratios
        self.__MarginTop = 0.10                       # Pad Margin Top
        self.__MarginBottom = 0.48                    # Pad Margin Bottom
        self.__MarginLeft = 0.19                      # Pad Margin Left
        self.__MarginRight = 0.05                     # Pad Margin Right


    ############################################################################
    # Add backgrounds that are merged to a stack and displayed as filled areas
    def addBackground(self, hist_, legendtext, color):
        hist = hist_.Clone()
        if self.debug: print("Add background")
        if self.rebin > 1:
            hist.Rebin(self.rebin)
        if self.divideByWidth:
            hist.Scale(1, "width")
        self.__hasBackground = True
        self.__NlegEntries += 1
        bkg = {}
        bkg["name"] = legendtext
        bkg["hist"] = hist
        bkg["hist"].SetFillColor(color)
        bkg["hist"].SetLineColor(color)
        self.__backgrounds.append(bkg)
        if self.__autoYrange and hist.GetMaximum() > self.__ymax:
            self.__ymax = hist.GetMaximum()
        # Check if min and max bounds of x-axis fit other histograms
        if self.__xmin == -1:
            self.__xmin = hist.GetXaxis().GetBinLowEdge(1)
        else:
            if hist.GetXaxis().GetBinLowEdge(1) != self.__xmin:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)
        if self.__xmax == -1:
            self.__xmax = hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2)
        else:
            if hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2) != self.__xmax:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)
    ############################################################################
    # Add signals that are displayed as lines
    def addSignal(self, hist_, legendtext, color, lineStyle=1, lineWidth=2):
        hist = hist_.Clone()
        if self.debug: print("Add signal")
        if self.rebin > 1:
            hist.Rebin(self.rebin)
        if self.divideByWidth:
            hist.Scale(1, "width")
        self.__hasSignal = True
        self.__NlegEntries += 1
        sig = {}
        sig["name"] = legendtext
        sig["hist"] = hist
        sig["hist"].SetFillColorAlpha(ROOT.kWhite, 0.0)
        sig["hist"].SetLineColor(color)
        sig["hist"].SetLineWidth(lineWidth)
        sig["hist"].SetLineStyle(lineStyle)
        sig["color"] = color
        sig["linestyle"] = lineStyle
        sig["linewidth"] = lineWidth
        self.__signals.append(sig)
        if self.__autoYrange and hist.GetMaximum() > self.__ymax:
            self.__ymax = hist.GetMaximum()
        # Check if min and max bounds of x-axis fit other histograms
        if self.__xmin == -1:
            self.__xmin = hist.GetXaxis().GetBinLowEdge(1)
        else:
            if hist.GetXaxis().GetBinLowEdge(1) != self.__xmin:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)
        if self.__xmax == -1:
            self.__xmax = hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2)
        else:
            if hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2) != self.__xmax:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)

    ############################################################################
    # Add data that are displayed with markers,
    # only one data histogram is allowed
    def addData(self, hist_, legendtext="Data"):
        hist = hist_.Clone()
        if self.debug: print("Add data")
        if self.rebin > 1:
            hist.Rebin(self.rebin)
        if self.divideByWidth:
            hist.Scale(1, "width")
        self.__NlegEntries += 1
        if self.__hasData:
            print("[Error]: Cannot add Data more than once.")
            sys.exit(1)
        self.__hasData = True
        self.__data["name"] = legendtext
        self.__data["hist"] = hist
        self.__data["hist"].SetLineColor(ROOT.kBlack)
        self.__data["hist"].SetMarkerColor(ROOT.kBlack)
        self.__data["hist"].SetMarkerStyle(8)
        self.__data["hist"].SetMarkerSize(1)
        if self.__autoYrange and hist.GetMaximum() > self.__ymax:
            self.__ymax = hist.GetMaximum()
        # Check if min and max bounds of x-axis fit other histograms
        if self.__xmin == -1:
            self.__xmin = hist.GetXaxis().GetBinLowEdge(1)
        else:
            if hist.GetXaxis().GetBinLowEdge(1) != self.__xmin:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)
        if self.__xmax == -1:
            self.__xmax = hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2)
        else:
            if hist.GetXaxis().GetBinUpEdge(hist.GetSize()-2) != self.__xmax:
                print("[Error]: Histogram has not the same lower bound as those already added.")
                sys.exit(1)

    ############################################################################
    # Add systematic
    def addSystematic(self, up_, down_, sysname, bkgname, from_norm=False):
        up = up_.Clone()
        down = down_.Clone()
        if self.debug: print("Add systematic")
        if not from_norm:
            # If this function is called from 'addNormSystematic()', do not
            # rebin again
            if self.rebin > 1:
                up.Rebin(self.rebin)
                down.Rebin(self.rebin)
        if self.divideByWidth:
            up.Scale(1, "width")
            down.Scale(1, "width")
        self.__doSystematics = True
        foundBackground = False
        for bkg in self.__backgrounds:
            if bkg["name"] == bkgname:
                foundBackground = True
                deltaUp = up.Clone()
                deltaUp.Add(bkg["hist"], -1)
                deltaDown = down.Clone()
                deltaDown.Add(bkg["hist"], -1)

                self.__sysDeltas.append( (sysname, bkgname, deltaUp, deltaDown) )
                self.__sysnames.append(sysname)
        if not foundBackground:
            print("[Error]: Trying to add %s systematic to %s, but could not find a background with name %s"%(sysname, bkgname, bkgname))
            sys.exit(1)

    ############################################################################
    # Add a normalization uncertainty
    def addNormSystematic(self, bkgname, size):
        self.__doSystematics = True
        foundBackground = False
        for bkg in self.__backgrounds:
            if bkg["name"] == bkgname:
                foundBackground = True
                up   = bkg["hist"].Clone()
                down = bkg["hist"].Clone()
                up.Scale(1.0+size)
                down.Scale(1.0-size)
                self.addSystematic(up, down, bkgname+"_norm", bkgname, from_norm=True)
        if not foundBackground:
            print("[Error]: Trying to add normalization systematic to %s, but could not find a background with name %s" %(sysname, bkgname, bkgname))
            sys.exit(1)

    ############################################################################
    # Customize min/max of x axis
    def setCustomXRange(self, min, max):
        self.__xmin_draw = min
        self.__xmax_draw = max
        self.__customXrange = True

    ############################################################################
    # Customize min/max of y axis
    def setCustomYRange(self, min, max):
        self.__ymin = min
        self.__ymax = max
        self.yfactor = 1
        self.__autoYrange = False

    ############################################################################
    # Add some text to the plot
    def addText(self, x, y, text, font=43, size=12):
        latex = ROOT.TLatex(3.5, 24, text)
        latex.SetNDC()
        latex.SetTextAlign(13)
        latex.SetTextFont(font)
        latex.SetTextSize(size)
        latex.SetX(x)
        latex.SetY(y)
        self.__latexTexts.append(latex)

    ############################################################################
    # Return the TGraph with total uncertaintis
    def getTotalSystematic(self):
        return self.__errorhist

    ############################################################################
    # Function to increase margin by a factor
    def increaseMargin(self, label, factor):
        if label == "top":
            self.__MarginTop *= factor
        elif label == "bottom":
            self.__MarginBottom *= factor
        elif label == "right":
            self.__MarginRight *= factor
        elif label == "left":
            self.__MarginLeft *= factor

    ############################################################################
    # Private function to set the binning for current plot
    def __storeBinning(self):
        binning = []
        Nbins = 0
        if self.__hasBackground:
            Nbins = self.__backgrounds[0]["hist"].GetSize()-2
            for i in range(Nbins):
                bin = i+1
                binning.append(self.__backgrounds[0]["hist"].GetXaxis().GetBinLowEdge(bin))
                if bin == Nbins:
                    binning.append(self.__backgrounds[0]["hist"].GetXaxis().GetBinUpEdge(bin))
        elif self.__hasSignal:
            Nbins = self.__signals[0]["hist"].GetSize()-2
            for i in range(Nbins):
                bin = i+1
                binning.append(self.__signals[0]["hist"].GetXaxis().GetBinLowEdge(bin))
                if bin == Nbins:
                    binning.append(self.__signals[0]["hist"].GetXaxis().GetBinUpEdge(bin))
        elif self.__hasData:
            Nbins = self.__data["hist"].GetSize()-2
            for i in range(Nbins):
                bin = i+1
                binning.append(self.__data["hist"].GetXaxis().GetBinLowEdge(bin))
                if bin == Nbins:
                    binning.append(self.__data["hist"].GetXaxis().GetBinUpEdge(bin))
        else:
            print("[Error]: Binning cannot be extracted since no histograms are defined")
        self.__binning = binning
        self.__Nbins = Nbins
        return
    ############################################################################
    # Private function to sort backgrounds by their integral, put them in a stack,
    # and fill legend with the names (in the correct order).
    # Also create a histogram with all backgrounds added.
    def __buildStack(self):
        bkg_list = []
        # First go through backgrounds and store the integral
        isFirst = True
        for bkg in self.__backgrounds:
            integral = bkg["hist"].Integral()
            bkg_list.append( (bkg["hist"], integral, bkg["name"]) )
            if isFirst:
                self.__bkgtotal = bkg["hist"].Clone()
                isFirst = False
            else:
                self.__bkgtotal.Add(bkg["hist"])
        # Sort by integral
        def takeSecond(elem):
            return elem[1]
        bkg_list.sort(key=takeSecond)
        # Now fill stack
        for hist, integral, bkgname in bkg_list:
            self.__stack.Add(hist)
        # Legend has to be filled in reverse order
        for hist, integral, bkgname in reversed(bkg_list):
            self.__legend.AddEntry(hist, bkgname, "f")
        # Also set a new ymax
        if self.__autoYrange and self.__stack.GetMaximum() > self.__ymax:
            self.__ymax = self.__stack.GetMaximum()

    ############################################################################
    # Private, add up all sys variations and MC stat
    # Use central for up/down if no Systematics are set
    def __getTotalUncertainty(self):
        for i in range(self.__Nbins):
            bin = i+1
            bincenter = self.__bkgtotal.GetXaxis().GetBinCenter(bin)
            bincontent = self.__bkgtotal.GetBinContent(bin)
            self.__errorhist.SetPoint(bin, bincenter, bincontent)
            staterr = self.__bkgtotal.GetBinError(bin)
            totalErrSquared_up   = pow(staterr, 2)
            totalErrSquared_down = pow(staterr, 2)
            # Go through all uncertainties and add up shifts connected to the
            # same sys but different backgrounds
            for sys in self.__sysnames:
                shift_up = 0
                shift_down = 0
                for (sysname, bkgname, deltaUp, deltaDown) in self.__sysDeltas:
                    if sys == sysname:
                        dup = deltaUp.GetBinContent(bin)
                        ddown = deltaDown.GetBinContent(bin)
                        # case where up and down variations are in opposite directions
                        if dup > 0 and ddown < 0:
                            shift_up += dup
                            shift_down += ddown
                        elif dup < 0 and ddown > 0:
                            shift_up += ddown
                            shift_down += dup
                        elif dup > 0 and ddown > 0:
                            shift_up += dup if abs(dup) > abs(ddown) else ddown
                        elif dup < 0 and ddown < 0:
                            shift_down += dup if abs(dup) > abs(ddown) else ddown
                totalErrSquared_up += pow(shift_up, 2)
                totalErrSquared_down += pow(shift_down, 2)
            # if the total error is 0.0, the plotter does weird stuff, so
            # set to super small value > 0
            if totalErrSquared_up < pow(10, -20):
                totalErrSquared_up = pow(10, -20)
            if totalErrSquared_down < pow(10, -20):
                totalErrSquared_down = pow(10, -20)
            # Get X errors:
            ex_low = self.__bkgtotal.GetXaxis().GetBinCenter(bin) - self.__bkgtotal.GetXaxis().GetBinLowEdge(bin)
            ex_up =  self.__bkgtotal.GetXaxis().GetBinUpEdge(bin) - self.__bkgtotal.GetXaxis().GetBinCenter(bin)
            self.__errorhist.SetPointError(bin, ex_low, ex_up, sqrt(totalErrSquared_down), sqrt(totalErrSquared_up))
    ############################################################################
    # Private, create the ratio plot
    def __getRatio(self, h1, h2, color=None, linestyle=1, linewidth=2):
        self.__ratioCounter += 1
        ratio = ROOT.TH1F("ratio"+str(self.__ratioCounter), "ratio", self.__Nbins, arr.array('d',self.__binning))
        for i in range(self.__Nbins):
            bin=i+1
            if h2.GetBinContent(bin)==0:
                r=-1
                e=0
            else:
                r = h1.GetBinContent(bin)/h2.GetBinContent(bin)
                e = h1.GetBinError(bin)/h2.GetBinContent(bin)
            ratio.SetBinContent(bin,r)
            ratio.SetBinError(bin,e)
        if color is not None:
            ratio.SetLineColor(color)
        ratio.SetLineStyle(linestyle)
        ratio.SetLineWidth(linewidth)
        return ratio
    ############################################################################
    # Private, create the ratio uncertainty plot
    def __getRatioUncert(self, errorgraph):
        ratio = ROOT.TGraphAsymmErrors()
        Npoints = errorgraph.GetN()
        for i in range(Npoints):
            point=i+1
            # d1, d2 = ROOT.Double(0.), ROOT.Double(0.)
            # errorgraph.GetPoint(point, d1, d2)
            # Xval = float(d1)
            # Yval = float(d2)
            import ctypes
            d1, d2 = ctypes.c_double(0.0),ctypes.c_double(0.0)
            errorgraph.GetPoint(point, d1, d2)
            Xval = float(d1.value)
            Yval = float(d2.value)
            eX_hi = errorgraph.GetErrorXhigh(point)
            eX_lo = errorgraph.GetErrorXlow(point)

            if Yval==0:
                eY_hi = 0
                eY_lo = 0
            else:
                eY_hi = errorgraph.GetErrorYhigh(point)/Yval
                eY_lo = errorgraph.GetErrorYlow(point)/Yval

            ratio.SetPoint(point, Xval, 1.0)
            ratio.SetPointError(point, eX_lo, eX_hi, eY_lo, eY_hi)

        return ratio
    ############################################################################
    # Private, create the ratio plot
    def __getRatioLine(self):
        line = ROOT.TH1F("line", "line", self.__Nbins, arr.array('d',self.__binning))
        for i in range(self.__Nbins):
            bin=i+1
            line.SetBinContent(bin,1.0)
            line.SetBinError(bin,0.0)
        return line
    ############################################################################
    # Private, set options for the CMS label
    def __getCMS(self):
        cmstext = ROOT.TLatex(3.5, 24, "CMS")
        cmstext.SetNDC()
        cmstext.SetTextAlign(13)
        cmstext.SetTextFont(62)
        if self.drawRatio: cmstext.SetTextSize(0.08)
        else:              cmstext.SetTextSize(0.06)
        cmstext.SetX(0.22)
        cmstext.SetY(0.85)
        return cmstext

    ############################################################################
    # Private, set options for the CMS label
    def __getSimLabel(self):
        simtext = ROOT.TLatex(3.5, 24, self.simtext)
        simtext.SetNDC()
        simtext.SetTextAlign(13)
        simtext.SetX(0.22)
        simtext.SetTextFont(52)
        if self.drawRatio:
            simtext.SetTextSize(0.06)
            simtext.SetY(0.78)
        else:
            simtext.SetTextSize(0.04)
            simtext.SetY(0.80)
        return simtext

    ############################################################################
    # Private, set options for the subtext of the CMS label
    def __getSubtitle(self):
        subtext = ROOT.TLatex(3.5, 24, self.subtext)
        subtext.SetNDC()
        subtext.SetTextAlign(13)
        subtext.SetX(0.22)
        subtext.SetTextFont(52)
        yshift = 0
        if self.simtext is not None:
            yshift = -0.05
        if self.drawRatio:
            subtext.SetTextSize(0.06)
            subtext.SetY(0.78+yshift)
        else:
            subtext.SetTextSize(0.04)
            subtext.SetY(0.80+yshift)

        return subtext

    ############################################################################
    # Private, set options for the lumi label
    def __getLumi(self):
        infotext = "%s fb^{-1} [13 TeV]" %(self.lumi)
        lumitext = ROOT.TLatex(3.5, 24, infotext)
        lumitext.SetNDC()
        lumitext.SetTextAlign(31)
        lumitext.SetTextFont(42)
        lumitext.SetX(1-self.__MarginRight)
        if self.drawRatio:
            lumitext.SetTextSize(0.055)
            lumitext.SetY(1-self.__MarginTop+0.013)
        else:
            lumitext.SetTextSize(0.0367)
            lumitext.SetY(1-self.__MarginTop+0.013)
        lumitext.Draw()
        return lumitext

    ############################################################################
    # Private, set draw options for the histograms
    def __setDrawOptions(self, hist):
        hist.SetTitle("")
        hist.GetXaxis().SetTitle(self.xtitle)
        hist.GetYaxis().SetTitle(self.ytitle)
        hist.GetYaxis().SetTitleOffset(1.3)
        hist.GetYaxis().SetTitleSize(0.06)
        if not self.log:
            hist.GetYaxis().SetNdivisions(505)
        else:
            hist.GetYaxis().SetNdivisions(510)
        if self.__customXrange:
            hist.GetXaxis().SetRangeUser(self.__xmin_draw, self.__xmax_draw)
        hist.GetYaxis().SetRangeUser(self.__ymin, self.yfactor*self.__ymax)
        hist.GetXaxis().SetTickLength(0.07)
        hist.GetXaxis().SetNdivisions(505)
        if not self.drawRatio:
            hist.GetXaxis().SetTitleSize(25)
            hist.GetXaxis().SetTitleFont(43)
            hist.GetXaxis().SetTitleOffset(1.2)
            hist.GetXaxis().SetLabelFont(43)
            hist.GetXaxis().SetLabelSize(21)
        else:
            hist.GetXaxis().SetLabelSize(0.)
            hist.GetYaxis().SetLabelSize(0.)

    ############################################################################
    # Private, set draw options for the error band
    def __setUncertDrawOptions(self, hist):
        self.__setDrawOptions(hist)
        hist.SetFillStyle(3245)
        hist.SetFillColor(13)
        hist.SetLineWidth(0)
        hist.SetMarkerStyle(0)

    ############################################################################
    # Private, set draw options for the ratio
    def __setRatioDrawOptions(self, ratio, isData=False):
        (ymin, ymax) = self.ratiorange
        ratio.SetTitle('')
        ratio.GetYaxis().SetTitle(self.ratiotitle)
        if self.__customXrange:
            ratio.GetXaxis().SetRangeUser(self.__xmin_draw, self.__xmax_draw)
        ratio.GetYaxis().SetRangeUser(ymin, ymax)
        ratio.GetYaxis().SetNdivisions(505)
        ratio.GetYaxis().CenterTitle()
        ratio.GetYaxis().SetTitleSize(22)
        ratio.GetYaxis().SetTitleFont(43)
        ratio.GetYaxis().SetTitleOffset(2.2)
        ratio.GetYaxis().SetLabelFont(43)
        ratio.GetYaxis().SetLabelSize(19)
        ratio.GetYaxis().SetLabelOffset(0.009)
        ratio.GetXaxis().SetTitle(self.xtitle)
        ratio.GetXaxis().SetTickLength(0.07)
        ratio.GetXaxis().SetTitleSize(25)
        ratio.GetXaxis().SetTitleFont(43)
        if self.isPYROOT:
            ratio.GetXaxis().SetTitleOffset(1.3)
        else:
            ratio.GetXaxis().SetTitleOffset(5.5)
        ratio.GetXaxis().SetLabelFont(43)
        ratio.GetXaxis().SetLabelSize(21)
        ratio.GetXaxis().SetLabelOffset(0.035)
        ratio.GetXaxis().SetNdivisions(505)
        if isData:
            ratio.SetLineColor(ROOT.kBlack)
            ratio.SetMarkerColor(ROOT.kBlack)
            ratio.SetMarkerStyle(8)
            ratio.SetMarkerSize(1)


    ############################################################################
    # Private, set draw options for the ratio that contains outside values
    def __setRatioOutsideDrawOptions(self, ratio):
        # first apply usual cosmetics
        self.__setRatioDrawOptions(ratio)
        # Now change marker
        ratio.SetMarkerSize(0)

    ############################################################################
    # Private, get a ratio histogram that contains values outside the y range
    def __getRatioOutside(self, ratio):
        # If there are points outside the y range, the error bar is not drawn,
        # even if it would reach into the plot-
        # To solve this, an additional histogram is drawn
        (ymin, ymax) = self.ratiorange
        ratio_outside = ROOT.TH1F("ratio_outside", "ratio_outside", self.__Nbins, arr.array('d',self.__binning))
        for i in range(self.__Nbins):
            bin = i+1
            central = ratio.GetBinContent(bin)
            error = ratio.GetBinError(bin)
            min = ratio.GetBinContent(bin)-error
            max = ratio.GetBinContent(bin)+error
            if central > ymax and min < ymax:
                # if point is above y range:
                # 1. move central value down to the edge of y range
                # 2. make error smaller to make the error bar end at the same value as before
                offset = central - ymax
                central_new = central-offset
                error_new   = error-offset

            elif central < ymin and max > ymin:
                # if point is below y range:
                # 1. move central value up to the edge of y range
                # 2. make error smaller to make the error bar end at the same value as before
                offset = ymin - central
                central_new = central+offset
                error_new   = error-offset
            else:
                central_new = 0.0
                error_new   = 0.0
            ratio_outside.SetBinContent(bin, central_new)
            ratio_outside.SetBinError(bin, error_new)
        return ratio_outside


    ############################################################################
    # This function draws and saves the final plot.
    # It takes care of which objects exist (backgrounds, signals, data) and
    # all cosmetics are steered from here
    def draw(self):
        if self.debug: print("Store binning")
        self.__storeBinning()
        if self.debug: print("Create canvas and pads")
        canvas = ROOT.TCanvas("canvas"+str(self.id), "canvas"+str(self.id), 600, 600)
        pady1 = 0.31 if self.drawRatio else 0.0
        pad1 = ROOT.TPad("pad1", "pad1", 0, pady1, 1, 1.0)
        if self.drawRatio: pad1.SetBottomMargin(0.02)
        else:              pad1.SetBottomMargin(self.__MarginBottom*0.25)
        pad1.SetTopMargin(self.__MarginTop)
        pad1.SetLeftMargin(self.__MarginLeft)
        pad1.SetRightMargin(self.__MarginRight)
        pad1.Draw()

        if self.drawRatio:
            pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
            pad2.SetLeftMargin(self.__MarginLeft)
            pad2.SetRightMargin(self.__MarginRight)
            pad2.SetTopMargin(0)
            pad2.SetBottomMargin(self.__MarginBottom)
            pad2.Draw()

        pad1.cd()
        if self.log:
            if self.__autoYrange:
                self.__ymin = 0.0011*self.__ymax
                self.yfactor *= 100
            pad1.SetLogy()
        if self.debug: print("Set up legends and draw")
        self.__legend = ROOT.TLegend(.51+self.legshift[0],.85+self.legshift[1]-self.__NlegEntries*0.075/2,.9+self.legshift[2],.85+self.legshift[3])
        if self.NcolumnsLegend > 1:
            self.__legend.SetNColumns(self.NcolumnsLegend)
        histdrawn = False # Keep track if "SAME" option should be used
        if self.__hasData:
            self.__legend.AddEntry(self.__data["hist"], self.__data["name"], "pel")
        if self.__hasBackground:
            self.__buildStack()
            self.__setDrawOptions(self.__backgrounds[0]["hist"])
            if self.debug: print("Draw backgrounds")
            self.__backgrounds[0]["hist"].Draw("HIST")
            histdrawn = True
            self.__stack.Draw("HIST SAME")
            # Uncertainty on MC
            self.__getTotalUncertainty()
            self.__setUncertDrawOptions(self.__errorhist)
            self.__errorhist.Draw("E2 HIST SAME")
            self.__legend.AddEntry(self.__errorhist, self.totalUncText,"f")
        if self.__hasSignal:
            for sig in self.__signals:
                if self.debug: print("Draw signals")
                self.__setDrawOptions(sig["hist"])
                self.__legend.AddEntry(sig["hist"], sig["name"], "l")
                if histdrawn: sig["hist"].Draw("HIST SAME")
                else:         sig["hist"].Draw("HIST ")
                histdrawn = True
        if self.__hasData:
            if self.debug: print("Draw data")
            self.__setDrawOptions(self.__data["hist"])
            if histdrawn: self.__data["hist"].Draw("P SAME")
            else:         self.__data["hist"].Draw("P")
            histdrawn = True

        # Now draw the ratio pad
        if self.drawRatio:
            if self.debug: print("Set up axis for ratio")
            axis = ROOT.TGaxis( self.__xmin, self.__ymin, self.__xmin, self.yfactor*self.__ymax, self.__ymin, self.yfactor*self.__ymax, 505,"")
            if self.log:
                axis = ROOT.TGaxis( self.__xmin, self.__ymin, self.__xmin, self.yfactor*self.__ymax, self.__ymin, self.yfactor*self.__ymax, 505,"G")
            axis.SetLabelOffset(0.01)
            axis.SetLabelFont(43)
            axis.SetLabelSize(21)
            axis.SetNdivisions(505)
            if self.log:
                axis.SetNdivisions(510)
            axis.Draw()

            if self.debug: print("Go into ratio pad")
            pad2.cd()
            if self.debug: print("Draw ratio line")
            ratioline = self.__getRatioLine()
            self.__setRatioDrawOptions(ratioline)
            ratioline.SetFillColor(0)
            ratioline.SetLineColor(15)
            ratioline.SetLineWidth(2)
            ratioline.Draw("HIST")
            if self.debug: print("Draw ratio error band")
            ratio_uncert = self.__getRatioUncert(self.__errorhist)
            self.__setUncertDrawOptions(ratio_uncert)
            ratio_uncert.Draw("E2 SAME")
            if self.__hasSignal:
                if self.debug: print("Draw signal ratios")
                ratios_sig = []
                for sig in self.__signals:
                    ratios_sig.append(self.__getRatio(sig["hist"], self.__bkgtotal, sig["color"], sig["linestyle"], sig["linewidth"]) )
                for r in ratios_sig:
                    self.__setRatioDrawOptions(r)
                    r.Draw("HIST SAME")
            if self.__hasData:
                if self.debug: print("Draw data ratio")
                ratio_data = self.__getRatio(self.__data["hist"], self.__bkgtotal)
                self.__setRatioDrawOptions(ratio_data, isData=True)
                ratio_data.Draw("P SAME")
                if self.debug: print("Draw data ratio outside")
                ratio_data_outside = self.__getRatioOutside(ratio_data)
                self.__setRatioOutsideDrawOptions(ratio_data_outside)
                ratio_data_outside.Draw("P SAME")

            ROOT.gPad.RedrawAxis()

        # Back to pad1 and draw labels and legend
        if self.debug: print("Draw labels")
        pad1.cd()
        if self.drawLogo:
            CMSlabel = self.__getCMS()
            CMSlabel.Draw()
            if self.simtext is not None:
                simlabel = self.__getSimLabel()
                simlabel.Draw()
            sublabel = self.__getSubtitle()
            sublabel.Draw()
        if self.lumi is not None:
            lumilable = self.__getLumi()
            lumilable.Draw()
        textsize = self.legtextsize if self.drawRatio else self.legtextsize*0.75
        self.__legend.SetTextSize(textsize)
        self.__legend.Draw()
        ROOT.gPad.RedrawAxis()

        # Now draw text boxes
        if self.debug: print("Draw Text")
        for text in self.__latexTexts:
            text.Draw()

        # Save plot
        if self.debug: print("Save plot")
        plotname = os.path.join(self.plot_dir, self.plotname+".pdf")
        canvas.Print(plotname)
