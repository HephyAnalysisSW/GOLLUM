from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Repo-relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from pdf.nnpdf.constants import LHAPDF_XGRID, XGRID
import lhapdf
from pdf.nnpdf.distance_def import *
import common.user as user
import common.syncer as syncer
import common.helpers as helpers

flavour = {
    1: "d",
    -1: r"$\bar{d}$",
    2: "u",
    -2: r"$\bar{u}$",
    3: "s",
    -3: r"$\bar{s}$",
    4: "c",
    -4: r"$\bar{c}$",
    5: "b",
    -5: "bbar",
    6: "t",
    21: "g"
}
tex_flavour = {1: 'd',  -1: '#bar{d}',2: 'u', -2: '#bar{u}', 3: 's', -3: '#bar{s}',4: 'c', -4: '#bar{c}',  5: 'b', -5: '#bar{b}', 6: 't', 21: 'g'}

_colors = {
    "black":         ROOT.TColor.GetColor("#000000"),
    "blue":          ROOT.TColor.GetColor("#0072B2"),
    "orange":        ROOT.TColor.GetColor("#E69F00"),
    "bluish_green":  ROOT.TColor.GetColor("#009E73"),
    "vermillion":    ROOT.TColor.GetColor("#D55E00"),
    "reddish_purple":ROOT.TColor.GetColor("#CC79A7"),
    "sky_blue":      ROOT.TColor.GetColor("#56B4E9"),
    "yellow":        ROOT.TColor.GetColor("#F0E442"),
    "gray":          ROOT.TColor.GetColor("#999999"),
    "brown":         ROOT.TColor.GetColor("#8B4513"),
}

colors = [ _colors[i] for i in ["black", "blue", "orange", "bluish_green",  "reddish_purple", "gray", "sky_blue", "yellow", "brown", "vermillion"] ]

# Define the x grid of interest
x_grid = LHAPDF_XGRID[54:] # points that are >= 0.05 <= 0.6

# Set Q scale
Q = 50

plot_directory = os.path.join( user.plot_directory, f"basis_plots_Q_{Q}")

# How many shapes per plot? 
n_var_per_plot = 5
pdf_basis = "250503_pod_basis_40k"
center_pdf = lhapdf.mkPDF(pdf_basis, 0)

for n_plot in range(20):

    wmin_tot_basis = []

    for i_basis in range(n_var_per_plot):
        wmin_tot_basis.append(lhapdf.mkPDF(pdf_basis, n_plot*n_var_per_plot+i_basis+1))

    center_pdf.xfxQ(21, 0.5, Q)

    # Use your x_grid selection for plotting
    plot_x = np.array(x_grid, dtype="float64")
    plot_Q = np.full_like(plot_x, Q, dtype="float64")

    # List of flavours to plot, excluding top
    flavs_to_plot = [pid for pid in flavour.keys() if abs(pid) != 6]

    # Canvas layout
    n_flavs = len(flavs_to_plot)
    n_cols, n_rows = 3, 4   # 3x4 = 12 pads, one spare if you have 11 flavours
    c = ROOT.TCanvas("c_all", "All flavours", 1600, 1600)
    c.Divide(n_cols, n_rows)


    # We'll also build a legend once (for the PDFs) and draw it in the last pad
    leg = ROOT.TLegend(0.05, 0.2, 0.95, 0.93)
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)


    pdfs_to_plot = [center_pdf] + wmin_tot_basis

    pdf_labels   = ["center"] 
    for i_label in  range(len(wmin_tot_basis)):
        n_var = n_plot*n_var_per_plot+i_label+1
        pdf_labels.append( f"basis {n_var}")

    # We need one set of graphs to attach to the legend; take from the first flavour
    graphs_for_legend = []
    stuff = []
    for ipad, pid in enumerate(flavs_to_plot, start=1):
        c.cd(ipad)
        pad = ROOT.gPad
        pad.SetLogx()
        stuff.append(pad)
        mg = ROOT.TMultiGraph()
        stuff.append(mg)
        for i, (pdf_obj, label) in enumerate(zip(pdfs_to_plot, pdf_labels)):
            # Vectorized call: list of dicts (one per x,Q pair)
            vals_dicts = pdf_obj.xfxQ(tuple(plot_x), tuple(plot_Q))
            y_vals = np.array([d.get(pid, 0.0) for d in vals_dicts], dtype="float64")

            gr = ROOT.TGraph(len(plot_x), plot_x, y_vals)
            gr.SetLineWidth(2)
            gr.SetLineColor(colors[i % len(colors)])
            mg.Add(gr, "L")
            stuff.append(gr)
            # Store graphs for legend only once, from the first pad
            if ipad == 1:
                graphs_for_legend.append((gr, label))

        mg.Draw("AL")
        mg.SetTitle("")
        mg.GetXaxis().SetTitle("x")
        mg.GetYaxis().SetTitle("x f(x, Q^{2})")
        mg.GetXaxis().SetMoreLogLabels(True)
        mg.GetXaxis().SetNoExponent(True)

        # Flavour label in the top-left of each pad
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.05)
        latex.DrawLatex(0.90, 0.3, tex_flavour[pid])

    # The last pad has the legend
    c.cd(ipad+1)
    # Build the legend (using graphs from the first flavour)
    for gr, label in graphs_for_legend:
        leg.AddEntry(gr, label, "l")

    # Draw legend in the last used pad
    leg.Draw()

    # Single-page PDF
    helpers.copyIndexPHP(plot_directory)
    out_pdf = os.path.join( plot_directory, f"pdf_flavours_{n_plot:02d}_logx.png")

    c.Print(out_pdf)
syncer.sync()


## Choose only gluon
#flavours = [21]
#center_pdf_grid = pdf_grid_allflav(center_pdf, flavours, x_grid, Q)
#
#class POD_PDF():
#    def __init__(self, central_pdf, modes):
#        self.central_pdf = central_pdf
#        self.modes = modes
#
#    def __call__(self, fl, x, Q, weights):
#        phi0 = self.central_pdf.xfxQ(fl, x, Q)
#        phis = [mode.xfxQ(fl, x, Q) for mode in self.modes]
#        
#        for i, w in enumerate(weights):
#            phis[i] = w * (phis[i] - phi0)
#
#        return phi0 + sum(phis)
#
## example of building and calling it
#pod_pdf = POD_PDF(center_pdf, wmin_tot_basis)
#
#weights = [0.1, 0.2, 0.3, -0.1, -0.2]
#
#pod_pdf(21, 0.5, Q, weights)


# Here we define the PDFs we want to try to reproduce
PDF_sets = {
    "CT18NNLO": 58,
    "NNPDF31_nnlo_as_0118": 100,
    "NNPDF40_nnlo_as_01180": 100,
    "MSHT20nnlo_as118": 64,
}


# Overlay central PDFs from PDF_sets in the same style as above

# Build list of central PDFs (member 0 of each set)
pdfs_to_plot  = []
pdf_labels    = []
for pdf_name in PDF_sets.keys():
    pdfs_to_plot.append(lhapdf.mkPDF(pdf_name, 0))
    pdf_labels.append(pdf_name)

# Use your x_grid selection for plotting
plot_x = np.array(x_grid, dtype="float64")
plot_Q = np.full_like(plot_x, Q, dtype="float64")

# List of flavours to plot, excluding top
flavs_to_plot = [pid for pid in flavour.keys() if abs(pid) != 6]

# Canvas layout (same as above)
n_flavs = len(flavs_to_plot)
n_cols, n_rows = 3, 4   # 3x4 = 12 pads, one spare if you have 11 flavours
c = ROOT.TCanvas("c_sets", "All flavours - PDF sets", 1600, 1600)
c.Divide(n_cols, n_rows)

# Legend (same style as above)
leg = ROOT.TLegend(0.05, 0.2, 0.95, 0.93)
leg.SetNColumns(2)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetTextSize(0.035)

# We need one set of graphs to attach to the legend; take from the first flavour
graphs_for_legend = []
stuff = []

for ipad, pid in enumerate(flavs_to_plot, start=1):
    c.cd(ipad)
    pad = ROOT.gPad
    pad.SetLogx()
    stuff.append(pad)
    mg = ROOT.TMultiGraph()
    stuff.append(mg)

    for i, (pdf_obj, label) in enumerate(zip(pdfs_to_plot, pdf_labels)):
        # Vectorized call: list of dicts (one per x,Q pair)
        vals_dicts = pdf_obj.xfxQ(tuple(plot_x), tuple(plot_Q))
        y_vals = np.array([d.get(pid, 0.0) for d in vals_dicts], dtype="float64")

        gr = ROOT.TGraph(len(plot_x), plot_x, y_vals)
        gr.SetLineWidth(2)
        gr.SetLineColor(colors[i % len(colors)])
        mg.Add(gr, "L")
        stuff.append(gr)

        # Store graphs for legend only once, from the first pad
        if ipad == 1:
            graphs_for_legend.append((gr, label))

    mg.Draw("AL")
    mg.SetTitle("")
    mg.GetXaxis().SetTitle("x")
    mg.GetYaxis().SetTitle("x f(x, Q^{2})")
    mg.GetXaxis().SetMoreLogLabels(True)
    mg.GetXaxis().SetNoExponent(True)

    # Flavour label in the top-left of each pad
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.90, 0.3, tex_flavour[pid])

# The last pad has the legend
c.cd(ipad + 1)
for gr, label in graphs_for_legend:
    leg.AddEntry(gr, label, "l")
leg.Draw()

# Save overlay plot
helpers.copyIndexPHP(plot_directory)
out_pdf_sets = os.path.join(plot_directory, "pdf_flavours_PDFsets_logx.png")
c.Print(out_pdf_sets)




#assert False, ""
#
#pdfs_target = {}
#
#for PDF_set, nreps in PDF_sets.items():
#    pdfs_target[PDF_set] = []
#
#    for i in range(nreps):
#        pdfs_target[PDF_set].append(lhapdf.mkPDF(PDF_set, i+1))
#
#distances = {}
#summed_distances = {}
#mean_squared_error = {}
#median_squared_error = {}
#
#basis_dims = [1, 3, 5]
#
#for pdf_target_name, pdf_target_replicas in pdfs_target.items():
#    
#    distances[pdf_target_name] = []
#    summed_distances[pdf_target_name] = []
#    mean_squared_error[pdf_target_name] = []
#    median_squared_error[pdf_target_name] = []
#
#    for basis_dim in basis_dims:
#        wmin_basis = wmin_tot_basis[:basis_dim]
#        
#        distance = []
#
#        for pdf_target_replica in pdf_target_replicas:
#            original, reco, w, d = wmin_distance(pdf_target_replica, center_pdf_grid, wmin_basis, flavours, x_grid, Q, dist_type=0)
#            distance.append(d)
#
#        distances[pdf_target_name].append((basis_dim, distance))
#        summed_distances[pdf_target_name].append(np.sum(distance))
#        mean_squared_error[pdf_target_name].append(np.mean(distance))
#        median_squared_error[pdf_target_name].append(np.median(distance))
#
#fig, ax = plt.subplots(figsize=(7, 5))
#
#for pdf_target_name, pdf_target_replicas in pdfs_target.items():
#    ax.plot(basis_dims, median_squared_error[pdf_target_name], "-o", label=f"{pdf_target_name}", linewidth=2.5)
#    
#
## Labels and legend 
#ax.set_xlabel("POD basis dimension", fontsize=16)
#ax.set_ylabel("Median distance", fontsize=16)
#ax.legend(frameon=False, fontsize=14, loc="upper right")
#
## Improve grid visibility
#ax.grid(True, linestyle="--", alpha=0.5)
#ax.set_yscale("log")
## Adjust layout and save
#plt.tight_layout()
#fig.savefig("median_distance_generalisation.pdf",  bbox_inches='tight', dpi=300)
##plt.show()
#
#
#pdf_target = "MSHT20nnlo_as118"
#with PdfPages(f"{pdf_target}.pdf") as pdf:
#    for j in range(50):
#        original, reco, w, d = wmin_distance(
#            pdfs_target[pdf_target][j], center_pdf_grid, wmin_tot_basis[:basis_dim], flavours, x_grid, Q, dist_type=0
#        )
#    
#        EPSILON = 1e-4
#        for i in range(len(flavours)):
#    
#            fig, [axup, axdown] = plt.subplots(
#                2, 1, sharex=True, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
#            )
#            
#            # Upper plot: original vs reconstructed PDF
#            axup.plot(x_grid, original[i], label="Original", linewidth=4)
#            axup.plot(x_grid, reco[i], label="Reconstructed", linewidth=3, linestyle="dashed")
#    
#            # Lower plot: Ratio plot
#            axdown.plot(x_grid, reco[i] / (original[i]+EPSILON), linewidth=3)
#            axdown.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)  # Reference line
#            axdown.set_ylim(0.5, 1.5)  # Adjust for better readability
#            axdown.set_xlabel("x", fontsize=16)
#            axdown.set_ylabel("Ratio", fontsize=16)
#    
#            # Formatting upper plot
#            axup.set_title(f"{flavour[flavours[i]]}(x) PDF {pdf_target} rep {j}", fontsize=16)
#            axup.set_xscale("log")
#            axup.set_ylabel(r"$x f(x)$", fontsize=16)
#            axup.legend(frameon=False, fontsize=14)
#    
#            # Improve grid visibility
#            axup.grid(True, linestyle="--", alpha=0.5)
#            axdown.grid(True, linestyle="--", alpha=0.5)
#    
#            # Save and close
#            pdf.savefig(bbox_inches="tight")
#            plt.close()
#
#pdf_target = "NNPDF31_nnlo_as_0118"
#with PdfPages(f"{pdf_target}.pdf") as pdf:
#    for j in range(50):
#        original, reco, w, d = wmin_distance(
#            pdfs_target[pdf_target][j], center_pdf_grid, wmin_tot_basis[:basis_dim], flavours, x_grid, Q, dist_type=0
#        )
#    
#        EPSILON = 1e-4
#        for i in range(len(flavours)):
#    
#            fig, [axup, axdown] = plt.subplots(
#                2, 1, sharex=True, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
#            )
#            
#            # Upper plot: original vs reconstructed PDF
#            axup.plot(x_grid, original[i], label="Original", linewidth=4)
#            axup.plot(x_grid, reco[i], label="Reconstructed", linewidth=3, linestyle="dashed")
#    
#            # Lower plot: Ratio plot
#            axdown.plot(x_grid, reco[i] / (original[i]+EPSILON), linewidth=3)
#            axdown.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)  # Reference line
#            axdown.set_ylim(0.5, 1.5)  # Adjust for better readability
#            axdown.set_xlabel("x", fontsize=16)
#            axdown.set_ylabel("Ratio", fontsize=16)
#    
#            # Formatting upper plot
#            axup.set_title(f"{flavour[flavours[i]]}(x) PDF {pdf_target} rep {j}", fontsize=16)
#            axup.set_xscale("log")
#            axup.set_ylabel(r"$x f(x)$", fontsize=16)
#            axup.legend(frameon=False, fontsize=14)
#    
#            # Improve grid visibility
#            axup.grid(True, linestyle="--", alpha=0.5)
#            axdown.grid(True, linestyle="--", alpha=0.5)
#    
#            # Save and close
#            pdf.savefig(bbox_inches="tight")
#            plt.close()
#
#
#
