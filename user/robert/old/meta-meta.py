#!/usr/bin/env python
"""
Standalone script to read all ROOT files in a directory, extract histograms from canvases,
compute their means and variances, and produce a histogram of means.
Also writes out a text file with the mean and variance of each histogram.

Usage:
    python mean_histogram.py /path/to/root/files
"""
import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import ROOT
import glob
from ROOT import TFile, TH1F
import common.user as user
import common.syncer
os.makedirs(os.path.join(user.plot_directory, "meta-meta"), exist_ok=True )
def main(directory):
    # Find all .root files in the given directory
    pattern = os.path.join(directory, "*.root")
    root_files = glob.glob(pattern)
    if not root_files:
        print(f"No ROOT files found in {directory}")
        return

    means = []
    txt_filename = os.path.join( user.plot_directory, 'meta-meta', "hist_stats.txt" )
    common.syncer.file_sync_storage.append( txt_filename )
    with open(txt_filename, "w") as txt_out:
        txt_out.write("file,histogram_name,mean,variance\n")

        # Loop over each ROOT file
        for rf in root_files:
            f = TFile.Open(rf)
            if not f or f.IsZombie():
                print(f"Failed to open {rf}")
                continue

            # Iterate over all objects in the file
            for key in f.GetListOfKeys():
                obj = key.ReadObj()
                # Check for canvas objects
                if obj.InheritsFrom("TCanvas"):
                    canvas = obj
                    primitives = canvas.GetListOfPrimitives()
                    for prim in primitives:
                        # Check for histogram primitives (TH1)
                        if prim.InheritsFrom("TH1"):
                            h = prim
                            m = h.GetMean()
                            rms = h.GetRMS()
                            means.append(m)
                            # Write stats to text file
                            txt_out.write(f"{os.path.basename(rf)},{h.GetName()},{m},{rms}\n")
            f.Close()

    if not means:
        print("No histograms found in any canvases.")
        return

    # Create histogram of the means
    min_mean = min(means)
    max_mean = max(means)
    # Number of bins set to number of entries for clarity
    n_bins = len(means)
    mean_hist = TH1F("meanHist", "Means of Histograms", n_bins, min_mean, max_mean)
    for m in means:
        mean_hist.Fill(m)

    # Save the mean-histogram to a new ROOT file
    out_file_name = os.path.join(user.plot_directory, "meta-meta", "mean_hist.root")
    out_file = TFile(out_file_name, "RECREATE")
    mean_hist.Write()
    out_file.Close()
    common.syncer.file_sync_storage.append( out_file_name  )

    c1 = ROOT.TCanvas()
    mean_hist.Draw()
    c1.Print( os.path.join(user.plot_directory, "meta-meta", "mean_hist.png") )
    c1.Print( os.path.join(user.plot_directory, "meta-meta", "mean_hist.pdf") )

    print(f"Processed {len(means)} histograms.")
    print(f"Mean histogram saved to mean_hist.root")
    print(f"Statistics written to {txt_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mean_histogram.py /path/to/root/files")
        sys.exit(1)
    main(sys.argv[1])

common.syncer.sync()
