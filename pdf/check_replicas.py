#!/usr/bin/env python3

import sys
import math

import ROOT
import lhapdf

# =============================================================================
# Configuration
# =============================================================================

# Relative thresholds around 1.0:
#   |LHEPdfWeight - 1| > LHE_THRESHOLD  -> LHE_diff = 1
#   |pdf_ratio   - 1| > PDF_THRESHOLD  -> PDF_diff = 1
LHE_THRESHOLD = 0.005   # 1% by default
PDF_THRESHOLD = 0.005   # 1% by default

# PDF set and dimensions
pdf_basis = "NNPDF31_nnlo_hessian_pdfas"
basis_dim = 102  # number of non-central members: 100 replicas + 2 alpha_s

DIFF_WARN_THRESHOLD_PERCENT = 1.0   # e.g. 1% by default


# =============================================================================
# Script body (top level, no main())
# =============================================================================

if len(sys.argv) < 2:
    print("Usage: python check_pdf_weights.py <input.root>")
    sys.exit(1)

input_fname = sys.argv[1]

# Open ROOT file and get Events tree
f = ROOT.TFile.Open(input_fname)
if not f or f.IsZombie():
    print(f"ERROR: could not open file '{input_fname}'")
    sys.exit(1)

tree = f.Get("Events")
if not tree:
    print("ERROR: could not find TTree 'Events' in file")
    sys.exit(1)

nentries = tree.GetEntries()
print(f"Loaded 'Events' with {nentries} entries")

# LHAPDF setup
print(f"Loading LHAPDF set: {pdf_basis}")
central_pdf = lhapdf.mkPDF(pdf_basis, 0)
replicas = [lhapdf.mkPDF(pdf_basis, i + 1) for i in range(basis_dim)]
pdf_members = [central_pdf] + replicas   # 0..102
expected_n_pdf = 1 + basis_dim           # 103

# Loop over first 5 events (or fewer if tree is small)
max_events = min(10, nentries)

for ientry in range(max_events):
    tree.GetEntry(ientry)

    id1 = int(tree.Generator_id1)
    id2 = int(tree.Generator_id2)
    
    if not( id1==21 and id2==21):
        continue

    x1 = float(tree.Generator_x1)
    x2 = float(tree.Generator_x2)
    Q = math.sqrt(float(tree.Generator_scalePDF))

    nLHEPdfWeight = int(tree.nLHEPdfWeight)

    if nLHEPdfWeight != expected_n_pdf:
        print(
            f"WARNING: Event {ientry}: nLHEPdfWeight = {nLHEPdfWeight}, "
            f"expected {expected_n_pdf}"
        )

    # Number of PDF weights actually available
    npdf = min(nLHEPdfWeight, expected_n_pdf)
    lhe_weights = [tree.LHEPdfWeight[i] for i in range(npdf)]

    # Compute denominator using central PDF: f0(id1,x1,Q) * f0(id2,x2,Q)
    if x1 <= 0.0 or x2 <= 0.0:
        print(
            f"WARNING: Event {ientry}: non-positive x1/x2 "
            f"(x1={x1}, x2={x2}), setting all ratios to 0"
        )
        pdf_ratios = [0.0] * npdf
    else:
        f0_1 = central_pdf.xfxQ(id1, x1, Q)
        f0_2 = central_pdf.xfxQ(id2, x2, Q)
        denom = f0_1 * f0_2

        if denom == 0.0 or not math.isfinite(denom):
            print(
                f"WARNING: Event {ientry}: central PDF product is zero or non-finite "
                f"(f0_1={f0_1}, f0_2={f0_2}), setting all ratios to 0"
            )
            pdf_ratios = [0.0] * npdf
        else:
            pdf_ratios = []
            for imem in range(npdf):
                pdf = pdf_members[imem]
                f1 = pdf.xfxQ(id1, x1, Q)
                f2 = pdf.xfxQ(id2, x2, Q)
                num = f1 * f2
                ratio = num / denom
                pdf_ratios.append(ratio)

    # Ensure central/central is exactly 1.0
    if npdf > 0:
        pdf_ratios[0] = 1.0

    # Print the table for this event
    print("\n" + "=" * 80)
    print(
        f"Event {ientry}: "
        f"id1={id1}, x1={x1:.5f}; id2={id2}, x2={x2:.5f}; Q={Q:.5f}"
    )
    print(f"nLHEPdfWeight = {nLHEPdfWeight}, using first {npdf} entries")
    print(
        "Columns:\n"
        " 1) index i\n"
        " 2) LHEPdfWeight[i]\n"
        " 3) variation_over_central[i]\n"
        " 4) |(LHE-1) - (ratio-1)| in %\n"
        " 5) WARN if col4 > {:.2f}%"
        .format(DIFF_WARN_THRESHOLD_PERCENT)
    )
    print("-" * 80)
    print(" idx   LHEPdfWeight      var/central    Δdev[%]   WARN")
    print("-" * 80)

    for i in range(npdf):
        lhe_val = float(lhe_weights[i])/float(lhe_weights[0])
        ratio_val = float(pdf_ratios[i])

        # Deviation from 1
        delta_lhe = lhe_val - 1.0
        delta_pdf = ratio_val - 1.0

        # Percent-level difference of the *deviation from 1*
        rel_diff_percent = abs(delta_lhe - delta_pdf) * 100.0

        warn = rel_diff_percent > DIFF_WARN_THRESHOLD_PERCENT

        line = (
            f"{i:4d}  "
            f"{lhe_val: 12.6f}   "
            f"{ratio_val: 12.6f}   "
            f"{rel_diff_percent:8.3f}   "
            f"{'WARN' if warn else '':>4}"
        )
        print(line)

f.Close()
print("\nDone.")

