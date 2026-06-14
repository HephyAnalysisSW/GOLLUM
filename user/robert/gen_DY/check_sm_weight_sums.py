#!/usr/bin/env python3

import argparse

import ROOT

import samples
import samples_postprocessed


ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    nargs="+",
    default=[
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
    ],
)
parser.add_argument("--max-files", type=int, default=None)
args = parser.parse_args()

results = []

for sample_name in args.samples:
    component = samples_postprocessed.samples_by_name[sample_name]
    xsec = samples.get_sample(sample_name).xsec
    files = component.files if args.max_files is None else component.files[: args.max_files]

    chain = ROOT.TChain("Events")
    n_added = 0
    for path in files:
        if chain.Add(path):
            n_added += 1

    if n_added == 0:
        raise RuntimeError(f"No files added for {sample_name}")

    rdf = ROOT.RDataFrame(chain)
    rdf = rdf.Define("sm_weight", "xsec_weight * LHEReweightingWeight[0]")
    has_born = rdf.Filter("dy_born_has_candidate > 0")
    plot = has_born.Filter("dy_born_mll >= 54. && dy_born_mll < 150. && dy_born_abs_yll < 3.4")

    result = {
        "sample": sample_name,
        "xsec": xsec,
        "files": n_added,
        "events": int(rdf.Count().GetValue()),
        "has_born": int(has_born.Count().GetValue()),
        "plot": int(plot.Count().GetValue()),
        "sum_sm_has_born": float(has_born.Sum("sm_weight").GetValue()),
        "sum_sm_plot": float(plot.Sum("sm_weight").GetValue()),
        "sum_plain_has_born": float(has_born.Sum("xsec_weight").GetValue()),
    }
    results.append(result)

    print()
    print(sample_name)
    print(f"  xsec = {xsec:.12g}")
    print(f"  files = {result['files']}")
    print(f"  events/has_born/plot = {result['events']}/{result['has_born']}/{result['plot']}")
    print(f"  sum_sm_plot = {result['sum_sm_plot']:.12g}")
    print(f"  sum_sm_has_born = {result['sum_sm_has_born']:.12g}")
    print(f"  sum_plain_has_born = {result['sum_plain_has_born']:.12g}")
    print(f"  sum_sm_plot/xsec = {result['sum_sm_plot'] / xsec:.12g}")
    print(f"  sum_sm_has_born/xsec = {result['sum_sm_has_born'] / xsec:.12g}")
    print(f"  sum_plain_has_born/xsec = {result['sum_plain_has_born'] / xsec:.12g}")

print()
print("Pairwise ratios")
for i, left in enumerate(results):
    for right in results[i + 1 :]:
        print(f"{left['sample']} / {right['sample']}")
        print(f"  xsec ratio = {left['xsec'] / right['xsec']:.12g}")
        for key in ["sum_sm_plot", "sum_sm_has_born", "sum_plain_has_born"]:
            denom = right[key]
            ratio = left[key] / denom if denom else float("nan")
            print(f"  {key} ratio = {ratio:.12g}")
