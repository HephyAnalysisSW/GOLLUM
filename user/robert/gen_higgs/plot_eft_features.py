#!/usr/bin/env python3

import argparse
import copy
import os
import sys

import numpy as np
import ROOT
from tqdm import tqdm

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

import common.helpers as helpers
import common.syncer as syncer
import common.user as user
from data import samples_eft


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def make_hist(feature_name: str, vmin: float, vmax: float) -> ROOT.TH1F:
    if feature_name == "nJets":
        lo = int(np.floor(vmin))
        hi = int(np.ceil(vmax))
        nbins = max(1, hi - lo + 1)
        hist = ROOT.TH1F(sanitize(feature_name), feature_name, nbins, lo - 0.5, hi + 0.5)
    else:
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            width = 1.0 if vmin == 0 else 0.05 * abs(vmin)
            vmin -= width
            vmax += width
        hist = ROOT.TH1F(sanitize(feature_name), feature_name, 50, float(vmin), float(vmax))
    hist.GetXaxis().SetTitle(feature_name)
    hist.GetYaxis().SetTitle("Weighted events")
    hist.SetLineWidth(2)
    hist.SetLineColor(ROOT.kAzure + 1)
    return hist


def fill_hist(hist: ROOT.TH1F, values: np.ndarray, weights: np.ndarray) -> None:
    finite = np.isfinite(values) & np.isfinite(weights)
    values = np.ascontiguousarray(values[finite], dtype=np.float64)
    weights = np.ascontiguousarray(weights[finite], dtype=np.float64)
    if len(values) == 0:
        return
    hist.FillN(len(values), values, weights)


def feature_ranges(sample, n_batches: int):
    mins = np.full(len(sample.feature_names), np.inf, dtype=np.float64)
    maxs = np.full(len(sample.feature_names), -np.inf, dtype=np.float64)
    sample.set_n_split(n_batches)
    for shard in tqdm(range(n_batches), desc="range pass", unit="batch"):
        features, = sample.materialize(shard, "f")
        if features.shape[0] == 0:
            continue
        finite = np.isfinite(features)
        for i in range(features.shape[1]):
            vals = features[finite[:, i], i]
            if vals.size == 0:
                continue
            mins[i] = min(mins[i], vals.min())
            maxs[i] = max(maxs[i], vals.max())
    return mins, maxs


def make_empty_hist(feature_name: str) -> ROOT.TH1F:
    hist = ROOT.TH1F(sanitize(feature_name), feature_name, 1, 0.0, 1.0)
    hist.SetBinContent(1, 0.0)
    return hist


def feature_ranges_limited(sample, n_batches: int, max_events: int):
    mins = np.full(len(sample.feature_names), np.inf, dtype=np.float64)
    maxs = np.full(len(sample.feature_names), -np.inf, dtype=np.float64)
    sample.set_n_split(n_batches)
    remaining = max_events
    for shard in tqdm(range(n_batches), desc="range pass", unit="batch"):
        if remaining <= 0:
            break
        features, = sample.materialize(shard, "f", n=remaining)
        remaining -= len(features)
        if features.shape[0] == 0:
            continue
        finite = np.isfinite(features)
        for i in range(features.shape[1]):
            vals = features[finite[:, i], i]
            if vals.size == 0:
                continue
            mins[i] = min(mins[i], vals.min())
            maxs[i] = max(maxs[i], vals.max())
    return mins, maxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="TT01j2l_EFT_2018", help="Sample object from data.samples_eft")
    parser.add_argument("--n-batches", type=int, default=20, help="Number of materialization batches")
    parser.add_argument("--small", action="store_true", help="Debug mode: a few files and at most 50000 events")
    args = parser.parse_args()

    if not hasattr(samples_eft, args.sample):
        raise AttributeError(f"Unknown sample '{args.sample}' in data.samples_eft")

    sample = copy.deepcopy(getattr(samples_eft, args.sample))
    max_events = None
    sample_label = args.sample
    if args.small:
        sample.set_max_files(3)
        max_events = 50000
        sample_label = f"{args.sample}_small"

    plot_dir = os.path.join(user.plot_directory, "EFT", "features", sample_label)
    os.makedirs(plot_dir, exist_ok=True)
    helpers.copyIndexPHP(os.path.join(user.plot_directory, "EFT"))
    helpers.copyIndexPHP(os.path.join(user.plot_directory, "EFT", "features"))
    helpers.copyIndexPHP(plot_dir)

    range_sample = copy.deepcopy(sample)
    if max_events is None:
        mins, maxs = feature_ranges(range_sample, args.n_batches)
    else:
        mins, maxs = feature_ranges_limited(range_sample, args.n_batches, max_events)
    hists = []
    for i_feature, feature_name in enumerate(sample.feature_names):
        if np.isfinite(mins[i_feature]) and np.isfinite(maxs[i_feature]):
            hist = make_hist(feature_name, mins[i_feature], maxs[i_feature])
        else:
            hist = make_empty_hist(feature_name)
        hists.append(hist)

    fill_sample = copy.deepcopy(sample)
    fill_sample.set_n_split(args.n_batches)
    remaining = max_events
    for shard in tqdm(range(args.n_batches), desc="fill pass", unit="batch"):
        if remaining is not None and remaining <= 0:
            break
        features, weights = fill_sample.materialize(shard, "fw", n=remaining)
        if remaining is not None:
            remaining -= len(features)
        if features.shape[0] == 0:
            continue
        for i_feature, hist in enumerate(hists):
            fill_hist(hist, features[:, i_feature], weights)

    for feature_name, hist in zip(sample.feature_names, hists):
        canvas = ROOT.TCanvas(f"c_{sanitize(feature_name)}", feature_name, 800, 700)
        canvas.SetLogy(True)
        hist.Draw("HIST")

        base = os.path.join(plot_dir, sanitize(feature_name))
        root_out = ROOT.TFile.Open(base + ".root", "RECREATE")
        hist.Write("hist")
        canvas.Write("canvas")
        root_out.Close()
        canvas.Print(base + ".png")
        canvas.Print(base + ".pdf")

    syncer.sync()


if __name__ == "__main__":
    main()
