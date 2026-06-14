#!/usr/bin/env python3

import argparse
import glob
import os
from collections import Counter

import ROOT


ROOT.gROOT.SetBatch(True)


def iter_root_files(paths, recursive=False):
    for path in paths:
        if os.path.isdir(path):
            pattern = os.path.join(path, "**", "*.root") if recursive else os.path.join(path, "*.root")
            for filename in sorted(glob.glob(pattern, recursive=recursive)):
                yield filename
        else:
            yield path


def file_weight_length(filename, tree_name, branch_name):
    tf = ROOT.TFile.Open(filename)
    if not tf or tf.IsZombie():
        return "ZOMBIE"
    tree = tf.Get(tree_name)
    if not tree:
        tf.Close()
        return "NO_EVENTS"
    if tree.GetEntries() <= 0:
        tf.Close()
        return 0
    if tree.GetBranch(branch_name):
        tree.GetEntry(0)
        value = int(getattr(tree, branch_name))
        tf.Close()
        return value
    if tree.GetBranch("LHEReweightingWeight"):
        tree.GetEntry(0)
        value = len(getattr(tree, "LHEReweightingWeight"))
        tf.Close()
        return value
    tf.Close()
    return "NO_BRANCH"


def main():
    parser = argparse.ArgumentParser(description="Find and optionally delete ROOT files with bad EFT weight vectors.")
    parser.add_argument("paths", nargs="+", help="ROOT files or directories to scan")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--branch", default="nLHEReweightingWeight")
    parser.add_argument("--expected-length", type=int, default=406)
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument("--delete", action="store_true", help="Delete files that do not match expected-length")
    args = parser.parse_args()

    counts = Counter()
    bad_files = []

    for filename in iter_root_files(args.paths, recursive=args.recursive):
        length = file_weight_length(filename, args.tree, args.branch)
        counts[length] += 1
        if length != args.expected_length:
            bad_files.append((filename, length))

    print(f"checked files: {sum(counts.values())}")
    print("length summary:")
    for key, value in sorted(counts.items(), key=lambda item: str(item[0])):
        print(f"  {key}: {value}")
    print(f"bad files: {len(bad_files)}")

    for filename, length in bad_files:
        print(f"{length}  {filename}")
        if args.delete and os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    main()
