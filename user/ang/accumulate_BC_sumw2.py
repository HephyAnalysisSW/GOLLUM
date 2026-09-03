#!/usr/bin/env python3
"""
Re-accumulate the spin observables per (Mtt, y_p) bin, keeping the
squared-weight sums needed for a SumW2 statistical uncertainty.

Each observable is a weighted mean of a per-event quantity a_i:

    X = sum_i w_i a_i / sum_i w_i

Numerator and denominator come from the same events, so the naive
"error on a ratio" is wrong -- the correlation must be kept. Propagating
it exactly gives the standard weighted-mean variance:

    Var(X) = [ sum w^2 a^2  -  2 X sum w^2 a  +  X^2 sum w^2 ] / (sum w)^2
           =   sum_i w_i^2 (a_i - X)^2 / (sum_i w_i)^2

so three squared-weight accumulators per observable are enough. This is
the SumW2 error in the ROOT sense, generalised from a plain histogram
(a_i = 1) to a weighted mean.

The per-event quantities a_i are defined directly on the FINAL observables
(including the sums such as C(r,k)+C(k,r) and B1+B2), not on their pieces:
the pieces are strongly correlated event by event, so their variances do
not simply add.

Sign convention matches compare_BC_theory_sim.py: the ntuple measures both
leptons along the same top-side axes, while the paper uses b_hat = -a_hat
for the antitop lepton, hence the +9 (not -9) and the (c+ - c-) difference.

By default reads the 13 TeV nominal (SC) sample and writes ttlep_SC_sumw2.json;
use --input-dir/--output to run on any other sample, e.g.

    ./accumulate_BC_sumw2.py --input-dir <ntuple dir> --output BC_nom_13p6.json

The known samples are collected in samples.py, so the usual invocation is

    ./accumulate_BC_sumw2.py --sample nom_13p6 --jobs 8
"""
import argparse
import glob
import json
import sys
import numpy as np
import uproot
from tqdm import tqdm

from samples import SAMPLES

INPUT_DIR = ('/groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples/'
             'RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-'
             'pythia8__106X_mc2017_realistic_v9-v1/')
OUTPUT = 'ttlep_SC_sumw2.json'

Mtt_cut = [2 * 172.5, 450, 600, 800, None]
y_cut = [-1.0, -0.5, 0.0, 0.5, 1.0]

AXES = ['n', 'r', 'k', 'r_star', 'k_star']

BRANCHES = ['parton_hasGenSpin', 'parton_Mtt', 'parton_cosTheta_t',
            'Generator_weight']
for _ax in AXES:
    for _sign in ['Plus', 'Minus']:
        BRANCHES.append(f'parton_cosTheta{_sign}_{_ax}')

OBSERVABLES = ['C(n,n)', 'C(r,r)', 'C(k,k)', 'C(r,k)+C(k,r)',
               'C(k,k*)', 'C(r*,k)+C(r,k*)',
               'B1(r)+B2(r)', 'B1(k)+B2(k)',
               'B1(r*)+B2(r*)', 'B1(k*)+B2(k*)', 'B1(n)+B2(n)']


def per_event_quantities(arrays):
    """Return {observable: a_i array} for the events in `arrays`."""
    def p(ax):
        return arrays[f'parton_cosThetaPlus_{ax}']

    def m(ax):
        return arrays[f'parton_cosThetaMinus_{ax}']

    return {
        'C(n,n)':            9.0 * p('n') * m('n'),
        'C(r,r)':            9.0 * p('r') * m('r'),
        'C(k,k)':            9.0 * p('k') * m('k'),
        'C(r,k)+C(k,r)':     9.0 * (p('r') * m('k') + p('k') * m('r')),
        'C(k,k*)':           9.0 * p('k') * m('k_star'),
        'C(r*,k)+C(r,k*)':   9.0 * (p('r_star') * m('k') + p('r') * m('k_star')),
        'B1(r)+B2(r)':       3.0 * (p('r') - m('r')),
        'B1(k)+B2(k)':       3.0 * (p('k') - m('k')),
        'B1(r*)+B2(r*)':     3.0 * (p('r_star') - m('r_star')),
        'B1(k*)+B2(k*)':     3.0 * (p('k_star') - m('k_star')),
        'B1(n)+B2(n)':       3.0 * (p('n') - m('n')),
    }


NM, NY = len(Mtt_cut) - 1, len(y_cut) - 1
NOBS = len(OBSERVABLES)


def accumulate_file(path):
    """Accumulators for one ROOT file.

    Returned as plain arrays rather than dicts so the result is cheap to ship
    back from a worker process: (sum_w, sum_w2, sum_wa, sum_w2a, sum_w2a2)
    with shapes (NM, NY) for the first two and (NOBS, NM, NY) for the rest.
    """
    sum_w = np.zeros((NM, NY))
    sum_w2 = np.zeros((NM, NY))
    sum_wa = np.zeros((NOBS, NM, NY))
    sum_w2a = np.zeros((NOBS, NM, NY))
    sum_w2a2 = np.zeros((NOBS, NM, NY))

    arrays = uproot.open(path)['Events'].arrays(BRANCHES, library='np')
    Mtt = arrays['parton_Mtt']
    yp = arrays['parton_cosTheta_t']
    w = arrays['Generator_weight']
    valid = arrays['parton_hasGenSpin'] == 1
    quantities = per_event_quantities(arrays)

    for ix in range(NM):
        xmin = Mtt_cut[ix]
        xmax = Mtt_cut[ix + 1] if Mtt_cut[ix + 1] is not None else np.inf
        for iy in range(NY):
            mask = (valid & (Mtt > xmin) & (Mtt < xmax)
                    & (yp > y_cut[iy]) & (yp < y_cut[iy + 1]))
            ww = w[mask]
            ww2 = ww * ww
            sum_w[ix, iy] = ww.sum()
            sum_w2[ix, iy] = ww2.sum()
            for io, obs in enumerate(OBSERVABLES):
                a = quantities[obs][mask]
                sum_wa[io, ix, iy] = (ww * a).sum()
                sum_w2a[io, ix, iy] = (ww2 * a).sum()
                sum_w2a2[io, ix, iy] = (ww2 * a * a).sum()

    return sum_w, sum_w2, sum_wa, sum_w2a, sum_w2a2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--sample', choices=sorted(SAMPLES),
                   help='named sample from samples.py (sets input dir/output)')
    p.add_argument('--input-dir', help='directory of gen ntuples to read')
    p.add_argument('--output', help='output JSON file')
    p.add_argument('--jobs', type=int, default=1,
                   help='number of worker processes over input files')
    p.add_argument('--max-files', type=int, default=None,
                   help='only read the first N files (for a quick test)')
    args = p.parse_args()

    if args.sample:
        s = SAMPLES[args.sample]
        args.input_dir = args.input_dir or s.ntuple_dir
        args.output = args.output or s.json
    args.input_dir = args.input_dir or INPUT_DIR
    args.output = args.output or OUTPUT
    return args


def main():
    args = parse_args()
    files = sorted(glob.glob(f'{args.input_dir}/*.root'))
    if not files:
        sys.exit(f'no ROOT files under {args.input_dir}')
    if args.max_files:
        files = files[:args.max_files]
    print(f'Found {len(files)} files in {args.input_dir}')

    totals = [np.zeros((NM, NY)), np.zeros((NM, NY))] + \
             [np.zeros((NOBS, NM, NY)) for _ in range(3)]

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            it = pool.imap_unordered(accumulate_file, files)
            for part in tqdm(it, total=len(files)):
                for tot, p in zip(totals, part):
                    tot += p
    else:
        for path in tqdm(files):
            for tot, p in zip(totals, accumulate_file(path)):
                tot += p

    sum_w, sum_w2 = totals[0], totals[1]
    sum_wa = {o: totals[2][io] for io, o in enumerate(OBSERVABLES)}
    sum_w2a = {o: totals[3][io] for io, o in enumerate(OBSERVABLES)}
    sum_w2a2 = {o: totals[4][io] for io, o in enumerate(OBSERVABLES)}

    res = {
        'input_dir': args.input_dir,
        'n_files': len(files),
        'Mtt_cut': [c if c is not None else None for c in Mtt_cut],
        'y_cut': y_cut,
        'observables': OBSERVABLES,
        'sum_w': sum_w.tolist(),
        'sum_w2': sum_w2.tolist(),
        'sum_wa': {o: sum_wa[o].tolist() for o in OBSERVABLES},
        'sum_w2a': {o: sum_w2a[o].tolist() for o in OBSERVABLES},
        'sum_w2a2': {o: sum_w2a2[o].tolist() for o in OBSERVABLES},
    }
    with open(args.output, 'w') as f:
        json.dump(res, f)
    print(f'wrote {args.output}')

    # quick look
    print(f"\n{'observable':<18}{'bin':>5}{'value':>10}{'stat err':>11}{'N_eff':>12}")
    for obs in OBSERVABLES:
        X = sum_wa[obs] / sum_w
        var = (sum_w2a2[obs] - 2 * X * sum_w2a[obs] + X**2 * sum_w2) / sum_w**2
        err = np.sqrt(var)
        neff = sum_w**2 / sum_w2
        for ix, iy in [(0, 0), (3, 1)]:
            print(f'{obs:<18}{ix}{iy:<4}{X[ix, iy]:>10.4f}'
                  f'{err[ix, iy]:>11.5f}{neff[ix, iy]:>12.3e}')


if __name__ == '__main__':
    main()
