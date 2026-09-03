#!/usr/bin/env python3
"""
Compare spin observables B and C from arXiv:2403.04371 (Tables 9, 11-21)
with the values extracted from simulation (BC_<sample>.json).

Theory: X = (N0 + N1) / (sigma0 + sigma1) per bin (the unexpanded ratio,
eq. (42) of the paper; the expanded alternative is eq. (43)), evaluated for
mu = mt/2, mt, 2mt. The mu = mt value is the central prediction; the
envelope of the three scales is drawn as a band.

Simulation: the BC_<sample>.json files written by accumulate_BC_sumw2.py store,
per (Mtt, y_p) bin, the weighted sums sum_w, sum_w2, sum_wa, sum_w2a, sum_w2a2
of the per-event quantities, from which the value and its SumW2 statistical
error are reconstructed (see sim_values_sumw2 below). One sample per file;
which ones are plotted comes from samples.py via --energy or --samples.

SIGN CONVENTION.  The ntuple projects BOTH leptons on the same (top-side)
axis a_hat -- gen_top/make_gen_top_ntuple.py lines 222-228 -- whereas the
paper's Table 1 defines the antitop axis as b_hat = -a_hat for every axis
(n, r, k, and likewise r*, k*).  With the paper's eq. (30),
cos(theta_+) = lhat_+ . a_hat and cos(theta_-) = lhat_- . b_hat, this means

    cosThetaPlus_X  = +cos(theta_+)
    cosThetaMinus_X = -cos(theta_-)

Feeding that into the paper's eq. (29),
    1/sigma d2sigma/dcos(theta_+)dcos(theta_-)
        = 1/4 (1 + B1 cos(theta_+) + B2 cos(theta_-) - C cos(theta_+)cos(theta_-))
gives B1 = 3<cos(theta_+)>, B2 = 3<cos(theta_-)>, C = -9<cos(theta_+)cos(theta_-)>,
and hence

    C_paper(a,b)        = +9 <cosThetaPlus_a cosThetaMinus_b>
    B1_paper + B2_paper = 3 <cosThetaPlus_axis - cosThetaMinus_axis>

verify_axis_convention.py checks the axis identification to machine
precision by re-running the ntuple's rotation logic against the paper's
eq. (35) + Table 1 on random events.  Consistency checks on the SC sample:
C(n,n) ~ +0.43 near threshold, C(k,k) < 0 at high Mtt in the central
region, and B1 ~ -B2 bin by bin as CP invariance requires (paper eq. (34)).
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from samples import SAMPLES, ORDER, ENERGY_LABEL

OBSERVABLES = ['C(n,n)', 'C(r,r)', 'C(k,k)', 'C(r,k)+C(k,r)',
               'C(k,k*)', 'C(r*,k)+C(r,k*)',
               'B1(r)+B2(r)', 'B1(k)+B2(k)',
               'B1(r*)+B2(r*)', 'B1(k*)+B2(k*)', 'B1(n)+B2(n)']

# ----------------------------------------------------------------------
# Theory tables from arXiv:2403.04371 (13.6 TeV), bins:
# Mtt: [2mt,450], [450,600], [600,800], [>800] GeV
# y_p = cosTheta_t*: [-1,-0.5], [-0.5,0], [0,0.5], [0.5,1]
# Rows below: 16 bins in order (Mtt bin outer, y_p bin inner).
# 6 columns: N0(mt/2), N0(mt), N0(2mt), N1(mt/2), N1(mt), N1(2mt)  [pb]
# For observables that vanish at LO only N1 (3 columns) is given.
# ----------------------------------------------------------------------

# Table 9: ttbar cross section, sigma0 and sigma1
SIGMA = np.array([
    [74.596, 59.133, 47.579, 30.112, 33.509, 33.911],
    [58.987, 46.915, 37.852, 22.347, 25.519, 26.202],
    [59.071, 46.971, 37.876, 18.071, 22.316, 23.847],
    [74.355, 58.987, 47.488, 29.831, 33.313, 33.780],
    [81.987, 63.546, 50.087, 28.030, 33.922, 35.307],
    [42.980, 33.454, 26.488,  2.410,  9.103, 12.139],
    [42.855, 33.371, 26.433,  2.273,  8.823, 12.025],
    [82.015, 63.521, 50.054, 28.667, 34.385, 35.554],
    [42.366, 31.993, 24.665, 12.792, 16.603, 17.532],
    [14.643, 11.144,  8.667, -2.538,  0.910,  2.622],
    [14.653, 11.151,  8.662, -2.118,  1.234,  2.858],
    [42.413, 32.029, 24.698, 12.066, 16.116, 17.234],
    [23.747, 17.292, 12.915,  4.418,  7.836,  8.750],
    [ 5.500,  4.072,  3.087, -2.769, -0.768,  0.241],
    [ 5.499,  4.069,  3.084, -2.832, -0.809,  0.221],
    [23.753, 17.301, 12.924,  4.526,  7.905,  8.762],
])

NUMERATORS = {}

# Table 11: N_nn  (C(n,n))
NUMERATORS['C(n,n)'] = np.array([
    [34.099, 26.853, 21.453, 13.612, 15.427, 15.728],
    [25.677, 20.277, 16.243,  8.558, 10.361, 10.918],
    [25.708, 20.301, 16.262,  6.724,  9.037,  9.928],
    [34.004, 26.777, 21.395, 13.698, 15.473, 15.740],
    [19.913, 15.398, 12.107,  8.343,  9.303,  9.339],
    [13.877, 10.798,  8.549,  0.230,  2.525,  3.650],
    [13.833, 10.765,  8.527,  0.208,  2.491,  3.588],
    [19.913, 15.396, 12.104,  8.491,  9.408,  9.432],
    [ 6.998,  5.291,  4.084,  2.371,  2.927,  3.013],
    [ 5.954,  4.548,  3.544, -1.334,  0.162,  0.916],
    [ 5.959,  4.550,  3.544, -1.137,  0.295,  1.019],
    [ 7.005,  5.296,  4.089,  2.191,  2.807,  2.926],
    [ 3.212,  2.352,  1.764,  0.168,  0.758,  0.973],
    [ 3.104,  2.300,  1.748, -1.784, -0.588,  0.023],
    [ 3.105,  2.302,  1.749, -1.817, -0.602,  0.017],
    [ 3.218,  2.355,  1.767,  0.203,  0.768,  0.982],
])

# Table 12: N_rr  (C(r,r))
NUMERATORS['C(r,r)'] = np.array([
    [ 23.907,  18.608,  14.684, 11.339, 12.228, 12.187],
    [  0.491,  -0.042,  -0.382,  3.378,  2.773,  2.259],
    [  0.466,  -0.062,  -0.400,  3.059,  2.502,  2.043],
    [ 23.824,  18.538,  14.631, 11.350, 12.261, 12.215],
    [  4.665,   3.468,   2.618,  6.238,  5.301,  4.514],
    [-16.485, -13.003, -10.423,  2.112, -1.292, -3.070],
    [-16.458, -12.972, -10.407,  2.108, -1.266, -3.054],
    [  4.666,   3.467,   2.620,  6.308,  5.334,  4.525],
    [ -2.359,  -1.824,  -1.438,  1.678,  0.739,  0.216],
    [ -8.431,  -6.459,  -5.051,  2.511,  0.211, -0.990],
    [ -8.440,  -6.463,  -5.054,  2.287,  0.046, -1.120],
    [ -2.356,  -1.823,  -1.436,  1.801,  0.823,  0.283],
    [ -2.457,  -1.809,  -1.364,  0.997,  0.171, -0.227],
    [ -3.866,  -2.870,  -2.183,  2.345,  0.810,  0.016],
    [ -3.863,  -2.870,  -2.182,  2.372,  0.838,  0.043],
    [ -2.464,  -1.811,  -1.366,  0.990,  0.154, -0.237],
])

# Table 13: N_kk  (C(k,k))
NUMERATORS['C(k,k)'] = np.array([
    [40.027, 31.090, 24.493, 11.974, 15.708, 17.026],
    [29.272, 22.926, 18.219,  8.820, 11.339, 12.214],
    [29.295, 22.939, 18.229,  6.653,  9.746, 10.986],
    [39.892, 30.981, 24.399, 11.936, 15.683, 17.043],
    [29.247, 22.194, 17.135,  9.381, 12.297, 13.144],
    [ 6.892,  5.178,  3.945, -0.336,  1.084,  1.759],
    [ 6.857,  5.152,  3.927, -0.499,  0.958,  1.667],
    [29.224, 22.182, 17.133,  9.631, 12.455, 13.243],
    [ 6.546,  4.768,  3.544,  3.885,  4.135,  3.986],
    [-2.559, -2.008, -1.606,  0.231, -0.254, -0.501],
    [-2.561, -2.009, -1.606,  0.199, -0.277, -0.519],
    [ 6.575,  4.787,  3.554,  3.972,  4.198,  4.043],
    [-2.166, -1.628, -1.259,  3.028,  1.677,  0.934],
    [-2.861, -2.130, -1.625,  1.458,  0.432, -0.097],
    [-2.861, -2.131, -1.627,  1.489,  0.447, -0.084],
    [-2.173, -1.630, -1.257,  2.985,  1.647,  0.899],
])

# Table 14: N_rk  (C(r,k)+C(k,r))
NUMERATORS['C(r,k)+C(k,r)'] = np.array([
    [-17.107, -13.884, -11.423, -6.623, -7.214, -7.258],
    [-10.935,  -8.831,  -7.231, -3.927, -4.506, -4.653],
    [-10.953,  -8.849,  -7.245, -2.802, -3.698, -4.062],
    [-17.092, -13.866, -11.410, -6.639, -7.229, -7.277],
    [-20.499, -16.052, -12.784, -5.027, -6.985, -7.670],
    [-10.670,  -8.364,  -6.671,  0.199, -1.670, -2.602],
    [-10.650,  -8.346,  -6.656,  0.193, -1.661, -2.580],
    [-20.505, -16.060, -12.788, -5.131, -7.058, -7.743],
    [-10.152,  -7.723,  -5.994, -0.561, -2.286, -3.009],
    [ -3.923,  -3.000,  -2.343,  1.136,  0.065, -0.486],
    [ -3.927,  -3.004,  -2.343,  1.046, -0.007, -0.541],
    [-10.148,  -7.719,  -5.996, -0.285, -2.086, -2.856],
    [ -4.592,  -3.373,  -2.539,  1.376, -0.038, -0.696],
    [ -1.329,  -0.988,  -0.752,  0.868,  0.319,  0.034],
    [ -1.328,  -0.988,  -0.752,  0.876,  0.323,  0.035],
    [ -4.600,  -3.377,  -2.542,  1.357, -0.052, -0.706],
])

# Tables 15-21: observables that vanish at LO -> N1 only (3 columns)
NUMERATORS['C(k,k*)'] = np.array([
    [ 0.001,  0.040,  0.070], [ 0.279,  0.221,  0.183],
    [-0.304, -0.242, -0.192], [-0.030, -0.019,  0.003],
    [ 0.366,  0.314,  0.278], [ 0.008,  0.016,  0.009],
    [ 0.195,  0.153,  0.120], [ 0.613,  0.520,  0.453],
    [-0.024,  0.000,  0.026], [-0.000,  0.001,  0.002],
    [ 0.005,  0.005,  0.005], [-0.037, -0.016,  0.000],
    [ 0.029,  0.049,  0.041], [-0.004, -0.002, -0.001],
    [ 0.001,  0.004,  0.003], [ 0.069,  0.073,  0.071],
])

NUMERATORS['C(r*,k)+C(r,k*)'] = np.array([
    [ 0.126,  0.121,  0.115], [-0.161, -0.137, -0.108],
    [ 0.042,  0.037,  0.031], [ 0.041,  0.052,  0.066],
    [ 0.138,  0.134,  0.126], [ 0.059,  0.046,  0.035],
    [-0.217, -0.168, -0.141], [-0.127, -0.065, -0.031],
    [-0.003,  0.012,  0.013], [ 0.006,  0.006,  0.002],
    [ 0.008,  0.006,  0.004], [ 0.042,  0.027,  0.029],
    [ 0.001, -0.001,  0.009], [-0.002, -0.002, -0.001],
    [ 0.003,  0.002,  0.000], [-0.000,  0.004,  0.005],
])

NUMERATORS['B1(r)+B2(r)'] = np.array([
    [0.021, 0.201, 0.357], [0.043, 0.096, 0.143],
    [0.043, 0.096, 0.143], [0.022, 0.202, 0.358],
    [0.121, 0.240, 0.347], [0.110, 0.113, 0.119],
    [0.111, 0.114, 0.120], [0.120, 0.239, 0.346],
    [0.096, 0.116, 0.139], [0.058, 0.049, 0.044],
    [0.058, 0.049, 0.044], [0.096, 0.117, 0.139],
    [0.061, 0.056, 0.056], [0.025, 0.019, 0.015],
    [0.025, 0.019, 0.015], [0.061, 0.056, 0.056],
])

NUMERATORS['B1(k)+B2(k)'] = np.array([
    [0.058, 0.285, 0.482], [0.047, 0.007, -0.025],
    [0.046, 0.007, -0.026], [0.061, 0.287,  0.485],
    [0.428, 0.630, 0.817], [0.181, 0.114,  0.063],
    [0.181, 0.114, 0.063], [0.426, 0.628,  0.815],
    [0.484, 0.531, 0.587], [0.150, 0.104,  0.071],
    [0.150, 0.104, 0.071], [0.485, 0.531,  0.588],
    [0.531, 0.494, 0.485], [0.121, 0.085,  0.060],
    [0.121, 0.085, 0.060], [0.532, 0.494,  0.485],
])

NUMERATORS['B1(r*)+B2(r*)'] = np.array([
    [-0.036, 0.013, 0.054], [-0.037, 0.013, 0.053],
    [-0.036, 0.013, 0.054], [-0.035, 0.014, 0.055],
    [-0.030, 0.002, 0.028], [-0.009, 0.007, 0.020],
    [-0.009, 0.007, 0.020], [-0.031, 0.001, 0.027],
    [-0.013, -0.003, 0.005], [0.007, 0.008, 0.009],
    [ 0.006, 0.008, 0.009], [-0.013, -0.003, 0.005],
    [-0.006, -0.003, 0.000], [0.009, 0.008, 0.006],
    [ 0.009, 0.008, 0.007], [-0.006, -0.003, 0.000],
])

NUMERATORS['B1(k*)+B2(k*)'] = np.array([
    [-0.040, 0.028, 0.083], [-0.008, 0.005, 0.015],
    [-0.008, 0.005, 0.015], [-0.038, 0.030, 0.086],
    [-0.023, 0.037, 0.086], [ 0.001, 0.005, 0.008],
    [ 0.001, 0.005, 0.008], [-0.025, 0.035, 0.084],
    [-0.006, 0.023, 0.047], [ 0.006, 0.005, 0.005],
    [ 0.006, 0.005, 0.005], [-0.005, 0.024, 0.048],
    [ 0.005, 0.021, 0.035], [ 0.009, 0.007, 0.005],
    [ 0.009, 0.007, 0.005], [ 0.005, 0.021, 0.035],
])

NUMERATORS['B1(n)+B2(n)'] = np.array([
    [0.260, 0.187, 0.138], [0.184, 0.132, 0.097],
    [0.184, 0.132, 0.097], [0.260, 0.187, 0.138],
    [0.761, 0.533, 0.384], [0.457, 0.320, 0.230],
    [0.457, 0.320, 0.230], [0.761, 0.533, 0.384],
    [0.546, 0.373, 0.263], [0.261, 0.179, 0.126],
    [0.261, 0.179, 0.126], [0.546, 0.373, 0.263],
    [0.310, 0.204, 0.140], [0.113, 0.075, 0.052],
    [0.113, 0.075, 0.052], [0.310, 0.204, 0.140],
])


def theory_values(name):
    """Return (central, lo, hi): unexpanded ratio N/(sigma) per bin,
    central at mu=mt, band = envelope of mu = mt/2, mt, 2mt."""
    N = NUMERATORS[name]
    if N.shape[1] == 6:                      # N0 and N1 given
        num = N[:, 0:3] + N[:, 3:6]
    else:                                    # N0 = 0, only N1 given
        num = N
    den = SIGMA[:, 0:3] + SIGMA[:, 3:6]
    X = num / den                            # columns: mt/2, mt, 2mt
    return X[:, 1], X.min(axis=1), X.max(axis=1)


# ----------------------------------------------------------------------
# Simulation values from the saved JSON (see module docstring for signs)
# ----------------------------------------------------------------------
def sim_values_sumw2(jsonfile):
    """Values and SumW2 statistical errors from accumulate_BC_sumw2.py.

    X = sum(w a)/sum(w) is a weighted mean, so numerator and denominator are
    fully correlated (same events). Keeping that correlation gives

        Var(X) = [S_w2a2 - 2 X S_w2a + X^2 S_w2] / (sum w)^2

    which is the SumW2 error generalised from a plain histogram to a mean.
    """
    with open(jsonfile) as f:
        d = json.load(f)
    sw = np.asarray(d['sum_w']).ravel()
    sw2 = np.asarray(d['sum_w2']).ravel()
    vals, errs = {}, {}
    for obs in d['observables']:
        swa = np.asarray(d['sum_wa'][obs]).ravel()
        sw2a = np.asarray(d['sum_w2a'][obs]).ravel()
        sw2a2 = np.asarray(d['sum_w2a2'][obs]).ravel()
        X = swa / sw
        vals[obs] = X
        errs[obs] = np.sqrt((sw2a2 - 2 * X * sw2a + X**2 * sw2) / sw**2)
    return vals, errs


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def draw_theory(ax, name, band_col, line_col):
    cen, lo, hi = theory_values(name)
    for i in range(16):
        ax.fill_between([i - 0.42, i + 0.42], lo[i], hi[i],
                        color=band_col, lw=0, zorder=1)
        ax.hlines(cen[i], i - 0.42, i + 0.42, color=line_col,
                  lw=1.6, zorder=2)
    return cen


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--energy', choices=sorted(ORDER),
                      help='overlay every sample of this energy (samples.py)')
    mode.add_argument('--samples', nargs='+', metavar='KEY',
                      help='explicit sample keys to overlay instead')
    p.add_argument('--output', help='output basename (default from --energy)')
    return p.parse_args()


def main():
    args = parse_args()

    keys = args.samples or ORDER[args.energy]
    entries = []
    for k in keys:
        s = SAMPLES[k]
        if not os.path.exists(s.json):
            raise SystemExit(f'{s.json} missing -- run '
                             f'./accumulate_BC_sumw2.py --sample {k}')
        vals, errs = sim_values_sumw2(s.json)
        entries.append((f'{s.label} [{s.ref}]',
                        s.color, s.marker, vals, errs))
        print(f'{k:<11} {s.json}')
    energy = args.energy or SAMPLES[keys[0]].energy
    out = args.output or f'BC_theory_vs_sim_{energy}TeV'
    header = f'{ENERGY_LABEL[energy]} simulations'

    mtt_labels = [r'$345-450$', r'$450-600$', r'$600-800$', r'$>800$']
    # theory drawn in neutral grey so it never collides with a sample colour
    band_col, line_col = '#dcdce0', '#9c9ca1'

    # spread the samples inside each bin so the error bars stay readable
    n = len(entries)
    offsets = (np.arange(n) - (n - 1) / 2) * (0.5 / max(n, 2))

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 14), sharex=True)
    axes = axes.ravel()
    x = np.arange(16)

    for ax, name in zip(axes, OBSERVABLES):
        draw_theory(ax, name, band_col, line_col)
        for (label, col, marker, vals, errs), dx in zip(entries, offsets):
            ax.errorbar(x + dx, vals[name],
                        yerr=errs[name],
                        fmt=marker, ms=4.0, color=col, ecolor=col,
                        elinewidth=1.1, capsize=2.0, zorder=3)

        for xsep in (3.5, 7.5, 11.5):
            ax.axvline(xsep, color='0.75', lw=0.8)
        ax.axhline(0, color='0.85', lw=0.8, zorder=0)
        ax.set_title(name, fontsize=12)
        ax.set_xlim(-0.7, 15.7)
        ax.tick_params(labelsize=9)
        ax.set_xticks([1.5, 5.5, 9.5, 13.5])
        ax.set_xticklabels(mtt_labels, fontsize=9)

    for ax in axes[8:11]:
        ax.set_xlabel(r'$m_{t\bar{t}}$ bin [GeV]', fontsize=10)

    # legend in the unused 12th panel
    axl = axes[11]
    axl.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=band_col),
               plt.Line2D([], [], color=line_col, lw=1.6)]
    labels = ['theory scale envelope ($\\mu = m_t/2,\\ m_t,\\ 2m_t$)',
              'theory, $\\mu = m_t$ (arXiv:2403.04371, 13.6 TeV)']
    for label, col, marker, _, errs in entries:
        handles.append(plt.Line2D([], [], color=col, marker=marker, ls='',
                                  ms=4.5))
        labels.append(label)
    axl.legend(handles, labels, loc='upper left', fontsize=9.5, frameon=False)
    axl.text(0.03, 0.92 - 0.075 * len(handles),
             'Within each $m_{t\\bar{t}}$ bin, the 4 points are the\n'
             '$y_p=\\cos\\theta_t^*$ bins: $(-1,-0.5)$, $(-0.5,0)$,\n'
             '$(0,0.5)$, $(0.5,1)$; samples are offset horizontally.\n\n'
             'Theory: unexpanded ratio $(N_0+N_1)/(\\sigma_0+\\sigma_1)$\n'
             'from Tables 11$-$21 over Table 9.\n\n'
             'Error bars are MC statistical only\n'
             '($\\sqrt{\\Sigma w^2}$-type, correlations between\n'
             'numerator and denominator kept).',
             transform=axl.transAxes, fontsize=9.5, va='top')

    fig.suptitle(r'$t\bar{t}$ spin observables: NLOW theory vs ' + header,
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=150)
    print(f'saved {out}.png / .pdf')

    # numeric summary: deviation from theory in units of the MC stat. error
    for label, col, marker, vals, errs in entries:
        print(f'\n{label}')
        print(f"{'observable':<18}{'max|sim-thy|':>13}{'median pull':>13}"
              f"{'max |pull|':>12}")
        for name in OBSERVABLES:
            cen, lo, hi = theory_values(name)
            diff = vals[name] - cen
            pull = diff / errs[name]
            print(f'{name:<18}{np.abs(diff).max():>13.4f}'
                  f'{np.median(np.abs(pull)):>13.1f}'
                  f'{np.abs(pull).max():>12.1f}')


if __name__ == '__main__':
    main()
