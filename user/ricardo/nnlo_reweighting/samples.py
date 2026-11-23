"""Registries for the parton-level tt-bar comparison: simulated samples, fixed-order predictions, variables.

The simulated-sample paths are copied from `user/ang/spinCo/samples.py`. That registry is deliberately
not imported: it lives untracked in another user's area and carries spin-study specific fields.
"""
import os
from collections import namedtuple

NNLO_REWEIGHTING_DIR = os.path.dirname(os.path.abspath(__file__))
HISTOGRAM_DIR = os.path.join(NNLO_REWEIGHTING_DIR, 'histograms')

# Reference inclusive NNLO tt-bar cross section at 13 TeV [pb] and the dilepton (incl. tau) cross section
# used for TTTo2L2Nu [pb]; their ratio is the effective dilepton branching ratio.
SIGMA_TTBAR_INCLUSIVE = 833.9
SIGMA_TTBAR_DILEPTON = 87.98
BR_DILEPTON = SIGMA_TTBAR_DILEPTON / SIGMA_TTBAR_INCLUSIVE

SimSample = namedtuple('SimSample', 'key label ntuple_dir sigma color marker')

_NIELS = '/groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples'
_ANG = '/groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples'

SIM_SAMPLES = {s.key: s for s in [
    SimSample('nom_13', 'POWHEG hvq + Py8 (NLO)',
              f'{_ANG}/RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v1',
              SIGMA_TTBAR_DILEPTON, '#5790fc', 'o'),
    SimSample('bb4l_13', 'POWHEG bb4l + Py8 (NLO, incl. tW)',
              f'{_NIELS}/RunIISummer20UL17NanoAODv9__BBLLNuNu_TuneCP5_13TeV-powheg-pythia8__bb4l_v2_106X_mc2017_realistic_v9-v2',
              95.57, '#f89c20', 'D'),
]}

Prediction = namedtuple('Prediction', 'key label generator order color linestyle')

_RENE_DIR = os.path.join(NNLO_REWEIGHTING_DIR, 'inputs_rene')
STRIPPER_ABSOLUTE_DIR = os.path.join(_RENE_DIR, 'data-for-CMS-PAS-TOP-20-006-mt172.5', 'mt172.5')
STRIPPER_NORMALIZED_DIR = os.path.join(_RENE_DIR, 'data-for-CMS-PAS-TOP-20-006-mt172.5-norm', 'mt172.5_norm')
MATRIX_DIR = os.path.join(NNLO_REWEIGHTING_DIR, 'inputs_niels')

PREDICTIONS = {p.key: p for p in [
    Prediction('stripper_lo', 'STRIPPER LO', 'stripper', 'LO', '#a96b59', ':'),
    Prediction('stripper_nlo', 'STRIPPER NLO', 'stripper', 'NLO', '#e42536', '--'),
    Prediction('stripper_nnlo', 'STRIPPER NNLO', 'stripper', 'NNLO', 'black', '-'),
    Prediction('matrix_nnlo', 'MATRIX NNLO', 'matrix', 'NNLO', '#964a8b', '-'),
]}

# Reference prediction for the ratio panel, per binning.
RATIO_REFERENCE = {'stripper': 'stripper_nnlo', 'matrix': 'matrix_nnlo'}

# One axis of a (multi-dimensional) observable. `absolute` histograms |branch|; `log_y` applies when the axis
# is the innermost (plotted) one.
Axis = namedtuple('Axis', 'key branch absolute label unit log_y')

AXES = {a.key: a for a in [
    Axis('m_ttbar', 'parton_ttbar_mass', False, r'$m(t\bar{t})$', 'GeV', True),
    Axis('y_ttbar', 'parton_ttbar_y', False, r'$y(t\bar{t})$', '', False),
    Axis('abs_y_ttbar', 'parton_ttbar_y', True, r'$|y(t\bar{t})|$', '', False),
    Axis('pt_ttbar', 'parton_ttbar_pt', False, r'$p_{T}(t\bar{t})$', 'GeV', True),
    Axis('pt_top', 'parton_Top_pt', False, r'$p_{T}(t)$', 'GeV', True),
    Axis('abs_y_top', 'parton_Top_y', True, r'$|y(t)|$', '', False),
]}

# A 1D, 2D or 3D observable; the key is the axis keys joined by '_x_', in the STRIPPER axis order (the last
# axis runs fastest in the files and is the one drawn on the x axis of each panel).
Variable = namedtuple('Variable', 'key axes stripper_observable matrix_file')


def make_variable(axes: tuple, stripper_observable: str, matrix_file: str | None = None) -> Variable:
    """Variable from AXES keys, with its key built from the axis keys."""
    return Variable('_x_'.join(axes), tuple(AXES[axis] for axis in axes), stripper_observable, matrix_file)


VARIABLES = {v.key: v for v in [
    make_variable(('y_ttbar',), 'yttbar'),
    make_variable(('m_ttbar',), 'mttbar', 'm_ttbar_MATRIX.dat'),
    make_variable(('pt_ttbar',), 'pTttbar'),
    make_variable(('m_ttbar', 'abs_y_ttbar'), 'mttbar_x_absyttbar'),
    make_variable(('m_ttbar', 'pt_ttbar'), 'mttbar_x_pTttbar'),
    make_variable(('abs_y_ttbar', 'pt_ttbar'), 'absyttbar_x_pTttbar'),
    make_variable(('m_ttbar', 'abs_y_top'), 'mttbar_x_absyt'),
    make_variable(('m_ttbar', 'pt_top'), 'mttbar_x_pTt'),
    make_variable(('abs_y_top', 'pt_top'), 'absyt_x_pTt'),
    make_variable(('pt_top', 'pt_ttbar'), 'pTt_x_pTttbar'),
    make_variable(('pt_ttbar', 'm_ttbar', 'abs_y_ttbar'), 'pTttbar_x_mttbar_x_absyttbar'),
]}


def histogram_path(sample_key: str) -> str:
    """Path of the accumulated histogram file of a simulated sample."""
    return os.path.join(HISTOGRAM_DIR, f'{sample_key}.npz')
