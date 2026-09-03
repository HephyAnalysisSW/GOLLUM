#!/usr/bin/env python3
"""
Registry of the parton-level gen ntuple samples used for the spin-observable
study, grouped by collision energy.

All samples use m_t = 172.5 GeV. The nominal one is POWHEG hvq + Pythia8; the
cross sections quoted below are the reference NNLO numbers for the energy
(bb4l is different because it also covers tW, so it is not a pure tt-bar
prediction).

The `ref` field is the arXiv number of the GENERATOR paper -- the CMS samples
themselves have no publication. Every sample is showered with Pythia8
(arXiv:1410.3012), which is therefore not repeated per sample.

`accumulate_BC_sumw2.py --sample <key>` produces `<key>`'s JSON, and
`compare_BC_theory_sim.py --energy 13p6` plots every sample of one energy in
`ORDER` order.
"""
from collections import namedtuple

Sample = namedtuple('Sample',
                    'key energy label ref ntuple_dir json color marker')

NIELS = '/groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples'
ANG = '/groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples'

# reference NNLO tt-bar cross sections per energy [pb]
SIGMA_NNLO = {'13p6': 103.4, '13': 87.98}
# bb4l also contains tW, hence a different total
SIGMA_BB4L = 95.57

# generator references
POWHEG_HVQ = 'arXiv:0707.3088'      # Frixione, Nason, Ridolfi -- hvq
MG5_AMC = 'arXiv:1405.0301'         # MadGraph5_aMC@NLO
FXFX = 'arXiv:1209.6215'            # Frederix, Frixione -- FxFx merging
MLM = 'arXiv:hep-ph/0611129'        # Mangano et al. -- MLM matching
BB4L = 'arXiv:1607.04538'           # Jezo et al. -- bb4l
PYTHIA8 = 'arXiv:1410.3012'         # Pythia 8.2, common to all samples

_SAMPLES = [
    # ---- 13.6 TeV (Run 3 Summer23) ----
    Sample('nom_13p6', '13p6', 'POWHEG (nominal)', POWHEG_HVQ,
           f'{NIELS}/Run3Summer23NanoAODv12__TTto2L2Nu_TuneCP5_13p6TeV_'
           'powheg-pythia8__130X_mcRun3_2023_realistic_v14-v2',
           'BC_nom_13p6.json', '#5790fc', 'o'),
    Sample('fxfx_13p6', '13p6', 'MG5 NLO + FxFx', f'{MG5_AMC}, {FXFX}',
           f'{NIELS}/Run3Summer23NanoAODv12__TTto2L2Nu-2Jets_TuneCP5_13p6TeV_'
           'amcatnloFXFX-pythia8__130X_mcRun3_2023_realistic_v14-v2',
           'BC_fxfx_13p6.json', '#f89c20', 's'),
    Sample('mlm_13p6', '13p6', 'MG5 LO + MLM', f'{MG5_AMC}, {MLM}',
           f'{NIELS}/Run3Summer23NanoAODv12__TTto2L2Nu-3Jets_TuneCP5_13p6TeV_'
           'madgraphMLM-pythia8__130X_mcRun3_2023_realistic_v15-v4',
           'BC_mlm_13p6.json', '#e42536', '^'),
    Sample('noSC_13p6', '13p6', 'POWHEG, spin corr. off', POWHEG_HVQ,
           f'{NIELS}/Run3Summer23NanoAODv12__TTto2L2Nu-noSpinCorr_TuneCP5_'
           '13p6TeV_powheg-pythia8__130X_mcRun3_2023_realistic_v15-v3',
           'BC_noSC_13p6.json', '#964a8b', 'v'),

    # ---- 13 TeV (RunII UL17) ----
    Sample('nom_13', '13', 'POWHEG (nominal)', POWHEG_HVQ,
           f'{ANG}/RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-'
           'powheg-pythia8__106X_mc2017_realistic_v9-v1',
           'BC_nom_13.json', '#5790fc', 'o'),
    Sample('bb4l_13', '13', 'POWHEG bb4l', BB4L,
           f'{NIELS}/RunIISummer20UL17NanoAODv9__BBLLNuNu_TuneCP5_13TeV-'
           'powheg-pythia8__bb4l_v2_106X_mc2017_realistic_v9-v2',
           'BC_bb4l_13.json', '#f89c20', 'D'),
    Sample('noSC_13', '13', 'POWHEG, spin corr. off', POWHEG_HVQ,
           f'{ANG}/RunIISummer20UL17NanoAODv9__TTTo2L2Nu-noSC_TuneCP5_13TeV-'
           'powheg-pythia8__106X_mc2017_realistic_v9-v2',
           'BC_noSC_13.json', '#964a8b', 'v'),
]

SAMPLES = {s.key: s for s in _SAMPLES}

# plotting order per energy: nominal first
ORDER = {
    '13p6': ['nom_13p6', 'fxfx_13p6', 'mlm_13p6', 'noSC_13p6'],
    '13': ['nom_13', 'bb4l_13', 'noSC_13'],
}

ENERGY_LABEL = {'13p6': '13.6 TeV', '13': '13 TeV'}


if __name__ == '__main__':
    import os
    for energy, keys in ORDER.items():
        print(f'--- {ENERGY_LABEL[energy]} '
              f'(sigma_NNLO = {SIGMA_NNLO[energy]} pb) ---')
        for k in keys:
            s = SAMPLES[k]
            print(f'  {k:<11}{s.label:<24}{s.ref:<38}'
                  f'ntuples {"OK" if os.path.isdir(s.ntuple_dir) else "MISSING"}'
                  f'  json {"OK" if os.path.exists(s.json) else "missing"}')
