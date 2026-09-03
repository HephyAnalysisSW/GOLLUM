# Spin observables: extracting B and C from simulation and comparing with theory

This directory holds everything needed to (1) produce a JSON file with the
extracted spin coefficients `B1+B2` and `C` per (m_tt, y_p) bin, and (2) plot
them against the NLO+ theory predictions of arXiv:2403.04371.

```
samples.py                       registry of all samples (paths, labels, styles)
gen_top/make_gen_top_ntuple.py   NanoAOD  ->  parton-level gen ntuples (ROOT)
accumulate_BC_sumw2.py           ntuples  ->  BC_<sample>.json      <- the JSON
compare_BC_theory_sim.py         JSONs    ->  BC_theory_vs_sim_<energy>TeV.png / .pdf
```

Everything is driven off the sample keys defined in `samples.py`:

| energy | key | sample |
|---|---|---|
| 13.6 TeV | `nom_13p6`  | POWHEG hvq (nominal) |
| 13.6 TeV | `fxfx_13p6` | MG5 NLO + FxFx (2 jets) |
| 13.6 TeV | `mlm_13p6`  | MG5 LO + MLM (3 jets) |
| 13.6 TeV | `noSC_13p6` | POWHEG, spin correlations off |
| 13 TeV   | `nom_13`    | POWHEG hvq (nominal) |
| 13 TeV   | `bb4l_13`   | POWHEG bb4l (also covers tW) |
| 13 TeV   | `noSC_13`   | POWHEG, spin correlations off |

`python samples.py` prints the table with, per sample, whether the ntuple
directory and the accumulated JSON are present.

## 0. Prerequisites

Use the shared group environment (do not modify it -- it is public):

```bash
source /software/2020/software/mamba/22.11.1-4/etc/profile.d/conda.sh && conda activate /groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu-2
```

It provides `numpy`, `uproot`, `tqdm` and `matplotlib`. Step 1 additionally
needs `ROOT` and a valid grid proxy (it reads NanoAOD over xrootd).

## 1. (Only if the ntuples do not exist yet) produce the gen ntuples

`gen_top/make_gen_top_ntuple.py` reads NanoAOD, reconstructs the tt-bar system
at parton level, and writes a flat `Events` tree with the branches the later
steps consume:

- `parton_hasGenSpin`, `parton_Mtt`, `parton_cosTheta_t` (= y_p), `Generator_weight`
- `parton_cosThetaPlus_X`, `parton_cosThetaMinus_X` for `X in {n, r, k, r_star, k_star}`

Run per input file or per dataset:

```bash
./gen_top/make_gen_top_ntuple.py --file root://cms-xrd-global.cern.ch//store/mc/.../XXXX.root
```

```bash
./gen_top/make_gen_top_ntuple.py --sample /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM
```

`gen_top/jobs.sh` is the expanded per-file job list used for the batch
submission. Output goes to `$output_directory/Top-gen-ntuples/<sample_id>/`.

Current existing samples:
```
13.6 TeV samples (NNLO cross section: 103.4 fb)
NLO Magraph sample with FxFx merging (suboptimal compared to the nominal): /groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples/Run3Summer23NanoAODv12__TTto2L2Nu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8__130X_mcRun3_2023_realistic_v14-v2
LO sample with additional jets and MLM matching with parton shower: /groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples/Run3Summer23NanoAODv12__TTto2L2Nu-3Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8__130X_mcRun3_2023_realistic_v15-v4
Simulation with spincorrelations turned off in Powheg: /groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples/Run3Summer23NanoAODv12__TTto2L2Nu-noSpinCorr_TuneCP5_13p6TeV_powheg-pythia8__130X_mcRun3_2023_realistic_v15-v3
nominal or baseline simulation: /groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples/Run3Summer23NanoAODv12__TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8__130X_mcRun3_2023_realistic_v14-v2

13 TeV samples (NNLO cross section: 87.98 fb)
bb4l (alternative simulation, also covers the tW process, so the total cross section is different: 95.57 fb): /groups/hephy/cms/niels.vandenbossche/SBIPDF/output/Top-gen-ntuples/RunIISummer20UL17NanoAODv9__BBLLNuNu_TuneCP5_13TeV-powheg-pythia8__bb4l_v2_106X_mc2017_realistic_v9-v2     
Simulation with spincorrelations turned off  in Powheg: /groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples/RunIISummer20UL17NanoAODv9__TTTo2L2Nu-noSC_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v2
nominal or baseline simulation: /groups/hephy/cms/ang.li/SBIPDF/output/Top-gen-ntuples/RunIISummer20UL17NanoAODv9__TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8__106X_mc2017_realistic_v9-v1

The nominal sample is the standard powheg hvq simulation, interfaced with pythia8 and the top mass in all samples is 172.5 GeV. 
```
so step 1 can normally be skipped.

## 2. Produce the JSON with the B and C coefficients

One JSON per sample, named by the key in `samples.py`:

```bash
./accumulate_BC_sumw2.py --sample nom_13p6 --jobs 12
```

All seven at once:

```bash
for k in nom_13p6 fxfx_13p6 mlm_13p6 noSC_13p6 nom_13 bb4l_13 noSC_13; do ./accumulate_BC_sumw2.py --sample $k --jobs 12; done
```

`--jobs N` spreads the input files over N worker processes (per-file partial
sums are simply added, so the result is independent of N); `--max-files N`
reads only the first N files for a quick test. Any directory not in the
registry can be run with `--input-dir <dir> --output <file.json>`; with no
arguments the script keeps its old behaviour (13 TeV nominal ->
`ttlep_SC_sumw2.json`). Each sample takes O(1 min) with `--jobs 12`.

Binning: `Mtt_cut = [345, 450, 600, 800, inf]`, `y_cut = [-1, -0.5, 0, 0.5, 1]`
-> 4 x 4 = 16 bins, m_tt outer, y_p inner.

Each observable is a weighted mean `X = sum(w a) / sum(w)` of a per-event
quantity `a`, defined directly on the *final* observable (e.g. on
`C(r,k)+C(k,r)` as a whole, not on its two pieces, which are correlated event
by event). The JSON stores the raw accumulators, not the ratios:

```
Mtt_cut, y_cut, observables       binning and the list of 11 observable names
sum_w, sum_w2                     [4][4] arrays
sum_wa[obs], sum_w2a[obs],        [4][4] arrays per observable
sum_w2a2[obs]
```

from which value and SumW2 statistical error follow as

```
X       = sum_wa / sum_w
Var(X)  = (sum_w2a2 - 2 X sum_w2a + X^2 sum_w2) / sum_w^2
```

i.e. the weighted-mean variance that keeps the numerator/denominator
correlation (a naive ratio error would be wrong). The script prints a short
value / error / N_eff table for two bins as a sanity check when it finishes.

The 11 observables stored are

```
C(n,n)  C(r,r)  C(k,k)  C(r,k)+C(k,r)  C(k,k*)  C(r*,k)+C(r,k*)
B1(r)+B2(r)  B1(k)+B2(k)  B1(r*)+B2(r*)  B1(k*)+B2(k*)  B1(n)+B2(n)
```

## 3. Make the comparison plots

One plot per collision energy, overlaying every sample of that energy:

```bash
./compare_BC_theory_sim.py --energy 13p6
```

```bash
./compare_BC_theory_sim.py --energy 13
```

Writes **`BC_theory_vs_sim_13p6TeV.png/.pdf`** and
**`BC_theory_vs_sim_13TeV.png/.pdf`**. `--samples KEY [KEY ...]` overlays an
arbitrary subset instead and `--output` sets the basename. One of
`--energy`/`--samples` is required; the samples are read from the
`BC_<key>.json` files written in step 2, so nothing but `numpy` and
`matplotlib` is needed to remake the plots.

Samples are offset horizontally inside each bin so their error bars stay
separable; colours and markers come from `samples.py`.

The theory numbers are hard-coded in the script from arXiv:2403.04371
(13.6 TeV): Table 9 for sigma_0/sigma_1, Tables 11-21 for the numerators
`N_0/N_1` at mu = m_t/2, m_t, 2 m_t. Each bin is drawn as

- a band = envelope of the three scales,
- a line = central value at mu = m_t,

with the unexpanded ratio `X = (N0+N1)/(sigma0+sigma1)` of eq. (42). For the
observables that vanish at LO only `N1` is tabulated (3 columns instead of 6);
`theory_values()` handles both shapes.

Layout: 4 x 3 panels, one per observable, 12th panel used for the legend.
The 16 x-points per panel are the 16 bins, m_tt bin outer (labelled ticks),
y_p bin inner.

Finally the script prints, for each sample, a numeric summary per observable:
`max|sim - theory|`, median |pull| and max |pull|, with
pull = (sim - theory) / (MC statistical error).

## What the plots show

- The four diagonal/off-diagonal C observables of the nominal, FxFx and MLM
  samples track the theory bands closely; the residual pulls are large only
  because the MC statistical errors are tiny (millions of effective events per
  bin), not because the values differ much -- `max|sim - theory|` stays below
  ~0.06 in absolute value for all of them.
- The spin-correlation-off samples sit at zero for `C(n,n)`, `C(r,r)`,
  `C(k,k)` and `C(r,k)+C(k,r)` in every bin, which is the intended behaviour
  and a good closure test of the whole extraction chain.
- The observables that vanish at LO (`C(k,k*)`, `C(r*,k)+C(r,k*)`, and all the
  `B1+B2`) are small everywhere and consistent between samples within their
  errors, except `B1(r)+B2(r)` and `B1(k)+B2(k)`, where bb4l differs clearly
  from the nominal.

## Caveats

- Errors on the simulation points are **MC statistical only**; the theory band
  is a scale envelope, not a statistical uncertainty. The printed pulls are
  therefore indicative, not a goodness-of-fit.
- The theory tables are 13.6 TeV. On the 13 TeV plot the bands are drawn
  unchanged, so part of any offset there is the energy difference.
- bb4l also contains tW (total cross section 95.57 pb instead of 87.98 pb), so
  it is not a pure tt-bar prediction and is not expected to match the tt-bar
  theory bands exactly.
- The cross sections in `samples.py` are only bookkeeping: every observable is
  a ratio of weighted sums within a sample, so no cross-section normalisation
  enters the plotted values.
