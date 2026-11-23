# NNLO reweighting: parton-level comparison (step 1)

Compare parton-level 1D (y(tt), m(tt), pT(tt)), 2D and 3D (e.g. pT(tt) x m(tt) x |y(tt)|) distributions in
the full phase space (no cuts) between the
nominal POWHEG hvq sample, POWHEG bb4l, and fixed-order STRIPPER LO/NLO/NNLO (`inputs_rene/`,
arXiv:2412.14910) and MATRIX NNLO (`inputs_niels/`, m(tt) only; MATRIX 3D results are in preparation).

```
samples.py           registries: simulated samples, predictions, variables
theory.py            readers for STRIPPER (1D/2D/3D) / MATRIX .dat files (pb/[X]/...), N-D rebinning
fill_histograms.py   gen ntuples -> histograms/<sample>.npz, in the fixed-order binnings
plot_comparison.py   npz + predictions -> <plot_directory>/nnlo_reweighting/parton_level/{absolute,normalized}/
```

Gen ntuples were produced with `user/robert/gen_top/make_gen_top_ntuple.py` (UL17, 13 TeV).

## Commands (from the repo root, hephy-ml-gpu-2 environment)

```bash
python user/ricardo/nnlo_reweighting/fill_histograms.py --sample nom_13  --jobs 12
python user/ricardo/nnlo_reweighting/fill_histograms.py --sample bb4l_13 --jobs 12
python user/ricardo/nnlo_reweighting/plot_comparison.py                # dsigma/dX [pb/unit]
python user/ricardo/nnlo_reweighting/plot_comparison.py --normalized   # 1/sigma dsigma/dX
python user/ricardo/nnlo_reweighting/plot_comparison.py --variables m_ttbar m_ttbar_x_pt_ttbar  # subset
```

## Variables and multi-dimensional binnings

`samples.AXES` defines the axes (branch, |.| or not, label, unit, log scale) and `samples.VARIABLES` the
1D/2D/3D observables built from them, in the STRIPPER axis order (`pt_ttbar_x_m_ttbar_x_abs_y_ttbar` <->
`pTttbar_x_mttbar_x_absyttbar`). Histograms are dicts with `edges` = tuple of per-axis edges and values of
shape (n_1, ..., n_d), differential in every axis; the 2D/3D STRIPPER files must be regular grids (checked).
Simulation is filled with `np.histogramdd` (per-axis under/overflow). Multi-dimensional figures show one
panel per bin of the outer axes, with the last axis on the x axis. The 2D/3D grids do not cover the full
phase space (e.g. m(tt) < 1500 GeV, |y| < 2.5), so their integrals are 1-3% below the total.
Plots use `mplhep` (CMS style); the simulation markers are offset within each bin (`xoffsets`).

`--max-files N` on `fill_histograms.py` for a quick test; `--pdf-band` on `plot_comparison.py`
adds the STRIPPER NNLO PDF band.

`--truncate VARIABLE=VALUE[,VALUE...]` cuts each axis of the plotted binning at the first edge >= VALUE
(e.g. `m_ttbar=1480` -> 1500 GeV; `none` keeps an axis, e.g. `m_ttbar_x_pt_ttbar=none,150`),
`--rebin VARIABLE=FACTOR[,FACTOR...]` then merges every FACTOR bins per axis (the last bin may merge
fewer). Both take one value per axis, apply to predictions and simulation via `theory.rebin` (cross
section conserved), are repeatable for several variables, and add `_<axis>upto<edge>` /
`_rebin<factors>` to the file name.

## Normalisation

- Simulation: `sigma_sample / BR_DILEPTON / sum(Generator_weight over all events)`, with
  `BR_DILEPTON = 87.98 / 833.9`, so the nominal sample integrates to the inclusive NNLO tt-bar
  cross section. bb4l uses 95.57 pb and is therefore higher (tW, non-resonant).
- STRIPPER: absolute files in `pb/[X]`; `--normalized` uses Rene's normalised files (correlated bands).
- MATRIX: `fb/GeV` converted to pb; normalised by its own integral (band scaled by the central integral).
- Ratio panel: X / STRIPPER NNLO (STRIPPER binning), X / MATRIX NNLO (MATRIX m(tt) binning).

## Rebinning between generators

`theory.rebin(histogram, target_edges)` fills a target bin only if both of its edges are edges of the
source binning (exact union of source bins); all other target bins are NaN and drawn as gaps.
Contents and scale/PDF bands are merged as volume-weighted sums (per axis for 2D/3D) (bands treated as fully correlated
across merged bins), MC errors in quadrature. `plot_comparison.py` rebins every prediction onto the
figure's binning and labels it "(rebinned)". A prediction with no rebinnable bin is not drawn and a
warning is logged; partially rebinnable predictions are drawn with gaps (also with a warning). Current inputs: MATRIX -> STRIPPER m(tt) works only for [1100, 1500] and
[1500, 2500] GeV (the MATRIX edges are 370/400, 475, 625, 825 instead of 380, 470, 620, 820);
STRIPPER -> MATRIX is impossible (bins would have to be split).

## Caveats

- Simulated tops are Pythia last copies (after ISR/FSR recoil); fixed-order tops are partonic.
  The pT(tt) comparison is the most affected.
- Simulated m(tt) extends below the first fixed-order edge (off-shell tops); such events land in
  the underflow, which is stored but not drawn. The fractions are logged by `fill_histograms.py`.
- Gen weights are float32 in the ntuples and are summed in float64 (with float32 accumulation the
  histograms were off by ~0.3% per bin).
- bb4l: ~4% of the weight has no pair of gen tops and is not histogrammed (but is in the normalisation).
- PDFs: STRIPPER uses NNPDF3.1 at the matching order, scales muR = muF = HT/4; MATRIX uses NNPDF3.0 NNLO.
