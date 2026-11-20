# NNLO reweighting: parton-level comparison (step 1)

Compare parton-level y(tt), m(tt) and pT(tt) in the full phase space (no cuts) between the
nominal POWHEG hvq sample, POWHEG bb4l, and fixed-order STRIPPER LO/NLO/NNLO (`inputs_rene/`,
arXiv:2412.14910) and MATRIX NNLO (`inputs_niels/`, m(tt) only).

```
samples.py           registries: simulated samples, predictions, variables
theory.py            readers for STRIPPER / MATRIX .dat files (all converted to pb/[X])
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
```

`--max-files N` on `fill_histograms.py` for a quick test; `--pdf-band` on `plot_comparison.py`
adds the STRIPPER NNLO PDF band.

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
Contents and scale/PDF bands are merged as width-weighted sums (bands treated as fully correlated
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
- bb4l: ~4% of the weight has no pair of gen tops and is not histogrammed (but is in the normalisation).
- PDFs: STRIPPER uses NNPDF3.1 at the matching order, scales muR = muF = HT/4; MATRIX uses NNPDF3.0 NNLO.
