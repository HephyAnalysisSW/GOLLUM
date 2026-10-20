# CMS integration and the frozen paper reference

The development branch merges paper commit `b7a9e9a` and CMS commit
`222f7b2`. `sbi-pdf-paper` stays unchanged and remains the authoritative
historical implementation. This is a real merge, preserving both histories.

## What was missing from the paper branch

The common ancestor is `b820690` (2026-04-11). Before integration there were
141 CMS-side commits and 9 paper-side commits after that ancestor. Comparing
tracked trees gave 265 identical files, 38 changed shared paths, 102 CMS-only
paths, and 113 paper-only paths (including historical configurations).

The merge brings in Autograd/SciPy fitting, combined configurations,
shape-only systematics, newer training/closure tools, CMS configurations,
and plotting/operational updates. Paper Delphes configurations, the paper
command archive, and native PDF-basis additions are retained. Four add/add
conflicts in `user/robert/gen_DY/` retain the paper versions; the CMS variants
remain available on the CMS branch. The PDF postfit conflict combines CMS
multi-config support with the paper's PDF modification path tag.

## Numerical differences that must remain explicit

**The paper unbinned reducer is not identical to the CMS reducer.**
The historical Numba kernel applies its truncated polynomial to every
`-1 < T < 1e-4`; the vectorized CMS implementation applies it only when
`abs(T) < 1e-4`. Installing Autograd exposes this difference because the
Numba path is disabled. This is an approximation difference, not merely
parallel reduction roundoff. For example, at the tested `nu_pu=0.1` point,
the paper value is about 1.03873 and the CMS value about 1.19554.

Use `--paper-numerics` to reproduce the historical reduction with a
differentiable implementation. Without it the development branch uses the
CMS formula. The compiled fallback now uses the same near-zero mask as CMS.
Compatibility-mode filenames include `_paperNumerics`, and result JSON
records both numerical mode and fitting backend.

**The paper Minuit precision override also affects uncertainties.**
The paper sets `Minuit.precision = 0.001`; CMS uses Minuit's default.
For the binned reference, the historical setting produces a covariance
reported by Minuit as forced positive definite. The modern setting reports
an accurate covariance. Their minima agree, but the errors differ
substantially. To reproduce the historical optimizer setting explicitly,
use `--minuit --minuit-precision 0.001`. A complete binned run with this
setting reproduces saved paper errors to within 0.005%. This is not the
development default.

Autograd provides Hessian errors; it does not provide MINOS intervals.
The CLI requires `--minuit` when `--minos` is requested. An ICH prediction
method and the safe log-ratio conversion were made differentiable while
retaining the same polynomial coefficients and diagonal-quadratic factors.
Existing `ICH.predict` behavior is unchanged.

## Re-run the paper checks

Configure `common/user.py` as in the README. Use the existing paper models,
rotations, and unbinned cache. Do not retrain models or use
`--overwrite all` for these checks: CMS also changes training precision and
sample selections, whose regenerated artifacts are not covered by this
comparison.

The binned inputs are those in the README. The unbinned check additionally
needs the matching model artifacts under `models/unbinned_v5D/SR/`, and
`NN2LCache/unbinned_delphes_7_RunII/unbinned_v5D/SR/tt2l_delphes.{h5,json}`
under the configured cache root. The HDF5 has 10,215,508 rows. Raw sample
directory discovery still uses `common/directories.py`; the existing CBE
locations must be accessible. Artifact packaging is deferred.

Run from the repository root after environment activation:

```bash
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python validation/check_paper_likelihood.py --fit binned \
  --rotation /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_binned_delphes_8_RunII_binned_delphes_RunII_v5D.json \
  --output /tmp/paper-binned-check.json
python validation/check_paper_likelihood.py --fit unbinned \
  --rotation /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_unbinned_delphes_7_RunII_unbinned_v5D.json \
  --output /tmp/paper-unbinned-check.json
```

Replace rotation paths with your copies if needed. The checker verifies
configuration/rotation hashes, parameter ordering, eight fixed-point
likelihood values, and every free gradient component against central finite
differences. It refuses a missing unbinned cache rather than rebuilding it.
Reference values are in `validation/paper_reference.json`.
Value tolerances are `atol=1e-8, rtol=1e-12`; gradient tolerances are
`atol=1e-4, rtol=5e-5`. Fixed points include nominal, individual POI/nuisance
displacements, and a reproducible mixed displacement.

An unbinned Autograd fit using the paper reduction is:

```bash
python fit/Likelihood.py configs/unbinned_v5D/unbinned_delphes_7_RunII.yaml \
  --rotate /scratch-cbe/users/robert.schoefbeck/SBIPDF/output/eigen_basis_unbinned_delphes_7_RunII_unbinned_v5D.json \
  --freezePOI 0.01 --paper-numerics --overwrite fit
```

For the historical Minuit settings, append
`--minuit --minuit-precision 0.001`.
Use your own configured output/plot directories. No paper artifact or
historical fit output is overwritten by the integration tests.

## Validation record

The checked environment is Python 3.10.18 with Autograd 1.8.0. Installing
Autograd with `pip install --no-deps autograd==1.8.0` added that package only;
all 227 pre-existing package versions were unchanged.

Tests run on `clip-login-0.cbe.vbc.ac.at` with BLAS/OpenMP/Numba thread
limits set to one for reproducible comparisons. This differs from the
earlier unrestricted-thread likelihood benchmark.

- All eight binned values match the frozen reference; maximum absolute
  difference is `2.38e-10`.
- All eight unbinned values in paper mode match; maximum absolute
  difference is `9.8e-9` at a likelihood value of about `113625`.
- Central differences check all 32 binned and 31 unbinned free parameters.
  Maximum gradient error scaled by `max(1, abs(finite_difference))` is
  `1.02e-5` binned and `7.98e-7` unbinned. The default CMS-mode
  unbinned gradient also passes all 31 central-difference checks, with
  maximum scaled error `7.98e-7`.
- Complete binned Autograd and Minuit/MINOS runs write finite covariance
  matrices, result JSON, and diagnostic plots. Both minima are zero.
  Modern Minuit and Autograd Hessian errors differ by at most 2.7%.
  They must not be described as reproducing the historical forced covariance.
- Full unbinned CMS-mode and paper-mode Autograd fits complete with exact
  Hessians, result JSON, positive-definite covariance, and plots. Both nominal
  Asimov minima are zero. Total wall times including setup, exact Hessian,
  and plots were approximately 468 and 512 seconds, respectively.
- New or modified Python sources pass syntax compilation. Training jobs,
  raw-data cache regeneration, and all new CMS analysis configurations
  are not individually validated by these checks.

Detailed logs and machine-readable measurements are retained on CBE under
`/scratch-cbe/users/robert.schoefbeck/SBIPDF/cms-paper-audit-q97yOs/`.

## Measured derivative speedup

For the 31-parameter unbinned mixed point, a full Autograd gradient takes
13.43 seconds. A central-difference gradient in the frozen paper evaluator,
using 62 likelihood calls, takes 292.06 seconds: a 21.75-fold derivative
speedup. The gradient vectors agree within the same validation tolerance.

Warmed forward calls average 4.67 seconds in the paper evaluator versus
6.52 seconds in the differentiable paper-mode evaluator in this measurement.
The speedup is in derivatives, not an individual forward call. These are
one-thread measurements on a shared login node, with other validation
processes running; they do not establish an end-to-end fit speedup.
Full Hessian construction is still substantial work. Binned Autograd and
Minuit/MINOS wall times are not directly comparable because MINOS computes
additional profile intervals.

Machine-readable measurements are in
[validation/results-2026-10-20.json](../validation/results-2026-10-20.json).
The README recipe was also executed from an independent clone with a private
home directory, synthetic username, and copied artifacts; the Autograd fit,
result metadata, covariance, and diagnostic plots passed.
