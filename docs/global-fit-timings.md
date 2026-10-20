# Unbinned likelihood: starting figures for a global-fit discussion

These measure the existing nominal-Asimov “7-1” PDF likelihood, with six
rotated PDF directions and 25 nuisance parameters. They do not benchmark
new EFT Wilson-coefficient dependence or PDF–EFT cross terms.

## Persistent evaluator

The README now provides a CBE recipe and `examples.unbinned_term.UnbinnedTerm`.
It loads existing models/cache once; subsequent calls use resident arrays.
`value_and_gradient` returns the value and all 31 derivatives together,
avoiding a separate forward call. It returns **-2 log L**, rather than log L.

A fresh-clone/private-home run on 2026-10-20 gave:

| Quantity | Measurement |
| --- | --- |
| Setup, once | 61.1 s |
| Warm value only | 4.80 s mean; 4.58–5.06 s |
| Warm joint value + 31 derivatives | 11.46 s mean; 11.02–11.70 s |
| Peak resident memory | 7.60 GiB |

Three repetitions were measured at the validated, nonzero mixed parameter
point. Hardware: `clip-login-1.cbe.vbc.ac.at`, KVM VM exposing 76 logical
Xeon Gold 6138 CPUs at nominal 2 GHz and 169 GiB RAM. BLAS/OpenMP/Numba
thread limits were **one**, and no GPU was used. The login node was shared;
these are starting measurements, not dedicated-machine throughput guarantees.

The measured term traverses 10,215,508 simulation rows. Its weighted nominal
yield is about 526,180 events. The log term and simulation normalization use
the same rows and are fused into one final reduction, not two independent
simulation samples. The response matrices have 35 PDF and 27 systematic
coefficient columns: about 633 million coefficient-product contributions
per call. Surrogate inference/training is already outside the timed call.

## Derivative gain and call budgets

An earlier matched one-thread comparison on `clip-login-0` measured the
31-component Autograd gradient at 13.43 s, versus 292.06 s for a central
difference gradient in the frozen paper evaluator (62 forward calls).
That is about **22 times faster for the gradient**, not for a whole fit.
A standalone differentiable forward evaluation is not faster than the
original compiled paper evaluator.

Extrapolating the new persistent-interface means, serially:

| Number of calls | Value only | Joint value + gradient |
| --- | --- | --- |
| 1,000 | 1.33 h | 3.18 h |
| 10,000 | 13.34 h | 31.84 h |

These exclude one-off setup, the other global-fit terms, optimizer line
searches, and any inner nuisance profiling. The timings expose all 31
parameters directly; a profiled likelihood point can require many calls.
Full nominal unbinned fits including setup, an exact Hessian, serialization,
and plots took approximately 468–512 s in the earlier checks. That full
Hessian cost is not paid for each value/gradient call.

Using the supplied NNPDF figure of 12–14 ms per evaluation, our warm scalar
call is roughly **340–400 times slower**. This is an order-of-magnitude
comparison: hardware, threading, and the amount of work in the NNPDF
measurement were not matched. No NNPDF gradient timing was supplied.

## Integration implications

A direct Python interface is usable now for exploration, provided the object
is kept alive across calls. The term would dominate the quoted evaluation
budget. Low-call-count optimizations are an accessible starting point;
large ensembles or extensive sampling need a substantial speedup, reduced
representation, or a larger computing budget. Batched/GPU kernels and
controlled thread tuning are candidates to measure, not speedups established
by these tests.

The default example includes nuisance constraints. If the global fit provides
them elsewhere, set `include_penalty=False` to avoid double counting.
Parameter ordering is exposed in `term.names`; it must be mapped to the
global fit's coordinates. An actual-data/EFT likelihood needs the appropriate
observations, response coefficients, and validation separately.

The example defaults to explicit historical-paper numerics. CMS numerics
remain selectable and are genuinely distinct away from the nominal point;
see [the integration findings](cms-integration.md). The frozen paper branch
and all external paper inputs remain unchanged. Packaging is deferred:
the exploration recipe explicitly links shared read-only models and cache.

Raw measurements, names, and the tested point/gradient are recorded in
[validation/nnpdf-exploration-2026-10-20.json](../validation/nnpdf-exploration-2026-10-20.json).
