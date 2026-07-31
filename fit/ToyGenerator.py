"""Toy dataset generation for pseudo-experiments.

Two independent axes: `source` (this module's "cache" vs "truth", see below) and
`mode` (N2LL's existing Asimov/observation distinction in fit/Likelihood.py,
untouched here). A toy is a plain dict, produced by `generate_toy`, persisted with
`save_toy`/`load_toy`, and applied to an N2LL via `N2LL.setToy`.

- `source="cache"`: the model (trained surrogates) is taken as truth. A toy is a
  Poisson draw on the cached MC's own intensity `w0 * (1 + T(hypothesis))`. Answers
  "given the surrogates are correct, are my intervals calibrated?"
- `source="truth"`: pseudo-data is generated from an exact reweighting, a
  per-process scale factor, or an alternate sample (`ProcessSource`), while the fit
  still uses the trained surrogates. Probes surrogate mismodelling as a bias in the
  fitted coefficients.

Known caveat (documented, not fixed): toys are drawn from the same cached/streamed
MC that supplies the expected-yield term, so toy and template fluctuations are
correlated. Sub-percent effect on interval widths at current MC statistics; see
user/ricardo/claude/plans/toy-generation-design-decisions.md for why this is not
corrected.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import importlib
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import h5py

# project root (this file lives in fit/) + ML/Calibration (for the shared UID-split helper)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "ML", "Calibration"))

from fit.Likelihood import N2LL, expand_pois_linear_quadratic, build_hypothesis_from_likelihood
from fit.Modeling import Hypothesis
import calibration_runner as cr  # _uid_split_interval

# common.derivative_providers is imported lazily inside _materialize_truth_weights
# (see below): it pulls in data/samples_eft.py, which eagerly constructs RDataLoaders
# for every declared EFT sample at import time, so importing it at module level would
# make cache-mode generation depend on the EFT sample data being reachable even though
# cache mode never touches it.

logger = logging.getLogger(__name__)


# ============================================================================
# Multi-process truth sources
# ============================================================================

@dataclass
class ProcessSource:
    """One process's contribution to a truth-mode toy in one region.

    w_truth = w_coefficients * scale_factor * prod(weight_branches) * weight_function(X)
    reducing to the nominal event weight when `coefficients` is None.
    """
    class_id: str
    sample_name: Optional[str] = None
    coefficients: Optional[dict] = None
    scale_factor: float = 1.0
    weight_branches: Optional[list] = None
    weight_function: Optional[Callable] = None


# ============================================================================
# RNG: stable per-(region[,class]) spawning
# ============================================================================

def _stable_str_to_int(s: str) -> int:
    """Process-stable string hash (Python's built-in hash() is salted per-process)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _spawn_rng(seed: int, *parts: str) -> np.random.Generator:
    """RNG for a single (seed, region_id[, class_id]) tuple, via a stable hash of the
    parts -- not sequential spawning off one Generator -- so adding a region or
    process does not perturb another's draws."""
    entropy = [int(seed) & 0xFFFFFFFF] + [_stable_str_to_int(p) & 0xFFFFFFFF for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ============================================================================
# constraint centers
# ============================================================================

def throw_constraint_centers(hypothesis, rng: np.random.Generator) -> dict:
    """Throw nu_obs ~ Normal(nu_true, 1) for every penalized nuisance; {name: nu_obs}."""
    return {p.name: float(rng.normal(p.val, 1.0)) for p in hypothesis.parameters if p.isPenalized}


# ============================================================================
# cache-mode unbinned
# ============================================================================

def _compute_cache_intensity(n2ll: N2LL, region_id: str, hypothesis) -> np.ndarray:
    """w0_i * (1 + T_i(hypothesis)) over the whole cached MC for one region -- the
    model's expected (Asimov) per-event intensity, unclipped. Shared by
    `generate_unbinned_toy_from_cache` (which draws Poisson toys from it) and the
    diagnostic Asimov reference histogram in plot/toys (which sums it as-is)."""
    N = n2ll._N_region.get(region_id, 0)
    if N == 0:
        return np.empty(0, dtype=np.float64)

    cA_per_class = n2ll._assemble_cA_per_class(region_id, hypothesis)
    nuA_per_group = n2ll._assemble_nuA_groups(region_id, hypothesis)
    nu_vals = {p.name: p.val for p in getattr(hypothesis, "parameters", []) if not p.isPOI}
    ln_bias = {
        cid: sum(a * nu_vals.get(nm, 0.0) for nm, a in n2ll._lnN_by_class.get((region_id, cid), []))
        for cid in n2ll._class_ids_by_region[region_id]
    }
    rate_shift = n2ll._assemble_rate_shift_per_class(region_id, hypothesis)

    class_ids = n2ll._class_ids_by_region[region_id]
    w0 = np.asarray(n2ll._h5[(region_id, class_ids[0])]["w0"], dtype=np.float64)
    T = n2ll._compute_T_chunk(region_id, cA_per_class, nuA_per_group, ln_bias, rate_shift, 0, N)
    return w0 * (1.0 + T)


def generate_unbinned_toy_from_cache(n2ll: N2LL, region_id: str, hypothesis, rng: np.random.Generator,
                                      *, allow_negative_weights: bool = False) -> dict:
    """Cache-mode unbinned toy: n_i ~ Poisson(w0_i * (1 + T_i(hypothesis))) over the
    whole cached MC (no UID split -- see the plan's "UID splitting" section for why
    cache mode uses everything).

    Negative intensity has two distinct causes, both real: at hypothesis=0, T is
    identically 0 for every event, so a negative w0_i * (1+T_i) there can only come
    from a negative-weight MC event (routine for NLO generators like aMC@NLO/POWHEG,
    not a modelling problem); away from hypothesis=0 it can also come from T < -1
    (the model extrapolating outside its valid range). Both are reported and treated
    the same way as truth mode's negative-weight handling: crash by default with the
    offending fraction, clip to zero under allow_negative_weights=True.

    Returns row indices and multiplicities for n_i > 0 (not sliced columns -- the
    surrogate columns are verbatim cache slices, rehydrated by `load_toy`).
    """
    N = n2ll._N_region.get(region_id, 0)
    if N == 0:
        return {"indices": np.empty(0, dtype=np.int64), "n": np.empty(0, dtype=np.float64)}

    lam = _compute_cache_intensity(n2ll, region_id, hypothesis)

    neg = lam < 0.0
    n_neg = int(np.sum(neg))
    if n_neg > 0:
        neg_weight_sum = float(np.sum(lam[neg]))
        if not allow_negative_weights:
            raise RuntimeError(
                f"[toy:cache:{region_id}] {n_neg}/{N} events ({n_neg / N:.4%}) have negative intensity "
                f"w0*(1+T) (summed negative intensity {neg_weight_sum:.4g}); pass "
                f"allow_negative_weights=True to clip them to zero."
            )
        logger.info("[toy:cache:%s] clipping %d/%d negative-intensity events (summed %.4g) to zero.",
                    region_id, n_neg, N, neg_weight_sum)
        lam = np.where(neg, 0.0, lam)

    n_i = rng.poisson(lam)
    keep = n_i > 0
    return {"indices": np.nonzero(keep)[0].astype(np.int64), "n": n_i[keep].astype(np.float64)}


def _rehydrate_cache_by_class(n2ll: N2LL, region_id: str, indices: np.ndarray) -> dict:
    """Rebuild by-class surrogate columns for a cache-mode toy by indexing the live
    cache. Structural binding: an out-of-range index (e.g. a retrained/shrunk cache)
    raises IndexError rather than silently misaligning columns."""
    class_ids = n2ll._class_ids_by_region[region_id]
    by_class = {}
    for cid in class_ids:
        cols = n2ll._h5[(region_id, cid)]
        by_class[cid] = {k: np.asarray(cols[k])[indices] for k in cols if k != "w0"}
    return by_class


def materialize_cache_region_diagnostics(n2ll: N2LL, region_id: str, hypothesis, feature_names: list,
                                          plot_opts: dict, extra_indices: Optional[np.ndarray] = None):
    """Single re-stream pass over a region's Asimov samples, for diagnostic plotting
    (see plot/toys/toy_diagnostic_plots.py). Cache-mode toys store only (indices,
    surrogate columns), not X, to avoid duplicating the surrogate cache, so this is
    the opt-in, costly path back to raw kinematics.

    Returns (asimov_hist, extra_X):
      - asimov_hist: {feature_name: weighted histogram of w0*(1+T) at `hypothesis`
        over the WHOLE cached MC, binned per `plot_opts`} -- what toys generated at
        this hypothesis should fluctuate around, always available since the
        surrogate cache is always built regardless of toy source. Negative
        intensities are clipped to zero for display (a diagnostic reference isn't
        the place to enforce generation's crash-by-default policy), with a logged
        count/fraction.
      - extra_X: raw feature matrix at `extra_indices` (e.g. the union of
        cache-mode toys' kept row indices), or None if `extra_indices` is None.
        Piggy-backs on the same pass since computing `asimov_hist` already visits
        every cached event once.

    `extra_indices`, if given, must be sorted ascending (true of both
    `generate_unbinned_toy_from_cache`'s and `np.unique`'s output).
    """
    region = _find_region(n2ll, region_id)

    lam = _compute_cache_intensity(n2ll, region_id, hypothesis)
    neg = lam < 0.0
    n_neg = int(np.sum(neg))
    if n_neg > 0:
        logger.info(
            "[toy:plot:%s] clipping %d/%d negative-intensity events (summed %.4g) to zero "
            "for the Asimov reference histogram.", region_id, n_neg, len(lam), float(np.sum(lam[neg])),
        )
        lam = np.where(neg, 0.0, lam)

    edges = {
        feat: np.linspace(*plot_opts[feat]["binning"][1:], int(plot_opts[feat]["binning"][0]) + 1)
        for feat in feature_names if feat in plot_opts
    }
    asimov_hist = {feat: np.zeros(len(e) - 1, dtype=np.float64) for feat, e in edges.items()}

    extra_indices = None if extra_indices is None else np.asarray(extra_indices, dtype=np.int64)
    if extra_indices is not None and not np.all(np.diff(extra_indices) >= 0):
        raise ValueError(f"[toy:plot:{region_id}] extra_indices must be sorted ascending.")
    extra_X = np.empty((len(extra_indices), len(feature_names)), dtype=np.float64) if extra_indices is not None else None

    col_positions = None
    row_ptr = 0
    global_offset = 0
    for feat_names, X, _w0 in n2ll._iter_asimov_batches(region):
        if col_positions is None:
            pos = {f: i for i, f in enumerate(feat_names)}
            missing = [f for f in feature_names if f not in pos]
            if missing:
                raise KeyError(f"[toy:plot:{region_id}] Feature(s) {missing} not in region samples.")
            col_positions = [pos[f] for f in feature_names]

        batch_hi = global_offset + len(X)
        batch_lam = lam[global_offset:batch_hi]
        for feat, e in edges.items():
            asimov_hist[feat] += np.histogram(X[:, pos[feat]], bins=e, weights=batch_lam)[0]

        if extra_indices is not None:
            while row_ptr < len(extra_indices) and extra_indices[row_ptr] < batch_hi:
                extra_X[row_ptr] = X[extra_indices[row_ptr] - global_offset, col_positions]
                row_ptr += 1
        global_offset = batch_hi

    if extra_indices is not None and row_ptr < len(extra_indices):
        raise RuntimeError(
            f"[toy:plot:{region_id}] {len(extra_indices) - row_ptr} indices exceeded the re-streamed "
            f"event count ({global_offset}); does the toy's cache match this config's samples?"
        )
    return asimov_hist, extra_X


# ============================================================================
# truth-mode: resolving sources, feature union, per-source truth weights
# ============================================================================

def _find_region(n2ll: N2LL, region_id: str) -> dict:
    for region in list(n2ll.regions) + list(n2ll.binned):
        if region["id"] == region_id:
            return region
    raise RuntimeError(f"[toy] Unknown region '{region_id}'.")


def _find_class(region: dict, class_id: str) -> dict:
    for C in region.get("classes", []) or []:
        if C["id"] == class_id:
            return C
    raise RuntimeError(f"[toy] Unknown class '{class_id}' in region '{region['id']}'.")


def _resolve_process_sources(classes: list, explicit_sources) -> dict:
    """Merge explicit ProcessSources with implicit ones (scale_factor=1 from the
    class's own sample) for every class with no explicit source."""
    by_cid: dict = {}
    for s in explicit_sources or []:
        if s.class_id in by_cid:
            raise RuntimeError(f"[toy] Duplicate ProcessSource for class '{s.class_id}'.")
        by_cid[s.class_id] = s
    known_cids = {C["id"] for C in classes}
    unknown = set(by_cid) - known_cids
    if unknown:
        raise RuntimeError(f"[toy] Unknown class id(s) in truth sources: {sorted(unknown)}")
    for C in classes:
        cid = C["id"]
        if cid not in by_cid:
            by_cid[cid] = ProcessSource(class_id=cid)
    return by_cid


def _region_feature_union(region: dict) -> list:
    """Union of every feature list the region's surrogates consume (classifier +
    every class's BIT + every class's PNNs), built from the loaded predictors'
    feature_names -- not from any job['features'] list."""
    predictors = []
    clf = region.get("_classifier_predictor")
    if clf is not None:
        predictors.append(clf)
    for C in region.get("classes", []) or []:
        poi_pred = (C.get("POI") or {}).get("predictor")
        if poi_pred is not None:
            predictors.append(poi_pred)
        for S in C.get("_pnn_systs", []) or []:
            pred = S.get("predictor")
            if pred is not None:
                predictors.append(pred)

    names, seen = [], set()
    for pred in predictors:
        for f in getattr(pred, "feature_names", []) or []:
            if f not in seen:
                seen.add(f)
                names.append(f)
    return names


def _class_bit_job(n2ll: N2LL, region_id: str, class_id: str) -> dict:
    region = _find_region(n2ll, region_id)
    C = _find_class(region, class_id)
    job_id = (C.get("POI") or {}).get("job")
    if not job_id:
        raise RuntimeError(f"[toy] Class '{region_id}/{class_id}' has no configured BIT job (POI.job).")
    jobs_by_id = getattr(n2ll, "_toy_jobs_by_id", None)
    if not jobs_by_id:
        raise RuntimeError(
            "[toy] n2ll has no attached job registry; ToyGenerator.__main__ must set n2ll._toy_jobs_by_id."
        )
    job = jobs_by_id.get(job_id)
    if job is None:
        raise RuntimeError(f"[toy] BIT job '{job_id}' not found in the loaded config's jobs.")
    return job


def _materialize_truth_weights(n2ll: N2LL, region_id: str, source: ProcessSource, feature_names: list,
                                split: Optional[str], hypothesis, allow_negative_weights: bool):
    """Stream one ProcessSource's sample, reconstruct the exact truth weight,
    restrict to `split`, rescale by the retained *weight* fraction, and (if
    `hypothesis` is given) multiply by the model's region-level (1+T) at that
    hypothesis -- the surrogate-route injection, evaluated on this source's own
    kinematics via the same `_eval_region_surrogates`/`_compute_T_from_columns`
    N2LL uses for real data (see the plan's "two POI injection routes").

    Returns (X (M,d), w_truth (M,), diagnostics dict).
    """
    region = _find_region(n2ll, region_id)
    C = _find_class(region, source.class_id)
    class_sample = C.get("sample")
    if not class_sample:
        raise RuntimeError(f"[toy] Class '{region_id}/{source.class_id}' has no configured 'sample'.")
    sample_name = source.sample_name or class_sample
    if source.sample_name is not None and source.sample_name != class_sample:
        logger.warning(
            "[toy:%s/%s] sample_name '%s' differs from the class's configured sample '%s' "
            "-- this is a deliberate mismodelling injection.",
            region_id, source.class_id, source.sample_name, class_sample,
        )

    provider = None
    required_observers: list = []
    if source.coefficients is not None:
        if hypothesis is not None:
            poi_vals = {p.name: p.val for p in getattr(hypothesis, "POIs", [])}
            overlap = [k for k in source.coefficients if poi_vals.get(k, 0.0) != 0.0]
            if overlap:
                raise ValueError(
                    f"[toy:{region_id}/{source.class_id}] POI(s) {overlap} injected via both "
                    f"'coefficients' and a nonzero hypothesis value; pick one route."
                )
        from common.derivative_providers import build_derivative_provider
        job = _class_bit_job(n2ll, region_id, source.class_id)
        provider = build_derivative_provider(job)
        # Set, not order: n2ll._poi_order is the fit's R_A column order (sorted, to match
        # the BIT's own alphabetized derivative columns -- see Likelihood.py). The truth
        # route below evaluates make_weight_matrix / expand_pois_linear_quadratic both in
        # provider.parameters order, so it is internally consistent in any order; only the
        # operator *set* has to agree with what the fit expects for this class.
        expected_pois = n2ll._poi_order.get((region_id, source.class_id), [])
        if expected_pois and set(provider.parameters) != set(expected_pois):
            raise RuntimeError(
                f"[toy:{region_id}/{source.class_id}] provider POIs {sorted(provider.parameters)} "
                f"!= class POIs {sorted(expected_pois)}."
            )
        required_observers = list(provider.required_observers)

    splitting_cfg = getattr(n2ll, "_toy_splitting_defaults", None)
    uid_fields = list((splitting_cfg or {}).get("uid_fields", ["run", "luminosityBlock", "event"]))
    uid_splitter = lo = hi = None
    if split is not None:
        if not splitting_cfg:
            raise RuntimeError(
                f"[toy:{region_id}/{source.class_id}] split='{split}' requested but n2ll has no attached "
                f"splitting config (ToyGenerator.__main__ sets n2ll._toy_splitting_defaults)."
            )
        split_names = [split] if isinstance(split, str) else list(split)
        uid_splitter, uid_fields, (lo, hi) = cr._uid_split_interval(splitting_cfg, *split_names)

    observer_names = list(dict.fromkeys(required_observers + uid_fields))
    if source.weight_branches:
        for weight_branch in source.weight_branches:
            observer_names.append(weight_branch)
    loader = n2ll.factory.get(sample_name).clone()
    loader.setFeatures(feature_names, observer_names=observer_names)

    Xs, Ws = [], []
    sum_w_all = 0.0
    sum_w_split = 0.0
    for shard in range(int(getattr(loader, "n_split", 1))):
        X, G, w = loader.materialize(shard=shard, what="fow")
        X = np.asarray(X, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        sum_w_all += float(np.sum(w))

        if provider is not None:
            deriv_w = provider.truth_weight_matrix(G, w, observer_names)  # (N, M), col 0 = nominal
            w_coef = deriv_w[:, 0] + deriv_w[:, 1:] @ expand_pois_linear_quadratic(
                provider.parameters, source.coefficients
            )
        else:
            w_coef = w

        w_truth = w_coef * source.scale_factor
        if source.weight_branches:
            branch_mask = N2LL.make_column_mask(observer_names, source.weight_branches)
            w_truth = w_truth * np.prod(G[:, branch_mask], axis=1)
        if source.weight_function is not None:
            w_truth = w_truth * np.asarray(source.weight_function(X, feature_names), dtype=np.float64)

        if split is not None:
            on2idx = {n: i for i, n in enumerate(observer_names)}
            G_uid = G[:, [on2idx[f] for f in uid_fields]]
            m_keep = uid_splitter.mask_from_np(G_uid, uid_fields, lo, hi)
        else:
            m_keep = np.ones(len(w), dtype=bool)
        sum_w_split += float(np.sum(w[m_keep]))

        Xs.append(X[m_keep])
        Ws.append(w_truth[m_keep])

    if sum_w_all <= 0:
        raise RuntimeError(f"[toy:{region_id}/{source.class_id}] sample has zero total weight.")
    f_weight = sum_w_split / sum_w_all
    if f_weight <= 0:
        raise RuntimeError(f"[toy:{region_id}/{source.class_id}] split='{split}' selected zero weight.")

    X_all = np.concatenate(Xs, axis=0) if Xs else np.empty((0, len(feature_names)), dtype=np.float64)
    w_truth_all = (np.concatenate(Ws, axis=0) if Ws else np.empty(0, dtype=np.float64)) / f_weight

    nominal_yield = sum_w_all
    if hypothesis is not None and len(X_all):
        by_class_all = n2ll._eval_region_surrogates(region_id, X_all, feature_names)
        cA_per_class = n2ll._assemble_cA_per_class(region_id, hypothesis)
        nuA_per_group = n2ll._assemble_nuA_groups(region_id, hypothesis)
        nu_vals = {p.name: p.val for p in getattr(hypothesis, "parameters", []) if not p.isPOI}
        ln_bias = {
            cid: sum(a * nu_vals.get(nm, 0.0) for nm, a in n2ll._lnN_by_class.get((region_id, cid), []))
            for cid in n2ll._class_ids_by_region[region_id]
        }
        rate_shift = n2ll._assemble_rate_shift_per_class(region_id, hypothesis)
        T = n2ll._compute_T_from_columns(region_id, by_class_all, cA_per_class, nuA_per_group,
                                          ln_bias, rate_shift, 0, len(X_all))
        w_truth_all = w_truth_all * (1.0 + T)

    neg = w_truth_all < 0.0
    n_neg = int(np.sum(neg))
    truth_yield = float(np.sum(w_truth_all))
    neg_yield_share = float(np.sum(w_truth_all[neg]) / truth_yield) if (n_neg and truth_yield) else 0.0
    if n_neg > 0:
        if not allow_negative_weights:
            raise RuntimeError(
                f"[toy:{region_id}/{source.class_id}] {n_neg}/{len(w_truth_all)} events have negative "
                f"truth weight (share of yield {neg_yield_share:.4f}); pass allow_negative_weights=True "
                f"to clip them to zero."
            )
        w_truth_all = np.where(neg, 0.0, w_truth_all)
        truth_yield = float(np.sum(w_truth_all))

    diag = {
        "f_weight": f_weight, "split": split,
        "nominal_yield": nominal_yield, "truth_yield": truth_yield,
        "n_negative": n_neg, "negative_yield_share": neg_yield_share,
    }
    return X_all, w_truth_all, diag


# ============================================================================
# truth-mode unbinned toy (multi-process)
# ============================================================================

def generate_unbinned_toy_from_truth(n2ll: N2LL, region_id: str, sources, rng: np.random.Generator, *,
                                      split: tuple[str] | None = ("c2st_train", "c2st_val"), hypothesis=None,
                                      allow_negative_weights: bool = False) -> dict:
    """Multi-process truth-mode unbinned toy (see module docstring / plan
    "Multi-process handling"). Per source (one per class): reconstruct the truth
    weight, throw n_i ~ Poisson(w_truth_i), keep n_i > 0, record an origin label
    (diagnostics only). Concatenate across sources and evaluate the region's
    surrogates once on the whole unlabeled set, exactly as setObservation does for
    real data.
    """
    region = _find_region(n2ll, region_id)
    classes = region.get("classes", []) or []
    by_cid = _resolve_process_sources(classes, sources)
    feature_names = _region_feature_union(region)

    Xs, Ns, origins, diagnostics = [], [], [], {}
    for i_class, (cid, source) in enumerate(by_cid.items()):
        X, w_truth, diag = _materialize_truth_weights(
            n2ll, region_id, source, feature_names, split, hypothesis, allow_negative_weights
        )
        n_i = rng.poisson(np.clip(w_truth, 0.0, None))
        keep = n_i > 0
        n_kept = n_i[keep]
        dup = n_kept > 1
        diag.update({
            "drawn_yield": float(n_kept.sum()),
            "n_multiplicity_gt1": int(dup.sum()),
            "multiplicity_gt1_yield_share": float(n_kept[dup].sum() / n_kept.sum()) if n_kept.sum() else 0.0,
        })
        diagnostics[cid] = diag
        logger.info("[toy:%s/%s] %s", region_id, cid, diag)

        Xs.append(X[keep])
        Ns.append(n_kept.astype(np.float64))
        origins.append(np.full(int(keep.sum()), i_class, dtype=np.int64))

    X_toy = np.concatenate(Xs, axis=0) if Xs else np.empty((0, len(feature_names)), dtype=np.float64)
    n_toy = np.concatenate(Ns, axis=0) if Ns else np.empty(0, dtype=np.float64)
    origin_toy = np.concatenate(origins, axis=0) if origins else np.empty(0, dtype=np.int64)

    by_class = n2ll._eval_region_surrogates(region_id, X_toy, feature_names)

    return {
        "X": X_toy, "w": n_toy, "by_class": by_class, "origin": origin_toy,
        "feature_names": feature_names, "diagnostics": diagnostics,
    }


# ============================================================================
# binned toy
# ============================================================================

def generate_binned_toy(n2ll: N2LL, region_id: str, rng: np.random.Generator, *,
                         hypothesis=None, sources=None, split: tuple[str] | None = ("c2st_train", "c2st_val"),
                         allow_negative_weights: bool = False) -> dict:
    """Binned toy: N_obs ~ Poisson(lambda) per bin, thrown once per bin (see the
    plan's "Binned" section for why histogram-first is exact and cheaper than
    per-event Poisson-then-histogram).

    Cache mode (sources is None): lambda = n2ll._compute_lambda_binned(...), the
    model's expected counts.
    Truth mode (sources given): lambda = the truth-weighted histogram summed over
    sources' events, into the same bin edges -- so it differs from the cache-mode
    lambda by exactly the ICH/ICPH mismodelling.
    """
    un = n2ll._binned_unroll[region_id]
    edges = un["edges"]
    Nflat = len(un["flat_bins"])

    if sources is None:
        if hypothesis is None:
            raise ValueError("[toy:binned] cache mode (sources=None) requires a hypothesis.")
        lam = n2ll._compute_lambda_binned(region_id, hypothesis)
    else:
        region = _find_region(n2ll, region_id)
        classes = region.get("classes", []) or []
        by_cid = _resolve_process_sources(classes, sources)
        axes = un["axes"]
        lam = np.zeros(Nflat, dtype=np.float64)
        for cid, source in by_cid.items():
            X, w_truth, diag = _materialize_truth_weights(
                n2ll, region_id, source, axes, split, hypothesis, allow_negative_weights
            )
            logger.info("[toy:%s/%s] %s", region_id, cid, diag)
            if len(edges) == 1:
                H, _ = np.histogram(X[:, 0], bins=edges[0], weights=w_truth)
            else:
                H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[edges[0], edges[1]], weights=w_truth)
            lam = lam + H.reshape(-1)

    neg = lam < 0.0
    n_neg = int(np.sum(neg))
    if n_neg > 0:
        if not allow_negative_weights:
            raise RuntimeError(
                f"[toy:binned:{region_id}] negative expected counts in {n_neg}/{Nflat} bins "
                f"(summed {float(np.sum(lam[neg])):.4g}); pass allow_negative_weights=True to clip them to zero."
            )
        lam = np.where(neg, 0.0, lam)

    counts = rng.poisson(lam)
    return {"counts": counts.astype(np.float64)}


# ============================================================================
# top-level
# ============================================================================

def generate_toy(n2ll: N2LL, seed: int, *, source: str, hypothesis=None, truth_sources=None,
                  split: tuple[str] | None = ("c2st_train", "c2st_val"), throw_nuisances: bool = False,
                  allow_negative_weights: bool = False) -> dict:
    """Generate one full toy (every unbinned + binned region of n2ll's likelihood).

    `source`: "cache" (model is truth) or "truth" (exact reweighting/scale/sample).
    `hypothesis`: the injection point. In cache mode, defaults to nominal
    (build_hypothesis_from_likelihood). In truth mode, None means no surrogate-route
    (1+T) multiplication -- ProcessSource modifiers are still applied.
    """
    if source not in ("cache", "truth"):
        raise ValueError(f"source must be 'cache' or 'truth', got {source!r}")
    if source == "cache" and truth_sources:
        raise ValueError("truth_sources is only valid with source='truth'.")

    gen_hypothesis = hypothesis if hypothesis is not None else build_hypothesis_from_likelihood(n2ll.lk, name="toy")
    constraint_centers = throw_constraint_centers(gen_hypothesis, _spawn_rng(seed, "__constraints__")) if throw_nuisances else {}

    unbinned_blocks: dict = {}
    binned_counts: dict = {}
    diagnostics: dict = {"unbinned": {}, "binned": {}}

    for region in n2ll.regions:
        rid = region["id"]
        region_rng = _spawn_rng(seed, rid)
        if source == "cache":
            toy = generate_unbinned_toy_from_cache(n2ll, rid, gen_hypothesis, region_rng,
                                                    allow_negative_weights=allow_negative_weights)
            unbinned_blocks[rid] = {
                "X": None, "w": toy["n"], "indices": toy["indices"],
                "by_class": _rehydrate_cache_by_class(n2ll, rid, toy["indices"]),
            }
        else:
            sources_for_region = (truth_sources or {}).get(rid, [])
            toy = generate_unbinned_toy_from_truth(
                n2ll, rid, sources_for_region, region_rng, split=split, hypothesis=hypothesis,
                allow_negative_weights=allow_negative_weights,
            )
            unbinned_blocks[rid] = {"X": toy["X"], "w": toy["w"], "by_class": toy["by_class"], "origin": toy["origin"]}
            diagnostics["unbinned"][rid] = toy["diagnostics"]

    for region in n2ll.binned:
        rid = region["id"]
        region_rng = _spawn_rng(seed, rid)
        if source == "cache":
            toy = generate_binned_toy(n2ll, rid, region_rng, hypothesis=gen_hypothesis,
                                       allow_negative_weights=allow_negative_weights)
        else:
            sources_for_region = (truth_sources or {}).get(rid, [])
            toy = generate_binned_toy(
                n2ll, rid, region_rng, sources=sources_for_region, hypothesis=hypothesis,
                split=split, allow_negative_weights=allow_negative_weights,
            )
        binned_counts[rid] = toy["counts"]

    # gen_hypothesis only carries the surrogate-route injection (or nominal, if none was
    # given). A coefficients-route injection (ProcessSource.coefficients, e.g. a point with
    # `injection:` and no `hypothesis:`) never touches gen_hypothesis, so on its own the
    # recorded metadata would read nominal regardless of what was actually injected. Fold
    # those coefficients in for the record.
    recorded_hypothesis = {p.name: float(p.val) for p in gen_hypothesis.parameters}
    if source == "truth" and truth_sources:
        for region_sources in truth_sources.values():
            for src in region_sources:
                for name, val in (src.coefficients or {}).items():
                    prior = recorded_hypothesis.get(name)
                    if prior is not None and prior != 0.0 and prior != float(val):
                        raise RuntimeError(
                            f"[toy] Conflicting injected value for '{name}': {prior} vs {val}."
                        )
                    recorded_hypothesis[name] = float(val)

    return {
        "seed": int(seed),
        "source": source,
        "hypothesis": recorded_hypothesis,
        "constraint_centers": constraint_centers,
        "unbinned_blocks": unbinned_blocks,
        "binned_counts": binned_counts,
        "diagnostics": diagnostics,
        "config_version": getattr(n2ll, "version", None),
    }


# ============================================================================
# persistence
# ============================================================================

def save_toy(path: str, toy: dict) -> None:
    """Persist a toy to HDF5. Cache-mode regions store only (indices, n) -- the
    surrogate columns are verbatim cache slices, rehydrated by load_toy rather than
    duplicated on disk. Truth-mode regions store the materialized X/by_class/origin.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["seed"] = int(toy["seed"])
        meta.attrs["source"] = toy["source"]
        meta.attrs["hypothesis"] = json.dumps(toy["hypothesis"])
        meta.attrs["config_version"] = str(toy.get("config_version") or "")
        meta.attrs["point"] = toy.get("point", "")

        cc = f.create_group("constraint_centers")
        for name, val in (toy.get("constraint_centers") or {}).items():
            cc.attrs[name] = float(val)

        ub = f.create_group("unbinned")
        for rid, block in toy["unbinned_blocks"].items():
            g = ub.create_group(rid)
            g.create_dataset("n", data=np.asarray(block["w"], dtype=np.float64))
            if block.get("indices") is not None:
                g.create_dataset("indices", data=np.asarray(block["indices"], dtype=np.int64))
            else:
                g.create_dataset("X", data=np.asarray(block["X"], dtype=np.float64))
                g.create_dataset("origin", data=np.asarray(block["origin"], dtype=np.int64))
                bc = g.create_group("by_class")
                for cid, comp in block["by_class"].items():
                    cg = bc.create_group(cid)
                    for col_name, col in comp.items():
                        cg.create_dataset(col_name, data=np.asarray(col, dtype=np.float64))

        bn = f.create_group("binned")
        for rid, counts in toy["binned_counts"].items():
            bn.create_dataset(rid, data=np.asarray(counts, dtype=np.float64))

    logger.info("[toy] Saved %s", path)


def load_toy(path: str, n2ll: N2LL) -> dict:
    """Load a toy saved by save_toy. Cache-mode regions are rehydrated by indexing
    n2ll's live cache -- a shrunk/retrained cache raises IndexError (or a
    _rehydrate_cache_by_class KeyError for a missing class) rather than silently
    misaligning columns."""
    with h5py.File(path, "r") as f:
        meta = f["meta"]
        seed = int(meta.attrs["seed"])
        source = str(meta.attrs["source"])
        hypothesis = json.loads(meta.attrs["hypothesis"])
        config_version = str(meta.attrs.get("config_version", ""))
        point = str(meta.attrs.get("point", ""))

        constraint_centers = {name: float(val) for name, val in f["constraint_centers"].attrs.items()}

        unbinned_blocks = {}
        for rid in f["unbinned"]:
            g = f["unbinned"][rid]
            n = np.asarray(g["n"])
            if "indices" in g:
                indices = np.asarray(g["indices"], dtype=np.int64)
                unbinned_blocks[rid] = {
                    "X": None, "w": n, "indices": indices,
                    "by_class": _rehydrate_cache_by_class(n2ll, rid, indices),
                }
            else:
                by_class = {}
                for cid in g["by_class"]:
                    cg = g["by_class"][cid]
                    by_class[cid] = {col_name: np.asarray(cg[col_name]) for col_name in cg}
                unbinned_blocks[rid] = {
                    "X": np.asarray(g["X"]), "w": n, "by_class": by_class,
                    "origin": np.asarray(g["origin"]),
                }

        binned_counts = {rid: np.asarray(f["binned"][rid]) for rid in f["binned"]}

    return {
        "seed": seed, "source": source, "hypothesis": hypothesis,
        "constraint_centers": constraint_centers,
        "unbinned_blocks": unbinned_blocks, "binned_counts": binned_counts,
        "config_version": config_version, "point": point,
    }


# ============================================================================
# spec file parsing + CLI
# ============================================================================

def _resolve_weight_function(dotted_path: str) -> Callable:
    module_name, func_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def _parse_injection(injection: dict) -> dict:
    truth_sources = {}
    for region_id, classes in (injection or {}).items():
        sources = []
        for class_id, fields in (classes or {}).items():
            fields = dict(fields or {})
            wf = fields.pop("weight_function", None)
            sources.append(ProcessSource(
                class_id=class_id,
                sample_name=fields.pop("sample_name", None),
                coefficients=fields.pop("coefficients", None),
                scale_factor=float(fields.pop("scale_factor", 1.0)),
                weight_branches=fields.pop("weight_branches", None),
                weight_function=_resolve_weight_function(wf) if wf else None,
            ))
            if fields:
                raise RuntimeError(f"Unknown ProcessSource field(s) {list(fields)} for {region_id}/{class_id}.")
        truth_sources[region_id] = sources
    return truth_sources


def _hypothesis_from_point(n2ll: N2LL, point_hyp) -> Optional[Hypothesis]:
    if not point_hyp:
        return None
    base = build_hypothesis_from_likelihood(n2ll.lk, name="toy_point")
    for name, val in point_hyp.items():
        if name not in base:
            raise KeyError(f"Unknown parameter '{name}' in point hypothesis.")
        base[name].val = float(val)
    return base


def _parse_seeds(spec: str) -> list:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


if __name__ == "__main__":
    import argparse
    import common.yaml_loader as yaml_loader
    from fit.Likelihood import load_likelihood

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Toy dataset generation for pseudo-experiments.")
    p.add_argument("configs", nargs="+", help="Path to one or more global YAML configs")
    p.add_argument("--toySpec", required=True, help="Path to the toy spec YAML")
    p.add_argument("--toyPoint", required=True, help="Name of the point in the spec to generate")
    p.add_argument("--seeds", required=True, help="Seed or range, e.g. '0-499' or '3,7,12'")
    p.add_argument("--outputDir", required=True, help="Directory to write <point>_toy<seed>.h5 files")
    p.add_argument("--overwrite", nargs="?", const="all", default=None, choices=["fit", "all"],
                   help="Overwrite the surrogate cache ('all') before generating.")
    p.add_argument("--plot", action="store_true",
                   help="After generating, write per-feature diagnostic plots (config's "
                        "default_features, overlaid across the requested seeds) under "
                        "<plot_directory>/toys/<version>/<point>/. Cache-mode toys pay the "
                        "cost of re-streaming the raw samples once to recover kinematics.")
    args = p.parse_args()

    list_configs = []
    for config_path in args.configs:
        aux_cfg = yaml_loader.load_yaml(config_path)
        yaml_loader.print_summary(aux_cfg, config_path, yaml_loader._INCLUDE_TRACE)
        yaml_loader.load_surrogates(aux_cfg, config_path, overwrite=False)
        list_configs.append(aux_cfg)
    cfg = yaml_loader.combine_configs(list_configs)

    like_info = load_likelihood(cfg)

    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])
    from common.yaml_loader import _resolve_features_list
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    factory = samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )

    base = "_".join(os.path.splitext(os.path.basename(c))[0] for c in args.configs)
    n2ll = N2LL(
        like_info, factory=factory,
        cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
        cache_root=None,
        overwrite=(args.overwrite == "all"),
    )

    spec = yaml_loader.load_yaml(args.toySpec)
    spec_source = spec.get("source", "cache")
    spec_split = spec.get("split", ("c2st_train","c2st_val"))
    if isinstance(spec_split, str):
        logger.info("spec_split is a string")
        spec_split = [part.strip() for part in spec_split.split(",") if part.strip()]
    else:
        logger.info("spec_split is a list")
    spec_throw_nuisances = bool(spec.get("throw_nuisances", False))
    spec_allow_negative = bool(spec.get("allow_negative_weights", False))

    n2ll.shuffle_features = None
    n2ll.build_cache()
    n2ll.prepare_runtime()

    n2ll.version = cfg.get("version")
    n2ll._toy_splitting_defaults = (cfg.get("defaults") or {}).get("splitting")
    n2ll._toy_jobs_by_id = {j["id"]: j for j in (cfg.get("jobs") or []) if j.get("id")}

    point = next((pt for pt in (spec.get("points") or []) if pt.get("name") == args.toyPoint), None)
    if point is None:
        available = [pt.get("name") for pt in (spec.get("points") or [])]
        raise RuntimeError(f"Point '{args.toyPoint}' not found in {args.toySpec}. Available: {available}")

    truth_sources = _parse_injection(point.get("injection")) if spec_source == "truth" else None
    hypothesis = _hypothesis_from_point(n2ll, point.get("hypothesis"))

    seeds = _parse_seeds(args.seeds)
    os.makedirs(args.outputDir, exist_ok=True)
    generated_toys = []
    for seed in seeds:
        toy = generate_toy(
            n2ll, seed, source=spec_source, hypothesis=hypothesis, truth_sources=truth_sources,
            split=spec_split, throw_nuisances=spec_throw_nuisances, allow_negative_weights=spec_allow_negative,
        )
        toy["point"] = args.toyPoint
        out_path = os.path.join(args.outputDir, f"{args.toyPoint}_{spec_source}_toy{seed}.h5")
        save_toy(out_path, toy)
        generated_toys.append(toy)

    if args.plot:
        if not features:
            raise RuntimeError("--plot requires defaults.default_features to be set in the config.")
        import common.user as user
        import common.syncer as syncer
        from plot.toys.toy_diagnostic_plots import plot_toy_feature_distributions

        plot_dir = os.path.join(user.plot_directory, "toys", str(cfg.get("version")), args.toyPoint)
        plot_toy_feature_distributions(n2ll, generated_toys, features, plot_dir)
        syncer.sync()
