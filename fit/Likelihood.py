from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from types import SimpleNamespace

try:
    import autograd.numpy as np
    from autograd import grad, hessian
    from autograd.tracer import getval
    _HAS_AUTOGRAD = True
except ImportError:
    import numpy as np
    _HAS_AUTOGRAD = False

    def grad(function):
        raise ImportError("autograd is required for gradient-based fits.")

    def hessian(function):
        raise ImportError("autograd is required for Hessian-based fits.")

    def getval(value):
        return value

from scipy.optimize import minimize

import os
import yaml
import json, math, importlib
import h5py

from math import ceil
from tqdm import tqdm

from collections import defaultdict, deque

import copy
import logging
logger = logging.getLogger(__name__)

import numpy as onp

import sys
sys.path.insert(0, '..')

from fit.Modeling import ModelParameter, Hypothesis, Rotated
from ML.Calibration.binned_calibration import apply_binned_calibration
import common.helpers as helpers
 
# ---- Likelihood wiring + model parameter scaffolding -----------------------

def _job_by_id(cfg, jid):
    return next((j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id") == jid), None)

def _predictor_from_job(job):
    # We rely on load_surrogates having attached job['predictor'] when available.
    return None if job is None else job.get("predictor", None)

# --- helper for predictor -> (axis_names, edges[]) ---
def _pred_binning_tuple(pred):
    names = tuple(getattr(pred, "axis_names", []) or [])
    edges = [ np.asarray(be, dtype=float) for be in (getattr(pred, "bin_edges", []) or []) ]
    return names, edges


def load_likelihood(cfg):
    """
    Parse cfg['likelihood'], attach predictors by job id, and collect
    - POI names (union across all classes, unbinned + binned)
    - nuisance names (union across all systematics, unbinned + binned)

    Returns a dict:
      {
        'regions':   [... enriched unbinned regions ...],   # may be empty
        'binned':    [... enriched binned regions ...],     # may be empty
        'pois':      sorted list of POI names,
        'nuisances': sorted list of nuisance names
        'floating': sorted list of floating nuisance parameter names
      }

    The function mutates the region dictionaries to include predictor hooks:
      Unbinned:
        region['classifier']['predictor']   (if tfmc)
        class['POI']['predictor']           (BIT)
        syst['predictor']                   (for type == 'pnn')
        syst['parameters'], syst['combinations'] propagated from job if missing
      Binned:
        class['POI']['predictor']           (ICH)
        syst['predictor']                   (for type == 'icph')
        syst['parameters'], syst['combinations'] propagated from job if missing
    """
    lk = cfg.get("likelihood", {}) or {}
    regions = list(lk.get("regions", []) or [])
    binned  = list(lk.get("binned",  []) or [])

    if not regions and not binned:
        logger.info("No likelihood regions (unbinned or binned) found.")
        return {'regions': [], 'binned': [], 'pois': [], 'nuisances': []}

    all_pois = set()
    all_nuis = set()
    floating = set()
    # convenience cache of jobs by id
    id2job = {j.get("id"): j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id")}

    # -------------------------
    # Unbinned regions (BIT/PNN)
    # -------------------------
    for R in regions:
        # classifier (TFMC)
        clf = R.get("classifier", {}) or {}
        if clf.get("type") == "tfmc":
            tfmc_id  = clf.get("job")
            tfmc_job = id2job.get(tfmc_id) or _job_by_id(cfg, tfmc_id)
            tfmc_pred = _predictor_from_job(tfmc_job)
            clf['predictor'] = tfmc_pred
            if tfmc_pred is None:
                logger.warning(f"[likelihood] TFMC '{tfmc_id}' has no predictor attached yet.")

        # classes
        classes = R.get("classes", []) or []
        for C in classes:
            poi = C.get("POI", {}) or {}
            poi_type = poi.get("type")
            # POI (BIT)
            if poi_type == "bit":
                poi_job_id = poi.get("job")
                if poi_job_id:
                    bit_job = id2job.get(poi_job_id) or _job_by_id(cfg, poi_job_id)
                    poi['predictor'] = _predictor_from_job(bit_job)
                    if poi['predictor'] is None:
                        logger.warning(f"[likelihood] BIT '{poi_job_id}' has no predictor attached yet.")
                    elif (bit_job or {}).get("eft") and not getattr(poi['predictor'], "expansion_point", None):
                        raise RuntimeError(
                            f"[likelihood] BIT '{poi_job_id}' has an 'eft' block but its predictor carries "
                            "no expansion_point; retrain it so the fit reads the point it was trained at."
                        )
            elif poi_type == "rate_shift":
                param_len = len(poi.get("parameters",[]))
                if len(poi.get("parameters"))!=1:
                    raise RuntimeError(f"A 'rate_shift' POI must have a single parameter. Found {param_len}.")
                if not poi.get("parameters")[0].startswith('rate_shift'):
                    raise RuntimeError("Rate shift parameter name must start with rate_shift.")
            elif poi_type is None:
                pass
            else:
                raise RuntimeError(f"Unknown POI type {poi_type}")

            # collect POI parameter names
            for nm in poi.get("parameters", []):
                all_pois.add(nm)

            # systematics
            for S in (C.get("systematics", []) or []):
                styp = S.get("type")
                if styp == "pnn":
                    pnn_id  = S.get("job")
                    pnn_job = id2job.get(pnn_id) or _job_by_id(cfg, pnn_id)
                    S['predictor'] = _predictor_from_job(pnn_job)
                    if S['predictor'] is None:
                        logger.warning(f"[likelihood] PNN '{pnn_id}' has no predictor attached yet.")

                    # propagate params/combinations from job if missing
                    pnn_params = list((pnn_job or {}).get("parameters", []) or [])
                    pnn_combs  = [tuple(c) for c in ((pnn_job or {}).get("combinations", []) or [])]
                    if 'parameters' not in S or not S['parameters']:
                        S['parameters'] = pnn_params
                    S['combinations'] = pnn_combs

                    if S.get('shape_only', False):

                        logger.warning(f"[likelihood] PNN '{pnn_id}' will be used only for shape variations.")
                        
                        if S['predictor'] is not None:
                            
                            if not S['predictor'].has_icp():
                                raise NotImplementedError("Currently, only allowing shape-only systematics for PNNs trained with ICP bias.")
                            
                            S['predictor'].remove_icp_bias()

                    else:
                        # optional: check PNN↔ICP consistency if referenced
                        try:
                            extras = (pnn_job or {}).get('extras', {}) or {}
                            icp_id = extras.get('use_icp')
                            if isinstance(icp_id, str) and icp_id in id2job:
                                icp_job = id2job[icp_id]
                                icp = icp_job.get('predictor', None)
                                if icp is not None:
                                    icp_params = list(getattr(icp, "parameters"))
                                    icp_combs  = [tuple(c) for c in getattr(icp, "combinations")]
                                    if not (pnn_params == icp_params and pnn_combs == icp_combs):
                                        logger.warning(f"[likelihood] PNN '{pnn_id}' params/combs differ from ICP '{icp_id}'.")
                        except Exception:
                            pass

                    # collect nuisance names
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)

                elif styp == "lnN":
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)
                else:
                    # future unbinned syst types
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)

    # -----------------------
    # Binned regions (ICH/ICPH)
    # -----------------------
    for R in binned:
        # classes
        classes = R.get("classes", []) or []
        for C in classes:
            # POI (ICH)
            poi = C.get("POI", {}) or {}
            poi_type = poi.get("type")
            if poi_type == 'ich':
                poi_job_id = poi.get("job")
                if poi_job_id:
                    ich_job = id2job.get(poi_job_id) or _job_by_id(cfg, poi_job_id)
                    poi['predictor'] = _predictor_from_job(ich_job)
                    if poi['predictor'] is None:
                        logger.warning(f"[likelihood] ICH '{poi_job_id}' has no predictor attached yet.")
            elif poi_type == "rate_shift":
                raise NotImplementedError
                #param_len = len(poi.get("parameters",[]))
                #if len(poi.get("parameters"))!=1:
                #    raise RuntimeError(f"A 'rate_shift' POI must have a single parameter. Found {param_len}.")
                #if not poi.get("parameters")[0].startswith('rate_shift'):
                #    raise RuntimeError("Rate shift parameter name must start with rate_shift.")
            elif poi_type is None:
                pass
            else:
                raise RuntimeError(f"Unknown POI type {poi_type}")

            for nm in poi.get("parameters", []):
                all_pois.add(nm)

            # systematics
            for S in (C.get("systematics", []) or []):
                styp = S.get("type")
                if styp == "icph":
                    icph_id  = S.get("job")
                    icph_job = id2job.get(icph_id) or _job_by_id(cfg, icph_id)
                    S['predictor'] = _predictor_from_job(icph_job)
                    if S['predictor'] is None:
                        logger.warning(f"[likelihood] ICPH '{icph_id}' has no predictor attached yet.")

                    # propagate params/combinations from job if missing
                    icph_params = list((icph_job or {}).get("parameters", []) or [])
                    icph_combs  = [tuple(c) for c in ((icph_job or {}).get("combinations", []) or [])]
                    if 'parameters' not in S or not S['parameters']:
                        S['parameters'] = icph_params
                    S['combinations'] = icph_combs

                    # collect nuisance names
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)

                elif styp == "lnN":
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)
                else:
                    # future binned syst types
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                        if S.get("floating", False):
                            floating.add(nm)

    # Binning consistency check across regions:
    for R in binned:
        rid = R.get('id', '?')
        ref_names, ref_edges = None, None
        offenders = []

        for C in (R.get('classes') or []):
            # POI (ICH)
            poi = (C.get('POI') or {})
            pred_poi = poi.get('predictor')
            if pred_poi is not None:
                names, edges = _pred_binning_tuple(pred_poi)
                if ref_names is None:
                    ref_names, ref_edges = names, edges
                elif not helpers._binning_equal(names, edges, ref_names, ref_edges):
                    offenders.append((f"{rid}/{C.get('id','?')}/POI", names, edges))

            # systematics (ICPH)
            for S in (C.get('systematics') or []):
                if S.get('type') != 'icph':
                    continue
                pred_sys = S.get('predictor')
                if pred_sys is None:
                    continue
                names, edges = _pred_binning_tuple(pred_sys)
                if ref_names is None:
                    ref_names, ref_edges = names, edges
                elif not helpers._binning_equal(names, edges, ref_names, ref_edges):
                    offenders.append((f"{rid}/{C.get('id','?')}/sys[{S.get('id','?')}]", names, edges))

        if offenders:
            msg = [f"[binned binning] Inconsistent binning within region '{rid}'. "
                   f"Reference = {ref_names} / {[e.tolist() for e in ref_edges]}"]
            for tag, nms, eds in offenders:
                msg.append(f"  - {tag}: {nms} / {[e.tolist() for e in eds]}")
            raise RuntimeError("\n".join(msg))

    # Keep deterministic order
    pois_list = sorted(all_pois)
    nuis_list = sorted(all_nuis)
    floating_list = sorted(floating)

    # Return both sections enriched
    return {'regions': regions, 'binned': binned, 'pois': pois_list, 'nuisances': nuis_list, 'floating': floating_list}

def build_hypothesis_from_likelihood(like_info, *, name=None,
                                     poi_init=0.0, nuis_init=0.0,
                                     ):
    """
    Convenience: construct a Hypothesis from load_likelihood(...) output.
    Includes parameters discovered in BOTH unbinned and binned sections.

    Heuristics:
      - POIs are marked isPOI=True if name starts with 'c'.
    """
    pois = like_info.get('pois', []) or []
    nuis = like_info.get('nuisances', []) or []
    floating = like_info.get('floating', []) or []

    params = []
    for nm in pois:
        is_wc = nm.startswith('c')
        params.append(ModelParameter(name=nm, val=poi_init, isPOI=True, isPenalized=False))
    for nm in nuis:
        params.append(ModelParameter(
            name=nm, val=nuis_init, isPOI=False,
            isPenalized=bool(nm not in floating)
        ))
    return Hypothesis(parameters=params, name=name or "from_yaml")


try:
    from numba import njit, prange
    _NUMBA = True
except Exception:
    _NUMBA = False

if _HAS_AUTOGRAD:
    _NUMBA = False

if _NUMBA:
    @njit(parallel=True, fastmath=True)
    def _weighted_sum_log1p_minus_x(x: np.ndarray, w: np.ndarray) -> float:
        n = x.size
        s = 0.0
        for i in prange(n):
            xi = x[i]
            wi = w[i]
            if -1.0 < xi < 1e-4:
                x2 = xi * xi
                # Taylor expansion of log1p(x) - x
                t = 0.5 + xi * (-1.0/3.0 + xi * (1.0/4.0 + xi * (-1.0/5.0 + xi * (1.0/6.0))))
                y = -x2 * t
            else:
                y = math.log1p(xi) - xi
            s += wi * y
        return s
else:
    def _weighted_sum_log1p_minus_x(x: np.ndarray, w: np.ndarray) -> float:
        x = np.array(x)
        w = np.array(w)
        small = (np.abs(x) < 1e-4) & (x > -1.0)
        s_small = x * x * (0.5 + x * (-1.0 / 3.0 + x * (1.0 / 4.0 + x * (-1.0 / 5.0 + x * (1.0 / 6.0)))))
        y_small = -s_small
        y_big = np.log1p(x) - x
        y = np.where(small, y_small, y_big)
        return np.sum(w * y, dtype=np.float64)


def expand_pois_linear_quadratic(poi_names: List[str], poi_values: Dict[str, float],
                                  reference_point: Dict[str, float]) -> np.ndarray:
    """Build the c_A vector from the Taylor variable t = c - r, r = reference_point.

    `reference_point` is required, not defaulted: a default of zero would silently
    return offset-space results at any call site that forgets to pass it, and that
    result looks like a normal fit. Pass {} explicitly for a POI with no rebase (e.g.
    the PDF POIs, expanded around zero).
    """
    N = len(poi_names)
    t = np.array([poi_values.get(n, 0.0) - reference_point.get(n, 0.0) for n in poi_names])
    quads = []

    #FIXME careful here, double sum
    # This is the logic:
    # BIT predicts and works with *derivatives*, so R = 1 + t_A R_A = 1 + Sum_a ta Ra + 1/2 Sum_{a, b} ta tb Ra Rb (Taylor expansion)
    # Now, the double sum is slow so we write
    # R = 1 + Sum_a ta Ra + Sum_{a, b>=a} factor ta tb Ra Rb where factor = 1/2 if a=b (same factor as before) but factor=1 if b>a (counting twice)
    # My silicon friend didn't see that. (For the PNN I rather work with unique ordered sequences, so no prefactor)
    for i in range(N):
        for j in range(i,N):
            quads.append((0.5 if i==j else 1) * t[i] * t[j])  # 1/2 t_i t_j
    return np.concatenate([t, np.array(quads)], axis=0) if quads else t


def nuis_to_A_vector(param_names: List[str], combinations: List[Tuple[str, ...]], values: Dict[str, float]) -> np.ndarray:
    if not combinations:
        return np.zeros(0)
    out = []
    for comb in combinations:
        v = 1.0
        for p in comb:
            v *= values.get(p, 0.0)
        out.append(v)
    return np.array(out)

def pois_jacobian_linear_quadratic(poi_names: List[str],
                                    poi_values: Dict[str, float],
                                    reference_point: Dict[str, float]) -> np.ndarray:
    """
    Build the Jacobian C_{Aa} = ∂c_A/∂c_a for the same A-basis and ordering
    as `expand_pois_linear_quadratic`.

    `reference_point` (r) is required for the same reason as in
    `expand_pois_linear_quadratic`; it shifts the Taylor variable t = c - r that the
    quadratic block is built from, and does not affect the linear block since r is
    constant in c.

    Shape:
      - n_par = len(poi_names)
      - nA    = n_par + n_par*(n_par+1)//2
      => C has shape (nA, n_par), rows = A, columns = a.
    """
    N = len(poi_names)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Taylor variable in the same order as in expand_pois_linear_quadratic
    t = np.array([poi_values.get(n, 0.0) - reference_point.get(n, 0.0) for n in poi_names],
                 dtype=np.float64)

    n_quads = N * (N + 1) // 2
    nA = N + n_quads

    C = np.zeros((nA, N), dtype=np.float64)

    # Linear part: c_A = t_a  ->  ∂c_A/∂c_a = δ_{Aa} (∂t_a/∂c_a = 1, r constant)
    # A = 0..N-1 corresponds to the linear pieces
    for a in range(N):
        C[a, a] = 1.0

    # Quadratic part: same ordering / convention as expand_pois_linear_quadratic
    # A runs from N onward, loops (i,j) with j>=i.
    row = N
    for i in range(N):
        for j in range(i, N):
            if i == j:
                # c_A = 0.5 * t_i^2 -> ∂/∂c_i = t_i
                C[row, i] = t[i]
            else:
                # c_A = t_i * t_j -> ∂/∂c_i = t_j, ∂/∂c_j = t_i
                C[row, i] = t[j]
                C[row, j] = t[i]
            row += 1

    # Sanity: row should end at nA
    # assert row == nA
    return C

def nuis_jacobian_A(param_names: List[str],
                     combinations: List[Tuple[str, ...]],
                     values: Dict[str, float]) -> np.ndarray:
    """
    Build the Jacobian N_{Ba} = ∂ν_B/∂ν_a for the nuisance A-basis used in
    `nuis_to_A_vector`.

    - param_names: column ordering for ν_a
    - combinations: list of tuples describing each monomial row B:
        comb = ('nu1',)            -> linear
        comb = ('nu1', 'nu2')      -> quadratic cross term
        comb = ('nu1', 'nu1')      -> quadratic diagonal, etc.
    - values: mapping name -> ν_name

    Shape:
      - n_par = len(param_names)
      - nB    = len(combinations)
      => N has shape (nB, n_par).
    """
    n_par = len(param_names)
    nB = len(combinations)
    if nB == 0 or n_par == 0:
        return np.zeros((nB, n_par), dtype=np.float64)

    N_mat = np.zeros((nB, n_par), dtype=np.float64)

    # Loop over rows (each monomial ν_B)
    for k, comb in enumerate(combinations):
        # Fetch the ν values for each factor in the comb
        vals = [values.get(p, 0.0) for p in comb]

        # For each parameter column a, compute ∂ν_B/∂ν_a
        for a_idx, pname in enumerate(param_names):
            # Find all positions where this pname appears in the comb
            indices = [idx for idx, p in enumerate(comb) if p == pname]
            if not indices:
                continue  # derivative is zero if pname not in comb

            # General case: handle possible multiple occurrences of pname
            dval = 0.0
            m = len(comb)
            for idx in indices:
                # product over all positions except idx
                prod = 1.0
                for j in range(m):
                    if j == idx:
                        continue
                    prod *= vals[j]
                dval += prod

            N_mat[k, a_idx] = dval

    return N_mat


def _predict_classifier(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        P = model.predict(X)
    else:
        raise RuntimeError("Classifier predictor lacks predict method.")
    P = np.asarray(P)
    if P.ndim != 2:
        raise RuntimeError("Classifier output must be (N, n_classes).")
    return P.astype(np.float64, copy=False)


def predict_bit_ratio(model, X: np.ndarray, max_n_tree: Optional[int] = None) -> np.ndarray:

    if max_n_tree is not None and max_n_tree > len(model.trees):
        raise ValueError(f"Using model with {len(model.trees)}, but asking prediction from larger number of trees {max_n_tree}. Double-check.")

    if hasattr(model, "predict"):
        Y = model.predict(X, max_n_tree)
    #elif hasattr(model, "predict_A"):
    #    Y = model.predict_A(X)
    else:
        raise RuntimeError("BIT predictor lacks predict.")
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y = Y.astype(np.float64, copy=False)

    # Applied here rather than in BIT.predict(): the calibration is derived from the
    # raw predictions, so calibrating inside predict() would feed the derivation its
    # own output. Both callers of this function run before the fit, so the cached R
    # matrix that N2LL.__call__ contracts is already calibrated.
    calibration = getattr(model, "binned_calibration", None)
    if calibration is not None:
        Y = apply_binned_calibration(Y, model.derivatives, calibration)
    return Y


def predict_pnn_deltaA(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "deltaA"):
        return np.asarray(model.deltaA(X), dtype=np.float64)
    else:
        raise RuntimeError("PNN predictor lacks deltaA(X).")


def class_index(classes: List[Dict[str, Any]], cid: str) -> int:
    for i, C in enumerate(classes):
        if C['id'] == cid:
            return i
    raise RuntimeError(f"class id '{cid}' not in classes order")

class N2LL:
    """
    Negative two log-likelihood (Asimov-only).

    Workflow:
      1) build_cache()  # stream inference and write per-(region,class) HDF5 + JSON meta
      2) prepare_runtime()  # open HDF5s, read meta, pre-resolve shapes and mappings (no params)
      3) __call__(hypothesis)  # blazing fast: no disk I/O, only HDF5 slicing + math
      4) close()  # optional: close all HDF5 files
    """
    def __init__(self,
                 likelihood: Dict[str, Any],
                 factory,
                 cache_subdir: str = "caches",
                 cache_root: Optional[str] = None,
                 overwrite: bool = False,
                 eval_chunk_size: int = 2000000_000):
        import importlib, os
        self.lk = likelihood
        self.regions = list(likelihood.get('regions', []))
        self.factory = factory 
        #self.module_samples = module_samples
        #self.samples_mod = importlib.import_module(module_samples)
        #self.default_features = default_features
        self.cache_subdir = cache_subdir
        self.overwrite = overwrite
        self.eval_chunk_size = int(eval_chunk_size)

        # ===== Binned likelihood support =====
        self.binned = list(likelihood.get('binned', []) or [])

        # Per-binned-region runtime state
        self._binned_regions_ids: list[str] = []
        self._binned_classes_by_region: dict[str, list[dict]] = {}   # rid -> [class dicts]
        self._binned_unroll: dict[str, dict] = {}  # rid -> { 'shape':(nb1[,nb2]), 'flat_bins':[( (xlo,xhi), (ylo,yhi)|None )], 'axes':[...], 'edges':[...]}
        self._binned_lambda0: dict[str, np.ndarray] = {}  # rid -> (Nflat,) nominal λ at (0,0)
        self._binned_asimov_lambda: dict[str, np.ndarray] = {}  # rid -> (Nflat,) λ' if setAsimov used

        # ----- Asimov (off-nominal) support -----
        self._asimov_hyp = None                       # Hypothesis used for Asimov (c', ν')
        self._asimov_active = False                   # quick guard: any nonzero param?
        self._asimov_T: Dict[str, list[np.ndarray]] = {}   # region_id -> list of T'(chunk) arrays

        # where caches live
        try:
            import common.user as user
            base_dir = user.cache_directory
        except Exception:
            base_dir = "./caches"
        self.cache_root = cache_root or os.path.join( base_dir, cache_subdir )
        os.makedirs(self.cache_root, exist_ok=True)

        # in-memory pointers
        self._poi_order: Dict[Tuple[str, str], List[str]] = {}         # (rid,cid) -> POI names order for R_A
        self._poi_reference: Dict[Tuple[str, str], Dict[str, float]] = {}  # (rid,cid) -> reference point r for c-r
        self._cache_paths: Dict[Tuple[str, str], Tuple[str, str]] = {} # (rid,cid) -> (h5_path, meta_path)

        # opened runtime state (filled by prepare_runtime)
        self._h5: Dict[Tuple[str, str], "h5py.File"] = {}
        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._N_region: Dict[str, int] = {}                            # region id -> number of events
        self._class_ids_by_region: Dict[str, List[str]] = {}           # region id -> [class ids in order]
        self._lnN_by_class: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self._rate_shift_by_class: Dict[Tuple[str, str], Optional[str]] = {} 

        # attach predictors from likelihood (already set by yaml_loader.load_likelihood)
        self._prepare_structure()

        # flow control
        self._runtime_prepared      = False
        self._asimov_hypothesis_set = False
        self._observation_set       = False

        # ---- external evaluation in-memory cache (per region) ----
        # rid -> {
        #   'feature_names': tuple[str, ...],
        #   'n_features': int,
        #   'by_class': { cid: {'g': (N,), 'R': (N,nA), 'Delta::<sid>': (N,nB), ...} },
        # }
        self._ext_eval_cache: dict[str, dict] = {}

    # --------- structure & sanity ---------
    def _prepare_structure(self):
        for R in self.regions:
            rid = R['id']
            clf_pred = (R.get('classifier') or {}).get('predictor', None)
            R['_classifier_predictor'] = clf_pred

            # Asimov sample list
            asimov_list = (R.get('classifier') or {}).get('asimov', [])
            if not isinstance(asimov_list, list) or not all(isinstance(s, str) for s in asimov_list):
                raise RuntimeError(f"[N2LL] Region '{rid}' needs classifier.asimov: [sample, ...]")
            for sname in asimov_list:
                if not self.factory.get( sname ): #hasattr(self.samples_mod, sname):
                    raise RuntimeError(f"[N2LL] Asimov sample '{sname}' not found.")
            R['_asimov_samples'] = asimov_list

            for C in R.get('classes', []):
                cid = C['id']
                key = (rid, cid)

                poi = C.get('POI', {}) or {}
                poi_type = poi.get('type', None)
                poi_pred = poi.get('predictor', None)

                # --- rate_shift: a single additive POI term per class ---
                rate_shift_param = None
                if poi_type == "rate_shift":
                    rs_params = list(poi.get("parameters", []) or [])
                    if len(rs_params) != 1:
                        raise RuntimeError(f"[N2LL] 'rate_shift' POI must have exactly one parameter for {rid}/{cid}.")
                    rate_shift_param = rs_params[0]
                    poi_names = []  # IMPORTANT: no BIT coefficients in this case
                else:
                    # BIT (or empty/None): keep the usual POI parameter list for c_A ⋅ R_A
                    poi_names = list(poi.get('parameters', []) or [])

                    # Only complain about predictor if we actually asked for BIT
                    if poi_type == "bit" and poi_pred is None:
                        print(f"[N2LL] No BIT predictor for {rid}/{cid}")
                    if poi_type == "bit" and not poi_names:
                        print(f"[N2LL] No POI parameter names for {rid}/{cid}")

                    # The BIT alphabetizes its derivative columns (GpuMultiNode.py:293-297),
                    # so its R_A order is sorted, not the config order. c_A is built from
                    # this list and contracted positionally against those columns, so it
                    # must be sorted too -- otherwise the operators are silently permuted.
                    poi_names = sorted(poi_names)

                self._poi_order[key] = poi_names
                # The point the BIT was trained around; {} for rate_shift/PDF POIs, which
                # expand around zero. Reading it off the predictor (not GENERATION_POINT)
                # makes a training/fit mismatch impossible by construction.
                self._poi_reference[key] = dict(getattr(poi_pred, "expansion_point", {}) or {})
                self._rate_shift_by_class[key] = rate_shift_param


                # keep only PNN systematics here
                sys_list = []
                for S in (C.get('systematics') or []):
                    if S.get('type') != 'pnn':  # lnN etc. ignored for this likelihood core
                        continue
                    if S.get('predictor', None) is None:
                        raise RuntimeError(f"[N2LL] Missing PNN predictor for {rid}/{cid} systematic '{S.get('id','?')}'")
                    sys_list.append(S)
                C['_pnn_systs'] = sys_list

                # --- stash lnN normalization pieces: (nuisance_name, log1p(alpha)) ---
                lnN_terms = []
                for S in (C.get('systematics') or []):
                    if S.get('type') == 'lnN':
                        alpha = float(S.get('value', 0.0))
                        # assume a single parameter name (usual pattern); if multiple, raise error
                        if len(S.get('parameters'))!=1: 
                            raise RuntimeError("Problem in this lnN uncertainty: %r"%S)
                        lnN_terms.append((S.get('parameters')[0], math.log1p(alpha)))
                self._lnN_by_class[key] = lnN_terms


    # --------- helpers: paths ----------
    def _region_cache_dir(self, region_id: str) -> str:
        import os
        d = os.path.join(
            self.cache_root, 
            region_id,
            ('shuffle_'+'_'.join(self.shuffle_features)) if (hasattr(self, "shuffle_features") and self.shuffle_features is not None) else ""
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _paths_for(self, rid: str, cid: str) -> Tuple[str, str]:
        import os, json
        h5_path = os.path.join(self._region_cache_dir(rid), f"{cid}.h5")
        meta_path = os.path.join(self._region_cache_dir(rid), f"{cid}.json")
        self._cache_paths[(rid, cid)] = (h5_path, meta_path)
        return h5_path, meta_path

    # --------- batched iter over asimov ----------
    def _iter_asimov_batches(self, region) -> Iterable[Tuple[List[str], np.ndarray, np.ndarray]]:
        """
        Yields (feat_names, X, w0) per shard over all Asimov samples in a region.
        Sets n_split=100 temporarily on RDataLoader-like objects.
        """
        feat_names_ref = None
        print(region['_asimov_samples'])
        for sname in region['_asimov_samples']:
            #L = getattr(self.samples_mod, sname)
            L = self.factory.get(sname)
            # enforce consistent features
            feat_names = list(getattr(L, "feature_names", []) or [])
            if feat_names_ref is None:
                feat_names_ref = feat_names
            elif feat_names != feat_names_ref:
                raise RuntimeError(f"[N2LL] Feature mismatch across Asimov samples in region '{region['id']}'")

            # set n_split=100 if available (temporary)
            reset_split = False
            old_split = None
            if hasattr(L, "n_split"):
                reset_split = True
                old_split = L.n_split
                try:
                    L.n_split = 100
                except Exception:
                    reset_split = False

            n_shards = len(getattr(L, "base", L))
            for shard in range(n_shards):
                X, w0 = L.materialize(shard=shard, what="fw", n=None)
                X = np.asarray(X, dtype=np.float64)      # allow copy if needed (NumPy 2.x safe)
                w0 = np.asarray(w0, dtype=np.float64)    # allow copy if needed
                if X is not None and len(X) > 0:
                    yield feat_names, X, w0

            if reset_split:
                try:
                    L.n_split = old_split
                except Exception:
                    pass

    # --------- dataset appends (HDF5) ---------
    @staticmethod
    def _append_1d(dset, arr: np.ndarray):
        n_old = dset.shape[0]
        n_add = arr.shape[0]
        dset.resize((n_old + n_add,))
        dset[n_old:n_old+n_add] = arr

    @staticmethod
    def _append_2d(dset, arr: np.ndarray):
        n_old = dset.shape[0]
        n_add = arr.shape[0]
        dset.resize((n_old + n_add, dset.shape[1]))
        dset[n_old:n_old+n_add, :] = arr

    @staticmethod
    def make_column_mask(all_features, wanted_features):
        pos = {f: i for i, f in enumerate(all_features)}
        missing = [f for f in wanted_features if f not in pos]
        if missing:
            raise KeyError(f"Features not found in source: {missing}")
        mask = np.zeros(len(all_features), dtype=bool)
        for f in wanted_features:
            mask[pos[f]] = True
        return mask

    # --------- cache builder (HDF5) ----------
    def build_cache(self):
        """
        Stream inference shard-by-shard, and write per-(region,class) HDF5 files + JSON meta.
        If a cache file exists and overwrite=False, we *skip building* that class.
        """
        import os, json, h5py, numpy as np, math
        from tqdm import tqdm

        for R in self.regions:
            rid = R['id']
            classes = list(R.get('classes', []))
            n_proc = len(classes)
            clf = R.get('_classifier_predictor', None)

            # Decide per-class overwrite
            needs = {}
            for C in classes:
                cid = C['id']
                h5_path, meta_path = self._paths_for(rid, cid)
                exists = os.path.exists(h5_path) and os.path.exists(meta_path)
                if exists and not self.overwrite:
                    print(f"[N2LL] Cache file {h5_path} found. Loading (skip rebuild).")
                    needs[cid] = False
                else:
                    if exists and self.overwrite:
                        print(f"[N2LL] Cache file {h5_path} found. Overwriting.")
                    else:
                        print(f"[N2LL] No cache for (region={rid}, class={cid}). Building.")
                    needs[cid] = True

            # If nothing to do -> continue (we'll open in prepare_runtime)
            if not any(needs.values()):
                continue

            # Prepare writers for needed classes
            writers = {}
            for C in classes:
                cid = C['id']
                if not needs[cid]:
                    continue
                h5_path, meta_path = self._paths_for(rid, cid)
                print(f"Cache file for {cid}: {h5_path}")
                print(f"Meta  file for {cid}: {meta_path}")

                # create HDF5 and datasets with resizable first dim
                f = h5py.File(h5_path, "w")
                writers[cid] = {
                    "file": f,
                    "w0": f.create_dataset("w0", (0,), maxshape=(None,), dtype="f8", chunks=True),
                    "g":  f.create_dataset("g",  (0,), maxshape=(None,), dtype="f8", chunks=True),
                    # R and Δ shapes known only after first batch
                    "R":  None,
                    "Delta": {},   # id -> dset
                    "meta": {"delta_groups": []}
                }

            # Stream batches once and write everything
            first_batch_shapes: Dict[str, Dict[str, int]] = {}  # cid -> {"nA":.., "nB::<sysid>":..}
            shard_counter = 0

            # Pre-count shards for UI
            total_shards = 0
            for sname in R['_asimov_samples']:
                #L = getattr(self.samples_mod, sname)
                L = self.factory.get(sname)
                total_shards += len(getattr(L, "base", L))

            # prepare column selection mask
            if clf is not None:
                clf.column_mask = self.make_column_mask(L.feature_names, clf.feature_names)
            for C in classes:
                poi_pred = (C.get('POI') or {}).get('predictor')
                if poi_pred:
                    C['POI']['column_mask'] = self.make_column_mask(L.feature_names, poi_pred.feature_names)
                for S in C['_pnn_systs']:
                    # I forgot to store the feature_names in the PNN class, so let's look it up from the job:
                    S['column_mask'] = self.make_column_mask(L.feature_names, S['predictor'].feature_names)

            with tqdm(total=total_shards, desc=f"[N2LL] cache {rid}", unit="shard", leave=False) as pbar:
                for feat_names, X, w0 in self._iter_asimov_batches(R):
                   
                    if hasattr( self, "shuffle_features" ) and ( self.shuffle_features is not None ):
                        for s_feature in self.shuffle_features:
                            if s_feature not in feat_names:
                                raise RuntimeError( f"Can't shuffle {s_feature} because that's not in {feat_names}" )
                            i_f = feat_names.index(s_feature)
                            np.random.shuffle( X[:,i_f] )

                    Nb = len(X)
                    if Nb == 0:
                        pbar.update(1)
                        continue

                    # Classifier probabilities for this batch (or ones)
                    if clf is None or n_proc <= 1:
                        G = np.ones((Nb, n_proc), dtype=np.float64)
                    else:
                        G = _predict_classifier(clf, X[:, clf.column_mask])  # (Nb, n_proc)
                        if G.shape[1] != n_proc:
                            raise RuntimeError(f"[N2LL] Classifier outputs {G.shape[1]} != {n_proc} classes in region '{rid}'")

                    # For each class that needs building: compute and append
                    for C in classes:
                        cid = C['id']
                        if not needs[cid]:
                            continue
                        p_index = class_index(classes, cid)
                        writer = writers[cid]
                        f = writer["file"]

                        # w0 and g slice
                        self._append_1d(writer["w0"], w0)
                        self._append_1d(writer["g"],  G[:, p_index].astype(np.float64, copy=False))

                        ## BIT R_A
                        #poi_pred = (C.get('POI') or {}).get('predictor')
                        #R_A = predict_bit_ratio(poi_pred, X[:, C['POI']['column_mask']])  # (Nb, nA)

                        #if writer["R"] is None:
                        #    nA = R_A.shape[1]
                        #    writer["R"] = f.create_dataset("R", (0, nA), maxshape=(None, nA), dtype="f8", chunks=True)
                        #    first_batch_shapes.setdefault(cid, {})["nA"] = nA
                        #self._append_2d(writer["R"], R_A)

                        # BIT R_A (could be absent)
                        poi_pred = (C.get('POI') or {}).get('predictor')
                        if poi_pred is None:
                            # No BIT: R_A has width 0
                            R_A = np.empty((Nb, 0), dtype=np.float64)
                        else:
                            # Use columns only if a mask was set
                            col_mask = (C.get('POI') or {}).get('column_mask', None)
                            X_in = X[:, col_mask] if col_mask is not None else X
                            max_n_tree = (C.get('POI') or {}).get('max_n_tree', None)
                            R_A = predict_bit_ratio(poi_pred, X_in, max_n_tree)  # (Nb, nA)

                        if writer["R"] is None:
                            nA = R_A.shape[1]
                            writer["R"] = f.create_dataset("R", (0, nA), maxshape=(None, nA), dtype="f8", chunks=True)
                            first_batch_shapes.setdefault(cid, {})["nA"] = nA
                        self._append_2d(writer["R"], R_A)

                        # PNN Δ groups
                        for S in C['_pnn_systs']:
                            sid = S['id']
                            pnn = S['predictor']
                            dA = predict_pnn_deltaA(pnn, X[:, S['column_mask']])  # (Nb, nB)
                            if sid not in writer["Delta"]:
                                nB = dA.shape[1]
                                dset_name = f"Delta::{sid}"
                                writer["Delta"][sid] = f.create_dataset(dset_name, (0, nB), maxshape=(None, nB), dtype="f8", chunks=True)
                                # record meta for this sys id once
                                writer["meta"]["delta_groups"].append({
                                    "id": sid,
                                    "params": list(S.get("parameters", []) or []),
                                    "combs":  [list(t) for t in (S.get("combinations", []) or [])],
                                    "dset": dset_name,
                                })
                                first_batch_shapes.setdefault(cid, {})[f"nB::{sid}"] = nB
                            self._append_2d(writer["Delta"][sid], dA)

                    shard_counter += 1
                    pbar.update(1)

            # write meta and close writers
            for C in classes:
                cid = C['id']
                if not needs[cid]:
                    continue
                h5_path, meta_path = self._paths_for(rid, cid)
                w = writers[cid]
                # finalize meta with shapes (nice to have)
                meta_out = w["meta"]
                with open(meta_path, "w") as f:
                    json.dump(meta_out, f, indent=2)
                w["file"].flush(); w["file"].close()
                print(f"[N2LL] Written cache HDF5: {h5_path}")
                print(f"[N2LL] Written meta JSON: {meta_path}")

    def _prepare_binned_structure(self):
        """Resolve binned regions: attach predictors, unroll bins, and cache λ0."""
        if not self.binned:
            return

        print("\n[N2LL.prepare_runtime] Preparing BINNED regions…")
        for R in self.binned:
            rid = R['id']
            self._binned_regions_ids.append(rid)

            # classes and predictors
            classes = []
            for C in R.get('classes', []) or []:
                cid = C['id']
                # ICH (POI)
                poi = C.get('POI', {}) or {}
                ich = poi.get('predictor', None)
                if ich is None:
                    raise RuntimeError(f"[binned] Missing ICH predictor for {rid}/{cid}")
                poi_params = list(poi.get('parameters', []) or [])
                # ICPH groups
                icph_groups = []
                for S in (C.get('systematics', []) or []):
                    if S.get('type') != 'icph':
                        # still stash lnN in shared map
                        if S.get('type') == 'lnN':
                            alpha = float(S.get('value', 0.0))
                            if len(S.get('parameters'))!=1:
                                raise RuntimeError("Problem in this lnN uncertainty: %r"%S)
                            self._lnN_by_class[(rid, cid)] = self._lnN_by_class.get((rid, cid), [])
                            self._lnN_by_class[(rid, cid)].append((S['parameters'][0], math.log1p(alpha)))
                        continue
                    pred = S.get('predictor', None)
                    if pred is None:
                        raise RuntimeError(f"[binned] Missing ICPH predictor for {rid}/{cid}/{S.get('id','?')}")
                    # stash a meta dict we'll enrich with deltas as numpy arrays for fast math
                    gm = {
                        'id': S['id'],
                        'params': list(S.get('parameters', []) or []),
                        'combs':  [list(t) for t in (getattr(pred, "combinations", []) or [])],
                        # Always copy so shape_only normalization below does not mutate the loaded surrogate
                        '_deltas': np.array(pred.deltas, dtype=np.float64),
                        '_obj': pred,
                        'shape_only': bool(S.get('shape_only', False)),
                    }
                    icph_groups.append({'_meta': gm})

                classes.append({'id': cid,
                                '_ich': ich,
                                '_poi_params': poi_params,
                                '_icph_systs': icph_groups})

                # keep POI order so we can build c-vectors
                self._poi_order[(rid, cid)] = poi_params
                # ICH POIs are PDF coefficients, expanded around zero; {} is correct here.
                self._poi_reference[(rid, cid)] = dict(getattr(ich, "expansion_point", {}) or {})

            self._binned_classes_by_region[rid] = classes

            # Unroll using the ICH (all classes have same binning)
            un = self._unroll_bins_from_ich(classes[0]['_ich'])
            self._binned_unroll[rid] = un
            Nflat = len(un['flat_bins'])

            # Cache nominal λ0 at (c=0, ν=0)
            # Build a zero-POI hypothesis view:
            class _Zero:
                def __init__(self, names): self.POIs=[type('P',(),{'name':n,'val':0.0})() for n in names]
                def __contains__(self, k): return False
            # assemble per-class zero vector in compute; simpler: pass zeros to ICH
            lam0 = np.zeros(Nflat, dtype=np.float64)
            for C in classes:
                ich = C['_ich']
                cvec0 = np.zeros(len(C['_poi_params']), dtype=np.float64)
                sig0 = ich.predict(cvec0)  # (nb1,) or (nb1,nb2)
                lam0 += sig0.reshape(-1)   # ν=0 → exp(0)=1; lnN at ν=0 adds nothing
            self._binned_lambda0[rid] = lam0

            # Debug print
            print(f"[binned] Region '{rid}': bins={Nflat}, axes={un['axes']}, shape={un['shape']}")

    def prepare_runtime(self):
        """
        Open all HDF5 cache files and load metadata once.
        Also print column mappings and first 5 rows for sanity checks.
        """
        import os, json, h5py, numpy as np

        # small helpers to build readable column names
        def _poi_A_names(poi_names):
            names = []
            # linear
            for a in poi_names:
                names.append(a)
            # symmetric quadratic (a<=b)
            for i, a in enumerate(poi_names):
                for j in range(i, len(poi_names)):
                    b = poi_names[j]
                    names.append(f"{a}*{b}")
            return names

        def _nuis_A_names(params, combs):
            if not combs:
                return []
            out = []
            for cmb in combs:
                if len(cmb) == 1:
                    out.append(cmb[0])
                else:
                    out.append("*".join(cmb))
            return out

        # close any previously opened files
        self.close()

        for R in self.regions:
            rid = R['id']
            classes = list(R.get('classes', []))
            self._class_ids_by_region[rid] = [C['id'] for C in classes if isinstance(C, dict) and 'id' in C]
            N_region = None

            print(f"\n[N2LL.prepare_runtime] Region '{rid}': opening caches and validating…")

            for C in classes:
                cid = C['id']
                h5_path, meta_path = self._paths_for(rid, cid)

                if not (os.path.exists(h5_path) and os.path.exists(meta_path)):
                    raise RuntimeError(f"[N2LL] Missing cache artifacts for ({rid},{cid}). Build them first.")

                # open HDF5, read meta JSON
                f = h5py.File(h5_path, "r")
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)

                # length consistency within region
                N = f['w0'].shape[0]
                if N_region is None:
                    N_region = N
                elif N != N_region:
                    f.close()
                    raise RuntimeError(f"[N2LL] Inconsistent length for ({rid},{cid}) vs region={rid} first class")

                self._h5[(rid, cid)] = dict([(key, f[key][:]) for key in f])
                self._meta[(rid, cid)] = meta

                # ---- Debug prints: columns + first 5 rows ----
                nshow = min(5, N)
                print(f"[N2LL.prepare_runtime]  Class '{cid}': N={N}")
                # w0
                w0_head = np.array(f['w0'][0:nshow])
                print(f"  - w0 shape={f['w0'].shape}  head[0:{nshow}]: {w0_head}")

                # g
                g_head = np.array(f['g'][0:nshow])
                print(f"  - g  shape={f['g'].shape}   head[0:{nshow}]: {g_head}")

                # R with POI A-basis headers
                poi_names = self._poi_order[(rid, cid)]
                A_names = _poi_A_names(poi_names)
                R_dset = f['R']
                print(f"  - R  shape={R_dset.shape}")
                print(f"    R columns (|A|={len(A_names)}): {A_names}")
                R_head = np.array(R_dset[0:nshow, :])
                # print rows neatly
                for i in range(nshow):
                    print(f"    R[{i}]: {R_head[i, :]}")

                # Delta groups
                dg = meta.get("delta_groups", [])
                if dg:
                    print("  - Δ groups:")
                for gm in dg:
                    dname = gm.get("dset", f"Delta::{gm['id']}")
                    params = list(gm.get("params", gm.get("parameters", [])) or [])
                    combs  = [tuple(c) for c in gm.get("combs", gm.get("combinations", [])) or []]
                    B_names = _nuis_A_names(params, combs)
                    D = f[dname]
                    print(f"    • id='{gm['id']}'  dset='{dname}'  shape={D.shape}")
                    print(f"      Δ columns (|B|={len(B_names)}): {B_names}")
                    D_head = np.array(D[0:nshow, :])
                    for i in range(nshow):
                        print(f"      Δ[{i}]: {D_head[i, :]}")

            self._N_region[rid] = N_region or 0

        # Finally, prepare binned regions (attach predictors, unroll, cache λ0)
        self._prepare_binned_structure()
        self._runtime_prepared = True

    def close(self):
        """Close all opened HDF5 files."""
        for f in list(self._h5.values()):
            try:
                f.close()
            except Exception:
                pass
        self._h5.clear()

    # ---------- BIN UTILS ----------
    @staticmethod
    def _unroll_bins_from_ich(ich) -> dict:
        """
        Build an unrolling map for an ICH/ICPH object:
          returns {'shape': (nb1,) or (nb1,nb2),
                   'flat_bins': [ ((xlo,xhi), None) ] or [((xlo,xhi),(ylo,yhi))],
                   'axes': [...],
                   'edges': [edges1] or [edges1, edges2] }
        """
        axes  = list(getattr(ich, "axis_names", []) or [])
        edges = [np.asarray(be, dtype=float) for be in getattr(ich, "bin_edges", []) or []]
        if not edges:
            raise RuntimeError("Binned region requires ICH/ICPH with bin_edges.")
        if len(edges) == 1:
            nb1 = len(edges[0]) - 1
            flat = [((edges[0][i], edges[0][i+1]), None) for i in range(nb1)]
            return {'shape': (nb1,), 'flat_bins': flat, 'axes': axes, 'edges': edges}
        elif len(edges) == 2:
            nb1 = len(edges[0]) - 1
            nb2 = len(edges[1]) - 1
            flat = []
            for i in range(nb1):
                for j in range(nb2):
                    flat.append(((edges[0][i], edges[0][i+1]), (edges[1][j], edges[1][j+1])))
            return {'shape': (nb1, nb2), 'flat_bins': flat, 'axes': axes, 'edges': edges}
        else:
            raise RuntimeError("Only 1D/2D binning supported.")

    @staticmethod
    def _safe_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        num = np.maximum(np.asarray(num, dtype=np.float64), eps)
        den = np.maximum(np.asarray(den, dtype=np.float64), eps)
        return np.log(num) - np.log(den)

    # ---------- Binned A-basis assembly ----------
    def _assemble_c_vector_for_ich(self, rid: str, hypothesis, cid: str) -> np.ndarray:
        """
        ICH expects the plain c-vector in the stored parameter order.
        Fail hard if a required POI is missing.
        """
        poi_names = self._poi_order.get((rid, cid), None)
        if poi_names is None:
            raise RuntimeError(f"[binned] Missing POI names order for ({rid}/{cid}).")

        h = getattr(hypothesis, "_base", hypothesis)

        missing = [name for name in poi_names if name not in h]
        if missing:
            raise RuntimeError(
                f"[binned] Missing POIs in hypothesis for ({rid}/{cid}): {missing}. "
                f"Expected order: {poi_names}"
            )

        return np.array([h[name].val for name in poi_names], dtype=np.float64)

    def _assemble_nuA_groups_binned(self, rid: str, hypothesis) -> dict[str, list[tuple[dict, np.ndarray]]]:
        """
        ν_A vector per ICPh group, per class, for a given hypothesis.
        """
        nu_vals = {p.name: p.val for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        out: dict[str, list[tuple[dict, np.ndarray]]] = {}
        for C in self._binned_classes_by_region.get(rid, []):
            cid = C['id']
            groups = []
            for S in C.get('_icph_systs', []):
                gm = S['_meta']  # filled in prepare
                params = list(gm.get("params", []))
                combs  = [tuple(c) for c in gm.get("combs", [])]
                nuA = nuis_to_A_vector(params, combs, nu_vals)
                groups.append((gm, nuA))
            out[cid] = groups
        return out

    # ---------- Binned λ builder ----------
    def _compute_lambda_binned(self, rid: str, hypothesis) -> np.ndarray:
        """
        Build λ_i(c,ν) for all bins in a binned region rid by summing processes.
        """
        # per-class lnN (same as unbinned)
        nu_vals = {p.name: p.val for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        ln_bias = {}
        for C in self._binned_classes_by_region[rid]:
            cid = C['id']
            ln_bias[cid] = sum(log1p_alpha * nu_vals.get(pname, 0.0)
                               for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))

        # ν_A per ICPh group
        nuA_per_group = self._assemble_nuA_groups_binned(rid, hypothesis)

        # Unrolling info
        un = self._binned_unroll[rid]
        Nflat = len(un['flat_bins'])
        lam = np.zeros(Nflat, dtype=np.float64)

        # For each process, predict the *binned* σ_hat(c) from ICH, then apply nuisances
        for C in self._binned_classes_by_region[rid]:
            cid = C['id']
            ich = C['_ich']
            cvec = np.array([p.val for p in getattr(hypothesis, 'POIs', []) if p.name in C['_poi_params']])
            # IMPORTANT: ICH.predict takes the plain c-vector in the same order as variables
            cvec = self._assemble_c_vector_for_ich(rid, hypothesis, cid)
            sigma_hist = ich.predict(cvec)  # shape (nb1,) or (nb1,nb2)

            # accumulate nuisance exponent per bin from all ICPh groups,
            # separating shape-only groups (exact runtime renormalization) from the rest
            if sigma_hist.ndim == 1:
                nb1 = sigma_hist.shape[0]
                expo_norm  = np.zeros(nb1, dtype=np.float64)
                expo_shape = np.zeros(nb1, dtype=np.float64)
                for gm, nuA in nuA_per_group[cid]:
                    dA = gm['_deltas']   # shape (nB, nb1)
                    if gm.get('shape_only', False):
                        expo_shape = expo_shape + (nuA @ dA)
                    else:
                        expo_norm  = expo_norm  + (nuA @ dA)
                shape_factor = np.exp(expo_shape)
                shape_sum = sigma_hist @ shape_factor
                if shape_sum > 0:
                    shape_factor *= sigma_hist.sum() / shape_sum
                lam = lam + sigma_hist * np.exp(expo_norm + ln_bias[cid]) * shape_factor
            else:
                nb1, nb2 = sigma_hist.shape
                expo_norm2d  = np.zeros((nb1, nb2), dtype=np.float64)
                expo_shape2d = np.zeros((nb1, nb2), dtype=np.float64)
                for gm, nuA in nuA_per_group[cid]:
                    dA = gm['_deltas']   # shape (nB, nb1, nb2)
                    contrib = np.tensordot(nuA, dA, axes=(0, 0))  # (nb1, nb2)
                    if gm.get('shape_only', False):
                        expo_shape2d = expo_shape2d + contrib
                    else:
                        expo_norm2d  = expo_norm2d  + contrib
                shape_factor2d = np.exp(expo_shape2d)
                sig_flat = sigma_hist.reshape(-1)
                shape_sum = sig_flat @ shape_factor2d.reshape(-1)
                if shape_sum > 0:
                    shape_factor2d *= sig_flat.sum() / shape_sum
                lam = lam + sig_flat * np.exp(expo_norm2d.reshape(-1) + ln_bias[cid]) * shape_factor2d.reshape(-1)

        return lam


    # ---- assemble A-basis for POIs and nuisances from a hypothesis ----
    def _assemble_cA_per_class(self, rid: str, hypothesis) -> Dict[str, np.ndarray]:
        """Build c_A vectors per class for a given hypothesis."""
        cA_per_class: Dict[str, np.ndarray] = {}
        c_vec = {p.name: p.val for p in getattr(hypothesis, 'POIs', [])}
        for cid in self._class_ids_by_region.get(rid, []):
            poi_names = self._poi_order[(rid, cid)]
            reference_point = self._poi_reference[(rid, cid)]
            cA_per_class[cid] = expand_pois_linear_quadratic(poi_names, c_vec, reference_point)
        return cA_per_class

    def _assemble_rate_shift_per_class(self, rid: str, hypothesis) -> Dict[str, float]:
        """
        rate_shift is a single additive POI per class: (rid,cid) -> parameter name.
        Returns cid -> float shift (default 0.0 if not configured).
        """
        out: Dict[str, float] = {}
        for cid in self._class_ids_by_region.get(rid, []):
            pname = self._rate_shift_by_class.get((rid, cid), None)
            if pname is None:
                out[cid] = 0.0
            else:
                out[cid] = hypothesis[pname].val if pname in hypothesis else 0.0
        return out

    def _assemble_nuA_groups(self, rid: str, hypothesis) -> Dict[str, list[tuple[dict, np.ndarray]]]:
        """Build ν_A vectors per Δ-group for a given hypothesis."""
        nu_vals = {p.name: p.val for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        nuA_per_group: Dict[str, list[tuple[dict, np.ndarray]]] = {}
        for cid in self._class_ids_by_region.get(rid, []):
            meta = self._meta[(rid, cid)]
            groups = []
            for gm in meta.get("delta_groups", []):
                params = list(gm.get("params", gm.get("parameters", [])) or [])
                combs  = [tuple(c) for c in gm.get("combs", gm.get("combinations", [])) or []]
                nuA    = nuis_to_A_vector(params, combs, nu_vals)
                groups.append((gm, nuA))
            nuA_per_group[cid] = groups
        return nuA_per_group

    def _compute_T_from_columns(self, rid: str, columns_by_class, cA_per_class, nuA_per_group,
                                 ln_bias_map, rate_shift_map, start: int, stop: int) -> np.ndarray:
        """
        Compute T(x; c, ν) on [start:stop) for a single region rid, summing over classes,
        from already-materialized per-class columns.

        `columns_by_class[cid]` may be either an open HDF5 group (the cache) or an
        in-memory dict of numpy arrays (an observed/toy dataset's by-class block) --
        both support `col[start:stop]` slicing and `dset_name in col`, which is all
        this needs, so one implementation serves Asimov, observation and toy generation.

        T_i = Σ_p g_p(x_i) * [ (c⋅R_p)(x_i) * e^{Σ_s ν_B Δ_{p,B}(x_i)} + (e^{...} - 1) ].
        """
        M = stop - start
        T = np.zeros(M, dtype=np.float64)

        for cid in self._class_ids_by_region[rid]:
            f = columns_by_class[cid]
            g_slice = np.asarray(f['g'][start:stop], dtype=np.float64)          # (M,)
            R_slice = np.asarray(f['R'][start:stop, :], dtype=np.float64)       # (M, nA)
            cA      = cA_per_class[cid]

            if R_slice.shape[1] == 0:
                # No BIT contribution
                c_dot_R = np.zeros(R_slice.shape[0], dtype=np.float64)
            else:
                if R_slice.shape[1] != cA.shape[0]:
                    raise RuntimeError(f"[N2LL] BIT dim {R_slice.shape[1]} != |A| {cA.shape[0]} for {rid}/{cid}")
                c_dot_R = R_slice @ cA

            # --- additive POI bias: rate_shift (scalar per class) ---
            rs = rate_shift_map.get(cid, 0.0)
            if rs != 0.0:
                c_dot_R = c_dot_R + rs

            # build exponent from all Δ-groups
            expo = np.zeros_like(g_slice)
            for gm, nuA in nuA_per_group[cid]:
                dset = gm.get("dset", f"Delta::{gm['id']}")
                if dset not in f:
                    raise RuntimeError(f"[N2LL] Missing '{dset}' for {rid}/{cid}.")
                dA = np.asarray(f[dset][start:stop, :], dtype=np.float64)       # (M, nB)
                if dA.shape[1] != nuA.shape[0]:
                    raise RuntimeError(f"[N2LL] Δ dim {dA.shape[1]} != ν_A dim {nuA.shape[0]} for {rid}/{cid}/{gm['id']}")
                expo = expo + (dA @ nuA)                # (M,)

            # include per-class lnN bias additively in exponent
            expo += ln_bias_map[cid] # (M,)
            T += g_slice * (c_dot_R * np.exp(expo) + np.expm1(expo))

        return T

    def _compute_T_chunk(self, rid: str, cA_per_class, nuA_per_group, ln_bias_map, rate_shift_map, start: int, stop: int) -> np.ndarray:
        """Compute T(x; c, ν) on [start:stop) for region rid from the cached HDF5 columns."""
        columns_by_class = {cid: self._h5[(rid, cid)] for cid in self._class_ids_by_region[rid]}
        return self._compute_T_from_columns(rid, columns_by_class, cA_per_class, nuA_per_group,
                                             ln_bias_map, rate_shift_map, start, stop)

    def _eval_region_surrogates(self, rid: str, X: np.ndarray, feature_names: list[str]) -> dict:
        """
        Compute, in memory, the per-class arrays needed to build T(x; c,nu):
          - classifier probs g(x) if a classifier is configured (else ones)
          - BIT basis R_A(x)
          - each PNN group's Δ_B(x) matrix
        Returns a by-class dict mirroring the on-disk cache layout:
          { cid: {'g': (N,), 'R': (N,nA), 'Delta::<sid>': (N,nB), ...}, ... }

        A class with no configured BIT predictor gets `R` of shape (N,0) rather than
        raising -- needed for classes with no `POI` block at all, and for `rate_shift`
        POI blocks that carry no `predictor`. This is permissive by necessity for
        setObservation/toy generation, at a real cost: any *other* caller relying on
        this method (e.g. evaluate_ratio) that expects a BIT predictor to always be
        present will now silently get c_dot_R == 0 for such a class instead of a
        loud failure.
        """
        # resolve the region cfg and class list
        region = next((R for R in self.regions if R['id'] == rid), None)
        if region is None:
            raise RuntimeError(f"[_eval_region_surrogates] Unknown region id '{rid}'.")
        classes = list(region.get('classes', []) or [])
        n_proc  = len(classes)

        # input features + mask utility
        feat_names = list(feature_names or [])
        if not feat_names:
            raise RuntimeError("[_eval_region_surrogates] feature_names must be provided.")
        X = np.asarray(X, dtype=np.float64, order='C')
        if X.ndim != 2:
            raise RuntimeError(f"[_eval_region_surrogates] X must be 2D, got shape {X.shape}.")
        N = X.shape[0]

        # classifier g(x)
        clf = region.get('_classifier_predictor', None)
        if clf is None or n_proc <= 1:
            g_all = np.ones((N, n_proc), dtype=np.float64)
        else:
            if not hasattr(clf, "feature_names"):
                raise RuntimeError("[_eval_region_surrogates] classifier predictor lacks feature_names.")
            mask = self.make_column_mask(feat_names, list(clf.feature_names))
            g_all = _predict_classifier(clf, X[:, mask])  # (N, n_proc)
            if g_all.shape[1] != n_proc:
                raise RuntimeError(f"[_eval_region_surrogates] classifier outputs {g_all.shape[1]} != {n_proc} classes for region '{rid}'.")

        # per-class outputs
        by_class: dict[str, dict] = {}

        for C in classes:
            cid = C['id']
            comp = {}

            # g for this process
            p_index = class_index(classes, cid)
            comp['g'] = np.asarray(g_all[:, p_index], dtype=np.float64, order='C')  # (N,)

            # BIT R_A(x); a class with no predictor contributes no POI ratio.
            poi_predictor = (C.get('POI') or {}).get('predictor')
            if poi_predictor is None:
                R_A = np.empty((N, 0), dtype=np.float64)
            else:
                mask_bit = self.make_column_mask(feat_names, list(poi_predictor.feature_names))
                max_n_tree = (C.get('POI') or {}).get('max_n_tree', None)
                R_A = predict_bit_ratio(poi_predictor, X[:, mask_bit], max_n_tree)  # (N, nA)
            comp['R'] = np.asarray(R_A, dtype=np.float64, order='C')

            # PNN Δ groups
            for S in C.get('_pnn_systs', []):
                sid = S['id']
                pnn = S.get('predictor', None)
                if pnn is None:
                    raise RuntimeError(f"[_eval_region_surrogates] Missing PNN predictor for {rid}/{cid}/{sid}.")
                mask_pnn = self.make_column_mask(feat_names, list(pnn.feature_names))
                dA = predict_pnn_deltaA(pnn, X[:, mask_pnn])  # (N, nB)
                comp[f"Delta::{sid}"] = np.asarray(dA, dtype=np.float64, order='C')

            by_class[cid] = comp

        return by_class

    def setAsimov(self, hypothesis=None) -> None:
        """
        Set an off-nominal Asimov hypothesis (c', ν') and precompute T'(x; c', ν')
        on the cached event set. If all parameters are zero, disables the bias.
        """
        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.setAsimov] Call prepare_runtime() before setting Asimov.")

        # flow control. We set this to true irrespective of whether we eventually need a non-zero hypothesis
        self._asimov_hypothesis_set = True

        # n2ll.setAsimov() defaults to the null hypothesis, hence no bias term 
        if hypothesis is None:
            self._asimov_active = False
            return

        # quick check: any parameter nonzero?
        any_nonzero = any(abs(getval(p.val)) > 0.0 for p in getattr(hypothesis, 'parameters', []))
        # _asimov_active controls whether we have (c',nu')!=(0,0). If that's false, we need not evaluate T(x;c',nu')
        self._asimov_active = bool(any_nonzero)
        self._asimov_hyp = hypothesis if self._asimov_active else None
        self._asimov_T.clear()

        if not self._asimov_active:
            return  # nothing to cache; bias term will be skipped

        # Precompute per region
        for R in self.regions:
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue

            # A-basis and lnN bias for Asimov hypothesis
            cA_per_class = self._assemble_cA_per_class(rid, hypothesis)
            nuA_per_group = self._assemble_nuA_groups(rid, hypothesis)
            ln_bias = {
                cid: sum(log1p_alpha * hypothesis[nm].val if nm in hypothesis else 0.0
                         for nm, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }
            
            rate_shift = self._assemble_rate_shift_per_class(rid, hypothesis)

            # chunked compute and store
            chunk = self.eval_chunk_size
            Ts: list[np.ndarray] = []
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                #T_chunk = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
                T_chunk = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, rate_shift, start, stop)
                Ts.append(T_chunk)
            self._asimov_T[rid] = Ts

        # ----- also precompute binned Asimov λ'(i) if binned regions exist -----
        self._binned_asimov_lambda.clear()
        if self._binned_regions_ids:
            for rid in self._binned_regions_ids:
                lam_prime = self._compute_lambda_binned(rid, hypothesis)  # vector (Nflat,)
                self._binned_asimov_lambda[rid] = lam_prime

    def setObservation(self,
                       unbinned_loaders: dict | None = None,
                       binned_loaders:   dict | None = None,
                       *,
                       ignore_weights: bool = True) -> None:
        """
        Register an observed dataset for likelihood evaluation.

        Parameters
        ----------
        unbinned_loaders : dict or None
            Mapping {region_id -> loader}, where each loader is an RDataLoader or
            SelectionView (anything exposing `.materialize(shard=..., what=..., n=...)`
            and `.n_split` and `.feature_names`).
            These events will be used *event-by-event* in the unbinned likelihood.

        binned_loaders : dict or None
            Mapping {region_id -> loader} (same loader interface) from which we will
            *compute* observed binned counts by histogramming into the region's ICH binning.

            NOTE: Even for binned, we expect *unbinned* events here; we internally
            histogram to the region's bins defined in `self._binned_unroll[rid]['edges']`.

        ignore_weights : bool (default: True)
            If True, do NOT even load weights from loaders; treat every event as weight 1.
            This switches the `what` argument in `materialize(...)` so weights are never read.

        Effects
        -------
        - Sets `self._observation_set = True` and disables any previously set Asimov bias.
        - Populates:
            * `self._obs_unbinned[rid] = {'X': features (N,d), 'w': weights (N,)}` for unbinned regions
            * `self._obs_binned_counts[rid] = counts_flat (Nflat,)` for binned regions
        - Clears any previous observation caches.
        """

        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.setObservation] Call prepare_runtime() before setting observation.")

        if not ignore_weights:
            print("[N2LL.setObservation] Using weighted sample in setObservation.")

        unbinned_blocks: dict = {}
        binned_counts: dict = {}

        # ------------- UNBINNED OBSERVATION -------------
        if unbinned_loaders:
            for rid, loader in (unbinned_loaders.items()):
                # Sanity: region exists and is unbinned-configured
                if rid not in {R['id'] for R in self.regions}:
                    raise RuntimeError(f"[setObservation:unbinned] Unknown region id '{rid}'.")
                # Collect features (+ optional weights) across shards
                nsplits = int(getattr(loader, "n_split", 1))
                want = "f" if ignore_weights else "fw"
                Xs, Ws = [], []
                for shard in range(nsplits):
                    outs = loader.materialize(shard=shard, what=want, n=None)
                    if ignore_weights:
                        (X,) = outs
                        w = np.ones(X.shape[0], dtype=np.float64)
                    else:
                        X, w = outs
                        w = np.asarray(w, dtype=np.float64, order='C')
                    X = np.asarray(X, dtype=np.float64, order='C')
                    if X.ndim != 2:
                        raise RuntimeError(f"[setObservation:unbinned:{rid}] Features must be 2D, got shape {X.shape}.")
                    if X.shape[0] != w.shape[0]:
                        raise RuntimeError(f"[setObservation:unbinned:{rid}] len(weights) != nEvents ({w.shape[0]} != {X.shape[0]}).")
                    Xs.append(X); Ws.append(w)

                    # checks of the existence of surrogates has been done before setObservation is called

                if Xs:
                    X_all = np.concatenate(Xs, axis=0)
                    w_all = np.concatenate(Ws, axis=0)
                else: # Q: shouldn't this raise an error ? in principle a shard should *always* have events
                    X_all = np.empty((0, len(getattr(loader, "feature_names", []))), dtype=np.float64)
                    w_all = np.empty((0,), dtype=np.float64)

                # pre-evaluating surrogates on observed events
                # to avoid evaluating everytime n2ll(hyp) is called
                # NB: in binned, evaluation is done directly in n2ll(hyp),
                # because ICH and ICPH already give the ratio for nominal vs. alternative
                by_class = self._eval_region_surrogates(rid, X_all, loader.feature_names)
                unbinned_blocks[rid] = {'X': X_all, 'w': w_all, 'by_class': by_class}

                region_info = next(R for R in self.regions if R['id'] == rid)
                n_classes = len(region_info['classes'])
                n_classifiers = 1 if (region_info.get('_classifier_predictor') is not None and n_classes > 1) else 0
                n_poi_predictors = sum(1 for C in region_info['classes'] if (C.get('POI') or {}).get('predictor') is not None)
                n_syst_predictors = sum(len(C.get('_pnn_systs', []) or []) for C in region_info['classes'])

                print(f"[setObservation] Unbinned region '{rid}': loaded {X_all.shape[0]:,} events "
                      f"({'unit weights' if ignore_weights else 'with weights'}).")
                print(f'[setObservation] Evaluated {n_classifiers} classifiers, and individual surrogates for {n_classes} classes. Total: {n_poi_predictors} POI predictors and {n_syst_predictors} systematics predictors.')

        # ------------- BINNED OBSERVATION (from unbinned events) -------------
        if binned_loaders:
            for rid, loader in (binned_loaders.items()):
                if rid not in self._binned_unroll:
                    raise RuntimeError(f"[setObservation:binned] Region '{rid}' has no binned definition in current likelihood.")
                un = self._binned_unroll[rid]  # contains 'edges' (list), 'axes' (names), etc.
                edges = un['edges']
                axes  = un['axes']  # feature names for the binnings

                # Resolve feature indices for the loader
                feat_names = list(getattr(loader, "feature_names", []) or [])
                if not feat_names:
                    raise RuntimeError(f"[setObservation:binned:{rid}] Loader does not expose feature_names.")
                try:
                    idx = [feat_names.index(ax) for ax in axes]
                except ValueError as e:
                    raise RuntimeError(f"[setObservation:binned:{rid}] Loader lacks required bin axis features {axes}.") from e

                nsplits = int(getattr(loader, "n_split", 1))
                want = "f" if ignore_weights else "fw"

                # Accumulate counts (sum of weights or 1s) per bin
                if len(edges) == 1:
                    nb1 = len(edges[0]) - 1
                    counts = np.zeros(nb1, dtype=np.float64)
                elif len(edges) == 2:
                    nb1 = len(edges[0]) - 1
                    nb2 = len(edges[1]) - 1
                    counts2d = np.zeros((nb1, nb2), dtype=np.float64)
                else:
                    raise RuntimeError("[setObservation:binned] Only 1D/2D binning supported.")

                for shard in range(nsplits):
                    outs = loader.materialize(shard=shard, what=want, n=None)
                    if ignore_weights:
                        (X,) = outs
                        w = None  # implies unit weights in histogramming
                    else:
                        X, w = outs
                        w = np.asarray(w, dtype=np.float64)

                    X = np.asarray(X, dtype=np.float64)
                    if len(edges) == 1:
                        x = X[:, idx[0]]
                        H, _ = onp.histogram(x, bins=edges[0], weights=(w if w is not None else None))
                        counts += H.astype(np.float64)
                    else:
                        x = X[:, idx[0]]
                        y = X[:, idx[1]]
                        H, _, _ = onp.histogram2d(x, y, bins=[edges[0], edges[1]], weights=(w if w is not None else None))
                        counts2d += H.astype(np.float64)

                flat_counts = counts if len(edges) == 1 else counts2d.reshape(-1)
                binned_counts[rid] = flat_counts
                print(f"[setObservation] Binned region '{rid}': filled {flat_counts.size} bins "
                      f"({'unit weights' if ignore_weights else 'with weights'}).")

        self.setObservationArrays(unbinned_blocks=unbinned_blocks or None, binned_counts=binned_counts or None)

    def setObservationArrays(self, unbinned_blocks: dict | None = None, binned_counts: dict | None = None) -> None:
        """
        Register an observed/toy dataset from already-evaluated arrays (no loaders).

        Parameters
        ----------
        unbinned_blocks : dict or None
            Mapping {region_id -> {'X': features (N,d) or None, 'w': weights (N,),
            'by_class': {cid: {'g': (N,), 'R': (N,nA), 'Delta::<sid>': (N,nB), ...}}}}.
            'X' may be None (e.g. cache-mode toys, which have no materialized X).
        binned_counts : dict or None
            Mapping {region_id -> counts_flat (Nflat,)}.

        Effects mirror setObservation: sets `self._observation_set = True`, disables
        any previously set Asimov bias, and populates `self._obs_unbinned` /
        `self._obs_binned_counts`. `setObservation` ends by calling this, so this is
        the single place that owns the observed/Asimov mode switch.
        """
        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.setObservationArrays] Call prepare_runtime() before setting observation.")

        # You can’t mix observed-data mode with Asimov in the same evaluation flow.
        # We allow switching, but make it explicit and clear.
        if getattr(self, "_asimov_hypothesis_set", False) and getattr(self, "_asimov_active", False):
            print("[N2LL.setObservationArrays] An Asimov hypothesis had been set; disabling it in favor of observed-data mode.")

        self._asimov_hypothesis_set = False
        self._asimov_active = False
        self._asimov_hyp = None

        # hypothesis for non-central Asimov
        self._asimov_T.clear()
        self._binned_asimov_lambda.clear()

        # Reset observation containers
        self._obs_unbinned = {}
        self._obs_binned_counts = {}

        # Flag we’re now in observed-data mode
        self._observation_set = True

        known_region_ids = {R['id'] for R in self.regions}
        if unbinned_blocks:
            for rid, block in unbinned_blocks.items():
                if rid not in known_region_ids:
                    raise RuntimeError(f"[setObservationArrays:unbinned] Unknown region id '{rid}'.")
                w = np.asarray(block['w'], dtype=np.float64)
                for cid, comp in block.get('by_class', {}).items():
                    for col_name, col in comp.items():
                        if len(col) != len(w):
                            raise RuntimeError(
                                f"[setObservationArrays:unbinned:{rid}/{cid}] "
                                f"len({col_name})={len(col)} != len(w)={len(w)}.")
                self._obs_unbinned[rid] = block

        if binned_counts:
            for rid, counts in binned_counts.items():
                if rid not in self._binned_unroll:
                    raise RuntimeError(f"[setObservationArrays:binned] Region '{rid}' has no binned definition in current likelihood.")
                self._obs_binned_counts[rid] = np.asarray(counts, dtype=np.float64)

    def setToy(self, toy: dict, hypothesis) -> None:
        """
        Register a toy (from fit/ToyGenerator.py's generate_toy/load_toy) as the
        observation, and apply its thrown constraint centres to `hypothesis`.
        """
        unbinned_blocks = {
            rid: {'X': block.get('X'), 'w': block['w'], 'by_class': block['by_class']}
            for rid, block in toy.get('unbinned_blocks', {}).items()
        }
        self.setObservationArrays(unbinned_blocks=unbinned_blocks or None,
                                   binned_counts=toy.get('binned_counts') or None)
        base = hypothesis._base if hasattr(hypothesis, '_base') else hypothesis
        base.set_constraint_centers(toy.get('constraint_centers', {}))

    def __call__(self, hypothesis) -> float:
        """
        Evaluate -2 log L for either:
          (A) OBSERVED data (registered via setObservation), or
          (B) ASIMOV expectation (registered via setAsimov).

        Exactly one of setObservation(...) or setAsimov(...) must be called.
        Asimov bias term is included only in case (B) and only if an off-nominal
        Asimov hypothesis was provided.
        """

        if not self._runtime_prepared:
            raise RuntimeError("[N2LL] Call prepare_runtime() before evaluating.")

        # Enforce exactly one mode active
        if bool(self._observation_set) == bool(self._asimov_hypothesis_set):
            raise RuntimeError("[N2LL] Exactly one of setObservation(...) or setAsimov(...) must be called "
                               "before evaluating, but not both (and not neither).")

        # ===================================================================
        # (A) OBSERVATION MODE
        # ===================================================================
        if self._observation_set and not self._asimov_hypothesis_set:
            total_unbinned = 0.0
            total_binned   = 0.0

            # ---------- UNBINNED (if provided) ----------
            if getattr(self, "_obs_unbinned", None):
                # ν values for lnN bias if we need to reconstruct T from by_class
                nu_vals = {p.name: p.val for p in getattr(hypothesis._base, 'parameters', []) if not p.isPOI}

                for rid, block in self._obs_unbinned.items():
                    byc = block['by_class']
                    W   = np.asarray(block['w'], dtype=np.float64)
                    N   = len(W)

                    # current hypothesis A-basis and ν_A groups
                    cA_per_class  = self._assemble_cA_per_class(rid, hypothesis._base)
                    nuA_per_group = self._assemble_nuA_groups(rid, hypothesis._base)
                    ln_bias = {
                        cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                                 for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                        for cid in self._class_ids_by_region[rid]
                    }
                    rate_shift = self._assemble_rate_shift_per_class(rid, hypothesis._base)

                    # first term in N2LL: weighted sum of T evaluated on Asimov events for each class
                    chunk = self.eval_chunk_size
                    class_ids = self._class_ids_by_region[rid]
                    
                    # N = 0 case may happen if we have regions with spurious
                    # data events without events in Asimov.
                    # in any case, the loop for the calculation 
                    # of the first term will simply not run
                    N_asimov =self._N_region.get(rid, 0)
                    for ichunk, start in enumerate(range(0, N_asimov, chunk)):
                        stop = min(start + chunk, N_asimov)
                        T_asimov_chunk = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, rate_shift, start, stop)
                        W_asimov_chunk = self._h5[(rid, class_ids[0])]['w0'][start:stop]
                        # IMPORTANT: event weights for all the events in Asimov are stored
                        # in the cache file for all of the classes
                        # So doing the below line with class_ids[-1] would also work.                                            
                        total_unbinned -= np.dot(W_asimov_chunk, T_asimov_chunk)

                    # second term in N2LL: divided into two parts
                    # 1: get T evaluated on the observed events, summing over predictions of surrogates for each class
                    # 2: make weighted sum of log1p(T), for observed weights can be one or not
                    # depends on whether setObservation was called with ignoreWeights=True
                    T = self._compute_T_from_columns(rid, byc, cA_per_class, nuA_per_group, ln_bias, rate_shift, 0, N)

                    total_unbinned += np.dot(W,np.log1p(T))

            # ---------- BINNED (always available if you provided columns for axes) ----------
            if getattr(self, "_binned_regions_ids", None) and getattr(self, "_obs_binned_counts", None):
                for rid in self._binned_regions_ids:
                    if rid not in self._obs_binned_counts:
                        continue  # region not histogrammed (e.g. missing axis columns)
                    lam0 = self._binned_lambda0[rid]                    # (Nflat,)
                    lam  = self._compute_lambda_binned(rid, hypothesis._base) # (Nflat,)
                    Nobs = self._obs_binned_counts[rid]                        # (Nflat,)

                    log_ratio = self._safe_log_ratio(lam, lam0)         # stable
                    total_binned += np.sum( -(lam - lam0) + Nobs * log_ratio, dtype=np.float64 )

            n2ll = -2.0 * (total_unbinned + total_binned)
            n2ll += hypothesis._base.penalty()
            return n2ll

        # ===================================================================
        # (B) ASIMOV MODE  
        # ===================================================================
        total_sum = 0.0   # Σ w * (log1p(T) - T)
        bias_sum  = 0.0   # Σ w * T'(asimov) * log1p(T)

        nu_vals = {p.name: p.val for p in getattr(hypothesis._base, 'parameters', []) if not p.isPOI}

        for R in self.regions:
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue

            cA_per_class  = self._assemble_cA_per_class(rid, hypothesis._base)
            nuA_per_group = self._assemble_nuA_groups(rid, hypothesis._base)
            ln_bias = {
                cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                         for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }
            rate_shift = self._assemble_rate_shift_per_class(rid, hypothesis._base)

            chunk = self.eval_chunk_size
            asimov_T_chunks = self._asimov_T.get(rid, None) if self._asimov_active else None
            for ichunk, start in enumerate(range(0, N, chunk)):
                stop = min(start + chunk, N)
                #T = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
                T = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, rate_shift, start, stop)
                # IMPORTANT: event weights for all the events in Asimov are stored
                # in the cache file for all of the classes
                # So doing the below line with class_ids[-1] would also work.
                W = self._h5[(rid, class_ids[0])]['w0'][start:stop]
                total_sum += _weighted_sum_log1p_minus_x(T, W)

                if asimov_T_chunks is not None:
                    Tprime = asimov_T_chunks[ichunk]
                    bias_sum += np.sum(W * np.log1p(T) * Tprime, dtype=np.float64)

        total_unbinned = total_sum + bias_sum

        total_binned = 0.0
        if getattr(self, "_binned_regions_ids", None):
            for rid in self._binned_regions_ids:
                lam0 = self._binned_lambda0[rid]
                lam  = self._compute_lambda_binned(rid, hypothesis._base)
                lam_asimov = self._binned_asimov_lambda.get(rid, lam0)

                log_ratio = self._safe_log_ratio(lam, lam0)
                total_binned += np.sum( -(lam - lam0) + lam_asimov * log_ratio, dtype=np.float64 )

        n2ll = -2.0 * (total_unbinned + total_binned)
        n2ll += hypothesis._base.penalty()
        return n2ll

from iminuit import Minuit

class _CovarianceMatrix:
    def __init__(self, matrix, parameter_names):
        self.matrix = onp.asarray(matrix, dtype=onp.float64)
        self.parameter_names = list(parameter_names)

    def correlation(self):
        diag = onp.sqrt(onp.clip(onp.diag(self.matrix), 0.0, None))
        denom = onp.outer(diag, diag)
        corr = onp.zeros_like(self.matrix)
        onp.divide(self.matrix, denom, out=corr, where=denom > 0)
        onp.fill_diagonal(corr, 1.0)
        return corr

    def __array__(self, dtype=None):
        return onp.asarray(self.matrix, dtype=dtype)


class _FitResult:
    def __init__(self, *, parameters, values, errors, covariance, fval, edm, niter, success, message, jac):
        self.parameters = list(parameters)
        self.values = onp.asarray(values, dtype=onp.float64)
        self.errors = onp.asarray(errors, dtype=onp.float64)
        self.covariance = _CovarianceMatrix(covariance, self.parameters)
        self.fval = float(fval)
        self.edm = float(edm)
        self.niter = int(niter)
        self.success = bool(success)
        self.message = str(message)
        self.jac = None if jac is None else onp.asarray(jac, dtype=onp.float64)

    def __repr__(self):
        status = "success" if self.success else "failed"
        return f"FitResult(fval={self.fval:.6e}, niter={self.niter}, status={status})"


def _scalar_value(value):
    try:
        return float(getval(value))
    except Exception:
        return float(value)


from iminuit import Minuit
def run_iminuit_fit(n2ll, hypothesis, *, step=None, print_every=25,
                   do_migrad=True, do_hesse=True, do_minos=False, minosNP=None ,verbosity=1):

    # -- collect free parameters (works for rotated or plain) --
    if isinstance(hypothesis, Rotated):
        free = [p for p in hypothesis.POIs if not p.isFrozen] + [p for p in hypothesis.nuisances
                                  if not p.isFrozen and not getattr(p, "isIgnored", False)]
        poi_names = {p.name for p in hypothesis.POIs if not p.isFrozen}
    else:
        free = [p for p in hypothesis.parameters
                if not p.isFrozen and not getattr(p, "isIgnored", False)]
        poi_names = set()

    if not free:
        raise RuntimeError("No free parameters to fit.")

    names = [p.name for p in free]
    x0    = [float(p.val) for p in free]

    # step defaults:
    #   None  -> plain: all 0.1
    #   float -> uniform
    #   dict  -> per-parameter overrides
    if step is None:
        steps = {p.name: 0.1 for p in free}
    elif isinstance(step, (int, float)):
        steps = {p.name: float(step) for p in free}
    elif isinstance(step, dict):
        steps = {p.name: 0.1 for p in free}
        for k, v in step.items():
            if k in steps:
                steps[k] = float(v)
    else:
        raise TypeError("step must be None, float, or dict{name: float}")

    eval_count = 0
    def fcn(*x):
        nonlocal eval_count
        # one-shot parameter map (avoid sequential setattr on Rotated)
        pars = {names[i]: float(x[i]) for i in range(len(names))}
        h_eval = hypothesis.cloneModify(**pars)   # absolute update in one go
        f = float(n2ll(h_eval))
        eval_count += 1
        if verbosity >= 2:
            if ((eval_count - 1) % max(1, int(print_every)) == 0) and print_every >= 0:
                print(f"\n[eval {eval_count:6d}] f = {f: .6e}")
                h_eval.print()  # print the actually evaluated point
        if math.isnan(f):
            raise RuntimeError("NaN likelihood!")
        return f

    # ---- Minuit with positional args and explicit names ----
    m = Minuit(fcn, *x0, name=names)
    m.errordef = 1.0
    m.strategy = 2

    # set user step and (if available) explicit FD step
    for i, nm in enumerate(names):
        s = steps[nm]
        m.errors[i] = s
        if hasattr(m, "set_initial_step"):
            m.set_initial_step(i, 0.3 * s)

    # rate_shift nuisances are bound 
    for name in m.parameters:
        if name.startswith("rate_shift"):
            m.limits[name] = (-1.0, None)

    if verbosity >= 1:
        print("\n[make_minuit] Floating parameters:")
        for i, nm in enumerate(names):
            print(f"  - {nm:>16s}  start = {m.values[i]: .6e}  step = {m.errors[i]: .3g}")

    if do_migrad:
        m.migrad();
        if verbosity >= 1:
            print("\n[MIGRAD]");  print(m)
    if do_hesse:
        m.hesse();
        if verbosity >= 1:
            print("\n[HESSE]");  print(m)
    if do_minos:
        minos_parameter_list = [p.name for p in getattr(hypothesis, "POIs", []) if p.name in m.parameters] or list(m.parameters)
        print("Running MINOS uncertainties for POIs.")
        if minosNP:
            if "all" in minosNP:
                print("Running MINOS uncertainties for all NPs.")
                minos_parameter_list = [p.name for p in free]
            else:
                print(f"Running MINOS uncertainties also for the following NPs: {minosNP}")
                minos_parameter_list+=minosNP
        m.minos(*minos_parameter_list)
        if verbosity >=1: 
            print("\n[MINOS]", minos_parameter_list);
            print(m)

    # write back best fit once (avoid repeated __setattr__ compounding)
    final_pars = {names[i]: float(m.values[i]) for i in range(len(names))}
    h_final = hypothesis.cloneModify(**final_pars)
    # copy final values onto the original object (single pass)
    for k, v in final_pars.items():
        setattr(hypothesis, k, v)
    if verbosity >= 1:
        print("\n[final] Best-fit hypothesis:")
        h_final.print()
    return m

def run_autograd_fit(n2ll, hypothesis, *, step=None, print_every=25,
                   do_migrad=True, do_hesse=True, do_minos=False, minosNP=None ,verbosity=1):

    # -- collect free parameters (works for rotated or plain) --
    if isinstance(hypothesis, Rotated):
        free = [p for p in hypothesis.POIs if not p.isFrozen] + [p for p in hypothesis.nuisances
                                  if not p.isFrozen and not getattr(p, "isIgnored", False)]
        poi_names = {p.name for p in hypothesis.POIs if not p.isFrozen}
    else:
        free = [p for p in hypothesis.parameters
                if not p.isFrozen and not getattr(p, "isIgnored", False)]
        poi_names = set()

    if not free:
        raise RuntimeError("No free parameters to fit.")

    names = [p.name for p in free]
    x0 = np.asarray([_scalar_value(p.val) for p in free], dtype=np.float64)

    # step defaults:
    #   None  -> plain: all 0.1
    #   float -> uniform
    #   dict  -> per-parameter overrides
    if step is None:
        steps = {p.name: 0.1 for p in free}
    elif isinstance(step, (int, float)):
        steps = {p.name: float(step) for p in free}
    elif isinstance(step, dict):
        steps = {p.name: 0.1 for p in free}
        for k, v in step.items():
            if k in steps:
                steps[k] = float(v)
    else:
        raise TypeError("step must be None, float, or dict{name: float}")

    eval_count = 0

    def fcn(x):
        nonlocal eval_count
        pars = {names[i]: x[i] for i in range(len(names))}
        h_eval = hypothesis.cloneModify(**pars)
        f = n2ll(h_eval)
        eval_count += 1
        if verbosity >= 2:
            if ((eval_count - 1) % max(1, int(print_every)) == 0) and print_every >= 0:
                print(f"\n[eval {eval_count:6d}] f = {_scalar_value(f): .6e}")
                h_eval.print()  # print the actually evaluated point
        if not onp.isfinite(_scalar_value(f)):
            raise RuntimeError("NaN likelihood!")
        return f

    bounds = [(-1.0, np.inf) if name.startswith("rate_shift") else (-np.inf, np.inf) for name in names]

    if verbosity >= 1:
        print("\n[fit] Floating parameters:")
        for i, nm in enumerate(names):
            print(f"  - {nm:>16s}  start = {x0[i]: .6e}  step = {steps[nm]: .3g}")

    if do_migrad:
        result = minimize(
            fcn,
            x0,
            method="L-BFGS-B",
            jac=grad(fcn),
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        x_best = onp.asarray(result.x, dtype=onp.float64)
        if verbosity >= 1:
            print("\n[MINIMIZE]")
            print(result)
    else:
        x_best = onp.asarray(x0, dtype=onp.float64)
        result = SimpleNamespace(success=True, message="minimization skipped", nit=0, fun=fcn(x_best), jac=grad(fcn)(x_best))

    # The objective is -2 log L.  Minuit's errordef=1 convention defines the
    # 1-sigma boundary as Δ(-2 log L) = 1, so C = 2 H⁻¹ (the factor of 2
    # absorbs the "-2" prefactor).  Apply the same scaling here.
    if do_hesse:
        try:
            hess = onp.asarray(hessian(fcn)(x_best), dtype=onp.float64)
            covariance = 2.0 * onp.linalg.pinv(hess)
        except Exception as exc:
            if verbosity >= 1:
                print(f"[warn] Exact Hessian failed ({exc}); falling back to optimizer covariance.")
            hess_inv = getattr(result, "hess_inv", None)
            if hess_inv is not None and hasattr(hess_inv, "todense"):
                covariance = 2.0 * onp.asarray(hess_inv.todense(), dtype=onp.float64)
            elif hess_inv is not None:
                covariance = 2.0 * onp.asarray(hess_inv, dtype=onp.float64)
            else:
                covariance = onp.eye(len(names), dtype=onp.float64)
    else:
        hess_inv = getattr(result, "hess_inv", None)
        if hess_inv is not None and hasattr(hess_inv, "todense"):
            covariance = 2.0 * onp.asarray(hess_inv.todense(), dtype=onp.float64)
        elif hess_inv is not None:
            covariance = 2.0 * onp.asarray(hess_inv, dtype=onp.float64)
        else:
            covariance = onp.eye(len(names), dtype=onp.float64)

    errors = onp.sqrt(onp.clip(onp.diag(covariance), 0.0, None))
    jac_best = onp.asarray(getattr(result, "jac", grad(fcn)(x_best)), dtype=onp.float64)
    edm = float(0.5 * onp.dot(jac_best, jac_best))

    final_pars = {names[i]: x_best[i] for i in range(len(names))}
    h_final = hypothesis.cloneModify(**final_pars)
    # copy final values onto the original object (single pass)
    for k, v in final_pars.items():
        setattr(hypothesis, k, v)
    if verbosity >= 1:
        print("\n[final] Best-fit hypothesis:")
        h_final.print()
    if do_minos and verbosity >= 1:
        print("[warn] MINOS uncertainties are not available in the autograd backend; using Hessian errors.")

    return _FitResult(
        parameters=names,
        values=x_best,
        errors=errors,
        covariance=covariance,
        fval=_scalar_value(fcn(x_best)),
        edm=edm,
        niter=getattr(result, "nit", getattr(result, "niter", 0)),
        success=getattr(result, "success", True),
        message=getattr(result, "message", ""),
        jac=jac_best,
    )

def serialize_result(m, base, version, args, out_path, toy_info=None ):
    """Write the fit result (values, errors, covariance) to JSON.

    'toy_info' carries the provenance of the toy dataset that was fitted
    (point, source, seed and the injected hypothesis), so downstream toy
    studies can compute pulls without reopening the toy HDF5. It is None
    for fits to data or Asimov.
    """

    result_payload = {
        "config_basename": base,
        "version": version,
        "no_syst": args.no_syst,
        "syst_only": args.syst_only,
        "fval": float(m.fval),
        "edm": float(getattr(m, "edm", np.nan)),
        "niter": int(getattr(m, "niter", -1)),
        "parameters": [
            {"name": name, "value": float(m.values[i]), "error": float(m.errors[i])}
            for i, name in enumerate(m.parameters)
        ],
        "free_parameter_order": list(m.parameters),
        "covariance": {
            "order": list(m.parameters),
            "matrix": np.asarray(m.covariance).tolist(),
        },
        "correlation": {
            "order": list(m.parameters),
            "matrix": np.asarray(m.covariance.correlation()).tolist(),
        },
    }
    if toy_info is not None:
        result_payload["toy"] = toy_info
    with open(out_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    print(f"[write] Fit result and covariance stored at:\n  {out_path}")

def pretty_par_name(name: str) -> str:
    # strip prefixes (only at the beginning), in the given order
    for pre in ("nu_", "CMS_"):
        if name.startswith(pre):
            name = name[len(pre):]
    # replacements
    return name.replace("res_j", "JER").replace("scale_j_Regrouped", "JES")

def plot_fit_summary_root(out_dir, base, rotated, hyp, fit_vals, fit_errs, suffix=""):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    pois = [p.name for p in (getattr(hyp, "POIs", None) or getattr(hyp, "pois", []))]
    nuis = [p.name for p in getattr(hyp, "nuisances", []) if not p.isFrozen]
    names = pois + nuis
    if args.no_syst:
        names = pois
    elif args.syst_only:
        names = nuis
    n_pois, n_nuis = len(pois), len(nuis)
    n = len(names)
    if n == 0:
        return

    xmin, xmax = -3.0, 3.0
    target = 2.5

    x = np.zeros(n, dtype=float)
    exl = np.zeros(n, dtype=float)
    exh = np.zeros(n, dtype=float)
    labels = list(names)

    for i, name in enumerate(names):
        if name in fit_vals:
            v = float(fit_vals[name])
            e = float(fit_errs.get(name, 0.0))
            lo = abs(float(fit_errs.get(f"{name}_lo", e)))
            hi = abs(float(fit_errs.get(f"{name}_hi", e)))
            lab = pretty_par_name(name)
        else:
            v,e,lo,hi = 0,0,0,0
            lab = pretty_par_name(name)+" (f.)"

        if i < n_pois:
            m = max(abs(v), lo, hi)
            if m > 0.0:
                k = int(np.floor(np.log10(target / m)))
                if k != 0:
                    s = 10.0 ** k
                    v *= s
                    lo *= s
                    hi *= s
                    lab = f"{lab}  (#times10^{{{k}}})"

        x[i], exl[i], exh[i] = v, lo, hi
        labels[i] = lab

    c = ROOT.TCanvas("c_fit", "fit", 950, 700)
    c.SetLeftMargin(0.35)
    c.SetRightMargin(0.06)
    c.SetTopMargin(0.12)
    c.SetBottomMargin(0.10)
    c.SetTickx(1)  # ticks also on top x-axis

    frame = ROOT.TH2F("frame_fit", "", 1, xmin, xmax, n, 0.5, n + 0.5)
    frame.GetXaxis().SetTitle("value")
    frame.GetYaxis().SetLabelSize(0.032)
    frame.GetXaxis().SetTitleSize(0.040)
    frame.GetXaxis().SetLabelSize(0.035)
    frame.GetXaxis().SetNdivisions(508)  # leave some whitespace / reduce clutter

    for i, lab in enumerate(labels):
        frame.GetYaxis().SetBinLabel(n - i, lab)

    frame.Draw("AXIS")

    g = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        y = float(n - i)
        g.SetPoint(i, float(x[i]), y)
        g.SetPointError(i, float(exl[i]), float(exh[i]), 0.0, 0.0)

    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetLineWidth(2)
    g.Draw("P SAME")

    # reference at 0
    l0 = ROOT.TLine(0.0, 0.5, 0.0, n + 0.5)
    l0.SetLineWidth(2)
    l0.Draw("SAME")

    # separator between POIs and nuisances
    ysep = None
    if n_pois and n_nuis:
        ysep=n
        if not args.syst_only:
            ysep = n - n_pois + 0.5
            lsep = ROOT.TLine(xmin, ysep, xmax, ysep)
            lsep.SetLineStyle(2)
            lsep.SetLineWidth(2)
            lsep.Draw("SAME")
        # prefit constraint band guides for nuisances: x=±1 from separator down to x-axis
        for xx in (-1.0, +1.0):
            lv = ROOT.TLine(xx, 0.5, xx, ysep)
            lv.SetLineStyle(2)
            lv.SetLineWidth(2)
            lv.Draw("SAME")

    # run-identifying strings
    t = ROOT.TLatex()
    t.SetNDC(True)
    t.SetTextSize(0.035)
    t.DrawLatex(0.36, 0.94, f"{base}{suffix}")
    t.SetTextSize(0.030)
    t.DrawLatex(0.36, 0.905, f"rotated: {'yes' if rotated else 'no'}")

    os.makedirs(out_dir, exist_ok=True)
    helpers.copyIndexPHP(out_dir)
    c.SaveAs(os.path.join(out_dir, f"{base}{suffix}_fit_summary.pdf"))
    c.SaveAs(os.path.join(out_dir, f"{base}{suffix}_fit_summary.png"))
    c.Close()

def plot_correlation_root(out_dir, base, rotated, names, corr, suffix=""):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    try:
        ROOT.gStyle.SetPalette(ROOT.kViridis)
    except Exception:
        pass

    n = len(names)
    if n == 0:
        return

    pretty = [pretty_par_name(nm) for nm in names]

    c = ROOT.TCanvas("c_corr", "corr", 1050, 950)
    c.SetLeftMargin(0.28)
    c.SetRightMargin(0.14)
    c.SetTopMargin(0.12)
    c.SetBottomMargin(0.30)

    h = ROOT.TH2D("h_corr", "", n, 0.5, n + 0.5, n, 0.5, n + 0.5)
    h.GetZaxis().SetRangeUser(-1.0, 1.0)
    h.GetZaxis().SetTitle("corr")
    h.GetZaxis().SetTitleOffset(1.1)

    for i, lab in enumerate(pretty, start=1):
        h.GetXaxis().SetBinLabel(i, lab)
        h.GetYaxis().SetBinLabel(i, lab)

    for i in range(n):
        for j in range(n):
            h.SetBinContent(i + 1, j + 1, float(corr[i][j]))

    h.GetXaxis().SetLabelSize(0.030)
    h.GetYaxis().SetLabelSize(0.030)
    h.GetXaxis().LabelsOption("v")  # keep x labels vertical; y labels horizontal
    h.Draw("COLZ")

    t = ROOT.TLatex()
    t.SetNDC(True)
    t.SetTextSize(0.035)
    t.DrawLatex(0.28, 0.94, f"{base}{suffix}")
    t.SetTextSize(0.030)
    t.DrawLatex(0.28, 0.905, f"rotated: {'yes' if rotated else 'no'}")

    os.makedirs(out_dir, exist_ok=True)
    c.SaveAs(os.path.join(out_dir, f"{base}{suffix}_correlation.pdf"))
    c.SaveAs(os.path.join(out_dir, f"{base}{suffix}_correlation.png"))
    helpers.copyIndexPHP(out_dir)
    c.Close()

if __name__ == "__main__":
    import common.syncer as syncer
    import contextlib

    # ---------------- args ----------------
    import argparse
    p = argparse.ArgumentParser(description="Likelihood fit")
    p.add_argument("configs", nargs="+",help="Path to one or more global YAML configs")
    p.add_argument("--base", help="Base name for fit result and cache directories")
    p.add_argument("--overwrite", nargs="?", const="all", default=None, choices=["fit", "all"],
                   help="Overwrite results: 'fit' overwrites fit JSON only; 'all' overwrites fit JSON and cache.")
    p.add_argument("--rotate", action="store", default=None, help="Point to a rotate JSON")
    p.add_argument("--freezePOI", type=float, default=None,
                   help="If used with --rotate, freeze rotated POIs with eigenvalue < threshold to 0.")
    p.add_argument("--no_syst", action="store_true", help="Disable all nuisances (freeze to 0).")
    p.add_argument("--syst_only", action="store_true", help="Disable all POIs (freeze to 0).")
    p.add_argument("--data", action="store_true", help="Fits to data defined in config.")
    p.add_argument("--toyFile", default=None,
                   help="Path to a toy HDF5 file generated by fit/ToyGenerator.py; fits to it instead of "
                        "data/Asimov. Generation-only: no spec parsing, seeding or point selection happens here.")
    p.add_argument("--asimov", nargs="+", default=None,  metavar=("PAR", "VAL"),
                   help="Set an off-nominal Asimov hypothesis via pairs: --asimov par1 val1 par2 val2 ...")
    p.add_argument("--shuffle", nargs="+", default=None,  help="Shuffle these features")
    p.add_argument("--verbosity", type=int, default=1, help="Verbosity passed to the fitter")
    p.add_argument("--minos", action="store_true", default=False,
                   help="Whether to use MINOS in the fit (POIs only). If not set, then use HESSE by default.")
    p.add_argument("--minosNP", nargs="+", default=None, help="NPs for which to derive MINOS uncertainties. Only works if fit is ran with --minos. 'all' runs MINOS for all NPs.")
    p.add_argument("--minuit", action="store_true", default=False,
                   help="Use the original iminuit/MIGRAD backend instead of the autograd+SciPy backend.")
    args = p.parse_args()

    import common.yaml_loader as yaml_loader

    if not (args.minuit or _HAS_AUTOGRAD):
        raise ImportError("Trying to run autograd fit but autograd was not imported properly.")

    # doing it this way, since print_summary and load_surrogates
    # use the path of the configs to give info to the user
    list_configs = []
    for config_path in args.configs:
        aux_cfg = yaml_loader.load_yaml(config_path)
        yaml_loader.print_summary(aux_cfg, config_path, yaml_loader._INCLUDE_TRACE)
        yaml_loader.load_surrogates(aux_cfg, config_path, overwrite=False)

        list_configs.append(aux_cfg)
    
    cfg = yaml_loader.combine_configs(list_configs)

    like_info = load_likelihood(cfg)
    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    rotated = bool(args.rotate)
    hyp_for_fit = Rotated(hyp, args.rotate, name="Fisher-basis") if rotated else hyp

    if args.no_syst and args.syst_only:
        raise ValueError("You cannot ask for a fit with --no_syst and --syst_only.")

    if args.no_syst:
        for p_ in hyp.nuisances + hyp_for_fit.nuisances:
            p_.val = 0.0
            p_.isFrozen = True
        print("[opts] --no_syst: all nuisances set to 0 and frozen.")
    elif args.syst_only:
        for p_ in hyp.POIs + hyp_for_fit.POIs:
            # p_.val = 0.0 # do I need this ? I don't think I do.
            p_.isFrozen = True
        print("[opts] --syst_only: all POIs frozen.")

    # currently rate parameters are implemented using unpenalized/floating lnN
    # which is defined as a systematic (fit should be run with --syst_only)
    # POI field (from ICH) still has to be in config to get nominal yields

    # code below avoids fitting rate and ICH POIs (e.g. forgetting --syst_only option)
    if not args.syst_only and not args.no_syst:
        # classes with active ICH POI and floating lnN parameters
        offenders: list[str] = []

        for section_name in ("regions", "binned"):
            for region_cfg in like_info[section_name]:
                region_id = region_cfg["id"]

                for class_cfg in region_cfg["classes"]:
                    class_id = class_cfg["id"]
                    poi_cfg = class_cfg.get("POI", {})
                    poi_type = poi_cfg.get("type")
                    poi_parameters = poi_cfg.get("parameters", [])

                    has_active_poi = (poi_type is not None) and (len(poi_parameters) > 0)

                    has_floating_lnn = any(
                        (syst_cfg.get("type") == "lnN") and bool(syst_cfg.get("floating", False))
                        for syst_cfg in class_cfg.get("systematics", [])
                    )

                    if has_active_poi and has_floating_lnn:
                        offenders.append(f"{section_name}:{region_id}/{class_id}")

        if offenders:
            raise RuntimeError(
                "Invalid configuration for non-syst-only fit: active POI with floating lnN in "
                f"{offenders}. Run with --syst_only or fix the configuration."
            )

    # -------- paths (fit + plots) --------
    import common.user as user
    
    # base from mangling together configs or given by user
    base_list = []
    for config_path in args.configs:
        base_list.append(os.path.splitext(os.path.basename(config_path))[0])
    
    base = "_".join(base_list) 
    
    if args.base:
        base = args.base

    version = str(cfg.get("version", "v0"))
    suffix = ("_nosyst" if args.no_syst else "") + ("_rotate" if rotated else "")
    if args.freezePOI is not None and (args.syst_only == False):
        suffix += f"_freezePOI{args.freezePOI:g}"
    if args.syst_only:
        suffix = "_systonly"
    if args.shuffle:
        suffix += "_" + "_".join(args.shuffle)
        print(f"Shuffling these features: {','.join(args.shuffle)}")
    if args.data:
        suffix += "_data"
        print("Fitting to data!")
    toy_info = None
    out_path = user.output_directory
    if args.toyFile:
        with h5py.File(args.toyFile, "r") as _toy_meta_f:
            _toy_point = str(_toy_meta_f["meta"].attrs.get("point", "")) or "toy"
            _toy_seed = int(_toy_meta_f["meta"].attrs["seed"])
            _toy_source = str(_toy_meta_f["meta"].attrs.get("source", ""))
            _toy_hypothesis = json.loads(str(_toy_meta_f["meta"].attrs["hypothesis"]))
        toy_info = {"point": _toy_point, "source": _toy_source,
                    "seed": _toy_seed, "hypothesis": _toy_hypothesis}
        suffix += f"_{_toy_point}_{_toy_source}_toy{_toy_seed}"
        # storing many toy fit results in their own folder
        out_path = os.path.join(out_path, f"{base}_{_toy_point}_{_toy_source}_toy_fits")
        print(f"Fitting to toy '{_toy_point}' seed {_toy_seed} from {args.toyFile} (source: {_toy_source})")

    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, f"{base}_{version}{suffix}_fit.json")

    plot_dir = os.path.join(user.plot_directory, "likelihood_fit", base)
    if toy_info:
        plot_dir = os.path.join(plot_dir, f"{toy_info['point']}_{toy_info['source']}_toy_fits")
    
    plot_dir = os.path.join(plot_dir, f"{version}{suffix}")
    os.makedirs(plot_dir, exist_ok=True)

    overwrite_fit = args.overwrite in ("fit", "all")
    overwrite_cache = args.overwrite == "all"

    fit_log_path = os.path.join(plot_dir, f"fit_log_{version}{suffix}.txt")

    if args.freezePOI is not None and not rotated:
        raise RuntimeError("--freeze-poi requires --rotate")

    # Make sample loader factory from default cfg
    samples_mod = importlib.import_module(cfg["defaults"]["module_samples"])

    from common.yaml_loader import _resolve_features_list
    default_features = cfg["defaults"].get("default_features", None)
    features = _resolve_features_list(default_features) if default_features else None
    factory = samples_mod.Factory(
        features=features,
        selection=cfg["defaults"].get("default_selection", None),
        selection_features=cfg["defaults"].get("default_selection_features", None),
    )

    fit = None
    fit_loaded = False
    with open(fit_log_path, "w", encoding="utf-8") as _fit_log:
        _tee = helpers.Tee(sys.stdout, _fit_log, ascii_only=True)
        with contextlib.redirect_stdout(_tee), contextlib.redirect_stderr(_tee):

            # -------- optional rotated-POI EV printout / freezing --------
            if rotated:
                with open(args.rotate, "r") as f:
                    rot_payload = json.load(f)

                basis_labels = list(rot_payload.get("basis_labels", []) or [])
                eigenvalues = list(rot_payload.get("eigenvalues", []) or [])

                if basis_labels and eigenvalues and len(basis_labels) != len(eigenvalues):
                    raise RuntimeError(
                        f"Rotation JSON mismatch: len(basis_labels)={len(basis_labels)} "
                        f"!= len(eigenvalues)={len(eigenvalues)}"
                    )

                ev_by_name = {lab: float(ev) for lab, ev in zip(basis_labels, eigenvalues)}

                print("[rotation] Rotated POIs and eigenvalues:")
                for var in hyp_for_fit.POIs:
                    ev = ev_by_name.get(var.name, None)
                    if ev is None:
                        print(f"  {var.name:>12s}   EV = <not provided>")
                    else:
                        print(f"  {var.name:>12s}   EV = {ev:.6e}")

                if args.freezePOI is not None:
                    thr = float(args.freezePOI)
                    frozen_rotated = []
                    kept_rotated = []

                    for var in hyp_for_fit.POIs:
                        ev = ev_by_name.get(var.name, None)
                        if ev is not None and ev < thr:
                            var.freeze(value=0.0)
                            frozen_rotated.append((var.name, ev))
                        else:
                            kept_rotated.append((var.name, ev))

                    print(f"[rotation] --freeze-poi = {thr:.6e}")
                    if frozen_rotated:
                        print("[rotation] Frozen rotated POIs:")
                        for nm, ev in frozen_rotated:
                            print(f"  {nm:>12s}   EV = {ev:.6e}")
                    else:
                        print("[rotation] No rotated POIs were frozen.")

                    print("[rotation] Rotated POIs kept floating:")
                    for nm, ev in kept_rotated:
                        if ev is None:
                            print(f"  {nm:>12s}   EV = <not provided>")
                        else:
                            print(f"  {nm:>12s}   EV = {ev:.6e}")

            # -------- load fit if available --------
            if (not overwrite_fit) and os.path.exists(out_path):
                fit = json.load(open(out_path))
                fit_loaded = True
                print(f"[info] Loaded existing fit result from {out_path}")
            else:
                n2ll = N2LL(
                    like_info,
                    factory=factory,
                    cache_subdir=os.path.join("NN2LCache", base, cfg["version"]),
                    cache_root=None,
                    overwrite=overwrite_cache,
                )
                n2ll.shuffle_features = args.shuffle
                n2ll.build_cache()
                n2ll.prepare_runtime()

                # allow unbinned and binned regions simultaneously
                # they should have different names
                # e.g. SR_2016 (unbinned) and CR_2016 (binned)

                # default for data, but can be changed below,
                # if one wants to pass a MC sample/toy as "data sample"
                ignore_weights = True
                # unbinned region data
                unbinned_dataloaders = {}
                for region in like_info['regions']:
                    region_id = region['id']
                    if 'data' in region:
                        loader = factory.get(region['data']['sample'])
                        # if 'selection' in region:
                        #     loader.addSelection(region['selection'])
                        unbinned_dataloaders.update({region_id: loader})
                        # if there's at least one region with a weighted sample for "data"
                        # will use weights. not an issue if mix weighted/unweighted as "data"
                        # weights are usually ~ 1
                        if (region['data'].get('ignore_weights', True) is False) and ignore_weights:
                            ignore_weights = False
                            

                # binned region data
                binned_dataloaders = {}
                for region in like_info['binned']:
                    region_id = region['id']
                    if 'data' in region:
                        loader = factory.get(region['data']['sample'])
                        # if 'selection' in region:
                        #     loader.addSelection(region['selection'])
                        binned_dataloaders.update({region_id: loader})
                        if (region['data'].get('ignore_weights', True) is False) and ignore_weights:
                            ignore_weights = False

                # ---- toy: generated separately by fit/ToyGenerator.py, loaded (never generated) here ----
                if args.toyFile:
                    if args.data or args.asimov is not None:
                        raise RuntimeError("--toyFile cannot be combined with --data or --asimov.")
                    from fit.ToyGenerator import load_toy
                    toy = load_toy(args.toyFile, n2ll)
                    n2ll.setToy(toy, hyp_for_fit)

                # data
                elif args.data:
                    if (not unbinned_dataloaders) and (not binned_dataloaders):
                        raise ValueError("You asked for a data fit, but did not define any dataset in your config. Exiting!")
                    n2ll.setObservation(unbinned_dataloaders, binned_dataloaders, ignore_weights=ignore_weights)

                # ---- on-nominal Asimov point ----
                elif args.asimov is None:
                    n2ll.setAsimov()

                # ---- optional off-nominal Asimov point ----
                else:
                    if rotated:
                        raise NotImplementedError

                    if len(args.asimov) % 2 != 0:
                        raise RuntimeError(
                            f"--asimov expects pairs PAR VAL (even number of tokens), got: {args.asimov}"
                        )

                    asimov_kwargs = {}
                    for i in range(0, len(args.asimov), 2):
                        par = args.asimov[i]
                        try:
                            val = float(args.asimov[i + 1])
                        except ValueError as e:
                            raise RuntimeError(
                                f"--asimov value for '{par}' must be a float, got '{args.asimov[i+1]}'"
                            ) from e
                        asimov_kwargs[par] = val

                    asimov_h = hyp.cloneModify(**asimov_kwargs)
                    print(f"[opts] --asimov: setting Asimov hypothesis to {asimov_kwargs}")
                    n2ll.setAsimov(asimov_h)

                # currently using default fit step=None
                # which sets step to 0.1 for all parameters
                # (see function definition)
                # step can also be a single value or a dictionary
                _fitter = run_iminuit_fit if args.minuit else run_autograd_fit
                if args.minuit:
                    print("[opts] Using iminuit/MIGRAD backend (--minuit)")
                else:
                    print("[opts] Using autograd+SciPy backend (default)")
                m = _fitter(
                    n2ll,
                    hyp_for_fit,
                    print_every=1,
                    do_migrad=True,
                    do_hesse=True,
                    do_minos=args.minos,
                    minosNP=args.minosNP,
                    verbosity=args.verbosity,
                )

                serialize_result(m, base, version, args, out_path, toy_info=toy_info)
                fit = json.load(open(out_path))

            print("Best -2logL =", fit["fval"])
            print("Correlation")
            print(np.asarray(fit["correlation"]["matrix"]))
            # -------- generic Minuit covariance diagnosis (fit basis only) --------
            cov = np.asarray(fit["covariance"]["matrix"], dtype=np.float64)
            cov = 0.5 * (cov + cov.T)  # symmetrize numerically
            names_cov = list(fit["free_parameter_order"])

            evals_cov, evecs_cov = np.linalg.eigh(cov)   # ascending
            lam_abs_max = float(np.max(np.abs(evals_cov))) if len(evals_cov) else 0.0

            tol_neg = 1e-10 * max(1.0, lam_abs_max)
            tol_tiny = 1e-8 * max(1.0, lam_abs_max)

            neg_idx = [i for i, lam in enumerate(evals_cov) if lam < -tol_neg]
            tiny_idx = [i for i, lam in enumerate(evals_cov) if abs(lam) <= tol_tiny]

            pos = evals_cov[evals_cov > tol_neg]
            cond = float(np.max(pos) / np.min(pos)) if len(pos) else np.inf

            print("[covariance] eigensystem diagnostics in fitted-parameter basis")
            print("  parameter order:", names_cov)
            print("  max |eigenvalue| =", lam_abs_max)
            print("  negative modes   =", neg_idx)
            print("  tiny modes       =", tiny_idx)
            print("  cond(pos part)   =", cond)

            # largest covariance eigenvalues = weakest constrained combinations
            n_show = min(5, len(evals_cov))

            print(f"[covariance] Largest {n_show} eigenvalues:")
            for k in range(len(evals_cov) - n_show, len(evals_cov)):
                lam = float(evals_cov[k])
                print(f"  mode {k:2d}: lambda = {lam:.6e}")

            print("[covariance] Best constrained combinations:")
            for k in range(n_show):
                lam = float(evals_cov[k])
                vec = evecs_cov[:, k]
                order = np.argsort(-np.abs(vec))

                print(f"  mode {k:2d}: lambda = {lam:.6e}")
                n_printed = 0
                for j in order:
                    if abs(vec[j]) < 0.15:
                        continue
                    print(f"    {names_cov[j]:>16s} : {vec[j]:+.4f}")
                    n_printed += 1
                    if n_printed >= 6:
                        break
                if n_printed == 0:
                    for j in order[:3]:
                        print(f"    {names_cov[j]:>16s} : {vec[j]:+.4f}")

            print("[covariance] Least constrained combinations:")
            for k in range(len(evals_cov) - n_show, len(evals_cov)):
                lam = float(evals_cov[k])
                vec = evecs_cov[:, k]
                order = np.argsort(-np.abs(vec))

                print(f"  mode {k:2d}: lambda = {lam:.6e}")
                n_printed = 0
                for j in order:
                    if abs(vec[j]) < 0.15:
                        continue
                    print(f"    {names_cov[j]:>16s} : {vec[j]:+.4f}")
                    n_printed += 1
                    if n_printed >= 6:
                        break

    # store fit log unless we loaded from disk
    if not fit_loaded:
        syncer.file_sync_storage.append(fit_log_path)

    # -------- plots --------
    names = fit["free_parameter_order"]

    fit_vals = {p["name"]: float(p["value"]) for p in fit["parameters"]}
    fit_errs = {p["name"]: float(p["error"]) for p in fit["parameters"]}
    for nm in names:
        fit_errs[f"{nm}_lo"] = fit_errs[nm]
        fit_errs[f"{nm}_hi"] = fit_errs[nm]

    plot_fit_summary_root(
        plot_dir,
        base,
        rotated=rotated,
        hyp=hyp_for_fit,
        fit_vals=fit_vals,
        fit_errs=fit_errs,
        suffix=f"_{version}{suffix}",
    )

    plot_correlation_root(
        plot_dir,
        base,
        rotated=rotated,
        names=names,
        corr=fit["correlation"]["matrix"],
        suffix=f"_{version}{suffix}",
    )
    try:
        syncer.sync()
    except Exception as e:
        print(f"[sync] Warning: failed to sync outputs ({e}).")
        # Prevent duplicate failing retries from the atexit sync hook.
        try:
            syncer.file_sync_storage = []
        except Exception:
            pass

