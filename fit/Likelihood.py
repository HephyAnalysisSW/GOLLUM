from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from iminuit import Minuit


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

import numpy as np

import sys
sys.path.insert(0, '..')

from fit.Modeling import ModelParameter, Hypothesis
from common.helpers import _binning_equal
 
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
            # POI (BIT)
            poi = C.get("POI", {}) or {}
            poi_job_id = poi.get("job")
            if poi_job_id:
                bit_job = id2job.get(poi_job_id) or _job_by_id(cfg, poi_job_id)
                poi['predictor'] = _predictor_from_job(bit_job)
                if poi['predictor'] is None:
                    logger.warning(f"[likelihood] BIT '{poi_job_id}' has no predictor attached yet.")
            # collect POI parameter names
            for nm in (poi.get("paramaters") or poi.get("parameters") or []):
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

                elif styp == "lnN":
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                else:
                    # future unbinned syst types
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)

    # -----------------------
    # Binned regions (ICH/ICPH)
    # -----------------------
    for R in binned:
        # classes
        classes = R.get("classes", []) or []
        for C in classes:
            # POI (ICH)
            poi = C.get("POI", {}) or {}
            poi_job_id = poi.get("job")
            if poi_job_id:
                ich_job = id2job.get(poi_job_id) or _job_by_id(cfg, poi_job_id)
                poi['predictor'] = _predictor_from_job(ich_job)
                if poi['predictor'] is None:
                    logger.warning(f"[likelihood] ICH '{poi_job_id}' has no predictor attached yet.")
            for nm in (poi.get("paramaters") or poi.get("parameters") or []):
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

                elif styp == "lnN":
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                else:
                    # future binned syst types
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)

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
                elif not _binning_equal(names, edges, ref_names, ref_edges):
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
                elif not _binning_equal(names, edges, ref_names, ref_edges):
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

    # Return both sections enriched
    return {'regions': regions, 'binned': binned, 'pois': pois_list, 'nuisances': nuis_list}

def build_hypothesis_from_likelihood(like_info, *, name=None,
                                     poi_init=0.0, nuis_init=0.0,
                                     penalize_nuisances=True):
    """
    Convenience: construct a Hypothesis from load_likelihood(...) output.
    Includes parameters discovered in BOTH unbinned and binned sections.

    Heuristics:
      - POIs are marked isPOI=True if name starts with 'c'.
      - Nuisances are marked penalized unless penalize_nuisances=False.
    """
    pois = like_info.get('pois', []) or []
    nuis = like_info.get('nuisances', []) or []

    params = []
    for nm in pois:
        is_wc = nm.startswith('c')
        params.append(ModelParameter(name=nm, val=poi_init, isPOI=True, isPenalized=False))
    for nm in nuis:
        params.append(ModelParameter(
            name=nm, val=nuis_init, isPOI=False,
            isPenalized=bool(penalize_nuisances)
        ))
    return Hypothesis(parameters=params, name=name or "from_yaml")


try:
    from numba import njit, prange
    _NUMBA = True
except Exception:
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
                t = 0.5 + xi * (-1.0/3.0 + xi * (1.0/4.0 + xi * (-1.0/5.0 + xi * (1.0/6.0))))
                y = -x2 * t
            else:
                y = math.log1p(xi) - xi
            s += wi * y
        return s
else:
    def _weighted_sum_log1p_minus_x(x: np.ndarray, w: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        y = np.empty_like(x)
        small = (np.abs(x) < 1e-4) & (x > -1.0)
        if np.any(small):
            xs = x[small]
            s_small = xs*xs * (0.5 + xs*(-1/3 + xs*(1/4 + xs*(-1/5 + xs*(1/6)))))
            y[small] = -s_small
        big = ~small
        if np.any(big):
            xb = x[big]
            y[big] = np.log1p(xb) - xb
        return float(np.sum(w * y, dtype=np.float64))


def _expand_pois_linear_quadratic(poi_names: List[str], poi_values: Dict[str, float]) -> np.ndarray:
    N = len(poi_names)
    c = np.array([float(poi_values.get(n, 0.0)) for n in poi_names], dtype=np.float64)
    quads = []

    #FIXME careful here, double sum
    # This is the logic:
    # BIT predicts and works with *derivatives*, so R = 1 + c_A R_A = 1 + Sum_a ca Ra + 1/2 Sum_{a, b} ca cb Ra Rb (Taylor expansion)
    # Now, the double sum is slow so we write
    # R = 1 + Sum_a ca Ra + Sum_{a, b>=a} factor ca cb Ra Rb where factor = 1/2 if a=b (same factor as before) but factor=1 if b>a (counting twice)
    # My silicon friend didn't see that. (For the PNN I rather work with unique ordered sequences, so no prefactor)
    for i in range(N):
        for j in range(i,N): 
            quads.append((0.5 if i==j else 1) * c[i] * c[j])  # 1/2 c_i c_j
    return np.concatenate([c, np.asarray(quads, dtype=np.float64)], axis=0) if quads else c


def _nuis_to_A_vector(param_names: List[str], combinations: List[Tuple[str, ...]], values: Dict[str, float]) -> np.ndarray:
    if not combinations:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(len(combinations), dtype=np.float64)
    for k, comb in enumerate(combinations):
        v = 1.0
        for p in comb:
            v *= float(values.get(p, 0.0))
        out[k] = v
    return out


def _predict_classifier(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        P = model.predict(X)
    else:
        raise RuntimeError("Classifier predictor lacks predict method.")
    P = np.asarray(P)
    if P.ndim != 2:
        raise RuntimeError("Classifier output must be (N, n_classes).")
    return P.astype(np.float64, copy=False)


def _predict_bit_ratio(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        Y = model.predict(X)
    #elif hasattr(model, "predict_A"):
    #    Y = model.predict_A(X)
    else:
        raise RuntimeError("BIT predictor lacks predict.")
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    return Y.astype(np.float64, copy=False)


def _predict_pnn_deltaA(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "deltaA"):
        return np.asarray(model.deltaA(X), dtype=np.float64)
    else:
        raise RuntimeError("PNN predictor lacks deltaA(X).")


def _class_index(classes: List[Dict[str, Any]], cid: str) -> int:
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
                 module_samples: str,
                 cache_subdir: str = "caches",
                 cache_root: Optional[str] = None,
                 overwrite: bool = False,
                 eval_chunk_size: int = 200_000):
        import importlib, os
        self.lk = likelihood
        self.regions = list(likelihood.get('regions', []))
        self.module_samples = module_samples
        self.samples_mod = importlib.import_module(module_samples)
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
        self.cache_root = cache_root or os.path.join(base_dir, cache_subdir)
        os.makedirs(self.cache_root, exist_ok=True)

        # in-memory pointers
        self._poi_order: Dict[Tuple[str, str], List[str]] = {}         # (rid,cid) -> POI names order for R_A
        self._cache_paths: Dict[Tuple[str, str], Tuple[str, str]] = {} # (rid,cid) -> (h5_path, meta_path)

        # opened runtime state (filled by prepare_runtime)
        self._h5: Dict[Tuple[str, str], "h5py.File"] = {}
        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._N_region: Dict[str, int] = {}                            # region id -> number of events
        self._class_ids_by_region: Dict[str, List[str]] = {}           # region id -> [class ids in order]
        self._lnN_by_class: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

        # attach predictors from likelihood (already set by yaml_loader.load_likelihood)
        self._prepare_structure()

        # flow control
        self._runtime_prepared      = False
        self._asimov_hypothesis_set = False
        self._observation_set       = False

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
                if not hasattr(self.samples_mod, sname):
                    raise RuntimeError(f"[N2LL] Asimov sample '{sname}' not found in {self.module_samples}")
            R['_asimov_samples'] = asimov_list

            for C in R.get('classes', []):
                cid = C['id']
                key = (rid, cid)

                poi = C.get('POI', {}) or {}
                poi_pred = poi.get('predictor', None)
                poi_names = list(poi.get('parameters', []) or [])
                if poi_pred is None:
                    raise RuntimeError(f"[N2LL] Missing BIT predictor for {rid}/{cid}")
                if not poi_names:
                    raise RuntimeError(f"[N2LL] No POI parameter names for {rid}/{cid}")
                self._poi_order[key] = poi_names

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
        d = os.path.join(self.cache_root, region_id)
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
        import numpy as np
        feat_names_ref = None
        for sname in region['_asimov_samples']:
            L = getattr(self.samples_mod, sname)

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

                print(h5_path, meta_path)
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
                L = getattr(self.samples_mod, sname)
                total_shards += len(getattr(L, "base", L))

            # prepare column selection mask
            if clf is not None:
                clf['column_mask'] = self.make_column_mask(L.feature_names, clf.feature_names)
            for C in classes:
                poi_pred = (C.get('POI') or {}).get('predictor')
                if poi_pred:
                    C['POI']['column_mask'] = self.make_column_mask(L.feature_names, poi_pred.feature_names)
                for S in C['_pnn_systs']:
                    # I forgot to store the feature_names in the PNN class, so let's look it up from the job:
                    S['column_mask'] = self.make_column_mask(L.feature_names, _job_by_id(cfg, S['job'])['features'])

            with tqdm(total=total_shards, desc=f"[N2LL] cache {rid}", unit="shard", leave=False) as pbar:
                for feat_names, X, w0 in self._iter_asimov_batches(R):
                    Nb = len(X)
                    if Nb == 0:
                        pbar.update(1)
                        continue

                    # Classifier probabilities for this batch (or ones)
                    if clf is None or n_proc <= 1:
                        G = np.ones((Nb, n_proc), dtype=np.float64)
                    else:
                        G = _predict_classifier(clf, X[:, clf['column_mask']])  # (Nb, n_proc)
                        if G.shape[1] != n_proc:
                            raise RuntimeError(f"[N2LL] Classifier outputs {G.shape[1]} != {n_proc} classes in region '{rid}'")

                    # For each class that needs building: compute and append
                    for C in classes:
                        cid = C['id']
                        if not needs[cid]:
                            continue
                        p_index = _class_index(classes, cid)
                        writer = writers[cid]
                        f = writer["file"]

                        # w0 and g slice
                        self._append_1d(writer["w0"], w0)
                        self._append_1d(writer["g"],  G[:, p_index].astype(np.float64, copy=False))

                        # BIT R_A
                        poi_pred = (C.get('POI') or {}).get('predictor')
                        R_A = _predict_bit_ratio(poi_pred, X[:, C['POI']['column_mask']])  # (Nb, nA)

                        if writer["R"] is None:
                            nA = R_A.shape[1]
                            writer["R"] = f.create_dataset("R", (0, nA), maxshape=(None, nA), dtype="f8", chunks=True)
                            first_batch_shapes.setdefault(cid, {})["nA"] = nA
                        self._append_2d(writer["R"], R_A)

                        # PNN Δ groups
                        for S in C['_pnn_systs']:
                            sid = S['id']
                            pnn = S['predictor']
                            dA = _predict_pnn_deltaA(pnn, X[:, S['column_mask']])  # (Nb, nB)
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
                # ICPh groups
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
                    # stash a meta dict we’ll enrich with deltas as numpy arrays for fast math
                    gm = {
                        'id': S['id'],
                        'params': list(S.get('parameters', []) or []),
                        'combs':  [list(t) for t in (getattr(pred, "combinations", []) or [])],
                        # We store deltas as (nB, nb1) or (nB, nb1, nb2) in float64
                        '_deltas': np.asarray(pred.deltas, dtype=np.float64),
                        '_obj': pred,
                    }
                    icph_groups.append({'_meta': gm})

                classes.append({'id': cid,
                                '_ich': ich,
                                '_poi_params': poi_params,
                                '_icph_systs': icph_groups})

                # keep POI order so we can build c-vectors
                self._poi_order[(rid, cid)] = poi_params

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

                self._h5[(rid, cid)] = f
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
        ICH expects the *plain* c-vector in parameter order (not expanded A-basis).
        """
        C = None
        # find POI params order for this class (already stored for unbinned BIT; reuse mapping)
        poi_names = self._poi_order.get((rid, cid), None)
        if poi_names is None:
            # For binned ICH we still stored _poi_order in prepare step (below)
            raise RuntimeError(f"[binned] Missing POI names order for ({rid}/{cid}).")
        C = np.array([float(getattr(hypothesis, name, self[name]).val) if name in hypothesis else float(self[name].val)
                      for name in poi_names], dtype=np.float64)
        return C

    def _assemble_nuA_groups_binned(self, rid: str, hypothesis) -> dict[str, list[tuple[dict, np.ndarray]]]:
        """
        ν_A vector per ICPh group, per class, for a given hypothesis.
        """
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        out: dict[str, list[tuple[dict, np.ndarray]]] = {}
        for C in self._binned_classes_by_region.get(rid, []):
            cid = C['id']
            groups = []
            for S in C.get('_icph_systs', []):
                gm = S['_meta']  # filled in prepare
                params = list(gm.get("params", []))
                combs  = [tuple(c) for c in gm.get("combs", [])]
                nuA = _nuis_to_A_vector(params, combs, nu_vals)
                groups.append((gm, nuA))
            out[cid] = groups
        return out

    # ---------- Binned λ builder ----------
    def _compute_lambda_binned(self, rid: str, hypothesis) -> np.ndarray:
        """
        Build λ_i(c,ν) for all bins in a binned region rid by summing processes.
        """
        # per-class lnN (same as unbinned)
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
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
            cvec = np.array([float(p.val) for p in getattr(hypothesis, 'POIs', []) if p.name in C['_poi_params']])
            # IMPORTANT: ICH.predict takes the plain c-vector in the same order as variables
            sigma_hist = ich.predict(cvec)  # shape (nb1,) or (nb1,nb2)

            # accumulate nuisance exponent per bin from all ICPh groups
            # we’ll form a flat vector exp(exponent + ln_bias[cid])
            if sigma_hist.ndim == 1:
                nb1 = sigma_hist.shape[0]
                expo = np.zeros(nb1, dtype=np.float64)
                for gm, nuA in nuA_per_group[cid]:
                    dA = gm['_deltas']   # shape (nB, nb1)
                    expo += (nuA @ dA).astype(np.float64)  # (nb1,)
                lam += sigma_hist * np.exp(expo + ln_bias[cid])
            else:
                nb1, nb2 = sigma_hist.shape
                expo2d = np.zeros((nb1, nb2), dtype=np.float64)
                for gm, nuA in nuA_per_group[cid]:
                    dA = gm['_deltas']   # shape (nB, nb1, nb2)
                    # tensordot over combination axis -> (nb1,nb2)
                    expo2d += np.tensordot(nuA, dA, axes=(0, 0)).astype(np.float64)
                lam += sigma_hist.reshape(-1) * np.exp(expo2d.reshape(-1) + ln_bias[cid])

        return lam


    # ---- assemble A-basis for POIs and nuisances from a hypothesis ----
    def _assemble_cA_per_class(self, rid: str, hypothesis) -> Dict[str, np.ndarray]:
        """Build c_A vectors per class for a given hypothesis."""
        cA_per_class: Dict[str, np.ndarray] = {}
        c_vec = {p.name: float(p.val) for p in getattr(hypothesis, 'POIs', [])}
        for cid in self._class_ids_by_region.get(rid, []):
            poi_names = self._poi_order[(rid, cid)]
            cA_per_class[cid] = _expand_pois_linear_quadratic(poi_names, c_vec)
        return cA_per_class

    def _assemble_nuA_groups(self, rid: str, hypothesis) -> Dict[str, list[tuple[dict, np.ndarray]]]:
        """Build ν_A vectors per Δ-group for a given hypothesis."""
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        nuA_per_group: Dict[str, list[tuple[dict, np.ndarray]]] = {}
        for cid in self._class_ids_by_region.get(rid, []):
            meta = self._meta[(rid, cid)]
            groups = []
            for gm in meta.get("delta_groups", []):
                params = list(gm.get("params", gm.get("parameters", [])) or [])
                combs  = [tuple(c) for c in gm.get("combs", gm.get("combinations", [])) or []]
                nuA    = _nuis_to_A_vector(params, combs, nu_vals)
                groups.append((gm, nuA))
            nuA_per_group[cid] = groups
        return nuA_per_group

    def _compute_T_chunk(self, rid: str, cA_per_class, nuA_per_group, ln_bias_map, start: int, stop: int) -> np.ndarray:
        """
        Compute T(x; c, ν) on [start:stop) for a single region rid, summing over classes.
        T_i = Σ_p g_p(x_i) * [ (c⋅R_p)(x_i) * e^{Σ_s ν_B Δ_{p,B}(x_i)} + (e^{...} - 1) ].
        """
        f_first = self._h5[(rid, self._class_ids_by_region[rid][0])]
        M = stop - start
        T = np.zeros(M, dtype=np.float64)

        for cid in self._class_ids_by_region[rid]:
            f = self._h5[(rid, cid)]
            g_slice = f['g'][start:stop]                # (M,)
            R_slice = f['R'][start:stop, :]             # (M, nA)
            cA      = cA_per_class[cid]
            if R_slice.shape[1] != cA.shape[0]:
                raise RuntimeError(f"[N2LL] BIT dim {R_slice.shape[1]} != |A| {cA.shape[0]} for {rid}/{cid}")
            c_dot_R = R_slice @ cA                      # (M,)

            # build exponent from all Δ-groups
            expo = np.zeros_like(g_slice)
            for gm, nuA in nuA_per_group[cid]:
                dset = gm.get("dset", f"Delta::{gm['id']}")
                dA   = f[dset][start:stop, :]           # (M, nB)
                if dA.shape[1] != nuA.shape[0]:
                    raise RuntimeError(f"[N2LL] Δ dim {dA.shape[1]} != ν_A dim {nuA.shape[0]} for {rid}/{cid}/{gm['id']}")
                expo += dA @ nuA                        # (M,)

            # include per-class lnN bias additively in exponent
            exp_expo = np.exp(expo + ln_bias_map[cid])  # (M,)
            T += g_slice * (c_dot_R * exp_expo + (exp_expo - 1.0))

        return T

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
        any_nonzero = any(abs(float(p.val)) > 0.0 for p in getattr(hypothesis, 'parameters', []))
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
                cid: sum(log1p_alpha * float(hypothesis[nm].val) if nm in hypothesis else 0.0
                         for nm, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }

            # chunked compute and store
            chunk = self.eval_chunk_size
            Ts: list[np.ndarray] = []
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                T_chunk = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
                Ts.append(T_chunk)
            self._asimov_T[rid] = Ts

        # ----- also precompute binned Asimov λ'(i) if binned regions exist -----
        self._binned_asimov_lambda.clear()
        if self._asimov_active and self._binned_regions_ids:
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
        import numpy as np

        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.setObservation] Call prepare_runtime() before setting observation.")

        # You can’t mix observed-data mode with Asimov in the same evaluation flow.
        # We allow switching, but make it explicit and clear.
        if getattr(self, "_asimov_hypothesis_set", False) and getattr(self, "_asimov_active", False):
            print("[N2LL.setObservation] An Asimov hypothesis had been set; disabling it in favor of observed-data mode.")
        self._asimov_hypothesis_set = False
        self._asimov_active = False
        self._asimov_hyp = None
        self._asimov_T.clear()
        self._binned_asimov_lambda.clear()

        # Reset observation containers
        self._obs_unbinned = {}
        self._obs_binned_counts = {}

        # Flag we’re now in observed-data mode
        self._observation_set = True

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

                if Xs:
                    X_all = np.concatenate(Xs, axis=0)
                    w_all = np.concatenate(Ws, axis=0)
                else:
                    X_all = np.empty((0, len(getattr(loader, "feature_names", []))), dtype=np.float64)
                    w_all = np.empty((0,), dtype=np.float64)

                self._obs_unbinned[rid] = {'X': X_all, 'w': w_all}
                print(f"[setObservation] Unbinned region '{rid}': loaded {X_all.shape[0]:,} events "
                      f"({'unit weights' if ignore_weights else 'with weights'}).")

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
                        H, _ = np.histogram(x, bins=edges[0], weights=(w if w is not None else None))
                        counts += H.astype(np.float64)
                    else:
                        x = X[:, idx[0]]
                        y = X[:, idx[1]]
                        H, _, _ = np.histogram2d(x, y, bins=[edges[0], edges[1]], weights=(w if w is not None else None))
                        counts2d += H.astype(np.float64)

                flat_counts = counts if len(edges) == 1 else counts2d.reshape(-1)
                self._obs_binned_counts[rid] = flat_counts
                print(f"[setObservation] Binned region '{rid}': filled {flat_counts.size} bins "
                      f"({'unit weights' if ignore_weights else 'with weights'}).")

    def __call__(self, hypothesis) -> float:
        """
        Evaluate -2 log L for either:
          (A) OBSERVED data (registered via setObservation), or
          (B) ASIMOV expectation (registered via setAsimov).

        Exactly one of setObservation(...) or setAsimov(...) must be called.
        Asimov bias term is included only in case (B) and only if an off-nominal
        Asimov hypothesis was provided.
        """
        import numpy as np

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
                nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}

                for rid, block in self._obs_unbinned.items():
                    # Mode A: direct T
                    if 'T' in block:
                        T = np.asarray(block['T'], dtype=np.float64)
                        W = np.asarray(block['w'], dtype=np.float64)
                        total_unbinned += _weighted_sum_log1p_minus_x(T, W)
                        continue

                    # Mode B: by_class arrays -> reconstruct T with current (c,nu)
                    byc = block['by_class']
                    W   = np.asarray(block['w'], dtype=np.float64)
                    N   = len(W)
                    T   = np.zeros(N, dtype=np.float64)

                    # current hypothesis A-basis and ν_A groups
                    cA_per_class  = self._assemble_cA_per_class(rid, hypothesis)
                    nuA_per_group = self._assemble_nuA_groups(rid, hypothesis)
                    ln_bias = {
                        cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                                 for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                        for cid in byc.keys()
                    }

                    for cid, comp in byc.items():
                        g_slice = np.asarray(comp['g'], dtype=np.float64)     # (N,)
                        R_slice = np.asarray(comp['R'], dtype=np.float64)     # (N, nA)
                        cA      = cA_per_class[cid]                           # (nA,)
                        if R_slice.shape[1] != cA.shape[0]:
                            raise RuntimeError(f"[N2LL:obs:unbinned] BIT dim {R_slice.shape[1]} != |A| {cA.shape[0]} for {rid}/{cid}")
                        c_dot_R = R_slice @ cA                                # (N,)

                        expo = np.zeros_like(g_slice)
                        for gm, nuA in nuA_per_group[cid]:
                            dset = gm.get("dset", f"Delta::{gm['id']}")
                            if dset not in comp:
                                raise RuntimeError(f"[N2LL:obs:unbinned] Missing '{dset}' for {rid}/{cid}.")
                            dA = np.asarray(comp[dset], dtype=np.float64)     # (N, nB)
                            if dA.shape[1] != nuA.shape[0]:
                                raise RuntimeError(f"[N2LL:obs:unbinned] Δ dim {dA.shape[1]} != ν_A dim {nuA.shape[0]} for {rid}/{cid}/{gm['id']}")
                            expo += dA @ nuA

                        exp_expo = np.exp(expo + ln_bias[cid])
                        T += g_slice * (c_dot_R * exp_expo + (exp_expo - 1.0))

                    total_unbinned += _weighted_sum_log1p_minus_x(T, W)

            # ---------- BINNED (always available if you provided columns for axes) ----------
            if getattr(self, "_binned_regions_ids", None) and getattr(self, "_obs_binned", None):
                for rid in self._binned_regions_ids:
                    if rid not in self._obs_binned:
                        continue  # region not histogrammed (e.g. missing axis columns)
                    lam0 = self._binned_lambda0[rid]                    # (Nflat,)
                    lam  = self._compute_lambda_binned(rid, hypothesis) # (Nflat,)
                    Nobs = self._obs_binned[rid]                        # (Nflat,)

                    log_ratio = self._safe_log_ratio(lam, lam0)         # stable
                    total_binned += np.sum( -(lam - lam0) + Nobs * log_ratio, dtype=np.float64 )

            n2ll = -2.0 * (total_unbinned + total_binned)
            n2ll += hypothesis.penalty()
            return float(n2ll)

        # ===================================================================
        # (B) ASIMOV MODE  
        # ===================================================================
        total_sum = 0.0   # Σ w * (log1p(T) - T)
        bias_sum  = 0.0   # Σ w * T'(asimov) * log1p(T)

        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}

        for R in self.regions:
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue

            cA_per_class  = self._assemble_cA_per_class(rid, hypothesis)
            nuA_per_group = self._assemble_nuA_groups(rid, hypothesis)
            ln_bias = {
                cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                         for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }

            chunk = self.eval_chunk_size
            asimov_T_chunks = self._asimov_T.get(rid, None) if self._asimov_active else None
            for ichunk, start in enumerate(range(0, N, chunk)):
                stop = min(start + chunk, N)
                T = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)
                W = self._h5[(rid, class_ids[0])]['w0'][start:stop]
                total_sum += _weighted_sum_log1p_minus_x(T, W)

                if asimov_T_chunks is not None:
                    Tprime = asimov_T_chunks[ichunk]
                    bias_sum += float(np.sum(W * np.log1p(T) * Tprime, dtype=np.float64))

        total_unbinned = total_sum + bias_sum

        total_binned = 0.0
        if getattr(self, "_binned_regions_ids", None):
            for rid in self._binned_regions_ids:
                lam0 = self._binned_lambda0[rid]
                lam  = self._compute_lambda_binned(rid, hypothesis)
                lam_asimov = self._binned_asimov_lambda.get(rid, lam0)

                log_ratio = self._safe_log_ratio(lam, lam0)
                total_binned += np.sum( -(lam - lam0) + lam_asimov * log_ratio, dtype=np.float64 )

        n2ll = -2.0 * (total_unbinned + total_binned)
        n2ll += hypothesis.penalty()
        return float(n2ll)

    def fisher_information(self, hypothesis, step_scale: float = 1e-4, verbose: bool = False) -> np.ndarray:
        """
        Fisher information I_{ab} = E[(∂_a log L)(∂_b log L)] in Asimov mode.
        Requires that setAsimov(...) has been called (A-simov if None, C-simov otherwise).
        Uses the same decomposition as __call__: unbinned + binned (+ prior penalty).

          - Unbinned score per event:
                s_k(x_i) = w_i * [(T'(x_i) - T(x_i)) / (1 + T(x_i))] * ∂T/∂θ_k (x_i)

          - Binned score per bin i:
                s_k(i) = [ λ'_i / λ_i  - 1 ] * ∂λ_i/∂θ_k

          - Prior (Gaussian penalty added to -2 log L) contributes Hessian of penalty/2.

        Only *free* parameters (not frozen / not ignored) are included.
        Returns:
            (n_free, n_free) np.ndarray
        """
        import numpy as np

        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.fisher_information] Call prepare_runtime() first.")
        if not self._asimov_hypothesis_set or self._observation_set:
            raise RuntimeError("[N2LL.fisher_information] Asimov mode required: call setAsimov(...), "
                               "and do not set an observation.")

        # Free parameters (order matches Minuit's)
        names = [p.name for p in hypothesis.parameters
                 if not p.isFrozen and not getattr(p, "isIgnored", False)]
        npar = len(names)
        if npar == 0:
            raise RuntimeError("[N2LL.fisher_information] No free parameters.")

        if verbose:
            print("[FI] Free parameters:", names)

        # Small helpers kept local (no API changes elsewhere)
        def _clone_shift(hyp, pname, delta):
            h = hyp.clone()
            for p in h.parameters:
                if p.name == pname:
                    p.val = float(p.val) + float(delta)
                    break
            return h

        def _eps_for(pname: str, val: float) -> float:
            base = abs(val) if abs(val) > 0 else 1.0
            return max(1e-8, step_scale * base)

        def _ln_bias_map_for(rid: str, hyp):
            # Build per-class lnN bias for a hypothesis
            nu_vals = {p.name: float(p.val) for p in hyp.parameters if not p.isPOI}
            class_ids = self._class_ids_by_region.get(rid, [])
            return {
                cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                         for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }

        FI = np.zeros((npar, npar), dtype=np.float64)

        # =========================
        # UNBinned contribution
        # =========================
        for R in self.regions:
            print(f"At unbinned region {R['id']}")
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue
            print(f"class_ids {class_ids} N {N}")

            # Current-hypothesis structures
            cA_now  = self._assemble_cA_per_class(rid, hypothesis)
            nuA_now = self._assemble_nuA_groups(rid, hypothesis)
            ln_bias_now = _ln_bias_map_for(rid, hypothesis)

            print ("cA_now")
            print (cA_now)
            print ("nuA_now")
            print (nuA_now)
            print ("ln_bias_now")
            print (ln_bias_now)

            # Asimov chunks T'(x) if C-simov; else None for A-simov
            asimov_T_chunks = self._asimov_T.get(rid, None) if self._asimov_active else None

            print("asimov_T_chunks", asimov_T_chunks)

            chunk = self.eval_chunk_size
            for ichunk, start in enumerate(range(0, N, chunk)):
                stop = min(start + chunk, N)

                print ("ichunk", ichunk,"from",start,"to",stop)

                # Baseline T and weights
                T0 = self._compute_T_chunk(rid, cA_now, nuA_now, ln_bias_now, start, stop)    # (M,)
                W  = self._h5[(rid, class_ids[0])]['w0'][start:stop]                           # (M,)

                print ("T0",T0, "W", W)

                # Coefficient (T' - T)/(1 + T)
                if asimov_T_chunks is not None:
                    Tprime = asimov_T_chunks[ichunk]
                    coeff  = (Tprime - T0) / (1.0 + T0)
                    print ("Tprime", Tprime)
                else:
                    coeff  = (-T0) / (1.0 + T0)  # A-simov: T' = 0
                    print ("Tprime", None)

                print ("coeff", coeff)

                # Build dT/dθ_k via central finite differences
                dT_list = []
                for k, pname in enumerate(names):
                    v   = float(getattr(hypothesis, pname, hypothesis[pname]).val)
                    eps = _eps_for(pname, v)

                    hyp_p = _clone_shift(hypothesis, pname, +eps)
                    hyp_m = _clone_shift(hypothesis, pname, -eps)

                    # Structures for +eps/-eps (must also update lnN bias if pname is a nuisance)
                    cA_p,  nuA_p  = self._assemble_cA_per_class(rid, hyp_p), self._assemble_nuA_groups(rid, hyp_p)
                    cA_m,  nuA_m  = self._assemble_cA_per_class(rid, hyp_m), self._assemble_nuA_groups(rid, hyp_m)
                    ln_p,  ln_m   = _ln_bias_map_for(rid, hyp_p), _ln_bias_map_for(rid, hyp_m)

                    T_p = self._compute_T_chunk(rid, cA_p, nuA_p, ln_p, start, stop)
                    T_m = self._compute_T_chunk(rid, cA_m, nuA_m, ln_m, start, stop)

                    dT_list.append((T_p - T_m) / (2.0 * eps))
                    print ("T_p", T_p)
                    print ("T_m", T_m)
                    print ("dT_list",dT_list)

                # Scores for this chunk: s_k = W * coeff * dT_k
                S = np.stack([W * coeff * dT_k for dT_k in dT_list], axis=0)  # (npar, M)
                FI += S @ S.T

        # =========================
        # BINNED contribution
        # =========================
        if getattr(self, "_binned_regions_ids", None):
            for rid in self._binned_regions_ids:
                lam0 = self._binned_lambda0[rid]
                lam  = self._compute_lambda_binned(rid, hypothesis)
                lam_asimov = self._binned_asimov_lambda.get(rid, lam0)  # C-simov if active, else A-simov

                ratio_factor = (lam_asimov / np.maximum(lam, 1e-15)) - 1.0  # (Nflat,)

                dlam_list = []
                for k, pname in enumerate(names):
                    v   = float(getattr(hypothesis, pname, hypothesis[pname]).val)
                    eps = _eps_for(pname, v)

                    hyp_p = _clone_shift(hypothesis, pname, +eps)
                    hyp_m = _clone_shift(hypothesis, pname, -eps)

                    lam_p = self._compute_lambda_binned(rid, hyp_p)
                    lam_m = self._compute_lambda_binned(rid, hyp_m)

                    dlam_list.append((lam_p - lam_m) / (2.0 * eps))

                S = np.stack([ratio_factor * dlam_k for dlam_k in dlam_list], axis=0)  # (npar, Nflat)
                FI += S @ S.T

        # =========================
        # PRIOR / PENALTY contribution
        # =========================
        # __call__ adds:  n2ll += hypothesis.penalty()
        # ⇒ log L includes a prior term  - penalty/2
        # ⇒ Fisher adds the Hessian of (penalty/2) over the free-parameter subspace.
        def _penalty(hyp):
            return float(hyp.penalty())

        if any(getattr(p, "isPenalized", False) and (not p.isFrozen) and (not getattr(p, "isIgnored", False))
               for p in hypothesis.parameters):
            Hpen = np.zeros((npar, npar), dtype=np.float64)
            p0 = _penalty(hypothesis)
            for a, na in enumerate(names):
                va  = float(getattr(hypothesis, na, hypothesis[na]).val)
                epa = _eps_for(na, va)

                ha_p = _clone_shift(hypothesis, na, +epa)
                ha_m = _clone_shift(hypothesis, na, -epa)

                p_ap = _penalty(ha_p)
                p_am = _penalty(ha_m)

                Hpen[a, a] = (p_ap - 2.0*p0 + p_am) / (epa*epa)

                for b in range(a+1, npar):
                    nb  = names[b]
                    vb  = float(getattr(hypothesis, nb, hypothesis[nb]).val)
                    epb = _eps_for(nb, vb)

                    h_pp = _clone_shift(ha_p, nb, +epb)
                    h_pm = _clone_shift(ha_p, nb, -epb)
                    h_mp = _clone_shift(ha_m, nb, +epb)
                    h_mm = _clone_shift(ha_m, nb, -epb)

                    p_pp = _penalty(h_pp)
                    p_pm = _penalty(h_pm)
                    p_mp = _penalty(h_mp)
                    p_mm = _penalty(h_mm)

                    Hab = (p_pp - p_pm - p_mp + p_mm) / (4.0 * epa * epb)
                    Hpen[a, b] = Hab
                    Hpen[b, a] = Hab

            FI += 0.5 * Hpen

        if verbose:
            print("[FI] done.")
        return FI


class _MinuitArrayAdapter:
    """Array-based FCN for Minuit (keeps names, prints progress)."""
    def __init__(self, n2ll, hypothesis, names, print_every=25):
        self.n2ll = n2ll
        self.hyp = hypothesis
        self.names = list(names)
        self.free = [p for p in hypothesis.parameters if not p.isFrozen and not getattr(p, "isIgnored", False)]
        self.eval = 0
        self.print_every = max(1, int(print_every))

    def __call__(self, x):
        for i, p in enumerate(self.free):
            p.val = float(x[i])
        self.eval += 1
        f = float(self.n2ll(self.hyp))
        if (self.eval - 1) % self.print_every == 0:
            print(f"\n[eval {self.eval:6d}] f = {f: .6e}")
            self.hyp.print()
        return f

def make_minuit(n2ll, hypothesis, *, step=0.1, print_every=25):
    free = [p for p in hypothesis.parameters if not p.isFrozen and not getattr(p, "isIgnored", False)]
    if not free:
        raise RuntimeError("No free parameters to fit.")
    names = [p.name for p in free]
    x0 = np.array([float(p.val) for p in free], dtype=float)

    adapter = _MinuitArrayAdapter(n2ll, hypothesis, names, print_every=print_every)

    # FCN for iminuit with positional args; keep names via name=...
    def _fcn_positional(*x):
        return adapter(np.asarray(x, dtype=float))

    m = Minuit(_fcn_positional, *x0, name=names)
    m.errordef = 1.0  # -2logL / chi2 objective

    # set step sizes
    if isinstance(step, dict):
        for i, n in enumerate(names):
            m.errors[i] = float(step.get(n, 0.1))
    else:
        for i in range(len(names)):
            m.errors[i] = float(step)

    print("\n[make_minuit] Floating parameters:")
    for i, n in enumerate(names):
        print(f"  - {n:>16s}  start = {m.values[i]: .6e}  step = {m.errors[i]: .3g}")
    return m, adapter

def run_minuit_fit(n2ll, hypothesis, *, step=0.1, print_every=25,
                   do_migrad=True, do_hesse=True, do_minos=False):
    m, adapter = make_minuit(n2ll, hypothesis, step=step, print_every=print_every)

    if do_migrad:
        print("\n[MIGRAD]"); m.migrad(); print(m)
    if do_hesse:
        print("\n[HESSE]"); m.hesse(); print(m)
    if do_minos:
        poi_names = [p.name for p in getattr(hypothesis, "POIs", []) if p.name in m.parameters]
        if not poi_names:
            poi_names = list(m.parameters)
        print("\n[MINOS]", poi_names); m.minos(*poi_names)

    # push best-fit back
    for i, p in enumerate(adapter.free):
        p.val = float(m.values[i])

    print("\n[final] Best-fit hypothesis:")
    hypothesis.print()
    return m, adapter

def serialize_result(m, base, version, args, out_path ):
    
    result_payload = {
        "config_basename": base,
        "version": version,
        "no_syst": bool(args.no_syst),
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
    with open(out_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    print(f"[write] Fit result and covariance stored at:\n  {out_path}")

if __name__ == "__main__":
    # ---------------- args ----------------
    import argparse
    p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
    p.add_argument("--no_syst", action="store_true", help="Disable all nuisances (freeze to 0).")
    args = p.parse_args()

    import common.yaml_loader as yaml_loader 

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    like_info = load_likelihood(cfg)

    hyp = build_hypothesis_from_likelihood(like_info, name="SR")

    # --- optionally disable all nuisances ---
    if args.no_syst:
        for p in hyp.nuisances:
            p.val = 0.0
            p.isFrozen = True
        print("[opts] --no_syst: all nuisances set to 0 and frozen.")

    hyp.print()

    n2ll = N2LL( like_info, 'data.samples',  os.path.join( "NN2LCache",  os.path.splitext(os.path.basename(args.config))[0], cfg['version']), cache_root=None, overwrite=args.overwrite)
    n2ll.build_cache()
    n2ll.prepare_runtime()

    # compute A-simov
    n2ll.setAsimov()

    # compute C-simov (POI or nuisance injection)
    #n2ll.setAsimov(hyp.cloneModify(c1=1))

    ## run Minuit; prints the model every 25 evaluations by default
    m, adapter = run_minuit_fit(n2ll, hyp, step=0.1, print_every=1, do_migrad=True, do_hesse=True, do_minos=False)

    # best-fit -2logL
    print("Best -2logL =", m.fval)

    print("Correlation")
    print(m.covariance.correlation())

    # -------- persist fit result + covariance --------
    import os, json, numpy as np
    import common.user as user

    base    = os.path.splitext(os.path.basename(args.config))[0]
    version = str(cfg.get("version", "v0"))
    suffix  = "_nosyst" if args.no_syst else ""
    os.makedirs(user.output_directory, exist_ok=True)
    out_path = os.path.join(user.output_directory, f"{base}_{version}{suffix}_fit.json")
    serialize_result(m, base, version, args, out_path)


