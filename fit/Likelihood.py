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

# ---- Likelihood wiring + model parameter scaffolding -----------------------

def _job_by_id(cfg, jid):
    return next((j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id") == jid), None)

def _predictor_from_job(job):
    # We rely on load_surrogates having attached job['predictor'] when available.
    return None if job is None else job.get("predictor", None)

def load_likelihood(cfg):
    """
    Parse cfg['likelihood'], attach predictors by job id, and collect
    - POI names (union across all classes)
    - nuisance names (union across all systematics)
    
    Returns a dict:
      {
        'regions': [... enriched likelihood regions ...],
        'pois':     sorted list of POI names,
        'nuisances':sorted list of nuisance names
      }
    The function mutates the region dictionaries to include predictor hooks:
      region['classifier']['predictor']
      class['POI']['predictor']
      syst['predictor']     (for type == 'pnn')
    """
    lk = cfg.get("likelihood", {}) or {}
    regions = list(lk.get("regions", []) or [])

    if not regions:
        logger.info("No likelihood regions found.")
        return {'regions': [], 'pois': [], 'nuisances': []}

    all_pois = set()
    all_nuis = set()

    # convenience cache of jobs by id
    id2job = {j.get("id"): j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id")}

    # Walk regions
    for R in regions:
        # classifier (TFMC)
        clf = R.get("classifier", {}) or {}
        if clf.get("type") == "tfmc":
            tfmc_id = clf.get("job")
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
            systs = C.get("systematics", []) or []
            for S in systs:
                styp = S.get("type")
                if styp == "pnn":
                    pnn_id = S.get("job")
                    pnn_job = id2job.get(pnn_id) or _job_by_id(cfg, pnn_id)
                    S['predictor'] = _predictor_from_job(pnn_job)
                    if S['predictor'] is None:
                        logger.warning(f"[likelihood] PNN '{pnn_id}' has no predictor attached yet.")

                    # NEW: expose PNN combinations (and ensure parameters present)
                    pnn_params = list((pnn_job or {}).get("parameters", []) or [])
                    pnn_combs  = [tuple(c) for c in ((pnn_job or {}).get("combinations", []) or [])]
                    if 'parameters' not in S or not S['parameters']:
                        S['parameters'] = pnn_params                  
                    S['combinations'] = pnn_combs                     

                    # Optional extra check: ensure PNN↔ICP match if PNN references an ICP by id in its extras
                    # (this duplicates the checker in load_surrogates, but keeps it close to likelihood, too)
                    try:
                        extras = (pnn_job or {}).get('extras', {}) or {}
                        icp_id = extras.get('use_icp')
                        if isinstance(icp_id, str) and icp_id in id2job:
                            icp_job = id2job[icp_id]
                            icp = icp_job.get('predictor', None)
                            if icp is not None:
                                pnn_params = list((pnn_job or {}).get("parameters", []) or [])
                                pnn_combs  = [tuple(c) for c in ((pnn_job or {}).get("combinations", []) or [])]
                                icp_params = list(getattr(icp, "parameters"))
                                icp_combs  = [tuple(c) for c in getattr(icp, "combinations")]
                                if not (pnn_params == icp_params and pnn_combs == icp_combs):
                                    logger.warning(f"[likelihood] PNN '{pnn_id}' params/combs differ from ICP '{icp_id}'.")
                    except Exception:
                        pass

                    # collect nuisance names from YAML
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)

                elif styp == "lnN":
                    # log-normal norm nuisances; they have 'parameters': [...]
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)
                else:
                    # Future syst types (jes/jer/etc.) can be added here
                    for nm in (S.get("parameters") or []):
                        all_nuis.add(nm)

    # Keep deterministic order
    pois_list = sorted(all_pois)
    nuis_list = sorted(all_nuis)

    # Write back the enriched regions for downstream consumers
    return {'regions': regions, 'pois': pois_list, 'nuisances': nuis_list}

def build_hypothesis_from_likelihood(like_info, *, name=None,
                                     poi_init=0.0, nuis_init=0.0,
                                     penalize_nuisances=True):
    """
    Convenience: construct a Hypothesis from load_likelihood(...) output.

    Heuristics:
      - POIs are marked isPOI=True if name starts with 'c'.
      - Nuisances are marked penalized unless penalize_nuisances=False.

    Returns Hypothesis instance.
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
                        G = _predict_classifier(clf, X)  # (Nb, n_proc)
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
                        R_A = _predict_bit_ratio(poi_pred, X)  # (Nb, nA)

                        if writer["R"] is None:
                            nA = R_A.shape[1]
                            writer["R"] = f.create_dataset("R", (0, nA), maxshape=(None, nA), dtype="f8", chunks=True)
                            first_batch_shapes.setdefault(cid, {})["nA"] = nA
                        self._append_2d(writer["R"], R_A)

                        # PNN Δ groups
                        for S in C['_pnn_systs']:
                            sid = S['id']
                            pnn = S['predictor']
                            dA = _predict_pnn_deltaA(pnn, X)  # (Nb, nB)
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

    def close(self):
        """Close all opened HDF5 files."""
        for f in list(self._h5.values()):
            try:
                f.close()
            except Exception:
                pass
        self._h5.clear()

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

    def setAsimov(self, hypothesis) -> None:
        """
        Set an off-nominal Asimov hypothesis (c', ν') and precompute T'(x; c', ν')
        on the cached event set. If all parameters are zero, disables the bias.
        """
        if not self._h5:
            raise RuntimeError("[N2LL.setAsimov] Call prepare_runtime() before setting Asimov.")

        # quick check: any parameter nonzero?
        any_nonzero = any(abs(float(p.val)) > 0.0 for p in getattr(hypothesis, 'parameters', []))
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

    def __call__(self, hypothesis) -> float:
        """
        Return  -2 * Σ_i w0_i * ( log1p(T_i) - T_i )   [baseline Asimov at (0,0)]
            plus bias term if an off-nominal Asimov hypothesis (c',ν') is set:

            +  -2 * Σ_i w0_i * T'_i * log1p(T_i)

        where:
          T_i  = T(x_i; c,  ν)  (current hypothesis)
          T'_i = T(x_i; c', ν') (precomputed via setAsimov)
        """
        import numpy as np

        if not self._h5:
            raise RuntimeError("[N2LL] Call prepare_runtime() before evaluating.")

        # ----- build A-basis for current hypothesis -----
        # nuisances for lnN bias
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}

        total_sum = 0.0         # Σ w * (log1p(T) - T)
        bias_sum  = 0.0         # Σ w * T'(asimov) * log1p(T)

        for R in self.regions:
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue

            # current hypothesis: c_A and ν_A groups
            cA_per_class  = self._assemble_cA_per_class(rid, hypothesis)
            nuA_per_group = self._assemble_nuA_groups(rid, hypothesis)
            ln_bias = {
                cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                         for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }

            # chunked evaluation
            chunk = self.eval_chunk_size
            asimov_T_chunks = self._asimov_T.get(rid, None) if self._asimov_active else None
            for ichunk, start in enumerate(range(0, N, chunk)):
                stop = min(start + chunk, N)

                # compute T for current hypothesis on this chunk
                T = self._compute_T_chunk(rid, cA_per_class, nuA_per_group, ln_bias, start, stop)  # (M,)
                W = self._h5[(rid, class_ids[0])]['w0'][start:stop]                                # (M,)

                # baseline contribution: Σ w * (log1p(T) - T)
                # (use the numerically-stable helper for the sum)
                total_sum += _weighted_sum_log1p_minus_x(T, W)

                # bias term only if an off-nominal Asimov has been set
                if asimov_T_chunks is not None:
                    Tprime = asimov_T_chunks[ichunk]   # (M,)
                    # add Σ w * T'(asimov) * log1p(T)
                    bias_sum += float(np.sum(W * np.log1p(T) * Tprime, dtype=np.float64))

        # assemble -2 log L
        n2ll = -2.0 * (total_sum + bias_sum)

        # nuisance penalty (gaussian prior)
        n2ll += hypothesis.penalty()

        return float(n2ll)

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

if __name__ == "__main__":
    # ---------------- args ----------------
    import argparse
    p = argparse.ArgumentParser(description="TFMC training (YAML-driven)")
    p.add_argument("config", help="Path to global YAML config")
    p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
    args = p.parse_args()

    import common.yaml_loader as yaml_loader 

    cfg = yaml_loader.load_yaml(args.config)
    yaml_loader.print_summary(cfg, args.config, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, args.config, overwrite=False, prefer_numba=False)

    like_info = load_likelihood(cfg)

    hyp = build_hypothesis_from_likelihood(like_info, name="SR")
    hyp.print()

    n2ll = N2LL( like_info, 'data.samples',  os.path.join( "NN2LCache",  os.path.splitext(os.path.basename(args.config))[0], cfg['version']), cache_root=None, overwrite=args.overwrite)
    n2ll.build_cache()
    n2ll.prepare_runtime()
    #n2ll = n2ll(hyp) 

    n2ll.setAsimov(hyp.cloneModify(nu_jes=0.5, c1=0.1))

    val = n2ll(hyp)
    ## run Minuit; prints the model every 25 evaluations by default
    m, adapter = run_minuit_fit(n2ll, hyp, step=0.1, print_every=1, do_migrad=True, do_hesse=True, do_minos=False)

    # best-fit -2logL
    print("Best -2logL =", m.fval)

    print("Correlation")
    print(m.covariance.correlation())
