import os
import yaml
from collections import defaultdict, deque

import copy
import logging
logger = logging.getLogger(__name__)

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
            tfmc_id = clf.get("id")
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


# --- cli ------------------------------------------------------------------
if __name__ == "__main__":
    import common.yaml_loader as yaml_loader 

    root = sys.argv[1]
    cfg = yaml_loader.load_yaml(root)
    yaml_loader.print_summary(cfg, root, yaml_loader._INCLUDE_TRACE)
    yaml_loader.load_surrogates(cfg, root, overwrite=False, prefer_numba=False)

    like_info = load_likelihood(cfg)

    hyp = build_hypothesis_from_likelihood(like_info, name="SR")
    hyp.print()
