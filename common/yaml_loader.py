import os
import yaml
from collections import defaultdict, deque
from typing import List, Iterable
from pathlib import Path

import sys
sys.path.insert(0, '..')
from common.helpers import _binning_equal

class IncludeError(Exception):
    pass

# --- tracing --------------------------------------------------------------
_INCLUDE_TRACE = []

def _record_include(parent_file: str, key: str, child_file: str):
    _INCLUDING = os.path.abspath(parent_file)
    _CHILD = os.path.abspath(child_file)
    _INCLUDE_TRACE.append((_INCLUDING, key, _CHILD))

# --- io / utils -----------------------------------------------------------
def _read_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve(base_file: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(os.path.dirname(os.path.abspath(base_file)), rel)

def _as_list(x):
    if isinstance(x, str): return [x]
    if isinstance(x, list) and all(isinstance(s, str) for s in x): return x
    raise IncludeError("'include' must be a string or a list of strings")

def _merge(values):
    if not values: return None
    if all(isinstance(v, list) for v in values):
        out = []; [out.extend(v) for v in values]; return out
    if all(isinstance(v, dict) for v in values):
        out = {}; [out.update(v) for v in values]; return out
    return values[-1]

# --- include expander -----------------------------------------------------
def _include_for_key(parent_key: str, sources, current_file: str, seen):
    results = []
    for src in _as_list(sources):
        path = _resolve(current_file, src)
        sig = (os.path.abspath(path), parent_key)
        if sig in seen:
            raise IncludeError(f"Cycle detected: {path} -> '{parent_key}'")
        _record_include(current_file, parent_key, path)
        seen.add(sig)
        doc = _read_yaml(path)
        if not isinstance(doc, dict) or parent_key not in doc:
            raise IncludeError(f"{path} does not contain key '{parent_key}'")
        results.append(_expand(doc[parent_key], path, seen, parent_key))
        seen.remove(sig)
    return _merge(results)

def _expand(node, current_file: str, seen, parent_key: str | None):
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, dict) and set(v.keys()) == {"include"}:
                node[k] = _expand(_include_for_key(k, v["include"], current_file, seen),
                                  current_file, seen, k)
            else:
                node[k] = _expand(v, current_file, seen, k)
        return node

    if isinstance(node, list):
        i = 0
        while i < len(node):
            item = node[i]
            if isinstance(item, dict) and set(item.keys()) == {"include"}:
                if not parent_key:
                    raise IncludeError("List-level 'include' needs a parent key")
                inserted = _include_for_key(parent_key, item["include"], current_file, seen)
                if isinstance(inserted, list):
                    node[i:i+1] = inserted
                    i += len(inserted)
                else:
                    node[i:i+1] = [inserted]
                    i += 1
            else:
                node[i] = _expand(item, current_file, seen, parent_key)
                i += 1
        return node
    return node

def load_yaml(path: str):
    doc = _read_yaml(path)
    cfg = _expand(doc, os.path.abspath(path), seen=set(), parent_key=None)
    _apply_defaults_and_checks(cfg)   # <<< NEW
    return cfg

# --- feature resolution (NEW) ---------------------------------------------
def _resolve_features_list(tokens: Iterable[str]) -> List[str]:
    """
    Expand a list like ["TOP_KINEMATICS", "ASYMMETRY", "tr_ttbar_pt"] into a flat
    unique list of feature column names using data.observables.

    Rules:
      - If token names a list defined in data.observables (e.g. TOP_KINEMATICS),
        extend by that list.
      - Else, if token is in data.observables.ALL_FEATURES, append it as a single feature.
      - Else, raise a descriptive error.
    """
    if not tokens:
        return []
    from data import observables as obs
    out = []
    seen = set()
    for t in tokens:
        if not isinstance(t, str):
            raise RuntimeError(f"[features] All entries must be strings, got {type(t)} for {t!r}")
        # try attribute list (e.g. TOP_KINEMATICS)
        if hasattr(obs, t):
            val = getattr(obs, t)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                for name in val:
                    if name not in seen:
                        out.append(name); seen.add(name)
                continue
        # else, treat as single feature name present in ALL_FEATURES
        if hasattr(obs, "ALL_FEATURES") and t in getattr(obs, "ALL_FEATURES"):
            if t not in seen:
                out.append(t); seen.add(t)
            continue
        raise RuntimeError(f"[features] '{t}' is neither a list in data.observables nor a known feature in ALL_FEATURES.")
    return out

def _apply_defaults_and_checks(cfg: dict):
    """
    - Resolve defaults.default_features via _resolve_features_list.
    - For each job of type ICH / ICPH:
        * If job.binning missing -> set to defaults
    - For each job of type scaler / classifier(tfmc) / pnn / bit:
        * If job.features missing -> set to defaults
        * Else resolve job.features via _resolve_features_list
    - If a PNN or TFMC has extras.use_scaler, ensure features identical to that scaler job.
    """
    defaults = cfg.get("defaults", {}) or {}
    default_tokens = (defaults.get("default_features") or [])
    default_features = _resolve_features_list(default_tokens)
    # keep a resolved copy (optional)
    cfg.setdefault("defaults", {})["_resolved_features"] = list(default_features)
    # default binning, if there is one
    default_binning = (defaults.get("default_binning") or None)

    jobs = cfg.get("jobs", []) or []
    # resolve per job
    for j in jobs:
        if not isinstance(j, dict):
            continue
        jtyp = j.get("type")
        if jtyp in {"ich", "icph"} and default_binning:
            if not "binning" in j:
                j["binning"] = default_binning
        if jtyp not in {"scaler", "pnn", "bit", "classifier"}:
            continue
        if jtyp == "classifier" and j.get("framework") != "tfmc":
            continue
        feat_tokens = j.get("features", None)
        if feat_tokens is None:
            j["features"] = list(default_features)
        else:
            j["features"] = _resolve_features_list(feat_tokens)

    # scaler-feature consistency for TFMC/PNN that reference a scaler
    id2job = {j.get("id"): j for j in jobs if isinstance(j, dict) and j.get("id")}
    for j in jobs:
        if not isinstance(j, dict):
            continue
        jtyp = j.get("type")
        if jtyp == "classifier" and j.get("framework") == "tfmc":
            extras = j.get("extras", {}) or {}
            sid = extras.get("use_scaler")
            if isinstance(sid, str) and sid in id2job:
                sj = id2job[sid]
                if sj.get("type") != "scaler":
                    raise RuntimeError(f"[features] TFMC '{j.get('id')}' extras.use_scaler='{sid}' is not a scaler job.")
                f_a = j.get("features", [])
                f_b = sj.get("features", [])
                if f_a != f_b:
                    raise RuntimeError(f"[features] TFMC '{j.get('id')}' features != scaler '{sid}' features.\n"
                                       f"  TFMC : {f_a}\n  Scaler: {f_b}")
        if jtyp == "pnn":
            extras = j.get("extras", {}) or {}
            sid = extras.get("use_scaler")
            if isinstance(sid, str) and sid in id2job:
                sj = id2job[sid]
                if sj.get("type") != "scaler":
                    raise RuntimeError(f"[features] PNN '{j.get('id')}' extras.use_scaler='{sid}' is not a scaler job.")
                f_a = j.get("features", [])
                f_b = sj.get("features", [])
                if f_a != f_b:
                    raise RuntimeError(f"[features] PNN '{j.get('id')}' features != scaler '{sid}' features.\n"
                                       f"  PNN   : {f_a}\n  Scaler: {f_b}")

# --- pretty printers ------------------------------------------------------
def _print_include_tree(root_file: str, trace):
    children = defaultdict(list)
    files = set()
    for inc, key, child in trace:
        children[(inc, key)].append(child)
        files.add(inc); files.add(child)
    root_abs = os.path.abspath(root_file)

    def dfs(file_path: str, indent=""):
        print(f"{indent}{os.path.basename(file_path)}")
        grouped = defaultdict(list)
        for (inc, key), lst in children.items():
            if inc == file_path:
                grouped[key].extend(lst)
        for key in sorted(grouped.keys()):
            print(f"{indent}  [{key}]")
            for ch in grouped[key]:
                dfs(ch, indent + "    ")

    if trace:
        print("Include tree:")
        dfs(root_abs, indent="  ")
    else:
        print("Include tree: (none)")

def _collect_id_deps(job):
    deps = []
    extras = job.get("extras", {}) or {}
    for k in ("use_scaler", "use_ic", "use_icp"):
        v = extras.get(k, None)
        if isinstance(v, str):
            deps.append(v)
    return deps

def _build_job_layers(jobs):
    idx = {j.get("id"): j for j in jobs if isinstance(j, dict) and j.get("id")}
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for j in jobs:
        if not isinstance(j, dict): 
            continue
        jid = j.get("id")
        if not jid:
            continue
        deps = _collect_id_deps(j)
        for d in deps:
            if d not in idx:
                continue
            adj[d].append(jid)
            indeg[jid] += 1
        indeg[jid] = indeg[jid]
    layers = []
    from collections import deque as _dq
    q = _dq(sorted([n for n in idx if indeg[n] == 0]))
    seen = set()
    while q:
        layer = []
        for _ in range(len(q)):
            u = q.popleft()
            if u in seen: 
                continue
            layer.append(u); seen.add(u)
            for v in sorted(adj[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if layer:
            layers.append(layer)
    remaining = [n for n in idx if n not in set().union(*layers)]
    if remaining:
        layers.append(sorted(remaining))
    return layers, idx, adj

def _print_jobs(cfg):
    jobs = cfg.get("jobs", [])
    if not isinstance(jobs, list):
        print("Jobs: (none or not a list)")
        return
    layers, idx, adj = _build_job_layers(jobs)
    total = len(idx)
    roots = len(layers[0]) if layers else 0
    print(f"Jobs overview: {total} total, {roots} root(s), {len(layers)} layer(s).")
    for li, layer in enumerate(layers):
        print(f"  Layer {li}: " + ", ".join(layer))
    print("Job dependencies:")
    for jid in sorted(idx.keys()):
        deps = _collect_id_deps(idx[jid])
        deps_str = ", ".join(deps) if deps else "—"
        jtype = idx[jid].get("type", "unknown")
        print(f"  - {jid} [{jtype}]  <=  {deps_str}")

def print_summary(cfg, root_file, trace):
    include_files = {os.path.abspath(a) for a,_,_ in trace} | {os.path.abspath(c) for _,_,c in trace}
    print(f"Overview: root={os.path.abspath(root_file)}, includes={len(trace)} edges across {len(include_files)} file(s).")
    _print_include_tree(root_file, trace)
    _print_jobs(cfg)

# --- helper: normalize YAML binning into (axis_names, [edges...]) ---
def _normalize_cfg_binning(job_binning):
    import numpy as _np
    axes = []
    edges = []
    for item in (job_binning or []):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise RuntimeError(f"Invalid binning entry: {item!r}")
        nm, ed = item[0], item[1]
        axes.append(str(nm))
        arr = _np.asarray([float(x) for x in ed], dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise RuntimeError(f"Binning edges must be 1D with >=2 entries for axis '{nm}'.")
        edges.append(arr)
    return tuple(axes), edges

# --- surrogate loader ------------
def load_surrogates(cfg, config_path, overwrite=False, prefer_numba=False):
    """
    Load artifacts and attach predictors; also checks ICH/ICPH binnings and PNN↔ICP consistency.
    """
    import os, sys
    sys.path.insert(0, '..'); sys.path.insert(0, '../..')

    cfg_full = os.path.abspath(os.path.expanduser(os.path.expandvars(config_path)))
    import common.user as user

    def job_by_id(jid):
        return next((j for j in (cfg.get("jobs") or []) if j.get("id") == jid), None)

    same_flags = []
    if overwrite: same_flags.append("--overwrite")
    FLAGS = " " + " ".join(same_flags) if same_flags else ""

    def cfg_base_for(job):
        ver = cfg.get("version", "default")
        reg = job.get("region", cfg.get("region", "default"))
        return os.path.join(ver, reg)

    def try_load_scaler(path):
        try:
            from ML.Scaler.Scaler import Scaler
            return Scaler.load(path)
        except Exception:
            return None

    def try_load_ic(path):
        try:
            from ML.IC.IC import InclusiveCrosssection
            return InclusiveCrosssection.load(path)
        except Exception:
            return None

    def try_load_ich(path):
        try:
            from ML.ICH.ICH import InclusiveCrosssectionHistogram
            return InclusiveCrosssectionHistogram.load(path)
        except Exception:
            return None

    def try_load_icp(path):
        try:
            from ML.ICP.ICP import InclusiveCrosssectionParametrization
            return InclusiveCrosssectionParametrization.load(path)
        except Exception:
            return None

    def try_load_icph(path):
        try:
            from ML.ICPH.ICPH import InclusiveCrosssectionParametrizationHistogram
            return InclusiveCrosssectionParametrizationHistogram.load(path)
        except Exception:
            return None

    def try_load_pnn(model_dir):
        try:
            from ML.PNN.PNN import PNN
            return PNN.load(model_dir)
        except Exception:
            return None

    def try_load_tfmc(model_dir):
        try:
            from ML.TFMC.TFMC import TFMC
            return TFMC.load(model_dir)
        except Exception:
            return None

    def try_load_bit(path):
        try:
            if prefer_numba:
                from ML.BIT.NumbaBIT import MultiBoostedInformationTree
            else:
                from ML.BIT.MultiBoostedInformationTree import MultiBoostedInformationTree
            return MultiBoostedInformationTree.load(path)
        except Exception:
            return None

    ml_dir = Path(__file__).resolve().parent.parent / "ML"

    ok, missing = [], []
    # ---------- First pass ----------
    for i_job, job in enumerate((cfg.get("jobs") or [])):
        if not isinstance(job, dict): 
            continue
        jid  = job.get("id")
        jtyp = job.get("type")
        if not jid or not jtyp: 
            continue

        base = cfg_base_for(job)

        if jtyp == "scaler":
            process = job.get("process")
            out     = job.get("output", {}) or {}
            fname   = out.get("filename", f"Scaler_{process}.pkl")
            outdir  = os.path.join(user.model_directory, base, "Scaler")
            path    = os.path.join(outdir, fname)
            loaded  = try_load_scaler(path)
            if loaded is not None:
                print(f"[OK] Scaler {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] Scaler {jid}  (expected at {path})")
                missing.append(f"python {ml_dir}/Scaler/scaler_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "ic":
            process = job.get("process")
            out     = job.get("output", {}) or {}
            fname   = out.get("filename", f"IC_{process}.pkl")
            outdir  = os.path.join(user.model_directory, base, "IC")
            path    = os.path.join(outdir, fname)
            loaded  = try_load_ic(path)
            if loaded is not None:
                print(f"[OK] IC {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] IC {jid}  (expected at {path})")
                missing.append(f"python {ml_dir}/IC/ic_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "ich":
            outdir = os.path.join(user.model_directory, base, "ICH")
            process = job.get("process")
            fname  = (job.get("output", {}) or {}).get("filename", f"ICH_{process}.pkl")
            path   = os.path.join(outdir, fname)
            loaded = try_load_ich(path)
            if loaded is not None:
                print(f"[OK] ICH {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] ICH {jid}  (expected at {path})")
                missing.append(f"python {ml_dir}/ICH/ich_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "icp":
            out     = job.get("output", {}) or {}
            fname   = out.get("filename", f"ICP_{jid}.pkl")
            outdir  = os.path.join(user.model_directory, base, "ICP")
            path    = os.path.join(outdir, fname)
            loaded  = try_load_icp(path)
            if loaded is not None:
                print(f"[OK] ICP {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] ICP {jid}  (expected at {path})")
                missing.append(f"python {ml_dir}/ICP/icp_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "icph":
            out     = job.get("output", {}) or {}
            fname   = out.get("filename", f"ICPH_{jid}.pkl")
            outdir  = os.path.join(user.model_directory, base, "ICPH")
            path    = os.path.join(outdir, fname)
            loaded  = try_load_icph(path)
            if loaded is not None:
                print(f"[OK] ICPH {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] ICPH {jid}  (expected at {path})")
                missing.append(f"python {ml_dir}/ICPH/icph_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "pnn":
            model_dir = os.path.join(user.model_directory, base, "PNN", jid)
            loaded = try_load_pnn(model_dir)
            if loaded is not None:
                print(f"[OK] PNN {jid}  -> {model_dir}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
                cfg['jobs'][i_job]['predictor'].feature_names = cfg['jobs'][i_job]['features'] 
            else:
                print(f"[MISS] PNN {jid}  (expected at {model_dir})")
                missing.append(f"python {ml_dir}/PNN/pnn_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "classifier" and job.get("framework") == "tfmc":
            model_dir = os.path.join(user.model_directory, base, "TFMC", jid)
            loaded    = try_load_tfmc(model_dir)
            if loaded is not None:
                print(f"[OK] TFMC {jid}  -> {model_dir}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
                cfg['jobs'][i_job]['predictor'].feature_names = cfg['jobs'][i_job]['features'] 
            else:
                print(f"[MISS] TFMC {jid}  (expected at {model_dir})")
                missing.append(f"python {ml_dir}/TFMC/tfmc_training.py {cfg_full}{FLAGS} --job {jid}")

        elif jtyp == "bit":
            outdir = os.path.join(user.model_directory, base, "BIT", jid)
            fname  = (job.get("output", {}) or {}).get("filename", "BIT.pkl")
            path   = os.path.join(outdir, fname)
            loaded = try_load_bit(path)
            if loaded is not None:
                print(f"[OK] BIT {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
                cfg['jobs'][i_job]['predictor'].feature_names = cfg['jobs'][i_job]['features'] 
            else:
                print(f"[MISS] BIT {jid}  (expected at {path})")
                nb = " --numba" if prefer_numba else ""
                missing.append(f"python {ml_dir}/BIT/pdf_bit_training.py {cfg_full}{FLAGS}{nb} --job {jid}")

        # ICH/ICPH binning consistency
        if jtyp in {"ich", "icph"} and cfg['jobs'][i_job].get("predictor") is not None:
            import numpy as _np
            pred = cfg['jobs'][i_job]["predictor"]
            try:
                cfg_names, cfg_edges = _normalize_cfg_binning(cfg['jobs'][i_job].get("binning", []))
            except Exception as e:
                raise RuntimeError(f"[surrogate binning] Invalid YAML binning in job '{job.get('id','?')}': {e}")
            pred_names = tuple(getattr(pred, "axis_names", []) or [])
            pred_edges = [_np.asarray(be, dtype=float) for be in (getattr(pred, "bin_edges", []) or [])]
            if not _binning_equal(cfg_names, cfg_edges, pred_names, pred_edges):
                raise RuntimeError(
                    f"[surrogate binning] Mismatch for job '{job.get('id','?')}'.\n"
                    f"  YAML axis/edges : {cfg_names} / {[e.tolist() for e in cfg_edges]}\n"
                    f"  File axis/edges : {pred_names} / {[e.tolist() for e in pred_edges]}"
                )

    # ---------- Second pass ----------
    id2job = {j.get("id"): j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id")}

    # Attach TFMC scaler/IC payloads by ID
    for j in (cfg.get("jobs") or []):
        if not isinstance(j, dict):
            continue
        if j.get("type") == "classifier" and j.get("framework") == "tfmc":
            extras = j.get("extras", {}) or {}
            sid = extras.get("use_scaler")
            if isinstance(sid, str) and sid in id2job:
                sj = id2job[sid]
                sc = sj.get("predictor", None)
                if sc is not None:
                    try:
                        means = list(getattr(sc, "feature_means"))
                        vars_ = list(getattr(sc, "feature_variances"))
                        j["scaler_means"] = means
                        j["scaler_vars"]  = vars_
                        print(f"[TFMC attach] {j['id']}: scaler <- {sid}")
                    except Exception:
                        pass
            iid = extras.get("use_ic")
            if isinstance(iid, str) and iid in id2job:
                ij = id2job[iid]
                ic = ij.get("predictor", None)
                if ic is not None:
                    try:
                        sumw = float(getattr(ic, "total_weight"))
                        j["ic_weight_sum"] = sumw
                        print(f"[TFMC attach] {j['id']}: ic <- {iid} (sum={sumw:g})")
                    except Exception:
                        pass

    # PNN ↔ ICP consistency check
    for j in (cfg.get("jobs") or []):
        if not isinstance(j, dict):
            continue
        if j.get("type") == "pnn":
            extras = j.get("extras", {}) or {}
            icp_id = extras.get("use_icp")
            if isinstance(icp_id, str) and icp_id in id2job:
                icp_job = id2job[icp_id]
                icp = icp_job.get("predictor", None)
                if icp is not None:
                    pnn_params = list(j.get("parameters", []) or [])
                    pnn_combs  = [tuple(c) for c in (j.get("combinations", []) or [])]
                    try:
                        icp_params = list(getattr(icp, "parameters"))
                        icp_combs  = [tuple(c) for c in getattr(icp, "combinations")]
                    except Exception:
                        print(f"[PNN↔ICP check] {j['id']} vs {icp_id}: unable to read ICP fields.")
                        j["icp_consistency_ok"] = False
                        continue
                    ok_params = (pnn_params == icp_params)
                    ok_combs  = (pnn_combs  == icp_combs)
                    j["icp_consistency_ok"] = bool(ok_params and ok_combs)
                    if j["icp_consistency_ok"]:
                        print(f"[PNN↔ICP check] {j['id']} vs {icp_id}: OK")
                    else:
                        print(f"[PNN↔ICP check] {j['id']} vs {icp_id}: MISMATCH")
                        if not ok_params:
                            print(f"  parameters: PNN={pnn_params}  ICP={icp_params}")
                        if not ok_combs:
                            print(f"  combinations: PNN={pnn_combs}  ICP={icp_combs}")

    print("\n=== SUMMARY ===")
    print(f"Found {len(ok)} ready artifact(s), {len(missing)} missing.")
    if missing:
        print("Train the missing ones with:")
        for cmd in missing:
            print(cmd)
    return missing

# --- cli ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1]
    cfg = load_yaml(root)
    print_summary(cfg, root, _INCLUDE_TRACE)
    load_surrogates(cfg, root, overwrite=False, prefer_numba=False)

