import os
import yaml
from collections import defaultdict, deque

class IncludeError(Exception):
    pass

# --- tracing --------------------------------------------------------------
# Records tuples of (including_file_abs, key, included_file_abs)
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
    return values[-1]  # keep it simple

# --- include expander -----------------------------------------------------
def _include_for_key(parent_key: str, sources, current_file: str, seen):
    results = []
    for src in _as_list(sources):
        path = _resolve(current_file, src)
        sig = (os.path.abspath(path), parent_key)
        if sig in seen:
            raise IncludeError(f"Cycle detected: {path} -> '{parent_key}'")
        # record the inclusion edge
        _record_include(current_file, parent_key, path)

        seen.add(sig)
        doc = _read_yaml(path)
        if not isinstance(doc, dict) or parent_key not in doc:
            raise IncludeError(f"{path} does not contain key '{parent_key}'")
        results.append(_expand(doc[parent_key], path, seen, parent_key))
        seen.remove(sig)
    return _merge(results)

def _expand(node, current_file: str, seen, parent_key: str | None):
    # Dicts: support either field includes (key: { include: ... }) or normal recursion.
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, dict) and set(v.keys()) == {"include"}:
                node[k] = _expand(_include_for_key(k, v["include"], current_file, seen),
                                  current_file, seen, k)
            else:
                node[k] = _expand(v, current_file, seen, k)
        return node

    # Lists: splice items like "- include: file.yaml"
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

    return node  # scalars unchanged

def load_yaml(path: str):
    doc = _read_yaml(path)
    return _expand(doc, os.path.abspath(path), seen=set(), parent_key=None)

# --- pretty printers ------------------------------------------------------
def _print_include_tree(root_file: str, trace):
    # Build adjacency by including file
    children = defaultdict(list)
    files = set()
    for inc, key, child in trace:
        children[(inc, key)].append(child)
        files.add(inc); files.add(child)
    root_abs = os.path.abspath(root_file)

    # Build a tree starting at the given root file
    def dfs(file_path: str, indent=""):
        print(f"{indent}{os.path.basename(file_path)}")
        # Show children grouped by key (e.g., jobs, datasets, …)
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
    """
    Dependency model (by IDs):
      - extras.use_scaler: <scaler_job_id>
      - extras.use_ic:     <ic_job_id>
      - extras.use_icp:    <icp_job_id>
    Only string values are treated as dependencies.
    """
    deps = []
    extras = job.get("extras", {}) or {}
    for k in ("use_scaler", "use_ic", "use_icp"):
        v = extras.get(k, None)
        if isinstance(v, str):
            deps.append(v)
    return deps

def _build_job_layers(jobs):
    # Build DAG from ID-based references (see _collect_id_deps)
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
                # Ignore edges to unknown nodes; they might be in other files not loaded here
                continue
            adj[d].append(jid)
            indeg[jid] += 1
        indeg[jid] = indeg[jid]  # ensure key exists

    # Kahn layering
    layers = []
    q = deque(sorted([n for n in idx if indeg[n] == 0]))
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

    # If there are remaining nodes (cycle or cross-refs), tack them on last
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
    # Per-job dependency listing (ID-based)
    print("Job dependencies:")
    for jid in sorted(idx.keys()):
        deps = _collect_id_deps(idx[jid])
        deps_str = ", ".join(deps) if deps else "—"
        jtype = idx[jid].get("type", "unknown")
        print(f"  - {jid} [{jtype}]  <=  {deps_str}")

def print_summary(cfg, root_file, include_trace):
    # Overview line
    include_files = {os.path.abspath(a) for a,_,_ in include_trace} | {os.path.abspath(c) for _,_,c in include_trace}
    print(f"Overview: root={os.path.abspath(root_file)}, includes={len(include_trace)} edges across {len(include_files)} file(s).")
    # Include tree
    _print_include_tree(root_file, include_trace)
    # Jobs layout
    _print_jobs(cfg)

def load_surrogates(cfg, config_path, overwrite=False, prefer_numba=False):
    """
    For each job in cfg (IC, Scaler, ICP, TFMC, BIT):
      - try to load its saved artifact
      - on failure, print the training command the user should run.

    After the first pass, do a small second pass:
      - For each TFMC job, pull referenced scaler/IC (by ID) from cfg and attach
        scaler_means/scaler_vars and ic_weight_sum onto the TFMC job entry.
      - For each PNN job with extras.use_icp, verify its (parameters, combinations)
        match the referenced ICP artifact; print a check result.
    """
    import os, sys
    sys.path.insert(0, '..'); sys.path.insert(0, '../..')  # project roots

    import common.user as user

    # helpers
    def cfg_base_for(job):
        ver = cfg.get("version", "default")
        reg = job.get("region", cfg.get("region", "default"))
        return os.path.join(ver, reg)

    def job_by_id(jid):
        return next((j for j in (cfg.get("jobs") or []) if j.get("id") == jid), None)

    # CLI flags to echo in suggestions
    same_flags = []
    if overwrite: same_flags.append("--overwrite")
    FLAGS = " " + " ".join(same_flags) if same_flags else ""

    # On-demand imports to avoid heavy deps until needed
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

    def try_load_icp(path):
        try:
            from ML.ICP.ICP import InclusiveCrosssectionParametrization
            return InclusiveCrosssectionParametrization.load(path)
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

    # Summary accumulators
    ok, missing = [], []

    # ---------- First pass: load artifacts ----------
    for i_job, job in enumerate((cfg.get("jobs") or [])):
        if not isinstance(job, dict): 
            continue
        jid  = job.get("id")
        jtyp = job.get("type")
        if not jid or not jtyp: 
            continue

        base = cfg_base_for(job)

        # ---------- SCALER ----------
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
                missing.append(f"python scaler_training.py {config_path}{FLAGS} --job {jid}")

        # ---------- IC ----------
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
                missing.append(f"python IC/ic_training.py {config_path}{FLAGS} --job {jid}")

        # ---------- ICP ----------
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
                missing.append(f"python ICP/icp_training.py {config_path}{FLAGS} --job {jid}")

        # ---------- PNN ----------
        elif jtyp == "pnn":
            model_dir = os.path.join(user.model_directory, base, "PNN", jid)
            loaded = try_load_pnn(model_dir)
            if loaded is not None:
                print(f"[OK] PNN {jid}  -> {model_dir}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] PNN {jid}  (expected at {model_dir})")
                missing.append(f"python PNN/pnn_training.py {config_path}{FLAGS} --job {jid}")

        # ---------- TFMC classifier ----------
        elif jtyp == "classifier" and job.get("framework") == "tfmc":
            model_dir = os.path.join(user.model_directory, base, "TFMC", jid)
            loaded    = try_load_tfmc(model_dir)
            if loaded is not None:
                print(f"[OK] TFMC {jid}  -> {model_dir}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] TFMC {jid}  (expected at {model_dir})")
                print_flags = FLAGS  # keep same overwrite/small echo
                missing.append(f"python TFMC/tfmc_training.py {config_path}{print_flags} --job {jid}")

        # ---------- BIT ----------
        elif jtyp == "bit":
            outdir = os.path.join(user.model_directory, base, "BIT", jid)
            fname  = (job.get("output", {}) or {}).get("filename", "BIT.pkl")
            path   = os.path.join(outdir, fname)
            loaded = try_load_bit(path)
            if loaded is not None:
                print(f"[OK] BIT {jid}  -> {path}")
                ok.append(jid)
                cfg['jobs'][i_job]['predictor'] = loaded
            else:
                print(f"[MISS] BIT {jid}  (expected at {path})")
                nb = " --numba" if prefer_numba else ""
                missing.append(f"python BIT/pdf_bit_training.py {config_path}{FLAGS}{nb} --job {jid}")

        # (ignore other types here)

    # ---------- Second pass ----------
    id2job = {j.get("id"): j for j in (cfg.get("jobs") or []) if isinstance(j, dict) and j.get("id")}

    # (A) Attach TFMC scaler/IC payloads by ID
    for j in (cfg.get("jobs") or []):
        if not isinstance(j, dict):
            continue
        if j.get("type") == "classifier" and j.get("framework") == "tfmc":
            extras = j.get("extras", {}) or {}
            # Scaler stats
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
            # IC total weight
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

    # (B) PNN ↔ ICP consistency check (parameters & combinations)
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
                    # PNN spec
                    pnn_params = list(j.get("parameters", []) or [])
                    pnn_combs  = [tuple(c) for c in (j.get("combinations", []) or [])]
                    # ICP artifact
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

    # ---- Summary block ----
    print("\n=== SUMMARY ===")
    print(f"Found {len(ok)} ready artifact(s), {len(missing)} missing.")
    if missing:
        print("Train the missing ones with:")
        for cmd in missing:
            print(cmd)

# --- cli ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1]
    cfg = load_yaml(root)
    print_summary(cfg, root, _INCLUDE_TRACE)
    load_surrogates(cfg, root, overwrite=False, prefer_numba=False)
