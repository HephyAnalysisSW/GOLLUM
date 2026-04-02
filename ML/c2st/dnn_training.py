from __future__ import annotations
import shutil
import os, sys, time, argparse, importlib, warnings, pickle, random, yaml, math
import numpy as np
import tensorflow as tf

import common.user as user
import common.syncer as syncer

from tqdm import trange, tqdm

from DNN import DNN
from ML.PNN.PNN import PNN

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

# ---------------- args ----------------
p = argparse.ArgumentParser(description="DNN C2ST training (YAML-driven, TRAIN ONLY)")
p.add_argument("config", help="Path to global YAML config")
p.add_argument("--job", default=None, help="DNN job id to run (omit to list)")
p.add_argument("--overwrite", action="store_true", help="Overwrite model directory?")
p.add_argument("--small", action="store_true", help="Only first shard for debugging")
p.add_argument("--for_debug", action="store_true", help="Use for_debug directories")
p.add_argument("--n_split", default=None, help="Set sample split")
p.add_argument("--train_seed", type=int, default=12345, help="seed for model init / TF randomness only")
p.add_argument("--trial", type=int, default=None)
args = p.parse_args()

# ---------------- cfg ----------------
cfg_path = os.path.expanduser(os.path.expandvars(args.config))
import common.yaml_loader as yaml_loader
CFG = yaml_loader.load_yaml(cfg_path)

D = CFG.get("defaults", {}) or {}
module_samples = D.get("module_samples", "data.samples")


def list_and_exit():
    jobs = [j for j in (CFG.get("jobs") or []) if j.get("type") == "dnn_c2st"]
    if not jobs:
        print("No DNN C2ST jobs found.")
        sys.exit(0)
    flags = []
    if args.overwrite: flags.append("--overwrite")
    if args.small:     flags.append("--small")
    if args.for_debug: flags.append("--for_debug")
    if args.n_split:   flags.append(f"--n_split {args.n_split}")
    script = os.path.basename(__file__)
    for j in jobs:
        print(f"python {script} {args.config} {' '.join(flags)} --job {j['id']}")
    sys.exit(0)


if args.job is None:
    list_and_exit()

J = next((j for j in (CFG.get("jobs") or []) if j.get("id") == args.job), None)
if J is None or J.get("type") != "dnn_c2st":
    raise RuntimeError(f"Job '{args.job}' not found or not type 'dnn_c2st'.")

# ---------------- seeds ----------------
np.random.seed(args.train_seed)
random.seed(args.train_seed)
tf.random.set_seed(args.train_seed)

# ---------------- resolve loaders ----------------
from data.RDataLoader import RDataLoader
from data.SelectionView import SelectionView

samples_mod = importlib.import_module(module_samples)

from data.UIDSplitter import UIDSplitter

UID_CFG = (J.get("splitting") or {})
uid_enabled   = bool(UID_CFG.get("enabled", False))
uid_fields    = UID_CFG.get("uid_fields", ["run", "luminosityBlock", "event"])
uid_seed      = int(UID_CFG.get("seed", 0))
uid_n_buckets = int(UID_CFG.get("n_buckets", 10000))
uid_scheme    = (UID_CFG.get("scheme") or {})

uid_intervals = None
uid_splitter = None
if uid_enabled:
    uid_splitter = UIDSplitter(
        uid_fields=tuple(uid_fields),
        seed=uid_seed,
        n_buckets=uid_n_buckets,
    )

    keys  = list(uid_scheme.keys())
    fracs = [float((uid_scheme[k] or {}).get("fraction", 0.0)) for k in keys]

    sizes = [int(math.floor(f * uid_n_buckets)) for f in fracs]
    sizes[-1] += uid_n_buckets - sum(sizes)

    uid_intervals = {}
    lo = 0
    for k, sz in zip(keys, sizes):
        uid_intervals[k] = (lo, lo + int(sz))
        lo += int(sz)

    c2st_train_key = "c2st_train"
    c2st_val_key   = "c2st_val"
    train_interval = uid_intervals[c2st_train_key]
    val_interval   = uid_intervals[c2st_val_key]

    print(f"[UID] enabled=True fields={uid_fields} seed={uid_seed} n_buckets={uid_n_buckets}")
    print(f"[UID] scheme intervals: {uid_intervals}")
    print(f"[UID] C2ST train split '{c2st_train_key}' -> {train_interval}")
    print(f"[UID] C2ST val   split '{c2st_val_key}' -> {val_interval}")
else:
    raise RuntimeError("C2ST training requires splitting.enabled: true")

# ---------------- resolve syst reference job (for base_points / shared extras) ----------------
extras = (J.get("extras") or {})
syst_id = extras.get("syst_job_id", None)
SJ = None
if syst_id:
    SJ = next((jj for jj in (CFG.get("jobs") or []) if jj.get("id") == syst_id), None)
    if SJ is None:
        raise RuntimeError(f"syst_job_id '{syst_id}' not found in CFG jobs.")

# ---------------- base_points: prefer job-level, else inherit from syst_job_id ----------------
bp_specs = J.get("base_points", None)
if bp_specs is None and SJ is not None:
    bp_specs = SJ.get("base_points", None)
if not bp_specs:
    raise RuntimeError("No base_points in job, and no base_points available via syst_job_id.")

base_points = [spec["coords"] for spec in bp_specs]

loaders = []

for i, spec in enumerate(bp_specs):
    nm = spec["loader"]
    if not hasattr(samples_mod, nm):
        raise RuntimeError(f"Loader/view '{nm}' not found in module {module_samples}.")
    base = getattr(samples_mod, nm)

    features = J.get("features", None) or (SJ.get("features", None) if SJ else None)
    if not features:
        raise RuntimeError("No features found in job or SJ (required for setFeatures).")
    base.setFeatures(features)


    remove = list(spec.get("removeweights", []) or [])
    add    = list(spec.get("addweights", []) or [])

    if not remove and not add:
        loaders.append(base)
        continue

    if isinstance(base, RDataLoader):
        base_weights = list(base.weight_branches or [])
        root_loader = base

    elif isinstance(base, SelectionView):
        if base._w_override is not None:
            base_weights = list(base._w_override)
        else:
            if not isinstance(base.base, RDataLoader):
                raise RuntimeError(
                    f"SelectionView '{base.name}' has a non-RDataLoader base. "
                    "Layered views are not supported in this job logic."
                )
            base_weights = list(base.base.weight_branches or [])
        root_loader = base.base if isinstance(base.base, RDataLoader) else None
        if root_loader is None:
            raise RuntimeError(f"Could not find underlying RDataLoader for SelectionView '{base.name}'.")
    else:
        raise RuntimeError(f"Loader/view '{nm}' has unsupported type {type(base)} for automatic weight variations.")

    new_weights = list(base_weights)

    for w in remove:
        if w in new_weights:
            new_weights.remove(w)
        else:
            warnings.warn(
                f"[job {SJ.get('id','<unknown>')}] weight '{w}' requested for removal "
                f"but not found in loader '{nm}' (current weights: {base_weights})."
            )

    for w in add:
        if w not in new_weights:
            new_weights.append(w)

    # Ensure the underlying loader reads any new branches
    if hasattr(root_loader, "_requested_branches"):
        for b in add:
            if b not in root_loader._requested_branches:
                root_loader._requested_branches.append(b)

    if root_loader.observer_names is None:
        root_loader.observer_names = list(add)
    else:
        for b in add:
            if b not in root_loader.observer_names:
                root_loader.observer_names.append(b)

    # Construct effective loader/view
    if isinstance(base, RDataLoader):
        vname = f"{nm}_wvar{i}"
        eff_loader = SelectionView(
            base=base,
            name=vname,
            selection_fn=None,
            feature_names=base.feature_names,
            observer_names=base.observer_names,
            selection_feature_names=None,
            weight=new_weights,
        )
    else:
        vname = f"{base.name}_wvar{i}"
        eff_loader = SelectionView(
            base=base.base,
            name=vname,
            selection_fn=base._selection_fns,
            feature_names=base._feature_names,
            observer_names=base._observer_names,
            selection_feature_names=base._sel_feats,
            weight=new_weights,
        )

    loaders.append(eff_loader)

# Optional selection
sel   = J.get("selection", None) or (SJ.get("selection", None) if SJ else None)
sel_f = (J.get("selection_features", None) or (SJ.get("selection_features", None) if SJ else None) or [])

if sel:
    for loader in loaders:
        if isinstance(loader, RDataLoader):
            loader.addSelection(sel, sel_f)
        else:
            loader.base.addSelection(sel, sel_f)

# Reset n_split
if args.n_split:
    print( f"Set the loaders to n_split {args.n_split}" ) 
    for l in loaders:
        if isinstance( l, RDataLoader):
            l.set_n_split( args.n_split )
        else:
            l.base.set_n_split( args.n_split )

# sanity: same features across loaders
feat_names = list(getattr(loaders[0], "feature_names", []))
if not feat_names:
    raise RuntimeError("First loader has no feature_names.")
for L in loaders[1:]:
    if list(getattr(L, "feature_names", [])) != feat_names:
        raise RuntimeError("Feature mismatch across base-point loaders.")

input_dim = len(feat_names)
print(f"[job {J['id']}] input_dim={input_dim}")

for L in loaders:
    obs_names = getattr(L, "observer_names", None)
    if obs_names is None:
        raise RuntimeError(f"Loader {L} has no observer_names, but UID splitting requires {uid_fields}")
    for f in uid_fields:
        if f not in obs_names:
            raise RuntimeError(f"Loader {L} is missing UID field '{f}' in observer_names={obs_names}")

# ---------------- find nominal & varied base point index ----------------
def is_zero_coords(coords):
    c = np.asarray(coords, dtype=float)
    return np.all(np.isclose(c, 0.0))


def is_positive_coords(coords):
    c = np.asarray(coords, dtype=float)
    return np.all(c > 0.0)


nom_idx = None
for i, coords in enumerate(base_points):
    if is_zero_coords(coords):
        nom_idx = i
        break
if nom_idx is None:
    raise RuntimeError("No nominal base point (all zeros) found in base_points.")

# ---------------- choose varied index ----------------
if "test_coords" in J:
    tc = np.asarray(J["test_coords"], dtype=float)
    found = None
    for i, coords in enumerate(base_points):
        if np.all(np.isclose(np.asarray(coords, dtype=float), tc)):
            found = i
            break
    if found is None:
        raise RuntimeError(f"test_coords {J['test_coords']} not found among base_points.")
    var_idx = found

else:
    # NEW DEFAULT: prefer +1 (positive) variation
    pos_indices = [
        i for i, coords in enumerate(base_points)
        if i != nom_idx and is_positive_coords(coords)
    ]
    if pos_indices:
        var_idx = pos_indices[0]
    else:
        # fallback: any non-nominal (should almost never happen)
        var_idx = next(i for i in range(len(base_points)) if i != nom_idx)

if var_idx == nom_idx:
    raise RuntimeError("var_idx equals nominal index; choose a non-zero base point to test.")

print(
    f"[job {J['id']}] nominal idx={nom_idx}, "
    f"varied idx={var_idx}, varied coords={base_points[var_idx]}"
)


# ---------------- artifacts: scaler ----------------
cfg_base = os.path.join(CFG.get("version", "default"), J["region"])

from ML.Scaler.Scaler import Scaler

# prefer c2st job extras, else inherit from syst job extras (pnn)
scaler_id = (extras.get("use_scaler", None) or ((SJ.get("extras") or {}).get("use_scaler", None) if SJ else None))

if scaler_id:
    sj     = next(jj for jj in (CFG.get("jobs") or []) if jj.get("id") == scaler_id)
    sname  = sj["output"]["filename"]
    spath  = os.path.join(user.model_directory, cfg_base, "Scaler", sname)
    sc     = Scaler.load(spath)
    scaler_means, scaler_vars = sc.feature_means, sc.feature_variances
    print(f"Loaded Scaler: {spath}")
else:
    scaler_means = np.zeros(input_dim, dtype=np.float64)
    scaler_vars  = np.ones(input_dim,  dtype=np.float64)
    print("No Scaler configured; using identity.")

# ---------------- load PNN for reweighting (test2/test3) ----------------
def maybe_load_pnn_for_reweighting():
    extras = (J.get("extras") or {})

    # gate by use_pnn (test1: False -> don't load)
    use_pnn = bool(extras.get("use_pnn", False))
    if not use_pnn:
        return None

    pnn_job_id = extras.get("pnn_job_id", None)
    if not pnn_job_id:
        raise RuntimeError(
            f"[job {J.get('id','<unknown>')}] use_pnn=true but extras.pnn_job_id is missing."
        )

    # get PNN job config (for use_icp, etc.)
    PJ = next((jj for jj in (CFG.get("jobs") or []) if jj.get("id") == pnn_job_id), None)
    if PJ is None:
        raise RuntimeError(f"PNN job id '{pnn_job_id}' not found in CFG jobs.")
    pextras = (PJ.get("extras") or {})

    pnn_model_dir = os.path.join(
        user.model_directory,
        cfg_base + ("_for_debug" if args.for_debug else ""),
        "PNN",
        pnn_job_id,
    )
    latest = tf.train.latest_checkpoint(pnn_model_dir)
    if not latest:
        raise RuntimeError(f"No checkpoint found for PNN in: {pnn_model_dir}")
    print(f"Loading PNN from {pnn_model_dir} (latest={latest})")

    pnn = PNN.load(pnn_model_dir)
    pnn.set_scaler(scaler_means, scaler_vars)

    icp_id = pextras.get("use_icp", None)
    if icp_id:
        from ML.ICP.ICP import InclusiveCrosssectionParametrization
        ij = next((jj for jj in (CFG.get("jobs") or []) if jj.get("id") == icp_id), None)
        if ij is None:
            raise RuntimeError(f"ICP job id '{icp_id}' not found in CFG jobs.")

        icp_fn   = ij["output"]["filename"]
        icp_path = os.path.join(user.model_directory, cfg_base, "ICP", icp_fn)
        icp      = InclusiveCrosssectionParametrization.load(icp_path)
        print(f"Loaded ICP: {icp_path}")

        _params = list(icp.parameters)
        _combs  = [tuple(c) for c in icp.combinations]
        _DeltaA = np.asarray(icp.DeltaA, dtype=np.float64)
        pnn.set_icp(parameters=_params, combinations=_combs, DeltaA=_DeltaA)

    return pnn

# ---------------- shard iterator ----------------
def iterate_epoch(split: str, shard_limit=None):
    shard_counts = [len(getattr(L, "base", L)) for L in loaders]
    n_shards = min(shard_counts)
    if shard_limit is not None:
        n_shards = min(n_shards, shard_limit)

    if split == "c2st_train":
        lo, hi = train_interval
    elif split == "c2st_val":
        lo, hi = val_interval
    else:
        raise ValueError(f"Unknown split '{split}'")

    for shard in range(n_shards):
        Xs_sel, Ws_sel = [], []

        for L in loaders:
            X, O, w = L.materialize(shard=shard, what="fow")
            w = w.astype(np.float32, copy=False)

            obs_names = L.observer_names
            uid_idx = [obs_names.index(f) for f in uid_fields]
            O_uid = O[:, uid_idx]

            m = uid_splitter.mask_from_np(O_uid, list(uid_fields), lo, hi)

            Xs_sel.append(X[m])
            Ws_sel.append(w[m])

        yield Xs_sel, Ws_sel

# ---------------- build C2ST dataset ----------------
def build_c2st_arrays(test_id: int, split: str, shard_limit=None):
    """
    User-required definitions:

      test1:
        - background/class0 = nominal dataset (X0, w0)
        - signal/class1     = systematic varied dataset (Xi, wi)

      test3:
        - background/class0 = nominal dataset (X0, w0)
        - signal/class1     = REWEIGHTED systematic varied dataset (Xi, wi * exp(dAi @ vk))

      test2:
        - same (X,y,w) construction as test3
        - split fixed once (train/val/test fixed)
        - ONLY y_train is shuffled per toy (n_trials times)
        - y_val and y_test remain fixed (never shuffled)
    """
    if test_id in (2, 3):
        pnn = maybe_load_pnn_for_reweighting()
        if pnn is None:
            raise RuntimeError("test2/test3 require extras.use_pnn=true and valid extras.pnn_job_id.")
        VkA = pnn.VkA
        vk = VkA[var_idx]
    else:
        pnn = None
        vk = None

    X_list, y_list, w_list = [], [], []
    n0_total, n1_total = 0, 0

    for Xs, Ws in tqdm(iterate_epoch(split=split, shard_limit=shard_limit), desc="Materialize", unit="shard"):
        X0, w0 = Xs[nom_idx], Ws[nom_idx]
        Xi, wi = Xs[var_idx], Ws[var_idx]

        if len(X0) == 0 or len(Xi) == 0:
            continue

        def _drop_neg(X, w):
            m = (w > 0)
            return X[m], w[m]

        # class0: nominal background
        Xc0 = X0
        wc0 = w0

        # class1: varied signal (optionally reweighted)
        Xc1 = Xi
        if test_id == 1:
            wc1 = wi
        else:
            # IMPORTANT: reweight the VARIED sample weights
            dAi = pnn.deltaA(Xi)  # (Ni, C)
            wc1 = wi * np.exp(-(dAi @ vk))

        if test_id == 3 or test_id == 2 or test_id == 1:
            Xc0, wc0 = _drop_neg(Xc0, wc0)
            Xc1, wc1 = _drop_neg(Xc1, wc1)
            if len(Xc0) == 0 or len(Xc1) == 0:
                continue

        X_list.append(Xc0); y_list.append(np.zeros(len(Xc0), dtype=np.int64)); w_list.append(wc0.astype(np.float64))
        X_list.append(Xc1); y_list.append(np.ones (len(Xc1), dtype=np.int64)); w_list.append(wc1.astype(np.float64))

        n0_total += len(Xc0)
        n1_total += len(Xc1)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    w = np.concatenate(w_list, axis=0)

    meta = {
        "test_id": test_id,
        "nom_idx": nom_idx,
        "var_idx": var_idx,
        "var_coords": base_points[var_idx],
        "n_class0": int(n0_total),
        "n_class1": int(n1_total),
    }
    return X, y, w, meta

def make_tf_dataset(
    X, y, w, batch_size: int,
    use_weights: bool = True,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    reshuffle_each_iteration: bool = True,
    ):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    if use_weights:
        w = tf.convert_to_tensor(w, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((X, y, w))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(X),
            seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def offline_weighted_accuracy(y_true, y_score, w, from_logits: bool):
    y_true = np.asarray(y_true).astype(np.int64)
    w = np.asarray(w).astype(np.float64)

    if from_logits:
        y_pred = (y_score >= 0.0).astype(np.int64)
    else:
        y_pred = (y_score >= 0.5).astype(np.int64)

    correct = (y_pred == y_true).astype(np.float64)
    return float(np.sum(w * correct) / np.sum(w))


def offline_weighted_auc(y_true, y_score, w):
    """
    精确的加权 ROC-AUC（排序实现），语义与常见 sample_weight AUC 一致：
    AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)，带权重。
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    w = np.asarray(w).astype(np.float64)

    # 只保留 0/1
    m = (y_true == 0) | (y_true == 1)
    y_true, y_score, w = y_true[m], y_score[m], w[m]

    w_pos = w[y_true == 1].sum()
    w_neg = w[y_true == 0].sum()
    if w_pos <= 0 or w_neg <= 0:
        return float("nan")

    # 按 score 升序排序
    order = np.argsort(y_score, kind="mergesort")
    y = y_true[order]
    ww = w[order]
    s = y_score[order]

    # 计算加权 Mann–Whitney U（处理 ties）
    # 思路：对每个 unique score 的组，先统计该组内 pos_weight，
    # 再用“之前累计的 neg_weight”贡献 U，同时 ties 贡献 0.5*pos*neg_in_group
    auc_u = 0.0
    cum_neg = 0.0

    # 遍历分组
    i = 0
    n = len(s)
    while i < n:
        j = i
        # 找到同分组 [i, j)
        while j < n and s[j] == s[i]:
            j += 1

        w_pos_g = ww[i:j][y[i:j] == 1].sum()
        w_neg_g = ww[i:j][y[i:j] == 0].sum()

        # pos 与之前所有 neg（严格更小分数） -> 全贡献
        auc_u += w_pos_g * cum_neg
        # ties：同分组 pos/neg 各贡献一半
        auc_u += 0.5 * w_pos_g * w_neg_g

        cum_neg += w_neg_g
        i = j

    return float(auc_u / (w_pos * w_neg))

def _normalize_output_activation(val):
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("none", "null", ""):
        return None
    return val

def compile_model_loss_only(dnn: DNN):
    from_logits = (dnn.output_activation is None)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    dnn.model.compile(
        optimizer=dnn.optimizer,
        loss=loss,
        jit_compile=False,
    )

def build_callbacks(optim_cfg: dict):
    callbacks = []
    es = optim_cfg.get("early_stopping", None)
    if isinstance(es, dict):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=str(es.get("monitor", "val_loss")),
                patience=int(es.get("patience", 10)),
                min_delta=float(es.get("min_delta", 0.0)),
                mode=str(es.get("mode", "min")),
                restore_best_weights=bool(es.get("restore_best_weights", True)),
                verbose=1
            )
        )
    return callbacks

def ensure_clean_dir(path: str, overwrite: bool):
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    if (not overwrite) and os.listdir(path):
        raise RuntimeError(f"Directory not empty and --overwrite not set: {path}")



# ---------------- main training logic ----------------
extras = (J.get("extras") or {})
test_type = extras.get("test_type", None)

# accept both styles: J["test_id"] or extras["test_type"]
if test_type is not None:
    t = str(test_type).strip().lower()
    if t == "test1":
        test_id = 1
    elif t == "test2":
        test_id = 2
    elif t == "test3":
        test_id = 3
    else:
        raise ValueError(f"Unknown extras.test_type='{test_type}', must be test1/test2/test3")
else:
    test_id = int(J.get("test_id", 1))

if test_id not in (1, 2, 3):
    raise ValueError("Job must specify test_id in {1,2,3} or extras.test_type in {test1,test2,test3}")

# model/optim config
model_cfg = (J.get("model") or {})
hidden_layers = tuple(model_cfg.get("hidden_layers", [512, 512, 256, 128]))
activation = str(model_cfg.get("activation", "relu"))
l1 = float(model_cfg.get("l1", 0.0))
l2 = float(model_cfg.get("l2", 0.0))
output_activation = _normalize_output_activation(model_cfg.get("output_activation", None))  # None -> logits

optim_cfg = (J.get("optim") or {})
epochs = int(optim_cfg.get("epochs", 200))
lr = float(optim_cfg.get("learning_rate", 1e-3))
batch_size = int(optim_cfg.get("batch_size", 4096))

# test2 toys
n_trials = int(extras.get("n_toys", 1000))
# dirs
base_model_dir = os.path.join(
    user.model_directory,
    cfg_base + ("_for_debug" if args.for_debug else ""),
    "DNN_C2ST",
)

# shard control
shard_limit = 1 if args.small else None

# build arrays (X,y,w fixed)
# --- TRAIN ---
Xtr, ytr, wtr, meta = build_c2st_arrays(
    test_id=test_id,
    split="c2st_train",
    shard_limit=shard_limit
)

# --- VAL ---
Xva, yva, wva, _ = build_c2st_arrays(
    test_id=test_id,
    split="c2st_val",
    shard_limit=shard_limit
)
Xtr = np.asarray(Xtr)
ytr = np.asarray(ytr).astype(np.int64)
wtr = np.asarray(wtr).astype(np.float64)

Xva = np.asarray(Xva)
yva = np.asarray(yva).astype(np.int64)
wva = np.asarray(wva).astype(np.float64)

eps = 1e-8
Xtr = (Xtr - scaler_means) / np.sqrt(scaler_vars + eps)
Xva = (Xva - scaler_means) / np.sqrt(scaler_vars + eps)

# helper: one training run (TRAIN ONLY)
def train_once(run_dir: str, ytr_use, yva_fixed, trial_idx: int = 0):
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)
    tf.random.set_seed(args.train_seed)

    ensure_clean_dir(run_dir, overwrite=args.overwrite)

    dnn = DNN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        learning_rate=lr,
        l1=l1,
        l2=l2,
        output_activation=output_activation,
    )
    compile_model_loss_only(dnn)
    callbacks = build_callbacks(optim_cfg)

    use_weights = True

    if test_id in (1, 3):
        tr_shuffle = True
        tr_shuffle_seed = args.train_seed
        tr_reshuffle_each_iteration = True
    elif test_id == 2:
        tr_shuffle = True
        tr_shuffle_seed = args.train_seed
        tr_reshuffle_each_iteration = False
    else:
        raise ValueError(f"Unsupported test_id={test_id}")

    ds_tr = make_tf_dataset(
        Xtr.astype(np.float32),
        ytr_use.astype(np.float32),
        wtr.astype(np.float32),
        batch_size=batch_size,
        use_weights=use_weights,
        shuffle=tr_shuffle,
        shuffle_seed=tr_shuffle_seed,
        reshuffle_each_iteration=tr_reshuffle_each_iteration,
    )

    ds_va = make_tf_dataset(
        Xva.astype(np.float32),
        yva_fixed.astype(np.float32),
        wva.astype(np.float32),
        batch_size=batch_size,
        use_weights=use_weights,
        shuffle=False,
    )

    t0 = time.time()
    hist = dnn.model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    t1 = time.time()

    H = hist.history
    if "val_loss" in H and len(H["val_loss"]) > 0:
        best_epoch = int(np.argmin(H["val_loss"]))
        best_val_loss = float(H["val_loss"][best_epoch])
        print(f"[BEST] epoch={best_epoch} val_loss={best_val_loss}")
    else:
        best_epoch = len(H.get("loss", [])) - 1

    from_logits = (output_activation is None)

    ds_va_x = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(Xva.astype(np.float32))
    ).batch(batch_size)

    y_score_va = dnn.model.predict(ds_va_x, verbose=0).reshape(-1)

    val_acc_w = offline_weighted_accuracy(yva_fixed, y_score_va, wva, from_logits=from_logits)
    val_auc_w = offline_weighted_auc(yva_fixed, y_score_va, wva)

    print(f"[FINAL VAL] weighted_acc={val_acc_w:.6f} weighted_auc={val_auc_w:.6f}")

    last_epoch = len(hist.history.get("loss", [])) - 1
    dnn.save(run_dir, epoch=last_epoch)

    with open(os.path.join(run_dir, "history.pkl"), "wb") as f:
        pickle.dump(dict(hist.history), f)

    with open(os.path.join(run_dir, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "job_id": J["id"],
                "test_id": test_id,
                "input_dim": input_dim,
                "hidden_layers": list(hidden_layers),
                "activation": activation,
                "lr": lr,
                "epochs_cap": epochs,
                "batch_size": batch_size,
                "output_activation": output_activation,
                "loader_meta": meta,
                "train_time_sec": float(t1 - t0),
                "best_epoch": int(best_epoch),
                "final_val_weighted_acc": float(val_acc_w),
                "final_val_weighted_auc": float(val_auc_w),
            },
            f,
        )

    print(f"[run {os.path.basename(run_dir)}] training done. last_epoch={last_epoch}")
    return dict(hist.history)

# ---------------- dispatch ----------------
if test_id == 3:
    trials = [0]
    for t in trials:
        model_dir = os.path.join(base_model_dir, f"{J['id']}_trial{t:04d}")
        os.makedirs(model_dir, exist_ok=True)
        run_dir = os.path.join(model_dir, "main")

        train_once(run_dir, ytr_use=ytr, yva_fixed=yva, trial_idx=t)
        with open(os.path.join(model_dir, "meta_dataset.pkl"), "wb") as f:
            pickle.dump(
                {
                    "job_id": J["id"],
                    "test_id": test_id,
                    "output_activation": output_activation,
                    "loader_meta": meta,
                },
                f,
            )

elif test_id == 1:
    trials = [0]
    for t in trials:
        model_dir = os.path.join(base_model_dir, f"{J['id']}_trial{t:04d}")
        os.makedirs(model_dir, exist_ok=True)
        run_dir = os.path.join(model_dir, "main")

        train_once(run_dir, ytr_use=ytr, yva_fixed=yva, trial_idx=t)
        with open(os.path.join(model_dir, "meta_dataset.pkl"), "wb") as f:
            pickle.dump(
                {
                    "job_id": J["id"],
                    "test_id": test_id,
                    "output_activation": output_activation,
                    "loader_meta": meta,
                },
                f,
            )

    syncer.sync()

elif test_id == 2:
    # test2: ONLY shuffle TRAIN labels, val/test fixed forever
    trials = [args.trial] if args.trial is not None else range(n_trials)
    for t in trials:
        model_dir = os.path.join(base_model_dir, f"{J['id']}_trial{t:04d}")
        os.makedirs(model_dir, exist_ok=True)
        run_dir = os.path.join(model_dir, "main")
        rng = np.random.default_rng(args.train_seed + t)
        ytr_shuf = ytr.copy()
        rng.shuffle(ytr_shuf)  # <-- ONLY TRAIN labels shuffled

        train_once(run_dir, ytr_use=ytr_shuf, yva_fixed=yva, trial_idx=t)
        with open(os.path.join(model_dir, "meta_dataset.pkl"), "wb") as f:
            pickle.dump(
                {
                    "job_id": J["id"],
                    "test_id": test_id,
                    "output_activation": output_activation,
                    "loader_meta": meta,
                },
                f,
            )

    syncer.sync()
