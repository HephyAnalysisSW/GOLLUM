import os, math, copy, time, argparse, resource, gc, sys
from array import array
from collections import OrderedDict

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)

import numpy as np

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import common.user as user
import common.helpers as helpers
import common.syncer as syncer

from data.plot_options import plot_options
from data.SelectionView import SelectionView
from data.samples_RunII import Factory
from variations import variations  # OrderedDict


# -----------------------------------------------------------------------------
# CLI options (parse_known_args so IPython/Jupyter extra args don't break)
# -----------------------------------------------------------------------------
p = argparse.ArgumentParser(description="Prefit plots with syst band (subsettable)")
p.add_argument("--plot_directory", default="v2-3-2_tr_isvalid_isOS_offZ",
               help="String-based selection")
p.add_argument("--variation", nargs="+", default=None,
               help="Subset of systematics to run (space-separated). 'nominal' is always included. "
                    "Example: --variation alphaS ren jes_abs_16")
p.add_argument("--eras", nargs="+", default=None,
               help="Subset of eras to run (space-separated). Example: --eras 2017 2018")
p.add_argument("--selection", default=" (lep1_pt>20) & tr_isvalid & isOS & offZ",
               help="String-based selection")
p.add_argument("--branches_for_selection", nargs="+", default=["lep1_pt", "tr_isvalid", "isOS", "offZ"],
               help="branches we need to make the selection")
p.add_argument("--features", nargs="+", default=[],
               help="branches we need to make the selection")
p.add_argument("--templates", nargs="*", default=None,
               help="Make per-variation template plots for the given process(es). "
                    "If passed without values, defaults to TTLep_pow.")
args, _unknown = p.parse_known_args()

# -----------------------------------------------------------------------------
# User-editable config (in IPython you can override before %run -i)
# -----------------------------------------------------------------------------
#base = "/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-3_nJ2p_nB2p_2l/"
from data.samples_RunII import BASE_DIRECTORY
base = str(BASE_DIRECTORY)

eras = ["2016", "2016APV", "2017", "2018"]
processes = ["TTLep_pow", "SingleTop", "TTSemi_pow", "DrellYan"]
signals = []

# shard splitting (default 10; override per-process if needed)
N_SPLIT_DEFAULT = 1
n_split = {"TTLep_pow": 10}

features = []
features += [
    "MET_phi", "MET_pt", "ht", "nBJet", "nSelJet",
    "lep0_charge", "lep0_eta", "lep0_phi", "lep0_pt",
    "lep1_charge", "lep1_eta", "lep1_phi", "lep1_pt",
    "jet0_pt", "jet0_eta", "jet1_pt", "jet1_eta",
    "dilep_eta", "dilep_mass", "dilep_phi", "dilep_pt", "dilep_dEta", "dilep_dAbsEta",
    "tr_Top_eta", "tr_Top_mass", "tr_Top_phi", "tr_Top_pt", "tr_Top_y",
    "tr_AntiTop_eta", "tr_AntiTop_mass", "tr_AntiTop_phi", "tr_AntiTop_pt", "tr_AntiTop_y",
    "tr_ttbar_pt", "tr_ttbar_eta", "tr_ttbar_mass", "tr_ttbar_phi", "tr_ttbar_y", "tr_ttbar_beta_plus", "tr_ttbar_dEta", "tr_ttbar_dAbsEta",
    "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",
    "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r",
    "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star",
    "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star",
    "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus", "tr_xi_rk_plus", "tr_xi_rk_minus",
    "tr_xi_nk_plus", "tr_xi_nk_minus", "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
    "tr_cos_phi", "tr_c_hel", "tr_c_han",
]

# Restrict to list, if provided
if args.features:
    features = [f for f in features if f in args.features]

from data.colors import colors as proc_colors
from data.samples_RunII import process_labels

# -----------------------------------------------------------------------------
# Apply CLI subsetting (eras + variations)
# -----------------------------------------------------------------------------
if args.eras is not None:
    bad = [e for e in args.eras if e not in eras]
    if bad:
        raise RuntimeError(f"Unknown era(s) in --eras: {bad}. Known eras: {eras}")
    eras = list(args.eras)

if args.variation is not None:
    bad = [v for v in args.variation if v not in variations]
    if bad:
        raise RuntimeError(f"Unknown variation(s) in --variation: {bad}. Known keys: {list(variations.keys())}")
    keep = set(args.variation)
    variations = OrderedDict((k, v) for (k, v) in variations.items() if (k == "nominal" or k in keep))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
t0_global = time.time()
_last_msg_time = 0.0

def progress(msg, force=False, every_s=5.0):
    global _last_msg_time
    now = time.time()
    if force or (now - _last_msg_time) >= every_s:
        dt = now - t0_global
        print(f"[{dt:8.1f}s] {msg}", flush=True)
        _last_msg_time = now

def rss_mb():
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")

def rssmax_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def mem_report(tag):
    progress(f"[mem] {tag:>22s}  RSS={rss_mb():8.1f} MB  RSS_max={rssmax_mb():8.1f} MB", every_s=0.0)

def edges_from_binning(binning):
    if isinstance(binning, (list, tuple)) and len(binning) == 3 and isinstance(binning[0], int):
        nb, x1, x2 = binning
        return np.linspace(float(x1), float(x2), nb + 1, dtype=np.float64)
    return np.array(list(binning), dtype=np.float64)

def hist_with_flow(values, weights, edges):
    nb = len(edges) - 1
    idx = np.searchsorted(edges, values, side="right") - 1

    in_range = (idx >= 0) & (idx < nb)
    sumw  = np.bincount(idx[in_range], weights=weights[in_range], minlength=nb).astype(np.float64)
    sumw2 = np.bincount(idx[in_range], weights=(weights[in_range] ** 2), minlength=nb).astype(np.float64)

    under = (idx < 0)
    if np.any(under):
        w = weights[under]
        sumw[0]  += np.sum(w)
        sumw2[0] += np.sum(w * w)

    over = (idx >= nb)
    if np.any(over):
        w = weights[over]
        sumw[-1]  += np.sum(w)
        sumw2[-1] += np.sum(w * w)

    return sumw, sumw2

def np_to_th1(name, edges, sumw, sumw2=None):
    h = ROOT.TH1D(name, name, len(edges) - 1, array("d", edges.tolist()))
    h.SetDirectory(0)
    if sumw2 is None:
        sumw2 = np.zeros_like(sumw)
    for ib in range(1, len(edges)):
        c = float(sumw[ib - 1])
        e2 = float(sumw2[ib - 1])
        h.SetBinContent(ib, c)
        h.SetBinError(ib, math.sqrt(e2) if e2 > 0 else 0.0)
    return h

def make_asymm_band_graph(edges, y, err_dn, err_up, fill_color=ROOT.kGray + 1, fill_style=3345):
    nb = len(edges) - 1
    g = ROOT.TGraphAsymmErrors(nb)
    for i in range(nb):
        x1, x2 = float(edges[i]), float(edges[i + 1])
        xc = 0.5 * (x1 + x2)
        ex = 0.5 * (x2 - x1)
        g.SetPoint(i, xc, float(y[i]))
        g.SetPointError(i, ex, ex, float(err_dn[i]), float(err_up[i]))
    g.SetFillColor(fill_color)
    g.SetFillStyle(fill_style)
    g.SetLineWidth(0)
    return g


# -----------------------------------------------------------------------------
# Progress header
# -----------------------------------------------------------------------------
progress(
    f"Config: eras={eras} (n={len(eras)}), processes={processes} (n={len(processes)}), "
    f"signals={signals} (n={len(signals)}), variations={list(variations.keys())} (n={len(variations)}), "
    f"features={len(features)}",
    force=True
)

# -----------------------------------------------------------------------------
# Binning + labels (fail loudly if missing)
# -----------------------------------------------------------------------------
progress("Building binning/label maps from plot_options...", force=True)
edges_map, xlabel_map = {}, {}
for f in features:
    if f not in plot_options:
        raise RuntimeError(f"Feature '{f}' not in plot_options.")
    if "binning" not in plot_options[f]:
        raise RuntimeError(f"plot_options['{f}'] has no 'binning'.")
    edges_map[f]  = edges_from_binning(plot_options[f]["binning"])
    xlabel_map[f] = plot_options[f].get("tex", f)


# -----------------------------------------------------------------------------
# Build loaders into variations dict
# -----------------------------------------------------------------------------
progress("Instantiating Factory...", force=True)
factory = Factory(BASE_DIRECTORY=base)

progress(
    f"Building loaders for {len(variations)} variations x {len(eras)} eras x {len(processes)} bkg processes (+ nominal signals)...",
    force=True
)

for var_name, variation in variations.items():
    variation["loaders"] = {}

    if var_name != "nominal":
        has_weight = ("weight_up" in variation) or ("weight_down" in variation)
        has_sys    = ("sys_up" in variation) or ("sys_down" in variation)
        if has_weight and has_sys:
            raise NotImplementedError(f"Variation '{var_name}' mixes weights and sys branches.")
        if has_weight and (("weight_up" not in variation) or ("weight_down" not in variation)):
            raise RuntimeError(f"Weight variation '{var_name}' must define BOTH weight_up and weight_down.")
        if has_sys and (("sys_up" not in variation) or ("sys_down" not in variation)):
            raise RuntimeError(f"Sys variation '{var_name}' must define BOTH sys_up and sys_down.")
        if (not has_weight) and (not has_sys):
            raise RuntimeError(f"Variation '{var_name}' is neither sys nor weight; define sys_* or weight_*.")

    for era in eras:
        if ("eras" in variation) and (era not in variation["eras"]):
            continue

        progress(f"[loaders] variation='{var_name}' era='{era}'", every_s=2.0)
        variation["loaders"][era] = {}

        # backgrounds
        for proc in processes:
            variation["loaders"][era][proc] = {}

            if var_name == "nominal":
                l = factory.get(proc, era)
                l.set_n_split( n_split.get(proc, N_SPLIT_DEFAULT))
                l.setFeatures(features)
                l.addSelection(args.selection, args.branches_for_selection)
                variation["loaders"][era][proc]["nominal"] = l
                continue

            # nominal pointer
            variation["loaders"][era][proc]["nominal"] = variations["nominal"]["loaders"][era][proc]["nominal"]

            # sys-based (alternate loaders)
            if "sys_up" in variation:
                l_up = factory.get(proc, era, variation["sys_up"])
                l_dn = factory.get(proc, era, variation["sys_down"])
                l_up.set_n_split( n_split.get(proc, N_SPLIT_DEFAULT))
                l_dn.set_n_split( n_split.get(proc, N_SPLIT_DEFAULT))
                l_up.setFeatures(features)
                l_dn.setFeatures(features)
                l_up.addSelection(args.selection, args.branches_for_selection)
                l_dn.addSelection(args.selection, args.branches_for_selection)
                variation["loaders"][era][proc]["up"] = l_up
                variation["loaders"][era][proc]["down"] = l_dn

            # weight-based (views)
            elif "weight_up" in variation:
                base_loader = variations["nominal"]["loaders"][era][proc]["nominal"]

                up_w = list(base_loader.weight_branches)
                dn_w = list(base_loader.weight_branches)

                rw = variation.get("removeweight", variation.get("remove_weight", None))
                if rw is not None:
                    if rw not in up_w or rw not in dn_w:
                        raise RuntimeError(f"Variation '{var_name}': remove_weight '{rw}' not in base weight_branches.")
                    up_w.remove(rw)
                    dn_w.remove(rw)

                up_w.append(variation["weight_up"])
                dn_w.append(variation["weight_down"])

                variation["loaders"][era][proc]["up"] = SelectionView(
                    base=base_loader,
                    name=f"{proc}_{era}_{var_name}_up",
                    selection_fn=None,
                    feature_names=base_loader.feature_names,
                    observer_names=base_loader.observer_names,
                    weight=up_w,
                )
                variation["loaders"][era][proc]["down"] = SelectionView(
                    base=base_loader,
                    name=f"{proc}_{era}_{var_name}_down",
                    selection_fn=None,
                    feature_names=base_loader.feature_names,
                    observer_names=base_loader.observer_names,
                    weight=dn_w,
                )

        # signals: nominal only
        if var_name == "nominal":
            for proc in signals:
                if proc in variation["loaders"][era]:
                    continue
                variation["loaders"][era][proc] = {}
                l = factory.get(proc, era)
                l.set_n_split(n_split.get(proc, N_SPLIT_DEFAULT))
                l.setFeatures(features)
                l.addSelection(args.selection, args.branches_for_selection)
                variation["loaders"][era][proc]["nominal"] = l


# -----------------------------------------------------------------------------
# Build loader_hist in the intended pattern:
#   era -> proc -> shard (nominal) -> variation -> features
#   - clear nominal cache once per shard to avoid accumulating shards
#   - close ONLY the nominal loader at end of (era,proc)
# -----------------------------------------------------------------------------
from tqdm import tqdm

loader_hist = {}  # (var, era, proc, which) -> {feat: (sumw, sumw2)}

def _zeros_for_feat(feat):
    return np.zeros(len(edges_map[feat]) - 1, dtype=np.float64)

def _fill_from_arrays(acc_sumw, acc_sumw2, X, W):
    if X.size == 0 or W.size == 0:
        return
    for i, feat in enumerate(features):
        x = X[:, i]
        m = np.isfinite(x) & np.isfinite(W)
        sw, sw2 = hist_with_flow(x[m], W[m], edges_map[feat])
        acc_sumw[feat]  += sw
        acc_sumw2[feat] += sw2

def _materialize_fw_once(loader, shard):
    ar = loader[shard]
    X = loader.scalar_branches(ar, features)
    if loader.weight_branches:
        Wcols = loader.scalar_branches(ar, loader.weight_branches).astype(np.float32, copy=False)
        W = np.prod(Wcols, axis=1).astype(np.float64, copy=False)
    else:
        W = np.ones(len(ar), dtype=np.float64)
    return X, W

progress("Filling per-(era,proc,var) histograms (shard-major)...", force=True)
for era in eras:
    for proc in (processes + signals):
        progress(f"[hists] era='{era}' proc='{proc}'", force=True)

        l_nom = variations["nominal"]["loaders"][era][proc]["nominal"]
        n_shards = len(l_nom)

        # active variations only for background processes
        vnames = []
        if proc in processes:
            for vname, vdef in variations.items():
                if vname == "nominal":
                    continue
                if ("eras" in vdef) and (era not in vdef["eras"]):
                    continue
                if era not in vdef.get("loaders", {}):
                    continue
                if proc not in vdef["loaders"][era]:
                    continue
                ldict = vdef["loaders"][era][proc]
                if ("up" in ldict) and ("down" in ldict):
                    vnames.append(vname)

        # accumulators
        nom_sumw  = {f: _zeros_for_feat(f) for f in features}
        nom_sumw2 = {f: _zeros_for_feat(f) for f in features}
        v_sumw  = {(v, w): {f: _zeros_for_feat(f) for f in features} for v in vnames for w in ("up", "down")}
        v_sumw2 = {(v, w): {f: _zeros_for_feat(f) for f in features} for v in vnames for w in ("up", "down")}

        mem_report(f"before shard loop {era}/{proc} (n_shards={n_shards})")

        for shard in tqdm(range(n_shards), desc=f"{era}:{proc}", leave=False):

            # nominal once per shard (this populates base cache for views)
            X_nom, W_nom = _materialize_fw_once(l_nom, shard)

            _fill_from_arrays(nom_sumw, nom_sumw2, X_nom, W_nom)

            # variations inside shard
            for vname in vnames:
                vdef = variations[vname]
                ldict = vdef["loaders"][era][proc]
                up = ldict["up"]
                dn = ldict["down"]

                # weight-based: views only need weights; reuse nominal features
                if "weight_up" in vdef:
                    (W_up,) = up.materialize(shard, "w")
                    (W_dn,) = dn.materialize(shard, "w")
                    W_up = np.asarray(W_up, dtype=np.float64)
                    W_dn = np.asarray(W_dn, dtype=np.float64)

                    if X_nom.size:
                        for i, feat in enumerate(features):
                            x = X_nom[:, i]
                            mu = np.isfinite(x) & np.isfinite(W_up)
                            md = np.isfinite(x) & np.isfinite(W_dn)
                            swu, sw2u = hist_with_flow(x[mu], W_up[mu], edges_map[feat])
                            swd, sw2d = hist_with_flow(x[md], W_dn[md], edges_map[feat])
                            v_sumw[(vname, "up")][feat]    += swu
                            v_sumw2[(vname, "up")][feat]   += sw2u
                            v_sumw[(vname, "down")][feat]  += swd
                            v_sumw2[(vname, "down")][feat] += sw2d

                # sys-based: alternate loaders; must load their own features+weights
                else:
                    X_u, W_u = _materialize_fw_once(up, shard)
                    X_d, W_d = _materialize_fw_once(dn, shard)
                    _fill_from_arrays(v_sumw[(vname, "up")],   v_sumw2[(vname, "up")],   X_u, W_u)
                    _fill_from_arrays(v_sumw[(vname, "down")], v_sumw2[(vname, "down")], X_d, W_d)

            del X_nom, W_nom
            gc.collect()

        mem_report(f"after shard loop  {era}/{proc}")

        # commit
        loader_hist[("nominal", era, proc, "nominal")] = {f: (nom_sumw[f], nom_sumw2[f]) for f in features}
        for vname in vnames:
            loader_hist[(vname, era, proc, "up")]   = {f: (v_sumw[(vname, "up")][f],   v_sumw2[(vname, "up")][f])   for f in features}
            loader_hist[(vname, era, proc, "down")] = {f: (v_sumw[(vname, "down")][f], v_sumw2[(vname, "down")][f]) for f in features}

        # close sys loaders too (weight views don't own resources)
        for vname in vnames:
            vdef = variations[vname]
            if "sys_up" in vdef:
                variations[vname]["loaders"][era][proc]["up"].close()
                variations[vname]["loaders"][era][proc]["down"].close()

        # close nominal loader last (views share it)
        l_nom.close()
        mem_report(f"after close loaders {era}/{proc}")

# -----------------------------------------------------------------------------
# Totals and uncertainties (no data I/O here; only loader_hist lookups)
# -----------------------------------------------------------------------------
tot_hist = {f: {"nominal": None} for f in features}
for f in features:
    for v in variations.keys():
        if v != "nominal":
            tot_hist[f][v] = {"up": None, "down": None}

unc_hist = {f: {"err_dn": None, "err_up": None} for f in features}

progress(f"Computing totals/uncertainties for {len(features)} features...", force=True)

for ifeat, feat in enumerate(features, start=1):
    progress(f"[totals] ({ifeat}/{len(features)}) feat='{feat}'", force=True)

    nb = len(edges_map[feat]) - 1
    y_nom = np.zeros(nb, dtype=np.float64)
    sw2_nom = np.zeros(nb, dtype=np.float64)

    for era in eras:
        for proc in processes:
            sw, sw2 = loader_hist[("nominal", era, proc, "nominal")][feat]
            y_nom += sw
            sw2_nom += sw2

    tot_hist[feat]["nominal"] = (y_nom, sw2_nom)

    for vname, vdef in variations.items():
        if vname == "nominal":
            continue

        y_up = np.zeros(nb, dtype=np.float64)
        y_dn = np.zeros(nb, dtype=np.float64)

        for era in eras:
            var_applies = (("eras" not in vdef) or (era in vdef["eras"]))
            for proc in processes:
                if var_applies and (vname, era, proc, "up") in loader_hist:
                    swu, _ = loader_hist[(vname, era, proc, "up")][feat]
                    swd, _ = loader_hist[(vname, era, proc, "down")][feat]
                else:
                    sw, _ = loader_hist[("nominal", era, proc, "nominal")][feat]
                    swu, swd = sw, sw
                y_up += swu
                y_dn += swd

        tot_hist[feat][vname]["up"] = y_up
        tot_hist[feat][vname]["down"] = y_dn

    sig2_up = np.zeros(nb, dtype=np.float64)
    sig2_dn = np.zeros(nb, dtype=np.float64)

    for vname in variations.keys():
        if vname == "nominal":
            continue
        d_up = tot_hist[feat][vname]["up"] - y_nom
        d_dn = tot_hist[feat][vname]["down"] - y_nom

        pos = np.maximum.reduce([d_up, d_dn, np.zeros_like(d_up)])
        neg = np.maximum.reduce([-d_up, -d_dn, np.zeros_like(d_up)])

        sig2_up += pos * pos
        sig2_dn += neg * neg

    sig2_up += sw2_nom
    sig2_dn += sw2_nom

    unc_hist[feat]["err_up"] = np.sqrt(sig2_up)
    unc_hist[feat]["err_dn"] = np.sqrt(sig2_dn)


# -----------------------------------------------------------------------------
# ROOT plotting objects (also kept globally for inspection)
# -----------------------------------------------------------------------------
root_objs = {}

postfix = "RunII" if not args.eras else "_".join(args.eras)
plot_directory = f"{args.plot_directory}_{postfix}"

outdir = os.path.join(user.plot_directory, "prefit_sys", plot_directory)
outdir_lin = os.path.join(outdir, "lin")
outdir_log = os.path.join(outdir, "log")

os.makedirs(outdir_lin, exist_ok=True)
os.makedirs(outdir_log, exist_ok=True)
helpers.copyIndexPHP(outdir)
helpers.copyIndexPHP(outdir_lin)
helpers.copyIndexPHP(outdir_log)

progress(f"Plotting {len(features)} features to: {outdir}", force=True)

for ifeat, feat in enumerate(features, start=1):
    progress(f"[plot] ({ifeat}/{len(features)}) feat='{feat}'", force=True)

    skip_empty_parton = feat.startswith("parton_")

    edges = edges_map[feat]
    x_title = xlabel_map[feat]
    nb = len(edges) - 1

    y_nom, sumw2_nom = tot_hist[feat]["nominal"]
    err_up = unc_hist[feat]["err_up"]
    err_dn = unc_hist[feat]["err_dn"]

    # per-process, summed over eras (stack)
    proc_sumw, proc_sumw2 = {}, {}
    for proc in processes:
        sw = np.zeros(nb, dtype=np.float64)
        sw2 = np.zeros(nb, dtype=np.float64)
        for era in eras:
            s, s2 = loader_hist[("nominal", era, proc, "nominal")][feat]
            sw += s
            sw2 += s2
        proc_sumw[proc] = sw
        proc_sumw2[proc] = sw2

    proc_has_content = {p: bool(np.any(proc_sumw[p] != 0.0)) for p in processes}

    h_procs, lbls, proc_keys = [], [], []
    for proc in processes:
        h = np_to_th1(f"h_{feat}_{proc}", edges, proc_sumw[proc], proc_sumw2[proc])
        h.SetFillColor(proc_colors.get(proc, ROOT.kGray))
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        h_procs.append(h)
        lbls.append(process_labels.get(proc, proc))
        proc_keys.append(proc)

    order = np.argsort([h.Integral() for h in h_procs])
    h_procs = [h_procs[i] for i in order]
    lbls = [lbls[i] for i in order]
    proc_keys = [proc_keys[i] for i in order]

    h_total = np_to_th1(f"h_{feat}_total", edges, y_nom, np.zeros_like(y_nom))

    # signals: nominal only (overlay, not in totals)
    h_signals, sig_labels, sig_has_content = [], [], []
    for sproc in signals:
        s_sumw = np.zeros(nb, dtype=np.float64)
        s_sumw2 = np.zeros(nb, dtype=np.float64)
        for era in eras:
            s, s2 = loader_hist[("nominal", era, sproc, "nominal")][feat]
            s_sumw += s
            s_sumw2 += s2
        sig_has_content.append(bool(np.any(s_sumw != 0.0)))

        hs_sig = np_to_th1(f"h_{feat}_{sproc}", edges, s_sumw, s_sumw2)
        hs_sig.SetFillStyle(0)
        hs_sig.SetFillColor(0)
        hs_sig.SetLineColor(proc_colors.get(sproc, ROOT.kMagenta + 2))
        hs_sig.SetLineWidth(3)
        hs_sig.SetLineStyle(1)
        h_signals.append(hs_sig)
        sig_labels.append(process_labels.get(sproc, sproc))

    # Asimov data
    h_data = h_total.Clone(f"h_{feat}_data")
    h_data.SetDirectory(0)
    h_data.SetMarkerStyle(20)
    h_data.SetMarkerSize(1.0)
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(2)
    for ib in range(1, nb + 1):
        c = h_data.GetBinContent(ib)
        h_data.SetBinError(ib, math.sqrt(c) if c > 0 else 0.0)

    g_band = make_asymm_band_graph(edges, y_nom, err_dn, err_up, fill_color=ROOT.kGray + 1, fill_style=3345)

    h_up_line = np_to_th1(f"h_{feat}_up_line", edges, y_nom + err_up, np.zeros_like(y_nom))
    h_dn_line = np_to_th1(f"h_{feat}_dn_line", edges, y_nom - err_dn, np.zeros_like(y_nom))
    for hh in (h_up_line, h_dn_line):
        hh.SetFillStyle(0)
        hh.SetFillColor(0)
        hh.SetLineColor(ROOT.kGray + 2)
        hh.SetLineWidth(1)

    canvas_name = f"c_prefit_{feat}"
    c_fit = ROOT.TCanvas(canvas_name, canvas_name, 800, 800)

    padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 1.0)
    padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

    padTop.SetBottomMargin(0.0)
    padTop.SetTopMargin(0.08)
    padTop.SetLeftMargin(0.10)
    padTop.SetRightMargin(0.05)
    padTop.SetTicks(1, 1)

    padBottom.SetTopMargin(0.0)
    padBottom.SetBottomMargin(0.30)
    padBottom.SetLeftMargin(0.10)
    padBottom.SetRightMargin(0.05)
    padBottom.SetTicks(1, 1)

    padTop.Draw()
    padBottom.Draw()

    # TOP
    padTop.cd()
    padTop.SetLogy(False)

    hs = ROOT.THStack(f"hs_{feat}", "")
    for h in h_procs:
        hs.Add(h)

    hs.Draw("HIST")
    hs.GetXaxis().SetTitle(x_title)
    hs.GetYaxis().SetTitle("Events")
    hs.GetYaxis().SetTitleSize(0.040)
    hs.GetYaxis().SetTitleOffset(1.15)
    hs.GetYaxis().SetLabelSize(0.035)
    hs.GetXaxis().SetLabelSize(0)
    hs.GetXaxis().SetTitleSize(0)

    max_y = max(hs.GetMaximum(), h_data.GetMaximum())
    for hh in h_signals:
        max_y = max(max_y, hh.GetMaximum())

    hs.SetMinimum(0.0)
    hs.SetMaximum(1.75 * max_y if max_y > 0 else 1.0)

    g_band.Draw("2 SAME")
    h_up_line.Draw("HIST SAME")
    h_dn_line.Draw("HIST SAME")
    h_data.Draw("E SAME")

    for hh in h_signals:
        hh.Draw("HIST SAME")

    leg = ROOT.TLegend(0.15, 0.75, 0.95, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(3)
    leg.SetTextSize(0.035)

    leg.AddEntry(h_data, "Data (Asimov)", "lep")

    for h, lbl, pk in zip(h_procs[::-1], lbls[::-1], proc_keys[::-1]):
        if skip_empty_parton and (not proc_has_content.get(pk, True)):
            continue
        leg.AddEntry(h, lbl, "f")

    leg.AddEntry(g_band, "Uncertainty", "f")

    for hh, lbl, ok in zip(h_signals, sig_labels, sig_has_content):
        if skip_empty_parton and (not ok):
            continue
        leg.AddEntry(hh, lbl, "l")

    leg.Draw()

    # BOTTOM (ratio band)
    padBottom.cd()

    h_ratio_central = h_total.Clone(f"h_{feat}_ratio_central")
    h_ratio_central.SetDirectory(0)
    h_ratio_central.Reset("ICES")
    for ib in range(1, nb + 1):
        h_ratio_central.SetBinContent(ib, 1.0)
        h_ratio_central.SetBinError(ib, 0.0)

    h_ratio_central.SetLineColor(ROOT.kBlack)
    h_ratio_central.SetLineWidth(2)
    h_ratio_central.SetTitle("")

    h_ratio_central.GetYaxis().SetTitle("var / nominal")
    h_ratio_central.GetYaxis().SetNdivisions(505)
    h_ratio_central.GetYaxis().SetTitleSize(0.09)
    h_ratio_central.GetYaxis().SetTitleOffset(0.5)
    h_ratio_central.GetYaxis().SetLabelSize(0.08)

    h_ratio_central.GetXaxis().SetTitle(x_title)
    h_ratio_central.GetXaxis().SetTitleSize(0.10)
    h_ratio_central.GetXaxis().SetLabelSize(0.08)

    ratio_boxes = []
    h_ratio_line_up = h_ratio_central.Clone(f"h_{feat}_ratio_up")
    h_ratio_line_dn = h_ratio_central.Clone(f"h_{feat}_ratio_dn")
    for hh in (h_ratio_line_up, h_ratio_line_dn):
        hh.SetDirectory(0)
        hh.SetFillStyle(0)
        hh.SetFillColor(0)
        hh.SetLineColor(ROOT.kGray + 2)
        hh.SetLineWidth(1)

    max_dev = 0.0
    for ib in range(1, nb + 1):
        x1 = float(edges[ib - 1])
        x2 = float(edges[ib])
        nom = float(y_nom[ib - 1])

        if nom > 0.0:
            rel_dn = float(err_dn[ib - 1]) / nom
            rel_up = float(err_up[ib - 1]) / nom
            y_low  = 1.0 - rel_dn
            y_high = 1.0 + rel_up

            box = ROOT.TBox(x1, y_low, x2, y_high)
            box.SetFillColor(ROOT.kGray + 1)
            box.SetFillStyle(3345)
            box.SetLineWidth(0)
            ratio_boxes.append(box)

            h_ratio_line_up.SetBinContent(ib, y_high)
            h_ratio_line_dn.SetBinContent(ib, y_low)

            max_dev = max(max_dev, abs(y_high - 1.0), abs(1.0 - y_low))
        else:
            h_ratio_line_up.SetBinContent(ib, 1.0)
            h_ratio_line_dn.SetBinContent(ib, 1.0)

    if max_dev <= 0.0:
        r_min, r_max = 0.9, 1.1
    else:
        half_range = 1.3 * max_dev
        r_min = 1.0 - half_range
        r_max = 1.0 + half_range

    # hard clamp
    r_min = max(r_min, 0.75)
    r_max = min(r_max, 1.25)

    h_ratio_central.SetMinimum(r_min)
    h_ratio_central.SetMaximum(r_max)

    h_ratio_central.Draw("HIST")
    for b in ratio_boxes:
        b.Draw("SAME")
    h_ratio_line_up.Draw("HIST SAME")
    h_ratio_line_dn.Draw("HIST SAME")

    line = ROOT.TLine(float(edges[0]), 1.0, float(edges[-1]), 1.0)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("SAME")

    c_fit.cd()
    c_fit.Update()

    # save linear
    padTop.SetLogy(False)
    hs.SetMinimum(0.0)
    hs.SetMaximum(1.75 * max_y if max_y > 0 else 1.0)
    padTop.Modified()
    c_fit.Modified()
    c_fit.Update()

    out_png_lin = os.path.join(outdir_lin, f"prefit_{feat}.png")
    out_pdf_lin = os.path.join(outdir_lin, f"prefit_{feat}.pdf")
    c_fit.SaveAs(out_png_lin)
    c_fit.SaveAs(out_pdf_lin)

    # save log
    padTop.SetLogy(True)
    hs.SetMinimum(0.5)
    hs.SetMaximum(15.0 * max_y if max_y > 0 else 1.0)
    padTop.Modified()
    c_fit.Modified()
    c_fit.Update()

    out_png_log = os.path.join(outdir_log, f"prefit_{feat}.png")
    out_pdf_log = os.path.join(outdir_log, f"prefit_{feat}.pdf")
    c_fit.SaveAs(out_png_log)
    c_fit.SaveAs(out_pdf_log)

    root_objs[feat] = {
        "canvas": c_fit,
        "padTop": padTop,
        "padBottom": padBottom,
        "stack": hs,
        "h_procs": h_procs,
        "labels": lbls,
        "proc_keys": proc_keys,
        "proc_has_content": proc_has_content,
        "h_total": h_total,
        "h_data": h_data,
        "band": g_band,
        "h_up_line": h_up_line,
        "h_dn_line": h_dn_line,
        "h_ratio_central": h_ratio_central,
        "ratio_boxes": ratio_boxes,
        "h_ratio_line_up": h_ratio_line_up,
        "h_ratio_line_dn": h_ratio_line_dn,
        "unity_line": line,
        "legend": leg,
        "h_signals": h_signals,
        "signal_labels": sig_labels,
        "signal_has_content": sig_has_content,
        "out_png_lin": out_png_lin,
        "out_pdf_lin": out_pdf_lin,
        "out_png_log": out_png_log,
        "out_pdf_log": out_pdf_log,
    }

    progress(f"[plot-done] feat='{feat}' -> lin:{out_png_lin}  log:{out_png_log}", every_s=0.0)
    mem_report(f"after plot {feat}")

progress(f"DONE. loader_hist entries={len(loader_hist)}", force=True)


# -----------------------------------------------------------------------------
# Per-variation template plots (no extra data I/O; uses loader_hist already built)
# -----------------------------------------------------------------------------
if args.templates is not None:
    from variations import syst_groups

    template_processes = list(args.templates) if len(args.templates) else ["TTLep_pow"]

    import cmsstyle
    colors = [
        cmsstyle.p10.kBlue,
        cmsstyle.p10.kYellow,
        cmsstyle.p10.kRed,
        cmsstyle.p10.kAsh,
        cmsstyle.p10.kViolet,
        cmsstyle.p10.kBrown,
        cmsstyle.p10.kOrange,
        cmsstyle.p10.kGreen,
        cmsstyle.p10.kGray,
        cmsstyle.p10.kCyan,
    ]
    legend_columns = 3

    group_to_vars = {
        gname: [v for v in vlist if (v in variations and v != "nominal")]
        for gname, vlist in syst_groups.items()
    }

    templates_root = os.path.join(user.plot_directory, "sys_templates")
    os.makedirs(templates_root, exist_ok=True)
    helpers.copyIndexPHP(templates_root)

    progress(f"[templates] producing template plots for processes={template_processes}", force=True)

    for era in eras:
        for group_name, vnames_in_group in group_to_vars.items():
            if not vnames_in_group:
                continue

            for proc in template_processes:
                if proc not in variations["nominal"]["loaders"].get(era, {}):
                    continue

                key_nom = ("nominal", era, proc, "nominal")
                if key_nom not in loader_hist:
                    continue

                active_vars = []
                for vname in vnames_in_group:
                    if (vname, era, proc, "up") in loader_hist and (vname, era, proc, "down") in loader_hist:
                        active_vars.append(vname)
                if not active_vars:
                    continue

                base_dir = os.path.join(templates_root, era, group_name, proc)
                outdir_lin = os.path.join(base_dir, "lin")
                outdir_log = os.path.join(base_dir, "log")
                os.makedirs(outdir_lin, exist_ok=True)
                os.makedirs(outdir_log, exist_ok=True)

                helpers.copyIndexPHP(os.path.join(templates_root, era))
                helpers.copyIndexPHP(os.path.join(templates_root, era, group_name))
                helpers.copyIndexPHP(os.path.join(templates_root, era, group_name, proc))
                helpers.copyIndexPHP(outdir_lin)
                helpers.copyIndexPHP(outdir_log)

                progress(f"[templates] era={era} group={group_name} proc={proc} n_vars={len(active_vars)}", every_s=2.0)

                for feat in features:
                    if feat not in loader_hist[key_nom]:
                        continue

                    edges = edges_map[feat]
                    nb = len(edges) - 1
                    x_title = xlabel_map.get(feat, feat)

                    sw_nom, _sw2_nom = loader_hist[key_nom][feat]
                    if feat.startswith("parton_") and float(np.sum(sw_nom)) == 0.0:
                        continue

                    h_central = np_to_th1(
                        f"h_tpl_{era}_{proc}_{group_name}_{feat}_nom",
                        edges, sw_nom, np.zeros_like(sw_nom)
                    )
                    h_central.SetLineColor(ROOT.kBlack)
                    h_central.SetLineWidth(2)
                    h_central.SetTitle("")

                    h_vars, labels = [], []
                    for iv, vname in enumerate(active_vars):
                        col = colors[iv % len(colors)]
                        for which, ls, tag in (("up", ROOT.kSolid, "+1#sigma"), ("down", ROOT.kDashed, "-1#sigma")):
                            key = (vname, era, proc, which)
                            sw, _ = loader_hist[key][feat]
                            if feat.startswith("parton_") and float(np.sum(sw)) == 0.0:
                                continue
                            h = np_to_th1(
                                f"h_tpl_{era}_{proc}_{group_name}_{feat}_{vname}_{which}",
                                edges, sw, np.zeros_like(sw)
                            )
                            h.SetLineColor(col)
                            h.SetLineStyle(ls)
                            h.SetLineWidth(1)
                            h_vars.append(h)
                            labels.append(f"{vname} {tag}")

                    if not h_vars:
                        continue

                    canvas_name = f"c_tpl_{era}_{proc}_{group_name}_{feat}"
                    c = ROOT.TCanvas(canvas_name, canvas_name, 800, 900)

                    padLegend = ROOT.TPad(canvas_name + "_legend", canvas_name + "_legend", 0.0, 0.80, 1.0, 1.0)
                    padTop    = ROOT.TPad(canvas_name + "_top",    canvas_name + "_top",    0.0, 0.30, 1.0, 0.80)
                    padBottom = ROOT.TPad(canvas_name + "_bottom", canvas_name + "_bottom", 0.0, 0.00, 1.0, 0.30)

                    padLegend.SetBottomMargin(0.05)
                    padLegend.SetTopMargin(0.10)
                    padLegend.SetLeftMargin(0.10)
                    padLegend.SetRightMargin(0.10)
                    padLegend.SetFillStyle(0)

                    padTop.SetBottomMargin(0.0)
                    padTop.SetTopMargin(0.08)
                    padTop.SetLeftMargin(0.10)
                    padTop.SetRightMargin(0.05)
                    padTop.SetTicks(1, 1)

                    padBottom.SetTopMargin(0.0)
                    padBottom.SetBottomMargin(0.30)
                    padBottom.SetLeftMargin(0.10)
                    padBottom.SetRightMargin(0.05)
                    padBottom.SetTicks(1, 1)

                    padLegend.Draw()
                    padTop.Draw()
                    padBottom.Draw()

                    legend = ROOT.TLegend(0.02, 0.10, 0.98, 0.90)
                    legend.SetBorderSize(0)
                    legend.SetFillStyle(0)
                    legend.SetNColumns(legend_columns)
                    legend.AddEntry(h_central, "nominal", "l")
                    for h, lbl in zip(h_vars, labels):
                        legend.AddEntry(h, lbl, "l")

                    def _draw_top(set_log):
                        padTop.cd()
                        padTop.SetLogy(bool(set_log))

                        h_central.GetXaxis().SetTitle(x_title)
                        h_central.GetYaxis().SetTitle("Events")
                        h_central.GetYaxis().SetTitleSize(0.06)
                        h_central.GetYaxis().SetLabelSize(0.045)
                        h_central.GetXaxis().SetLabelSize(0.0)
                        h_central.GetXaxis().SetTitleSize(0.0)

                        max_y = max(h_central.GetMaximum(), max(h.GetMaximum() for h in h_vars))
                        if set_log:
                            h_central.SetMinimum(0.5)
                            h_central.SetMaximum(15.0 * max_y if max_y > 0 else 1.0)
                        else:
                            h_central.SetMinimum(0.0)
                            h_central.SetMaximum(1.20 * max_y if max_y > 0 else 1.0)

                        h_central.Draw("HIST")
                        for h in h_vars:
                            h.Draw("HIST SAME")

                    # bottom: ratios
                    padBottom.cd()

                    h_ratio_central = h_central.Clone(h_central.GetName() + "_ratio_central")
                    h_ratio_central.SetDirectory(0)
                    h_ratio_central.Reset("ICES")
                    for ib in range(1, nb + 1):
                        h_ratio_central.SetBinContent(ib, 1.0)
                        h_ratio_central.SetBinError(ib, 0.0)

                    h_ratio_central.SetLineColor(ROOT.kBlack)
                    h_ratio_central.SetLineWidth(2)
                    h_ratio_central.SetTitle("")

                    h_ratio_central.GetYaxis().SetTitle("var / nominal")
                    h_ratio_central.GetYaxis().SetNdivisions(505)
                    h_ratio_central.GetYaxis().SetTitleSize(0.09)
                    h_ratio_central.GetYaxis().SetTitleOffset(0.5)
                    h_ratio_central.GetYaxis().SetLabelSize(0.08)

                    h_ratio_central.GetXaxis().SetTitle(x_title)
                    h_ratio_central.GetXaxis().SetTitleSize(0.10)
                    h_ratio_central.GetXaxis().SetLabelSize(0.08)

                    h_ratio_vars = []
                    max_dev = 0.0
                    for h in h_vars:
                        hr = h.Clone(h.GetName() + "_ratio")
                        hr.SetDirectory(0)
                        hr.Reset("ICES")
                        for ib in range(1, nb + 1):
                            nomv = float(h_central.GetBinContent(ib))
                            vv   = float(h.GetBinContent(ib))
                            rv = (vv / nomv) if nomv > 0.0 else 1.0
                            hr.SetBinContent(ib, rv)
                            hr.SetBinError(ib, 0.0)
                            max_dev = max(max_dev, abs(rv - 1.0))
                        hr.SetLineColor(h.GetLineColor())
                        hr.SetLineStyle(h.GetLineStyle())
                        hr.SetLineWidth(1)
                        h_ratio_vars.append(hr)

                    if max_dev <= 0.0:
                        r_min, r_max = 0.9, 1.1
                    else:
                        half_range = 1.3 * max_dev
                        r_min = 1.0 - half_range
                        r_max = 1.0 + half_range

                    r_min = max(r_min, 0.75)
                    r_max = min(r_max, 1.25)

                    h_ratio_central.SetMinimum(r_min)
                    h_ratio_central.SetMaximum(r_max)

                    h_ratio_central.Draw("HIST")
                    for hr in h_ratio_vars:
                        hr.Draw("HIST SAME")

                    line = ROOT.TLine(float(edges[0]), 1.0, float(edges[-1]), 1.0)
                    line.SetLineStyle(ROOT.kDashed)
                    line.SetLineColor(ROOT.kBlack)
                    line.Draw("SAME")

                    padLegend.cd()
                    legend.Draw()

                    # save lin
                    _draw_top(False)
                    c.cd(); c.Update()
                    c.SaveAs(os.path.join(outdir_lin, f"{feat}.png"))
                    c.SaveAs(os.path.join(outdir_lin, f"{feat}.pdf"))

                    # save log
                    _draw_top(True)
                    c.cd(); c.Update()
                    c.SaveAs(os.path.join(outdir_log, f"{feat}.png"))
                    c.SaveAs(os.path.join(outdir_log, f"{feat}.pdf"))

syncer.sync()

