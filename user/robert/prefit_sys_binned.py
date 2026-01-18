import os, math, copy
from array import array

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(1)

import numpy as np

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import time
import argparse
from collections import OrderedDict

import common.user as user
import common.helpers as helpers
import common.syncer as syncer

from data.plot_options import plot_options
from data.SelectionView import SelectionView
from data.samples_RunII import Factory
from variations import variations  # your OrderedDict


# -----------------------------------------------------------------------------
# CLI options (parse_known_args so IPython/Jupyter extra args don't break)
# -----------------------------------------------------------------------------
p = argparse.ArgumentParser(description="Prefit plots with syst band (subsettable)")
p.add_argument("--plot_directory", default="v2-3_tr_isvalid_isOS_offZ",
               help="String-based selection")
p.add_argument("--variation", nargs="+", default=None,
               help="Subset of systematics to run (space-separated). 'nominal' is always included. "
                    "Example: --variation alphaS ren jes_abs_16")
p.add_argument("--eras", nargs="+", default=None,
               help="Subset of eras to run (space-separated). Example: --eras 2017 2018")
p.add_argument("--selection", default="tr_isvalid & isOS & offZ",
               help="String-based selection")
p.add_argument("--branches_for_selection", nargs="+", default=["tr_isvalid", "isOS", "offZ"],
               help="branches we need to make the selection")
args, _unknown = p.parse_known_args()

# -----------------------------------------------------------------------------
# User-editable config (in IPython you can override before %run -i)
# -----------------------------------------------------------------------------
base = "/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-3_nJ2p_nB2p_2l/"

eras = ["2016", "2016APV", "2017", "2018"]
processes = ["SingleTop", "TTSemi_pow", "TTLep_pow", "DrellYan"]

signals   = [ "EtaS", "EtaP" ]

features = []

features += [
    "MET_phi", "MET_pt", "ht", "nBJet", "nSelJet",
    "lep0_charge", "lep0_eta", "lep0_phi", "lep0_pt",
    "lep1_charge", "lep1_eta", "lep1_phi", "lep1_pt",
    "jet0_pt", "jet0_eta", "jet1_pt", "jet1_eta", "jet2_pt", "jet2_eta", "jet3_pt", "jet3_eta",
    "dilep_eta", "dilep_mass", "dilep_phi", "dilep_pt", "dilep_dEta", "dilep_dAbsEta",
    "tr_Top_eta", "tr_Top_mass", "tr_Top_phi", "tr_Top_pt", "tr_Top_y",
    "tr_AntiTop_eta", "tr_AntiTop_mass", "tr_AntiTop_phi", "tr_AntiTop_pt", "tr_AntiTop_y",
    "tr_Wm_eta", "tr_Wm_mass", "tr_Wm_phi", "tr_Wm_pt",
    "tr_Wp_eta", "tr_Wp_mass", "tr_Wp_phi", "tr_Wp_pt",
    "tr_antib_eta", "tr_antib_phi", "tr_antib_pt",
    "tr_antilep_eta", "tr_antilep_phi", "tr_antilep_pt",
    "tr_antinu_eta", "tr_antinu_phi", "tr_antinu_pt",
    "tr_b_eta", "tr_b_phi", "tr_b_pt",
    "tr_lep_eta", "tr_lep_phi", "tr_lep_pt",
    "tr_nu_eta", "tr_nu_phi", "tr_nu_pt",
    "tr_ttbar_pt", "tr_ttbar_eta", "tr_ttbar_mass", "tr_ttbar_phi", "tr_ttbar_y", "tr_ttbar_dEta", "tr_ttbar_dAbsEta",
    "tr_cos_phi_lab", "tr_abs_delta_phi_ll_lab",
    "tr_cosThetaPlus_n", "tr_cosThetaMinus_n", "tr_cosThetaPlus_r", "tr_cosThetaMinus_r",
    "tr_cosThetaPlus_k", "tr_cosThetaMinus_k", "tr_cosThetaPlus_r_star", "tr_cosThetaMinus_r_star",
    "tr_cosThetaPlus_k_star", "tr_cosThetaMinus_k_star",
    "tr_xi_nn", "tr_xi_rr", "tr_xi_kk", "tr_xi_nr_plus", "tr_xi_nr_minus", "tr_xi_rk_plus", "tr_xi_rk_minus",
    "tr_xi_nk_plus", "tr_xi_nk_minus", "tr_xi_r_star_k", "tr_xi_k_r_star", "tr_xi_kk_star",
    "tr_cos_phi", "tr_c_hel", "tr_c_han",
]
features += [
    "parton_Top_pt", "parton_Top_eta", "parton_Top_y", "parton_Top_phi", "parton_Top_mass",
    #"parton_Top_f1_pt", "parton_Top_f1_eta", "parton_Top_f2_pt", "parton_Top_f2_eta", "parton_Top_b_pt", "parton_Top_b_eta", "parton_Top_W_pt", "parton_Top_W_eta",
    "parton_AntiTop_pt", "parton_AntiTop_eta", "parton_AntiTop_y", "parton_AntiTop_phi", "parton_AntiTop_mass",
    #"parton_AntiTop_f1_pt", "parton_AntiTop_f1_eta", "parton_AntiTop_f2_pt", "parton_AntiTop_f2_eta", "parton_AntiTop_b_pt", "parton_AntiTop_b_eta", "parton_AntiTop_W_pt", "parton_AntiTop_W_eta",
    "parton_ttbar_pt", "parton_ttbar_mass", "parton_ttbar_eta", "parton_ttbar_y", "parton_ttbar_dEta", "parton_ttbar_dAbsEta",
    "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star",
    "parton_xi_nn", "parton_xi_rr", "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus",
    "parton_xi_r_star_k", "parton_xi_k_r_star", "parton_xi_kk_star",
    "parton_c_hel", "parton_c_han", "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",
]

# Process style (tune as you like)
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
    # Always keep nominal
    variations = OrderedDict((k, v) for (k, v) in variations.items() if (k == "nominal" or k in keep))


# -----------------------------------------------------------------------------
# Helpers (kept small by design)
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
progress(f"Config: eras={eras} (n={len(eras)}), processes={processes} (n={len(processes)}), signals={signals} (n={len(signals)}), "
         f"variations={list(variations.keys())} (n={len(variations)}), features={len(features)}", force=True)


# -----------------------------------------------------------------------------
# Binning + labels (fail loudly if missing)
# -----------------------------------------------------------------------------
progress("Building binning/label maps from plot_options...", force=True)
edges_map = {}
xlabel_map = {}
for f in features:
    if f not in plot_options:
        raise RuntimeError(f"Feature '{f}' not in plot_options.")
    if "binning" not in plot_options[f]:
        raise RuntimeError(f"plot_options['{f}'] has no 'binning'.")
    edges_map[f] = edges_from_binning(plot_options[f]["binning"])
    xlabel_map[f] = plot_options[f].get("tex", f)

# -----------------------------------------------------------------------------
# Build all loaders into variations dict (strict up/down requirements)
# -----------------------------------------------------------------------------
progress("Instantiating Factory...", force=True)
factory = Factory(base)

progress(f"Building loaders for {len(variations)} variations x {len(eras)} eras x {len(processes)} bkg processes (+ nominal signals)...", force=True)
for iv, (var_name, variation) in enumerate(variations.items(), start=1):
    progress(f"[loaders] ({iv}/{len(variations)}) variation='{var_name}'", force=True)
    variation["loaders"] = {}

    if var_name != "nominal":
        has_weight = ("weight_up" in variation) or ("weight_down" in variation)
        has_sys    = ("sys_up" in variation) or ("sys_down" in variation)
        if has_weight and has_sys:
            raise NotImplementedError(f"Variation '{var_name}' mixes weights and sys branches.")
        if has_weight:
            if ("weight_up" not in variation) or ("weight_down" not in variation):
                raise RuntimeError(f"Weight variation '{var_name}' must define BOTH weight_up and weight_down.")
        if has_sys:
            if ("sys_up" not in variation) or ("sys_down" not in variation):
                raise RuntimeError(f"Sys variation '{var_name}' must define BOTH sys_up and sys_down.")
        if (not has_weight) and (not has_sys):
            raise RuntimeError(f"Variation '{var_name}' is neither sys nor weight; define sys_* or weight_*.")

    for era in eras:
        if "eras" in variation and era not in variation["eras"]:
            continue

        progress(f"[loaders] variation='{var_name}' era='{era}'", every_s=2.0)
        variation["loaders"][era] = {}

        # backgrounds: all variations (nominal/up/down depending on var)
        for process in processes:
            variation["loaders"][era][process] = {}

            if var_name == "nominal":
                l = factory.get(process, era)
                l.setFeatures(features)
                l.addSelection(args.selection, args.branches_for_selection)
                variation["loaders"][era][process]["nominal"] = l
                continue

            # nominal pointer always
            variation["loaders"][era][process]["nominal"] = variations["nominal"]["loaders"][era][process]["nominal"]

            if "sys_up" in variation:
                l_up = factory.get(process, era, variation["sys_up"])
                l_up.setFeatures(features)
                l_up.addSelection(args.selection, args.branches_for_selection)
                l_dn = factory.get(process, era, variation["sys_down"])
                l_dn.setFeatures(features)
                l_dn.addSelection(args.selection, args.branches_for_selection)

                variation["loaders"][era][process]["up"] = l_up
                variation["loaders"][era][process]["down"] = l_dn

            elif "weight_up" in variation:
                base_loader = variations["nominal"]["loaders"][era][process]["nominal"]

                up_w = copy.deepcopy(base_loader.weight_branches)
                dn_w = copy.deepcopy(base_loader.weight_branches)

                # strict handling of remove_weight/removeweight key
                rw = variation.get("removeweight", variation.get("remove_weight", None))
                if rw is not None:
                    if rw not in up_w or rw not in dn_w:
                        raise RuntimeError(f"Variation '{var_name}': requested to remove weight '{rw}' but not in branches.")
                    up_w.remove(rw)
                    dn_w.remove(rw)

                up_w.append(variation["weight_up"])
                dn_w.append(variation["weight_down"])

                variation["loaders"][era][process]["up"] = SelectionView(
                    base=base_loader,
                    name=f"{process}_{era}_{var_name}_up",
                    selection_fn=None,
                    feature_names=base_loader.feature_names,
                    observer_names=base_loader.observer_names,
                    weight=up_w,
                )
                variation["loaders"][era][process]["down"] = SelectionView(
                    base=base_loader,
                    name=f"{process}_{era}_{var_name}_down",
                    selection_fn=None,
                    feature_names=base_loader.feature_names,
                    observer_names=base_loader.observer_names,
                    weight=dn_w,
                )
            else:
                raise RuntimeError(f"Internal: variation '{var_name}' reached inconsistent state.")

        # signals: ONLY nominal (no systematics, no inclusion in totals)
        if var_name == "nominal":
            for process in signals:
                if process in variation["loaders"][era]:
                    continue
                variation["loaders"][era][process] = {}
                l = factory.get(process, era)
                l.setFeatures(features)
                l.addSelection(args.selection, args.branches_for_selection)
                variation["loaders"][era][process]["nominal"] = l


# -----------------------------------------------------------------------------
# Histogram cache (dict-based) + explicit per-(var,era,proc,which) histogram dict
# -----------------------------------------------------------------------------
loader_cache = {}   # id(loader) -> {feat: (sumw, sumw2)}
loader_hist = {}    # (var, era, proc, which) -> {feat: (sumw, sumw2)}

_materialize_calls = 0  # purely for progress

def _fill_loader_hist_if_needed(var_name, era, proc, which):
    global _materialize_calls
    key = (var_name, era, proc, which)
    if key in loader_hist:
        return

    ldict = variations[var_name]["loaders"][era][proc]
    if which not in ldict:
        raise RuntimeError(f"Missing loader: variation='{var_name}' era='{era}' proc='{proc}' which='{which}'")
    loader = ldict[which]

    lid = id(loader)
    if lid not in loader_cache:
        _materialize_calls += 1
        progress(f"[materialize #{_materialize_calls}] var={var_name} era={era} proc={proc} which={which} "
                 f"(cached loaders={len(loader_cache)})")

        feats, _, w = loader.materialize(0, "fow")

        try:
            n_ev = len(w)
        except Exception:
            n_ev = None
        if n_ev is not None:
            progress(f"[materialize-done #{_materialize_calls}] var={var_name} era={era} proc={proc} which={which} N={n_ev}",
                     every_s=0.0)

        feats = np.asarray(feats)
        w = np.asarray(w, dtype=np.float64)

        per_feat = {}
        n_ev = feats.shape[0]

        for i, feat_name in enumerate(features):
            edges = edges_map[feat_name]

            x = feats[:, i]
            m = np.isfinite(x) & np.isfinite(w)   # per-feature drop (also guard against bad weights)

            if n_ev > 0:
                frac_drop = 1.0 - (float(m.sum()) / float(n_ev))
            else:
                frac_drop = 0.0

            if frac_drop > 0.0:
                progress(f"[nan-drop] var={var_name} era={era} proc={proc} which={which} feat={feat_name} "
                         f"dropped {n_ev - int(m.sum())}/{n_ev} = {frac_drop:.3%}", every_s=0.0)

            sw, sw2 = hist_with_flow(x[m], w[m], edges)
            per_feat[feat_name] = (sw, sw2)

        loader_cache[lid] = per_feat

    loader_hist[key] = loader_cache[lid]


# -----------------------------------------------------------------------------
# Totals and uncertainties (global dicts you can inspect)
# -----------------------------------------------------------------------------
tot_hist = {f: {"nominal": None} for f in features}
for f in features:
    for v in variations.keys():
        if v != "nominal":
            tot_hist[f][v] = {"up": None, "down": None}

unc_hist = {f: {"err_dn": None, "err_up": None} for f in features}


# -----------------------------------------------------------------------------
# Sum nominal and all variations (strict for applicable eras)
# (signals are intentionally NOT included)
# -----------------------------------------------------------------------------
progress(f"Computing totals/uncertainties for {len(features)} features...", force=True)

for ifeat, feat in enumerate(features, start=1):
    progress(f"[totals] ({ifeat}/{len(features)}) feat='{feat}'", force=True)

    edges = edges_map[feat]
    nb = len(edges) - 1

    y_nom = np.zeros(nb, dtype=np.float64)
    sumw2_nom = np.zeros(nb, dtype=np.float64)

    for era in eras:
        for proc in processes:
            _fill_loader_hist_if_needed("nominal", era, proc, "nominal")
            sw, sw2 = loader_hist[("nominal", era, proc, "nominal")][feat]
            y_nom += sw
            sumw2_nom += sw2

    tot_hist[feat]["nominal"] = (y_nom, sumw2_nom)

    # each variation totals (backgrounds only)
    n_var_eff = len(variations) - 1
    ivar_eff = 0
    for var_name, variation in variations.items():
        if var_name == "nominal":
            continue
        ivar_eff += 1
        progress(f"[totals] feat='{feat}' variation ({ivar_eff}/{n_var_eff}) '{var_name}'", every_s=3.0)

        y_up = np.zeros(nb, dtype=np.float64)
        y_dn = np.zeros(nb, dtype=np.float64)

        for era in eras:
            var_applies = (("eras" not in variation) or (era in variation["eras"]))

            for proc in processes:
                if var_applies:
                    _fill_loader_hist_if_needed(var_name, era, proc, "up")
                    _fill_loader_hist_if_needed(var_name, era, proc, "down")
                    swu, _ = loader_hist[(var_name, era, proc, "up")][feat]
                    swd, _ = loader_hist[(var_name, era, proc, "down")][feat]
                    y_up += swu
                    y_dn += swd
                else:
                    _fill_loader_hist_if_needed("nominal", era, proc, "nominal")
                    sw, _ = loader_hist[("nominal", era, proc, "nominal")][feat]
                    y_up += sw
                    y_dn += sw

        tot_hist[feat][var_name]["up"] = y_up
        tot_hist[feat][var_name]["down"] = y_dn

    sig2_up = np.zeros(nb, dtype=np.float64)
    sig2_dn = np.zeros(nb, dtype=np.float64)

    for var_name in variations.keys():
        if var_name == "nominal":
            continue
        y_up = tot_hist[feat][var_name]["up"]
        y_dn = tot_hist[feat][var_name]["down"]

        d_up = y_up - y_nom
        d_dn = y_dn - y_nom

        pos = np.maximum.reduce([d_up, d_dn, np.zeros_like(d_up)])
        neg = np.maximum.reduce([-d_up, -d_dn, np.zeros_like(d_up)])

        sig2_up += pos * pos
        sig2_dn += neg * neg

    sig2_up += sumw2_nom
    sig2_dn += sumw2_nom

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

    # ---- nominal per-process, summed over eras (for stack) ----
    proc_sumw = {}
    proc_sumw2 = {}
    for proc in processes:
        proc_sumw[proc] = np.zeros(nb, dtype=np.float64)
        proc_sumw2[proc] = np.zeros(nb, dtype=np.float64)
        for era in eras:
            _fill_loader_hist_if_needed("nominal", era, proc, "nominal")
            sw, sw2 = loader_hist[("nominal", era, proc, "nominal")][feat]
            proc_sumw[proc] += sw
            proc_sumw2[proc] += sw2

    # for parton_* features: mark processes that are completely empty (all-zero) after NaN removal
    proc_has_content = {p: bool(np.any(proc_sumw[p] != 0.0)) for p in processes}

    h_procs = []
    lbls = []
    proc_keys = []
    for proc in processes:
        h = np_to_th1(f"h_{feat}_{proc}", edges, proc_sumw[proc], proc_sumw2[proc])
        h.SetFillColor(proc_colors.get(proc, ROOT.kGray))
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)
        h_procs.append(h)
        lbls.append(process_labels.get(proc, proc))
        proc_keys.append(proc)

    # sort for stable stack (small -> large)
    order = np.argsort([h.Integral() for h in h_procs])
    h_procs = [h_procs[i] for i in order]
    lbls = [lbls[i] for i in order]
    proc_keys = [proc_keys[i] for i in order]

    h_total = np_to_th1(f"h_{feat}_total", edges, y_nom, np.zeros_like(y_nom))

    # ---- SIGNALS: nominal only, summed over eras, overlaid (NOT in stack, NOT in totals) ----
    h_signals = []
    sig_labels = []
    sig_has_content = []
    for sproc in signals:
        s_sumw = np.zeros(nb, dtype=np.float64)
        s_sumw2 = np.zeros(nb, dtype=np.float64)
        for era in eras:
            _fill_loader_hist_if_needed("nominal", era, sproc, "nominal")
            sw, sw2 = loader_hist[("nominal", era, sproc, "nominal")][feat]
            s_sumw += sw
            s_sumw2 += sw2

        sig_has_content.append(bool(np.any(s_sumw != 0.0)))

        hs_sig = np_to_th1(f"h_{feat}_{sproc}", edges, s_sumw, s_sumw2)
        hs_sig.SetFillStyle(0)
        hs_sig.SetFillColor(0)
        hs_sig.SetLineColor(proc_colors.get(sproc, ROOT.kMagenta + 2))
        hs_sig.SetLineWidth(3)
        hs_sig.SetLineStyle(1)
        h_signals.append(hs_sig)
        sig_labels.append(process_labels.get(sproc, sproc))

    # optional "Asimov data" for look-and-feel (inspect/change as you like)
    h_data = h_total.Clone(f"h_{feat}_data")
    h_data.SetDirectory(0)
    h_data.SetMarkerStyle(20)
    h_data.SetMarkerSize(1.0)
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(2)
    for ib in range(1, nb + 1):
        c = h_data.GetBinContent(ib)
        h_data.SetBinError(ib, math.sqrt(c) if c > 0 else 0.0)

    # band + boundary lines
    g_band = make_asymm_band_graph(edges, y_nom, err_dn, err_up, fill_color=ROOT.kGray + 1, fill_style=3345)

    h_up_line = np_to_th1(f"h_{feat}_up_line", edges, y_nom + err_up, np.zeros_like(y_nom))
    h_dn_line = np_to_th1(f"h_{feat}_dn_line", edges, y_nom - err_dn, np.zeros_like(y_nom))
    for hh in (h_up_line, h_dn_line):
        hh.SetFillStyle(0)
        hh.SetFillColor(0)
        hh.SetLineColor(ROOT.kGray + 2)
        hh.SetLineWidth(1)

    # ---- canvas and pads ----
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

    # ---- TOP PAD ----
    padTop.cd()
    padTop.SetLogy(False)  # draw initially in linear; we save both lin/log later

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

    # y-range headroom (consider also signals)
    max_y = max(hs.GetMaximum(), h_data.GetMaximum())
    for hh in h_signals:
        max_y = max(max_y, hh.GetMaximum())

    hs.SetMinimum(0.0)
    hs.SetMaximum(1.75 * max_y if max_y > 0 else 1.0)

    g_band.Draw("2 SAME")
    h_up_line.Draw("HIST SAME")
    h_dn_line.Draw("HIST SAME")
    h_data.Draw("E SAME")

    # overlay signals on top
    for hh in h_signals:
        hh.Draw("HIST SAME")

    leg = ROOT.TLegend(0.15, 0.75, 0.95, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(3)
    leg.SetTextSize(0.035)

    leg.AddEntry(h_data, "Data (Asimov)", "lep")

    # backgrounds: skip empty only for parton_* features
    for h, lbl, pk in zip(h_procs[::-1], lbls[::-1], proc_keys[::-1]):
        if skip_empty_parton and (not proc_has_content.get(pk, True)):
            continue
        leg.AddEntry(h, lbl, "f")

    leg.AddEntry(g_band, "Uncertainty", "f")

    # signals: skip empty only for parton_* features
    for hh, lbl, ok in zip(h_signals, sig_labels, sig_has_content):
        if skip_empty_parton and (not ok):
            continue
        leg.AddEntry(hh, lbl, "l")

    leg.Draw()

    # ---- BOTTOM PAD: ratio band around 1 ----
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

    # ---- save LINEAR ----
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

    # ---- save LOG ----
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

progress(f"DONE. unique materializations={_materialize_calls}, cached loaders={len(loader_cache)}, loader_hist entries={len(loader_hist)}",
         force=True)

syncer.sync()

