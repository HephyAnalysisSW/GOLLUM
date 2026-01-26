#!/usr/bin/env python3
import time
import numpy as np
import awkward as ak
import uproot

FILE = "/groups/hephy/cms/robert.schoefbeck/CMGRDF_ntuples/v2-3_nJ2p_nB2p_2l/2016/TTLep_pow_nominal.root"
TREE = "Events"

# keep this small on purpose (I/O + materialization debug)
#FEATURES = ["tr_ttbar_mass", "tr_ttbar_eta"]   # must exist in the file
FEATURES = [
    "MET_phi", "MET_pt", "ht", "nBJet", "nSelJet",
    "lep0_charge", "lep0_eta", "lep0_phi", "lep0_pt",
    "lep1_charge", "lep1_eta", "lep1_phi", "lep1_pt",
    "jet0_pt", "jet0_eta", "jet1_pt", "jet1_eta", #"jet2_pt", "jet2_eta", "jet3_pt", "jet3_eta",
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

N_SHARDS = 10
N_VIEWS  = 4

REOPEN_EACH_SHARD = True   # True mimics repeated uproot.open(...) overhead

def ak_mb(ar):
    # Awkward 2.6.x: sum bytes of underlying buffers
    return sum(buf.nbytes for buf in ak.to_buffers(ar)[2].values()) / 1024.0 / 1024.0


def rss_mb():
    # Linux RSS in MB
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")

def dt_ms(t0):  # perf_counter delta in ms
    return 1e3 * (time.perf_counter() - t0)

if __name__ == "__main__":
    f0 = uproot.open(FILE, object_cache=None, array_cache=None)
    t0 = f0[TREE]
    n_entries = t0.num_entries
    bounds = [(i * n_entries // N_SHARDS, (i + 1) * n_entries // N_SHARDS) for i in range(N_SHARDS)]

    class ProxyRDataLoader:
        def __init__(self, file_path, tree_name, feature_names):
            self.file_path = file_path
            self.tree_name = tree_name
            self.feature_names = feature_names

            self._file = None
            self._tree = None
            if not REOPEN_EACH_SHARD:
                self._file = f0
                self._tree = t0

            self._cache_i = None
            self._cache_ar = None
            self._cache_X  = None
            self._cache_w  = None

        def __len__(self):
            return N_SHARDS

        def __iter__(self):
            for i in range(len(self)):
                yield i

        def __getitem__(self, i):
            if self._cache_i == i and self._cache_ar is not None:
                return self._cache_ar

            lo, hi = bounds[i]

            # evict: single-shard cache (awkward + optional materialized arrays)
            self._cache_i = i
            self._cache_ar = None
            self._cache_X = None
            self._cache_w = None

            if REOPEN_EACH_SHARD:
                f = uproot.open(self.file_path, object_cache=None, array_cache=None)
                t = f[self.tree_name]
                self._cache_ar = t.arrays(self.feature_names, entry_start=lo, entry_stop=hi, library="ak")
                f.close()
            else:
                self._cache_ar = self._tree.arrays(self.feature_names, entry_start=lo, entry_stop=hi, library="ak")

            return self._cache_ar

        def materialize(self, i):
            if self._cache_i != i:
                _ = self[i]

            if self._cache_X is None:
                if self._cache_ar is None:
                    _ = self[i]
                ar = self._cache_ar

                cols = [ak.to_numpy(ar[n]).astype(np.float32, copy=False) for n in self.feature_names]
                X = np.stack(cols, axis=1)
                w = np.ones(X.shape[0], dtype=np.float32)

                self._cache_X = X
                self._cache_w = w
                self._cache_ar = None  # keep your memory drop

            return self._cache_X, self._cache_w

    class ProxySelectionView:
        # no I/O, no caching: 1:1 proxy (for now)
        def __init__(self, base, name):
            self.base = base
            self.name = name

        def materialize(self, shard):
            return self.base.materialize(shard)

    base = ProxyRDataLoader(FILE, TREE, FEATURES)
    views = [ProxySelectionView(base, f"view{i}") for i in range(N_VIEWS)]

    print(f"FILE: {FILE}")
    print(f"TREE: {TREE}  entries={n_entries}")
    print(f"FEATURES: {FEATURES}")
    print(f"SHARDS: {N_SHARDS}  VIEWS: {N_VIEWS}  REOPEN_EACH_SHARD={REOPEN_EACH_SHARD}")
    print(f"RSS start: {rss_mb():.1f} MB")
    print("-" * 120)
    print("shard  lo:hi           n   load_ms  mat_ms  views_ms   ak_MB    X_MB    w_MB    RSS_MB   X_id_same_across_views")
    print("-" * 120)

    t_total0 = time.perf_counter()

    for shard in range(N_SHARDS):
        lo, hi = bounds[shard]

        tL = time.perf_counter()
        ar = base[shard]
        load_ms = dt_ms(tL)

        ak_mb_val = ak_mb(ar)

        tM = time.perf_counter()
        X, w = base.materialize(shard)

        mat_ms = dt_ms(tM)

        X_mb = X.nbytes / 1024.0 / 1024.0
        w_mb = w.nbytes / 1024.0 / 1024.0

        tV = time.perf_counter()
        ids = []
        for v in views:
            Xv, wv = v.materialize(shard)
            ids.append(id(Xv))
        views_ms = dt_ms(tV)

        same = (len(set(ids)) == 1)

        # touch the arrays (prevents dead-code elimination when profiling)
        _ = float(np.sum(X[:, 0])) if X.size else 0.0

        print(f"{shard:5d}  {lo:7d}:{hi:<7d}  {len(ar):7d}  "
              f"{load_ms:7.1f}  {mat_ms:6.1f}  {views_ms:8.1f}  "
              f"{ak_mb_val:6.1f}  {X_mb:6.1f}  {w_mb:6.1f}  {rss_mb():7.1f}  {same}")

    print("-" * 120)
    print(f"TOTAL wall: {dt_ms(t_total0)/1e3:.3f} s  |  RSS end: {rss_mb():.1f} MB")

