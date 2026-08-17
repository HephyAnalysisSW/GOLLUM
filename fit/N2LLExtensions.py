# fit/Extras.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import sys
sys.path.insert(0, '..')

# Import base (Asimov stays here)
from fit.Likelihood import * 

class N2LLExtensions(N2LL):
    """
    Extension layer that adds:
      - fisher_information(...)
      - evaluate_ratio(...)
    """

    # ---------- in-memory cache for evaluate_ratio ----------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # region-scoped, ephemeral cache; keys = region id
        self._mem_ratio_cache: Dict[str, Dict[str, Any]] = {}

    def _assemble_cA_C_per_class(self, rid: str, hypothesis) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build c_A and C_{Aa} for each class in a region for the given hypothesis.

        Returns
        -------
        out : dict
          rid -> {
             cid -> {
               "cA": np.ndarray (nA,),
               "C":  np.ndarray (nA, n_par_class),
               "poi_names": [names in class order]
             }
          }
        """
        # POI values (global dict)
        poi_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'POIs', [])}

        out: Dict[str, Dict[str, np.ndarray]] = {}
        class_dict: Dict[str, Dict[str, np.ndarray]] = {}

        for cid in self._class_ids_by_region.get(rid, []):
            poi_names = self._poi_order[(rid, cid)]  # class-specific order
            reference_point = self._poi_reference[(rid, cid)]

            # A-basis vector and Jacobian
            cA = expand_pois_linear_quadratic(poi_names, poi_vals, reference_point)
            C  = pois_jacobian_linear_quadratic(poi_names, poi_vals, reference_point)

            class_dict[cid] = {
                "cA": cA,
                "C":  C,
                "poi_names": poi_names,
            }

        out = class_dict
        return out

    def _assemble_cA_C_per_class(self, rid: str, hypothesis) -> Dict[str, dict]:
        """
        Build, for a given region rid and hypothesis, the structures
        per class:
          - cA:   A-basis POI vector c_A
          - C:    Jacobian C_{Aa} = ∂c_A/∂c_a
          - names: local POI names for this class

        Returns:
          { cid -> { 'cA': cA, 'C': C, 'poi_names': poi_names } }
        """
        # global POI values by name
        c_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'POIs', [])}
        out: Dict[str, dict] = {}
        for cid in self._class_ids_by_region.get(rid, []):
            poi_names = self._poi_order[(rid, cid)]
            reference_point = self._poi_reference[(rid, cid)]
            # A-basis vector and Jacobian in the SAME ordering
            cA = expand_pois_linear_quadratic(poi_names, c_vals, reference_point)
            C  = pois_jacobian_linear_quadratic(poi_names, c_vals, reference_point)
            out[cid] = {
                'cA': cA,
                'C': C,
                'poi_names': poi_names,
            }
        return out

    def _assemble_nuA_N_per_group(self, rid: str, hypothesis) -> Dict[str, list]:
        """
        Build, for a given region rid and hypothesis, the per-class list of
        nuisance A-basis structures per Δ-group:
          - gm:     meta dict for the group (contains 'id', 'dset', 'params', 'combs')
          - params: list of fundamental ν-parameter names for this group
          - nuA:    A-basis vector ν_B
          - N:      Jacobian N_{Ba} = ∂ν_B/∂ν_a, a over this group's 'params'

        Returns:
          { cid -> [ { 'gm': gm, 'params': params, 'nuA': nuA, 'N': N_mat }, ... ] }
        """
        # global nuisance values by name
        nu_vals = {p.name: float(p.val)
                   for p in getattr(hypothesis, 'parameters', [])
                   if not p.isPOI}
        out: Dict[str, list] = {}
        for cid in self._class_ids_by_region.get(rid, []):
            meta = self._meta[(rid, cid)]
            groups = []
            for gm in meta.get("delta_groups", []):
                params = list(gm.get("params", gm.get("parameters", [])) or [])
                combs  = [tuple(c) for c in (gm.get("combs", gm.get("combinations", [])) or [])]
                nuA    = nuis_to_A_vector(params, combs, nu_vals)
                N_mat  = nuis_jacobian_A(params, combs, nu_vals)
                groups.append({
                    'gm': gm,
                    'params': params,
                    'nuA': nuA,
                    'N': N_mat,
                })
            out[cid] = groups
        return out

    def fisher_information(self, hypothesis, step_scale: float = 1e-4, verbose: bool = False) -> np.ndarray:
        """
        Fisher information I_{ab} = E[(∂_a log L)(∂_b log L)] in Asimov mode.

        Unbinned part (Poisson point process with R = 1+T):
            I_{ab}^{(unb)} = L_0 ∫ dσ_0 R t_a t_b
                           ≈ Σ_i w0_i R_i t_{a,i} t_{b,i}
                           = Σ_i w0_i (∂_a T_i)(∂_b T_i)/(1+T_i),

        with T_i ≡ T(x_i; θ), R_i = 1+T_i, and
            t_{a,i} = ∂_a log R_i = (∂_a T_i)/(1+T_i).

        We compute ∂_a T_i analytically using:
          - C_{Aa} = ∂c_A/∂c_a for POIs,
          - N_{Ba} = ∂ν_B/∂ν_a for nuisances,
          - the stored R_A(x), Δ_B(x) and lnN biases.

        Binned part (independent Poissons):
            I_{ab}^{(bin)} = Σ_i (1/λ_i) (∂_a λ_i)(∂_b λ_i),
        using finite-difference derivatives of λ_i.

        Prior adds the Hessian of (penalty / 2) over free parameters.
        """
        import numpy as np

        if not self._runtime_prepared:
            raise RuntimeError("[N2LL.fisher_information] Call prepare_runtime() first.")
        if not self._asimov_hypothesis_set or self._observation_set:
            raise RuntimeError("[N2LL.fisher_information] Asimov mode required: call setAsimov(...), "
                               "and do not set an observation.")

        # Free parameters
        free_params = [p for p in hypothesis.parameters
                       if not p.isFrozen and not getattr(p, "isIgnored", False)]
        names = [p.name for p in free_params]
        npar = len(names)
        if npar == 0:
            raise RuntimeError("[N2LL.fisher_information] No free parameters.")

        if verbose:
            print("[FI] Free parameters:", names)

        # Maps for quick lookup
        param_by_name = {p.name: p for p in hypothesis.parameters}
        poi_name_set  = {p.name for p in hypothesis.parameters if p.isPOI}

        # Helpers
        def _clone_shift(hyp, pname, delta):
            h = hyp.clone()
            for p in h.parameters:
                if p.name == pname:
                    p.val = float(p.val) + float(delta)
                    break
            return h

        def _eps_for(val: float) -> float:
            base = abs(val) if abs(val) > 0 else 1.0
            return max(1e-8, step_scale * base)

        def _ln_bias_map_for(rid: str, hyp):
            nu_vals = {p.name: float(p.val) for p in hyp.parameters if not p.isPOI}
            class_ids = self._class_ids_by_region.get(rid, [])
            return {
                cid: sum(log1p_alpha * nu_vals.get(pname, 0.0)
                         for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))
                for cid in class_ids
            }

        # Fisher matrix
        FI = np.zeros((npar, npar), dtype=np.float64)

        # =========================
        # UNBINNED contribution (analytic dT/dθ)
        # =========================
        for R in self.regions:
            rid = R['id']
            class_ids = self._class_ids_by_region.get(rid, [])
            if not class_ids:
                continue
            N = self._N_region.get(rid, 0)
            if N == 0:
                continue

            if verbose:
                print(f"[FI:unbinned] Region '{rid}', N={N}, classes={class_ids}")

            # Structures at current hypothesis
            c_struct   = self._assemble_cA_C_per_class(rid, hypothesis)
            nu_struct  = self._assemble_nuA_N_per_group(rid, hypothesis)
            ln_bias    = _ln_bias_map_for(rid, hypothesis)

            # lnN gradient d(ln_bias_cid)/dν_pname
            ln_bias_grad = {cid: {} for cid in class_ids}
            for cid in class_ids:
                for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []):
                    ln_bias_grad[cid][pname] = ln_bias_grad[cid].get(pname, 0.0) + log1p_alpha

            chunk = self.eval_chunk_size
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                M = stop - start
                if M <= 0:
                    continue

                # w0 from any class (same for all)
                W = self._h5[(rid, class_ids[0])]['w0'][start:stop]  # (M,)

                # Build T(x) and cache per-class intermediates
                T0 = np.zeros(M, dtype=np.float64)
                per_cid = {}

                for cid in class_ids:
                    f = self._h5[(rid, cid)]
                    g_slice = f['g'][start:stop]          # (M,)
                    R_slice = f['R'][start:stop, :]       # (M, nA)

                    cA = c_struct[cid]['cA']
                    C  = c_struct[cid]['C']
                    poi_names_local = c_struct[cid]['poi_names']

                    if R_slice.shape[1] != cA.shape[0]:
                        raise RuntimeError(f"[FI:unbinned] BIT dim {R_slice.shape[1]} != |A| {cA.shape[0]} for {rid}/{cid}")

                    c_dot_R = R_slice @ cA                # (M,)

                    expo = np.zeros(M, dtype=np.float64)
                    dA_cache = {}                         # gm_id -> dA (M, nB)

                    for grp in nu_struct[cid]:
                        gm     = grp['gm']
                        nuA    = grp['nuA']
                        dset   = gm.get("dset", f"Delta::{gm['id']}")
                        dA     = f[dset][start:stop, :]    # (M, nB)
                        if dA.shape[1] != nuA.shape[0]:
                            raise RuntimeError(f"[FI:unbinned] Δ dim {dA.shape[1]} != ν_A dim {nuA.shape[0]} for {rid}/{cid}/{gm['id']}")
                        dA_cache[gm['id']] = dA
                        expo += dA @ nuA

                    exp_expo = np.exp(expo + ln_bias[cid])  # (M,)

                    T_cid = g_slice * (c_dot_R * exp_expo + (exp_expo - 1.0))
                    T0 += T_cid

                    per_cid[cid] = {
                        "g": g_slice,
                        "R": R_slice,
                        "c_dot_R": c_dot_R,
                        "exp_expo": exp_expo,
                        "dA_cache": dA_cache,
                        "C": C,
                        "poi_names": poi_names_local,
                        "nu_groups": nu_struct[cid],
                    }

                # Build weights for Fisher integrand: w0 / (1 + T)
                denom = 1.0 + T0
                if np.any(denom <= 0.0):
                    raise RuntimeError("[FI:unbinned] Encountered 1+T <= 0; R=1+T must be >0.")
                #print( "W",W)
                #print( "minW",min(W))
                #print( "maxW",max(W))
                #print( "denom",denom)

                weight = W / denom
                # Compute dT/dθ_k analytically for all free parameters
                dT_list = []
                for pname in names:
                    p_obj = param_by_name[pname]
                    is_poi = p_obj.isPOI
                    dT_k = np.zeros(M, dtype=np.float64)

                    if is_poi:
                        # POI derivative: ∂T = Σ_cid g * (R @ C_col) * exp_expo
                        for cid in class_ids:
                            data      = per_cid[cid]
                            poi_names = data["poi_names"]
                            if pname not in poi_names:
                                continue
                            local_idx = poi_names.index(pname)
                            R_slice   = data["R"]
                            C         = data["C"]
                            g_slice   = data["g"]
                            exp_expo  = data["exp_expo"]

                            # ∂(c_A R_A)/∂c_a = Σ_A C_{Aa} R_A(x)
                            d_cR = R_slice @ C[:, local_idx]  # (M,)
                            dT_k += g_slice * d_cR * exp_expo

                    else:
                        # Nuisance (shape or lnN) derivative:
                        #   ∂T = Σ_cid g * e^Φ (c⋅R + 1) * ∂Φ/∂ν_a
                        for cid in class_ids:
                            data     = per_cid[cid]
                            g_slice  = data["g"]
                            exp_expo = data["exp_expo"]
                            c_dot_R  = data["c_dot_R"]

                            dPhi = np.zeros(M, dtype=np.float64)

                            # Contributions from PNN Δ-groups via ν_A(ν)
                            for grp in data["nu_groups"]:
                                gm     = grp["gm"]
                                params = grp["params"]
                                N_mat  = grp["N"]          # (nB, n_params_local)

                                if pname not in params:
                                    continue
                                loc = params.index(pname)

                                dnuA = N_mat[:, loc]      # (nB,)
                                gm_id = gm["id"]
                                dA    = data["dA_cache"][gm_id]  # (M, nB)

                                dPhi += dA @ dnuA

                            # lnN normalization contribution
                            dPhi += ln_bias_grad[cid].get(pname, 0.0)

                            dT_k += g_slice * exp_expo * (c_dot_R + 1.0) * dPhi

                    dT_list.append(dT_k)

                # Stack derivatives into (npar, M) and accumulate FI for this chunk:
                # I_ab += Σ_i w_i * (∂_a T_i ∂_b T_i)/(1+T_i)
                D = np.stack(dT_list, axis=0)  # (npar, M)
                FI += (D * weight) @ D.T

        # =========================
        # BINNED contribution (finite-diff dλ; analytic Fisher weight 1/λ)
        # =========================
        if getattr(self, "_binned_regions_ids", None):
            for rid in self._binned_regions_ids:
                lam = self._compute_lambda_binned(rid, hypothesis)
                lam = np.asarray(lam, dtype=np.float64).reshape(-1)
                if lam.size == 0:
                    continue

                if verbose:
                    print(f"[FI:binned] Region '{rid}', Nbins={lam.size}, mean(λ)={float(np.mean(lam)):.3e}")

                inv_lam = 1.0 / np.maximum(lam, 1e-15)
                sqrt_inv_lam = np.sqrt(inv_lam, dtype=np.float64)

                dlam_list = []
                for pname in names:
                    v   = float(param_by_name[pname].val)
                    eps = _eps_for(v)

                    hyp_p = _clone_shift(hypothesis, pname, +eps)
                    hyp_m = _clone_shift(hypothesis, pname, -eps)

                    lam_p = np.asarray(self._compute_lambda_binned(rid, hyp_p), dtype=np.float64).reshape(-1)
                    lam_m = np.asarray(self._compute_lambda_binned(rid, hyp_m), dtype=np.float64).reshape(-1)

                    dlam = (lam_p - lam_m) / (2.0 * eps)
                    dlam_list.append(dlam)

                S_bin = np.stack([sqrt_inv_lam * dlam_k for dlam_k in dlam_list], axis=0)  # (npar, Nflat)
                FI += S_bin @ S_bin.T

        # =========================
        # PRIOR / PENALTY contribution
        # =========================
        def _penalty(hyp):
            return float(hyp.penalty())

        if any(getattr(p, "isPenalized", False) and (not p.isFrozen) and (not getattr(p, "isIgnored", False))
               for p in hypothesis.parameters):
            Hpen = np.zeros((npar, npar), dtype=np.float64)
            p0 = _penalty(hypothesis)

            # Diagonal terms
            for a, na in enumerate(names):
                va  = float(param_by_name[na].val)
                epa = _eps_for(va)

                ha_p = _clone_shift(hypothesis, na, +epa)
                ha_m = _clone_shift(hypothesis, na, -epa)

                p_ap = _penalty(ha_p)
                p_am = _penalty(ha_m)

                Hpen[a, a] = (p_ap - 2.0 * p0 + p_am) / (epa * epa)

                # Off-diagonal terms
                for b in range(a + 1, npar):
                    nb  = names[b]
                    vb  = float(param_by_name[nb].val)
                    epb = _eps_for(vb)

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

    def _assemble_nuA_groups_from_cfg(self, rid: str, hypothesis) -> dict[str, list[tuple[dict, np.ndarray]]]:
        """
        Build ν_A vectors for each PNN Δ-group directly from the loaded likelihood cfg
        (no HDF5 meta). Mirrors _assemble_nuA_groups(...) but reads the groups from
        C['_pnn_systs'] prepared in _prepare_structure.
        Returns: {cid: [ (group_meta, nuA), ... ], ... }.
        """
        import numpy as np

        # collect current nuisance values (lowercase indices), then expand to capital-B basis
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}

        out: dict[str, list[tuple[dict, np.ndarray]]] = {}
        # find the region object
        region = next((R for R in self.regions if R['id'] == rid), None)
        if region is None:
            raise RuntimeError(f"[evaluate_ratio] Unknown region id '{rid}'.")

        for C in (region.get('classes') or []):
            cid = C['id']
            groups = []
            for S in C.get('_pnn_systs', []):
                # Build a small meta dict compatible with the shapes we will compute
                gm = {
                    'id': S['id'],
                    # parameters/combinations were propagated by load_likelihood(...)
                    'params': list(S.get('parameters', []) or []),
                    'combs':  [list(t) for t in (S.get('combinations', []) or [])],
                    # dataset name tag to mirror on-disk convention
                    'dset': f"Delta::{S['id']}",
                }
                params = list(gm['params'])
                combs  = [tuple(c) for c in gm['combs']]
                nuA    = nuis_to_A_vector(params, combs, nu_vals)  # (nB,)
                groups.append((gm, nuA))
            out[cid] = groups
        return out

    # _eval_region_surrogates moved to the base N2LL class (fit/Likelihood.py),
    # since setObservation/toy generation need it too.

    def evaluate_ratio(self, rid: str, X: np.ndarray, feature_names: list[str],
                       hypothesis, *, cached: bool = True, return_T: bool = False,
                       chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Evaluate R(x; c, nu) = 1 + T(x; c, nu) for an external feature matrix X
        belonging to a likelihood region 'rid'. If return_T=True, return T instead.

        Caching:
          - If cached=True, the surrogate outputs (g, R, Δ) for this region are
            stored in memory on first call and reused on subsequent calls as long
            as feature_names and X.shape[1] match. We assume X itself is unchanged.
          - If cached=False, surrogates are recomputed.

        Parameters
        ----------
        rid : str
            Region id (must be present in self.regions)
        X : array (N, d)
            External feature matrix in the order given by feature_names.
        feature_names : list[str]
            Names matching the columns of X.
        hypothesis : Hypothesis
            Provides parameter values (POIs and nuisances).
        cached : bool
            Use per-region in-memory cache for surrogate outputs.
        return_T : bool
            If True, return T; else return 1 + T.
        chunk_size : int or None
            Optional chunking over events to reduce peak memory. If None, vectorized.

        Returns
        -------
        np.ndarray
            Vector of length N with R(x) or T(x).
        """
        import numpy as np

        if not self._runtime_prepared:
            raise RuntimeError("[evaluate_ratio] Call prepare_runtime() first so predictors/structure are ready.")

        # -------- cache lookup / fill --------
        feat_tuple = tuple(feature_names or [])
        cache_hit = False
        if cached and (rid in self._ext_eval_cache):
            ent = self._ext_eval_cache[rid]
            if ent.get('feature_names') == feat_tuple and ent.get('n_features') == (X.shape[1] if X.ndim==2 else None):
                by_class = ent['by_class']
                cache_hit = True
            else:
                # feature set changed -> discard previous cache for this region
                self._ext_eval_cache.pop(rid, None)

        if not cache_hit:
            by_class = self._eval_region_surrogates(rid, X, feature_names)
            if cached:
                self._ext_eval_cache[rid] = {
                    'feature_names': feat_tuple,
                    'n_features': X.shape[1],
                    'by_class': by_class
                }

        # -------- assemble parameters (A-basis) --------
        # lnN bias per class
        nu_vals = {p.name: float(p.val) for p in getattr(hypothesis, 'parameters', []) if not p.isPOI}
        ln_bias = {}
        region = next((R for R in self.regions if R['id'] == rid), None)
        class_ids = [C['id'] for C in (region.get('classes') or [])]
        for cid in class_ids:
            ln_bias[cid] = sum(log1p_alpha * nu_vals.get(pname, 0.0)
                               for pname, log1p_alpha in self._lnN_by_class.get((rid, cid), []))

        # c_A per class in the stored POI order
        cA_per_class = self._assemble_cA_per_class(rid, hypothesis)

        # ν_A per PNN group from cfg (no disk)
        nuA_per_group = self._assemble_nuA_groups_from_cfg(rid, hypothesis)

        # -------- compute T (optionally chunked) --------
        N = X.shape[0]
        T = np.zeros(N, dtype=np.float64)
        idx_iter = range(0, N, int(chunk_size)) if chunk_size else ( (0,) )
        for start in idx_iter:
            if chunk_size:
                stop = min(start + int(chunk_size), N)
                sl = slice(start, stop)
            else:
                sl = slice(None)

            # accumulate over classes
            for cid in class_ids:
                comp = by_class[cid]
                g_slice = np.asarray(comp['g'][sl], dtype=np.float64)
                R_slice = np.asarray(comp['R'][sl, :], dtype=np.float64)
                cA      = cA_per_class[cid]
                if R_slice.shape[1] != cA.shape[0]:
                    raise RuntimeError(f"[evaluate_ratio] BIT dim {R_slice.shape[1]} != |A| {cA.shape[0]} for {rid}/{cid}")
                c_dot_R = R_slice @ cA  # (M,)

                # build exponent from Δ-groups
                expo = np.zeros_like(g_slice)
                for gm, nuA in nuA_per_group.get(cid, []):
                    dset = gm.get('dset', f"Delta::{gm['id']}")
                    if dset not in comp:
                        raise RuntimeError(f"[evaluate_ratio] Missing '{dset}' for {rid}/{cid} in by_class cache.")
                    dA = np.asarray(comp[dset][sl, :], dtype=np.float64)  # (M, nB)
                    if dA.shape[1] != nuA.shape[0]:
                        raise RuntimeError(f"[evaluate_ratio] Δ dim {dA.shape[1]} != ν_A dim {nuA.shape[0]} for {rid}/{cid}/{gm['id']}")
                    expo += dA @ nuA

                exp_expo = np.exp(expo + ln_bias[cid])
                T[sl] += g_slice * (c_dot_R * exp_expo + (exp_expo - 1.0))

        return T if return_T else (1.0 + T)

