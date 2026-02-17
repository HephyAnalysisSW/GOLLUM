import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------- Parameter & Hypothesis scaffolding ----------
class ModelParameter:
    """
    Lightweight model parameter representation with convenient helpers.
    """
    def __init__(self, name, val=0.0, *, isPOI=False,
                 isFrozen=False, isPenalized=False):
        self.name        = str(name)
        self.val         = float(val)
        self.isPOI       = bool(isPOI)
        self.isFrozen    = bool(isFrozen)
        self.isPenalized = bool(isPenalized)

    def __repr__(self):
        tags = []
        if self.isPOI:       tags.append("POI")
        else:                tags.append("Nuis.")
        if self.isFrozen:    tags.append("frozen")
        if self.isPenalized: tags.append("pen.")
        return f"<{self.name}({','.join(tags)})={self.val:.6e}>"

    def __str__(self):
        return self.__repr__().lstrip('<').rstrip('>')

    def __call__(self):
        return self.val

    def freeze(self, value=None):
        if value is not None:
            self.val = float(value)
        self.isFrozen = True
        return self

    def unfreeze(self):
        self.isFrozen = False
        return self

    def penalize(self):
        self.isPenalized = False
        return self

    def float(self):
        self.isPenalized = False
        self.isFrozen = False
        return self

    def set(self, value):
        self.val = float(value)
        return self

    def __float__(self):
        return float(self.val)

    # optional: for numpy arrays etc.
    def __array__(self, dtype=None):
        return np.asarray(self.val, dtype=dtype)

class Hypothesis:
    """
    Container of ModelParameters with convenient accessors and cloning helpers.

    Features:
      - hyp['c0']      → ModelParameter named 'c0'
      - hyp.c0         → same (attribute-style access; good for tab completion)
      - hyp.c0 = 0.1   → sets value of parameter 'c0' (unless frozen)
      - 'c0' in hyp    → True if parameter exists
    """
    def __init__(self, parameters, name=None):
        # Bypass __setattr__ for core attributes during init
        object.__setattr__(self, "parameters", list(parameters or []))
        object.__setattr__(self, "name", name)
        self._check()
        self._base = self # A rotated basis has a base. This completes the interface; can do hyp._base on both types.

    def _check(self):
        # POIs should not be penalized (guard user mistakes)
        for p in self.parameters:
            if p.isPOI and p.isPenalized:
                logger.warning("POI %s marked 'penalized'; clearing penalty.", p.name)
                p.isPenalized = False
        # Unique names
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate parameter names in hypothesis: {names}")

    # ---------- mapping-style access ----------
    def __contains__(self, key):
        return any(p.name == key for p in self.parameters)

    def __getitem__(self, key):
        for p in self.parameters:
            if p.name == key:
                return p
        raise KeyError(key)

    # ---------- attribute-style access ----------
    def __getattr__(self, name):
        """
        Called only if normal attribute lookup fails.
        If `name` matches a parameter, return that ModelParameter.
        Otherwise raise AttributeError listing available parameter names.
        """
        try:
            params = object.__getattribute__(self, "parameters")
        except AttributeError:
            # During very early init / weird states, just behave like normal
            raise AttributeError(f"'Hypothesis' object has no attribute '{name}'") from None

        for p in params:
            if p.name == name:
                return p

        available = ", ".join(p.name for p in params)
        raise AttributeError(
            f"Hypothesis has no parameter '{name}'. "
            f"Available parameters: {available}"
        )

    def __setattr__(self, name, value):
        """
        - Core attributes ('parameters', 'name', '_...') are set normally.
        - If `name` matches a parameter, treat `hyp.name = v` as setting p.val = v.
        - Otherwise create/overwrite a normal attribute.
        """
        if name in ("parameters", "name") or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # Try to treat it as a parameter assignment
        try:
            params = object.__getattribute__(self, "parameters")
        except AttributeError:
            # parameters not yet set (very early init) → just set attribute
            object.__setattr__(self, name, value)
            return

        for p in params:
            if p.name == name:
                if p.isFrozen:
                    raise RuntimeError(f"Parameter {name} is frozen; cannot assign.")
                p.val = float(value)
                return

        # Fallback: ordinary attribute
        object.__setattr__(self, name, value)

    def __dir__(self):
        """
        Improve tab-completion: include parameter names in dir(hyp).
        """
        base = set(super().__dir__())
        try:
            params = object.__getattribute__(self, "parameters")
        except AttributeError:
            return sorted(base)
        base.update(p.name for p in params)
        return sorted(base)

    # ---------- properties ----------
    @property
    def POIs(self):
        return [p for p in self.parameters if p.isPOI]

    @property
    def nuisances(self):
        return [p for p in self.parameters if not p.isPOI]

    @property
    def penalized(self):
        return [p for p in self.parameters if p.isPenalized]

    # ---------- penalty ----------
    def penalty(self):
        """Compute the penalty (sum v**2) from all penalized nuisances."""
        return sum(p.val**2 for p in self.parameters if p.isPenalized)

    # ---------- mutators ----------
    def modify(self, **kwargs):
        """
        hyp.modify(c1=0.2, nu_pu=0.0, ...)
        """
        for k, v in kwargs.items():
            self[k].set(v)
        return self

    def set_nuisance_frozen(self, name, isFrozen):
        getattr(self      , name).isFrozen = isFrozen

    # ---------- cloners ----------
    def clone(self):
        return copy.deepcopy(self)

    def cloneModify(self, **kwargs):
        h = self.clone()
        return h.modify(**kwargs)

    # ---------- pretty print ----------
    def print(self):
        title = self.name if self.name else "unnamed"
        print(f"Hypothesis ({title})\n")
        for i, p in enumerate(self.POIs):
            print(f"{i:02d}  {p}")
        print()
        for j, p in enumerate(self.nuisances, start=len(self.POIs)):
            print(f"{j:02d}  {p}")
import json
import numpy as np
import json
import numpy as np

class Rotated(Hypothesis):
    """
    Rotated view over a base Hypothesis:
      - Exposes *all* base parameters.
        * POIs: as rotated parameters d = D_full @ c (same count as base POIs).
        * Nuisances: passthrough with original names.
      - Setting d writes back to base POIs (checks frozen).
      - Setting a nuisance writes back to base nuisance (checks frozen).
      - Penalty is forwarded to the base.
    """

    def __init__(self, base: Hypothesis, json_filename: str, name: str | None = None, *, rcond: float = 1e-12, normalize_euclidean = False):
        if base is None:
            raise RuntimeError("[Rotated] base Hypothesis must be provided.")
        self._base = base
        self._json_path = str(json_filename)
        self._rcond = float(rcond)

        # ---- Base parameter partitions ----
        base_pois = [p for p in base.parameters if p.isPOI]
        base_nuis = [p for p in base.parameters if not p.isPOI]
        if not base_pois:
            raise RuntimeError("[Rotated] Base hypothesis has no POIs to rotate.")
        self._c_names = [p.name for p in base_pois]            # POI names (length K)
        self._nuis_names = [p.name for p in base_nuis]          # nuisance names
        K = len(self._c_names)

        # ---- Load rotation JSON ----
        with open(self._json_path, "r") as f:
            payload = json.load(f)

        poi_order_json = list(payload.get("poi_order", []) or [])
        if not poi_order_json:
            raise RuntimeError("[Rotated] JSON must contain 'poi_order' listing c-space parameter names.")

        D_raw = payload.get("D", None)
        if D_raw is None:
            raise RuntimeError("[Rotated] JSON must contain 'D'")
        D_raw = np.asarray(D_raw, dtype=np.float64)

        m_raw, k_raw = D_raw.shape
        if k_raw != len(poi_order_json):
            raise RuntimeError("[Rotated] D/V_new column count must match length of 'poi_order'.")

        # ---- Embed JSON columns into base POI order; add identity rows for missing POIs ----
        json_name_to_col = {nm: j for j, nm in enumerate(poi_order_json)}
        used_cols = set()
        D_cols = np.zeros((m_raw, K), dtype=np.float64)
        for k, cname in enumerate(self._c_names):
            if cname in json_name_to_col:
                j = json_name_to_col[cname]
                D_cols[:, k] = D_raw[:, j]
                used_cols.add(cname)

        missing = [c for c in self._c_names if c not in used_cols]
        u = len(missing)
        if u > 0:
            I_missing = np.zeros((u, K), dtype=np.float64)
            for r, cname in enumerate(missing):
                k = self._c_names.index(cname)
                I_missing[r, k] = 1.0
            D_full = np.vstack([D_cols, I_missing])  # (m_raw + u, K)
        else:
            D_full = D_cols

        # ---- Optional: normalize each rotated direction to unit Euclidean length in c space (d=Dc) ----
        if normalize_euclidean:
            norms = np.linalg.norm(D_full, axis=1)
            norms[norms == 0.0] = 1.0  # safety for any all-zero row
            D_full = D_full / norms[:, None]

        self._D = D_full

        # ---- Names for rotated POIs (d) ----
        basis_labels = payload.get("basis_labels", None)
        if basis_labels is not None and len(basis_labels) != m_raw:
            logger.warning("[Rotated] Ignoring JSON 'basis_labels' (len=%d) != provided rows m=%d.",
                           len(basis_labels), m_raw)
            basis_labels = None
        head_labels = basis_labels if basis_labels is not None else [f"d{i}" for i in range(m_raw)]
        tail_labels = list(missing)  # passthrough rows get their POI names
        self._d_names = head_labels + tail_labels               # len = self._D.shape[0]
        self._d_index = {nm: i for i, nm in enumerate(self._d_names)}
        self._nuis_index = {nm: i for i, nm in enumerate(self._nuis_names)}

        # Warn about JSON c-names not found in base (ignored)
        extra_in_json = [nm for nm in poi_order_json if nm not in self._c_names]
        if extra_in_json:
            logger.warning("[Rotated] Ignoring %d JSON c-names not present in base: %s",
                           len(extra_in_json), ", ".join(extra_in_json))

        # ---- Build parameter list (rotated POIs + passthrough nuisances) ----
        d_vals = self._compute_d_from_base()
        d_params = [
            ModelParameter(name=self._d_names[j], val=float(d_vals[j]), isPOI=True, isPenalized=False)
            for j in range(self._D.shape[0])
        ]
        # Nuisances mirror base (values and penalization flags for printing)
        nuis_params = [
            ModelParameter(name=nm,
                           val=float(getattr(self._base, nm).val),
                           isPOI=False,
                           isPenalized=bool(getattr(self._base, nm).isPenalized))
            for nm in self._nuis_names
        ]
        all_params = d_params + nuis_params

        super().__init__(parameters=all_params, name=(name or f"Rotated({base.name or 'base'})"))
        # Need to re-set self._base because the base constructor sets it to self.
        self._base = base

    # ---------- internals ----------
    def _current_c_vector(self) -> np.ndarray:
        return np.array([float(getattr(self._base, nm).val) for nm in self._c_names], dtype=np.float64)

    def _compute_d_from_base(self) -> np.ndarray:
        return self._D @ self._current_c_vector()

    def _solve_c_from_d(self, d_target: np.ndarray) -> np.ndarray:
        D = self._D
        m, K = D.shape
        if m == K:
            try:
                return np.linalg.solve(D, d_target)
            except np.linalg.LinAlgError:
                pass
        return np.linalg.pinv(D, rcond=self._rcond) @ d_target

    def _apply_d_update(self, d_new: np.ndarray, *, tol_frozen: float = 0.0) -> None:
        d_new = np.asarray(d_new, dtype=np.float64)
        if d_new.shape != (self._D.shape[0],):
            raise RuntimeError("[Rotated] d_new has wrong shape.")

        c_old = self._current_c_vector()
        c_new = self._solve_c_from_d(d_new)

        # check frozen POIs on base
        for nm, vo, vn in zip(self._c_names, c_old, c_new):
            bp = getattr(self._base, nm)
            if bp.isFrozen and (abs(float(vn) - float(vo)) > tol_frozen):
                raise RuntimeError(f"[Rotated] Attempt to change frozen base POI '{nm}': {vo} → {vn}")

        # write POIs back to base
        for nm, vn in zip(self._c_names, c_new):
            getattr(self._base, nm).val = float(vn)

        # sync local displayed d values
        for j, p in enumerate(self.parameters[:self._D.shape[0]]):
            p.val = float(d_new[j])

    # ---------- public API ----------
    def penalty(self) -> float:
        return self._base.penalty()

    def base(self) -> Hypothesis:
        return self._base

    def set_vector(self, names: list[str], values: list[float], *, tol_frozen: float = 0.0):
        if len(names) != len(values):
            raise RuntimeError("[Rotated.set_vector] names and values length mismatch.")

        # start from current values
        d_cur = self._compute_d_from_base()
        # We will also track nuisance updates
        nuis_updates = {}

        for nm, v in zip(names, values):
            if nm in self._d_index:
                d_cur[self._d_index[nm]] = float(v)
            elif nm in self._nuis_index:
                nuis_updates[nm] = float(v)
            else:
                raise KeyError(f"[Rotated.set_vector] Unknown parameter '{nm}' (neither rotated POI nor nuisance).")

        # apply rotated POI updates
        self._apply_d_update(d_cur, tol_frozen=tol_frozen)

        # apply nuisance updates to base
        for nm, val in nuis_updates.items():
            bp = getattr(self._base, nm)
            bp.val = float(val)

        # sync local nuisance values for printing
        n0 = self._D.shape[0]
        for i, nm in enumerate(self._nuis_names):
            self.parameters[n0 + i].val = float(getattr(self._base, nm).val)

    def modify(self, **kwargs):
        if not kwargs:
            return self
        norm = {}
        for k, v in kwargs.items():
            norm[k] = v
        self.set_vector(list(norm.keys()), list(norm.values()))
        return self

    def set_nuisance_frozen(self, name, isFrozen):
        getattr(self._base, name).isFrozen = isFrozen
        getattr(self      , name).isFrozen = isFrozen

    def __setattr__(self, name, value):
        # core/private
        if name in ("_base", "_json_path", "_rcond", "_D", "_c_names", "_d_names", "_d_index", "_nuis_names", "_nuis_index") or name.startswith("_"):
            object.__setattr__(self, name, value); return
        if name in ("parameters", "name"):
            object.__setattr__(self, name, value); return

        # rotated POI by attribute name
        if hasattr(self, "_d_index") and name in self._d_index:
            d_cur = self._compute_d_from_base()
            d_cur[self._d_index[name]] = float(value)
            self._apply_d_update(d_cur)
            return

        # nuisance passthrough by attribute name
        if hasattr(self, "_nuis_index") and name in self._nuis_index:
            bp = getattr(self._base, name)
            bp.val = float(value)
            # sync local cached nuisance value (for printing)
            n0 = self._D.shape[0]
            self.parameters[n0 + self._nuis_index[name]].val = float(bp.val)
            return

        # fallback: ordinary attribute
        object.__setattr__(self, name, value)

    def print(self):
        # sync rotated POIs from base
        d_vals = self._compute_d_from_base()
        for j, p in enumerate(self.parameters[:self._D.shape[0]]):
            p.val = float(d_vals[j])
        # sync nuisances from base
        n0 = self._D.shape[0]
        for i, nm in enumerate(self._nuis_names):
            self.parameters[n0 + i].val = float(getattr(self._base, nm).val)

        title = self.name if self.name else "rotated"
        print(f"Rotated Hypothesis ({title})")
        print("\n  [rotated POIs]")
        for i, p in enumerate(self.parameters[:self._D.shape[0]]):
            print(f"{i:02d}  {p}")
        print("\n  [nuisances (passthrough)]")
        for j, p in enumerate(self.parameters[self._D.shape[0]:], start=self._D.shape[0]):
            print(f"{j:02d}  {p}")

        print("\n[base] Current base POIs:")
        for nm in self._c_names:
            print(f"    {nm:>16s} = {float(getattr(self._base, nm).val): .6e}")

    def clone(self):
        base_clone = self._base.clone()
        return Rotated(base_clone, self._json_path, name=self.name, rcond=self._rcond)

    def cloneModify(self, **kwargs):
        h = self.clone()
        return h.modify(**kwargs)

    @property
    def POIs(self):
        # rotated POIs only
        return list(self.parameters[:self._D.shape[0]])

    @property
    def nuisances(self):
        # passthrough nuisances
        return list(self.parameters[self._D.shape[0]:])

    @property
    def penalized(self):
        # printing/helper use only; penalty() forwards to base
        return [p for p in self.nuisances if p.isPenalized]

