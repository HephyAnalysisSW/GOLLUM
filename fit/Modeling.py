import copy
import logging

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
        if self.isFrozen:
            raise RuntimeError(f"Parameter {self.name} is frozen.")
        self.val = float(value)
        return self

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

    # ---------- cloners ----------
    def clone(self):
        return copy.deepcopy(self)

    def cloneModify(self, **kwargs):
        h = self.clone()
        return h.modify(**kwargs)

    def cloneSM(self):
        """
        Clone with all parameter values reset to 0 (keeps frozen flags as-is).
        """
        h = self.clone()
        for p in h.parameters:
            if p.val != 0.0:
                if p.isFrozen:
                    logger.warning(
                        "Resetting frozen parameter %s from %g to 0.",
                        p.name, p.val
                    )
                p.val = 0.0
        return h

    def cloneFreeze(self, **fixed):
        """
        Clone and freeze given parameters: hyp.cloneFreeze(c1=0.0, nu_pu=0.0)
        """
        h = self.clone()
        for k, v in fixed.items():
            h[k].val = float(v)
            h[k].isFrozen = True
        return h

    # ---------- pretty print ----------
    def print(self):
        title = self.name if self.name else "unnamed"
        print(f"Hypothesis ({title})\n")
        for i, p in enumerate(self.POIs):
            print(f"{i:02d}  {p}")
        print()
        for j, p in enumerate(self.nuisances, start=len(self.POIs)):
            print(f"{j:02d}  {p}")

