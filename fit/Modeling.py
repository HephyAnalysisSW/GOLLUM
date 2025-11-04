import copy
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

    def set(self, value):
        if self.isFrozen:
            raise RuntimeError(f"Parameter {self.name} is frozen.")
        self.val = float(value)
        return self

    @classmethod
    def makePenalizedNuisance(cls, name, val=0.0):
        return cls(name=name, val=val, isPenalized=True, isPOI=False)

class Hypothesis:
    """
    Container of ModelParameters with convenience accessors and cloning helpers.
    No numerics here; this is purely structural for now.
    """
    def __init__(self, parameters, name=None):
        self.parameters = list(parameters or [])
        self.name = name
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

    def __contains__(self, key):
        return any(p.name == key for p in self.parameters)

    def __getitem__(self, key):
        for p in self.parameters:
            if p.name == key:
                return p
        raise KeyError(key)

    # Properties
    @property
    def POIs(self):
        return [p for p in self.parameters if p.isPOI]

    def penalty( self ):
        ''' Compute the penalty (sum v**2) from all penalized nuisance
        '''
        return sum( [ p.val**2 for p in self.parameters if p.isPenalized ] )

    @property
    def nuisances(self):
        return [p for p in self.parameters if not p.isPOI]

    @property
    def penalized(self):
        return [p for p in self.parameters if p.isPenalized]

    # Mutators
    def modify(self, **kwargs):
        """
        hyp.modify(c1=0.2, nu_pu=0.0, ...)
        """
        for k, v in kwargs.items():
            self[k].set(v)
        return self

    # Cloners
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
                    logger.warning("Resetting frozen parameter %s from %g to 0.", p.name, p.val)
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

    # Pretty print
    def print(self):
        title = self.name if self.name else "unnamed"
        print(f"Hypothesis ({title})\n")
        for i, p in enumerate(self.POIs):
            print(f"{i:02d}  {p}")
        print()
        for j, p in enumerate(self.nuisances, start=len(self.POIs)):
            print(f"{j:02d}  {p}")
