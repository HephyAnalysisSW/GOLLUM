# ----------------------------------------------------------------------
# UID splitter (YAML-agnostic)
# ----------------------------------------------------------------------
import numpy as np
import awkward as ak

class UIDSplitter:
    """
    Compute a deterministic boolean mask for a given UID bucket interval [lo, hi).

    - Does NOT read YAML
    - Does NOT know any split names / fractions / assignment
    - Only needs:
        * uid_fields: ("run","luminosityBlock","event") by default
        * seed: int (provided by training script from YAML)
        * interval: [lo, hi) in bucket space (provided by training script)
    """

    def __init__(
        self,
        uid_fields: tuple[str, str, str] = ("run", "luminosityBlock", "event"),
        seed: int = 0,
        n_buckets: int = 10000,
    ) -> None:
        self.uid_fields = tuple(uid_fields)
        if len(self.uid_fields) != 3:
            raise ValueError(f"uid_fields must have length 3, got {self.uid_fields}")
        self.seed = int(seed)
        self.n_buckets = int(n_buckets)
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive")

    # ---- splitmix64-based stable 64-bit hash (vectorized) ----
    def _uid_hash_u64(self, run: np.ndarray, lumi: np.ndarray, event: np.ndarray) -> np.ndarray:
        run = np.asarray(run).astype(np.uint64, copy=False)
        lumi = np.asarray(lumi).astype(np.uint64, copy=False)
        event = np.asarray(event).astype(np.uint64, copy=False)

        x = run
        x ^= lumi * np.uint64(0x9E3779B97F4A7C15)
        x ^= event * np.uint64(0xBF58476D1CE4E5B9)
        x ^= np.uint64(self.seed)

        # splitmix64
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = z ^ (z >> np.uint64(31))
        return z

    def _bucketize(self, run: np.ndarray, lumi: np.ndarray, event: np.ndarray) -> np.ndarray:
        h = self._uid_hash_u64(run, lumi, event)
        # 0 .. n_buckets-1
        return (h % np.uint64(self.n_buckets)).astype(np.uint32, copy=False)

    # ---- public API ----
    def mask_from_arrays(self, run: np.ndarray, lumi: np.ndarray, event: np.ndarray, lo: int, hi: int) -> np.ndarray:
        lo = int(lo)
        hi = int(hi)
        if not (0 <= lo <= hi <= self.n_buckets):
            raise ValueError(f"invalid interval [lo,hi)=({lo},{hi}) for n_buckets={self.n_buckets}")

        bucket = self._bucketize(run, lumi, event)
        return (bucket >= lo) & (bucket < hi)

    def mask_from_ak(self, ar: "ak.Array", lo: int, hi: int) -> np.ndarray:
        rname, lname, ename = self.uid_fields
        if rname not in ar.fields or lname not in ar.fields or ename not in ar.fields:
            missing = [k for k in self.uid_fields if k not in ar.fields]
            raise KeyError(f"UID fields missing in awkward array: {missing}")

        run = ak.to_numpy(ar[rname])
        lumi = ak.to_numpy(ar[lname])
        event = ak.to_numpy(ar[ename])
        return self.mask_from_arrays(run, lumi, event, lo=lo, hi=hi)

    def mask_from_np(self, G_sel: np.ndarray, names: list[str], lo: int, hi: int) -> np.ndarray:
        """
        Compute UID mask from numpy observer matrix G_sel.

        Parameters
        ----------
        G_sel : np.ndarray
            shape (N, M), produced by base.observers(...).
        names : list[str]
            column names corresponding to G_sel columns.
        lo, hi : int
            bucket interval [lo, hi).
        """
        rname, lname, ename = self.uid_fields
        try:
            ir = names.index(rname)
            il = names.index(lname)
            ie = names.index(ename)
        except ValueError as e:
            raise KeyError(f"UID fields {self.uid_fields} not found in selection feature names: {names}") from e

        # observers can come as float; cast safely to int64 first
        run  = np.asarray(G_sel[:, ir]).astype(np.int64, copy=False)
        lumi = np.asarray(G_sel[:, il]).astype(np.int64, copy=False)
        event= np.asarray(G_sel[:, ie]).astype(np.int64, copy=False)

        return self.mask_from_arrays(run, lumi, event, lo=lo, hi=hi)