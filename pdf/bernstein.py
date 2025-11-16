import numpy as np
from math import comb

class BernsteinBasis:
    """
    Bernstein basis (degree n).

    Usage
    -----
    B = BernsteinBasis(n)

    # get a callable for the k-th basis polynomial B_{k,n}(x):
    b_k = B[k]           # b_k(x) is vectorized: accepts scalar or ndarray x

    # evaluate the entire basis on x:
    X = np.linspace(0,1,201)
    M = B.values(X)      # shape (n+1, X.size)  -> row i is B_{i,n}(X)

    """

    def __init__(self, n: int):
        if int(n) != n or n < 0:
            raise ValueError("degree n must be a nonnegative integer")
        self._n = int(n)

    @property
    def degree(self) -> int:
        return self._n

    def _basis_frame(self, x):
        """
        Internal: produce an array shape (n+1, m) of B_{i,n}(x) values,
        where m = len(x) (or 1 for scalar x).
        """
        x = np.asarray(x)
        # If scalar, make 1-D array but remember to return scalar when requested.
        is_scalar = (x.ndim == 0)
        xflat = x.reshape(-1)  # 1-D view

        n = self._n
        rows = []
        # vectorized computation of each basis i
        for i in range(n + 1):
            rows.append(comb(n, i) * (xflat**i) * ((1.0 - xflat)**(n - i)))
        print(rows)
        M = np.stack(rows,axis=1)    # shape (m,n+1)
        print(M.shape)
        if is_scalar:
            # If user passed scalar, return 1D column (n+1,) for convenience
            return M[0,:]
        return M

    def values(self, x):
        """
        Evaluate all Bernstein basis polynomials B_{i,n}(x) for i=0..n.

        Returns array shape (n+1, len(x)) for array x, or shape (n+1,) for scalar x.
        """
        return self._basis_frame(x)

    def __getitem__(self, k):
        """
        Return a callable f(x) that evaluates B_{k,n}(x).

        Example:
            b2 = B[2]
            y = b2(np.linspace(0,1,100))
        """
        n = self._n
        if not (0 <= k <= n):
            raise IndexError("basis index out of range")
        k = int(k)
        # return a small vectorized function
        def basis_k(x):
            x = np.asarray(x)
            # compute using binomial formula
            return comb(n, k) * (x**k) * ((1.0 - x)**(n - k))
        return basis_k

    @classmethod
    def sequence(cls, min_degree: int, max_degree: int):
        """
        Yield BernsteinBasis for degrees min_degree..max_degree (inclusive).
        """
        if int(min_degree) != min_degree or int(max_degree) != max_degree:
            raise ValueError("degrees must be integers")
        mn, mx = int(min_degree), int(max_degree)
        if mn > mx:
            raise ValueError("min_degree must be <= max_degree")
        for d in range(mn, mx + 1):
            yield cls(d)

    def __repr__(self):
        return f"BernsteinBasis(degree={self._n})"
