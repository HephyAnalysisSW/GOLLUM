import itertools
import logging
from typing import Sequence, Union

import numpy as np
from numpy.polynomial.chebyshev import chebvander

logger = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]

class PDFParametrization:
    """
    Chebyshev expansion on y(x) = 1 - 2*sqrt(x):

        f(x) = 1 + sum_{i=0}^{n} c_i * T_i( y(x) )

    - Coefficients are (c0, ..., c_n), length n+1.
    - If n = 0, then f(x) = 1 + c0.
    """

    def __init__(self, n: int = 5):
        self.n = int(n)
        logger.debug(f"Using Chebyshev polynomials up to order {self.n}")
        self.variables = [f"c{i}" for i in range(self.n + 1)]

    # 2nd entry in Fig. 1 here: https://arxiv.org/pdf/1211.1215
    @staticmethod
    def y(x: ArrayLike) -> ArrayLike:
        x_arr = np.asarray(x, dtype=float)
        return 1.0 - 2.0 * np.sqrt(x_arr)

    def evaluate(self, x: ArrayLike, id: ArrayLike, coeffs: Sequence[float]) -> ArrayLike:
        """
        Compute f(x, id) = 1 + [sum_{i=0}^{n} c_i * T_i( y(x) )] * (id == 21).
        """
        if len(coeffs) != self.n + 1:
            raise ValueError(f"Expected {self.n + 1} coefficients, got {len(coeffs)}")

        x_arr = np.asarray(x, dtype=float)
        y_vals = self.y(x_arr)

        # chebvander returns columns [T0(y), T1(y), ..., Tn(y)]
        V = chebvander(y_vals, self.n)
        c = np.asarray(coeffs, dtype=float)

        cheb_sum = np.tensordot(V, c, axes=([-1], [0]))
        mask = (np.asarray(id) == 21).astype(float)

        out = 1.0 + cheb_sum * mask
        scalar_out = (np.ndim(x) == 0) and (np.ndim(id) == 0)
        return float(out) if scalar_out else out

    def make_combinations(self, order: int = 2):
        """
        Simple helper to build combinations of variable names up to 'order'.
        """
        combos = []
        for o in range(order + 1):
            combos.extend(itertools.combinations_with_replacement(self.variables, o))
        return combos

    @property
    def combinations(self):
        if not hasattr(self, "_combinations"):
            self._combinations = self.make_combinations(order=2)
        return self._combinations

    def product_parametrizations(self, x1, x2, id1, id2, coeffs):
        """Return f(x1; c) * f(x2; c) using the same coefficient vector c."""
        return self.evaluate(x1, id1, coeffs) * self.evaluate(x2, id2, coeffs)

    __call__ = product_parametrizations  # allow pdf(x, coeffs)

    def derivatives(self, x1, x2, id1, id2):
        """
        Compute all derivatives of F(c) = f(x1,id1;c)*f(x2,id2;c) at c=0
        in the order given by self.combinations:
          (), ('c0',),...,('c_n',), ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n').

        Returns a list of arrays/scalars aligned with self.combinations.
        """
        import numpy as np
        from numpy.polynomial.chebyshev import chebvander

        y1 = self.y(np.asarray(x1, float))
        y2 = self.y(np.asarray(x2, float))
        m1 = (np.asarray(id1) == 21).astype(float)
        m2 = (np.asarray(id2) == 21).astype(float)
        y1, y2, m1, m2 = np.broadcast_arrays(y1, y2, m1, m2)

        V1 = chebvander(y1, self.n)  # T_k(y1)
        V2 = chebvander(y2, self.n)  # T_k(y2)

        ones = np.ones_like(y1, float)
        # First derivatives: g_k = m1*T_k(y1) + m2*T_k(y2)
        g = m1[..., None] * V1 + m2[..., None] * V2  # shape [..., n+1]

        # Second derivatives: H_ab = m1*m2 * ( T_a(y1)T_b(y2) + T_b(y1)T_a(y2) )
        outer = V1[..., :, None] * V2[..., None, :]  # [..., a, b]
        H = (m1 * m2)[..., None, None] * (outer + np.swapaxes(outer, -2, -1))

        out = []
        # ()
        out.append(ones)
        # ('c0',) .. ('c_n',)
        for k in range(self.n + 1):
            out.append(g[..., k])
        # ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n')
        for i in range(self.n + 1):
            for j in range(i, self.n + 1):
                out.append(H[..., i, j])

        # If scalar inputs, return Python floats
        if all(np.ndim(v) == 0 for v in (x1, x2, id1, id2)):
            out = [float(v) for v in out]
        return np.transpose(np.array(out))

if __name__ == "__main__":
    import numpy as np

    # --- n = 0 case: f(x, id) = 1 + c0 * (id == 21) ---
    pdf0 = PDFParametrization(n=0)
    c0 = (0.7,)  # length n+1 = 1
    print("n=0, gluon (id=21):   ", pdf0.evaluate(0.3, 21, c0))  # -> 1 + 0.7 = 1.7
    print("n=0, quark (id=1):    ", pdf0.evaluate(0.3, 1,  c0))  # -> 1.0

    # --- n = 3, vectorized x and mixed ids ---
    pdf = PDFParametrization(n=3)
    x = np.linspace(0.0, 1.0, 6)  # vector input
    coeffs = (0.1, 0.2, -0.05, 0.01)  # (c0..c3), length n+1 = 4

    # Mixed ids: some gluons, some not
    ids_mixed = np.array([21, 21, 1, 2, 21, 3])
    vals_mixed = pdf.evaluate(x, ids_mixed, coeffs)
    print("\nn=3, vector x with mixed ids:")
    print("x         :", x)
    print("ids       :", ids_mixed)
    print("f(x, id)  :", vals_mixed)

    # Scalar id broadcasting: all gluons vs all non-gluons
    vals_all_g = pdf.evaluate(x, 21, coeffs)
    vals_all_q = pdf.evaluate(x, 1,  coeffs)
    print("\nBroadcast id=21 (all gluons):", vals_all_g)
    print("Broadcast id=1 (all non-g):  ", vals_all_q)  # should be all ones

    # --- Products with shared coefficients ---
    # 1) Both gluons: product is f(x)^2
    prod_gg = pdf.product_parametrizations(x, x, 21, 21, coeffs)
    print("\nProduct gg (id1=21, id2=21):", prod_gg)

    # 2) Gluon × non-gluon: product is f(x) * 1 = f(x)
    prod_gq = pdf.product_parametrizations(x, x, 21, 1, coeffs)
    print("Product gq (id1=21, id2=1): ", prod_gq)

    # 3) Mixed vector ids vs scalar id
    ids_alt = np.array([21, 1, 21, 1, 21, 1])
    prod_mixed = pdf.product_parametrizations(x, x, ids_alt, 21, coeffs)
    print("Product mixed (ids_alt vs 21):", prod_mixed)

    # 4) Scalar inputs
    xs = 0.25
    print("\nScalar x, gg:",
          pdf.product_parametrizations(xs, xs, 21, 21, coeffs))
    print("Scalar x, gq:",
          pdf.product_parametrizations(xs, xs, 21, 1, coeffs))

    print("Taylor expansion test:")
    import numpy as np

    # Taylor reconstruction using derivatives returned as a single np.array
    # with shape (..., len(pdf.combinations)), i.e. last axis indexes combinations.
    def taylor_reconstruct(pdf, x1, x2, id1, id2, coeffs):
        derivs = pdf.derivatives(x1, x2, id1, id2)  # shape (..., M)
        total = derivs[..., 0]  # () term

        # First-order terms: indices 1 .. n+1 map to ('c0',)..('c_n',)
        offset = 1
        for k in range(pdf.n + 1):
            total = total + derivs[..., offset + k] * coeffs[k]

        # Second-order terms follow in the order:
        # ('c0','c0'), ('c0','c1'), ..., ('c0','c_n'), ('c1','c1'), ..., ('c_n','c_n')
        idx = offset + (pdf.n + 1)
        for i in range(pdf.n + 1):
            for j in range(i, pdf.n + 1):
                w = 0.5 if i == j else 1.0
                total = total + w * derivs[..., idx] * coeffs[i] * coeffs[j]
                idx += 1
        return total

    # ---- Nontrivial vector test ----
    pdf = PDFParametrization(n=3)
    x1 = np.linspace(0.0, 1.0, 7)
    x2 = np.linspace(1.0, 0.0, 7)
    id1 = np.array([21, 1, 21, 2, 3, 21, 4])
    id2 = np.array([1, 21, 2, 21, 21, 3, 5])
    coeffs = (0.15, -0.07, 0.02, 0.01)  # c0..c3

    F_nom = pdf.product_parametrizations(x1, x2, id1, id2, coeffs)
    F_taylor = taylor_reconstruct(pdf, x1, x2, id1, id2, coeffs)
    print("Vector test: max |F_nom - F_taylor| =",
          float(np.max(np.abs(F_nom - F_taylor))))
    assert np.allclose(F_nom, F_taylor, rtol=1e-12, atol=1e-12)

    # ---- Scalar test (all gluons) ----
    xs1, xs2 = 0.25, 0.6
    ids1, ids2 = 21, 21
    F_nom_s = pdf.product_parametrizations(xs1, xs2, ids1, ids2, coeffs)
    F_taylor_s = taylor_reconstruct(pdf, xs1, xs2, ids1, ids2, coeffs)
    print("Scalar gg test: |F_nom - F_taylor| =",
          abs(F_nom_s - F_taylor_s))
    assert np.allclose(F_nom_s, F_taylor_s, rtol=1e-12, atol=1e-12)

    # ---- Scalar test (g×q -> linear only) ----
    ids1, ids2 = 21, 1
    F_nom_s2 = pdf.product_parametrizations(xs1, xs2, ids1, ids2, coeffs)
    F_taylor_s2 = taylor_reconstruct(pdf, xs1, xs2, ids1, ids2, coeffs)
    print("Scalar gq test: |F_nom - F_taylor| =",
          abs(F_nom_s2 - F_taylor_s2))
    assert np.allclose(F_nom_s2, F_taylor_s2, rtol=1e-12, atol=1e-12)

    print("All Taylor tests passed ✅")
