import sys
sys.path.insert(0, '..')

import logging
from typing import Sequence, Union
import itertools
import numpy as np
logger = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]

from pdf.nnpdf.constants import LHAPDF_XGRID, XGRID
import lhapdf

central_set   = "NNPDF31_nnlo_as_0118"

variation_set = "250503_pod_basis_40k"
hand_picked = [1,2,4,5,10,12,16,18,23,25,31]

# Computed the maximum variations for cross-normalization of PDF effects (see __main__ block)
max_amplitudes = {1: 159.27605180060988,
 2: 127.30473550676739,
 3: 5.359663062903383,
 4: 275.5362709147004,
 5: 264.44453980151735,
 6: 3.0839264003173215,
 7: 4.201746426939554,
 8: 6.198739771688117,
 9: 2.818421244651188,
 10: 61.264902413190384,
 11: 2.567267236624119,
 12: 81.10730434834116,
 13: 4.397071149742556,
 14: 4.019107518848609,
 15: 8.046264009693557,
 16: 37.14507557109136,
 17: 4.349244689458295,
 18: 13.408219361909715,
 19: 4.6011686879308735,
 20: 4.96496485599619,
 21: 4.760468823793103,
 22: 4.539657283712429,
 23: 13.67322159065316,
 24: 4.921189063688713,
 25: 10.83024472260793,
 26: 4.487580633689011,
 27: 4.734808250846411,
 28: 4.858311670344503,
 29: 4.734521959870874,
 30: 4.768238807452397,
 31: 9.30204862446099,
 32: 5.858426195711921,
 33: 4.72631235822778,
 34: 4.769920163965911,
 35: 4.756720809708991,
 36: 4.774455677467385,
 37: 4.763381003708858,
 38: 7.110380503621367,
 39: 5.1674342765089,
 40: 4.786255535841878,
 41: 4.754648285458536,
 42: 4.780756272485305,
 43: 4.783909431744489,
 44: 4.772601198272527,
 45: 4.7695099821939895,
 46: 4.790153216525275,
 47: 4.675428185622534,
 48: 4.750433052563139,
 49: 4.778597107669818,
 50: 4.773328774935295,
 51: 4.779789068221233,
 52: 4.775416223529802,
 53: 4.778074095474507,
 54: 4.75750615651149,
 55: 4.775530722643126,
 56: 4.581592117071566,
 57: 5.056381569344865,
 58: 4.777910366004922,
 59: 4.782859377885328,
 60: 4.778489908582197,
 61: 4.778068532385993,
 62: 4.6735389090207775,
 63: 4.772365282206208,
 64: 4.783890626019172,
 65: 4.7766388817573215,
 66: 4.776195248793192,
 67: 4.778857712622747,
 68: 4.776661443063431,
 69: 4.772668656394232,
 70: 4.774089865591365,
 71: 4.7882018081527535,
 72: 4.699955247475348,
 73: 4.77660017828423,
 74: 4.77715585798063,
 75: 4.777499816543703,
 76: 4.7770488881432245,
 77: 4.777867512231812,
 78: 4.777721074787995,
 79: 4.7601905236099675,
 80: 4.776836398684664,
 81: 4.778708791336929,
 82: 4.777218590368816,
 83: 4.776846849918851,
 84: 4.777303057481887,
 85: 4.776987456261569,
 86: 4.777105359668355,
 87: 4.777630754531709,
 88: 4.764699455430999,
 89: 4.775692908958504,
 90: 4.77737893963624,
 91: 4.777062203405971,
 92: 4.7772172714756875,
 93: 4.776811200101928,
 94: 4.777069051651837,
 95: 4.776997454208865,
 96: 4.771277042897167,
 97: 4.777252027658723,
 98: 4.777160248946143,
 99: 4.777222655552484,
 100: 4.777202809317785}


class PODBasis:

    def __init__( self, variations = hand_picked, active_pids=[21]):

        self.central_pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)

        self.original_variations = variations
        self.nvariations= len(self.original_variations)
        self.var_pdfs   = [ lhapdf.mkPDF("250503_pod_basis_40k", var) for var in self.original_variations ]
        self.scale_c = np.array([1./max_amplitudes[var] for var in self.original_variations])
        self.active_pids = active_pids

        self.typ = "PODBasis" 
        logger.debug(f"Using {self.typ} with basis vectors {self.original_variations}")
        self.variables = [f"c{i}" for i in range(self.nvariations)]

    def evaluate(self, x: ArrayLike, id: ArrayLike, Q: ArrayLike,
                 coeffs: Sequence[float] = None, return_derivative=False) -> ArrayLike:

        mask = np.isin(id, self.active_pids).astype(float)
        central_vals = np.array([t.get(id_) for t, id_ in
                                 zip(self.central_pdf.xfxQ(tuple(x), tuple(Q)), id)])
        var_vals = np.array([[t.get(id_) for t, id_ in
                              zip(self.var_pdfs[i_var].xfxQ(tuple(x), tuple(Q)), id)]
                             for i_var in range(self.nvariations)])

        shifts = np.array(self.scale_c[:,np.newaxis]*(var_vals - central_vals)*mask)
        coeffs = np.array(coeffs)
        if not return_derivative:
            #print (central_vals.shape, central_vals)
            #print (self.scale_c.shape, self.scale_c)
            #print (coeffs.shape, coeffs)
            #print (shifts.shape, shifts) 
            return central_vals + coeffs @ shifts 
        else:
            rel = (shifts/central_vals)
            return np.moveaxis(rel, 0, -1)

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

    def product_parametrizations(self, x1, x2, id1, id2, coeffs, Q):
        """Return f(x1; c) * f(x2; c) using the same coefficient vector c."""
        return self.evaluate(x1, id1, Q=Q, coeffs=coeffs) * self.evaluate(x2, id2, Q=Q, coeffs=coeffs)

    __call__ = product_parametrizations  # allow pdf(x, coeffs)

    def derivatives(self, x1, x2, id1, id2, Q):
        """
        Compute all *relative* derivatives of
          F(c) = f(x1,id1;c) * f(x2,id2;c)
        at c=0, in the order:
          (), ('c0',),...,('c_n',), ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n').

        Returns a list of arrays aligned with self.combinations.
        """

        # relative derivatives r = (1/f0) df/dc_i, last axis = coeff index
        r1 = self.evaluate(x1, id1, Q, return_derivative=True)
        r2 = self.evaluate(x2, id2, Q, return_derivative=True)

        ones = np.ones_like(r1[..., 0])

        # first relative derivatives of the product: (F'/F) = r1 + r2
        g = r1 + r2                                      # shape (..., nvariations)

        # second relative derivatives: (1/F) d^2F/dc_a dc_b = r1_a r2_b + r1_b r2_a
        outer = r1[..., :, None] * r2[..., None, :]      # (..., a, b)
        H = outer + np.swapaxes(outer, -2, -1)           # symmetrised in (a,b)

        out = [ones]
        for k in range(self.nvariations):
            out.append(g[..., k])
        for i in range(self.nvariations):
            for j in range(i, self.nvariations):
                out.append(H[..., i, j])

        return np.array(out).transpose()


#if __name__ == "__main__":
#    pod=PODBasis(variations=range(1,101))
#    max_amplitudes = { i_var+1: max(map(abs, pod.var_pdfs[i_var].xfxQ(21, np.linspace(0.05,0.5,100),[1.65]*100))) for i_var in range(0,100)}

if __name__ == "__main__":
    import numpy as np

    basis = PODBasis()  # uses default variations and active_pids
    print("nvariations =", basis.nvariations)

    # --- Taylor reconstruction using relative derivatives ---
    # derivatives(...) returns a *list* of arrays, ordered as:
    # (), ('c0',)..('c_n',), ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n')
    def taylor_reconstruct(pdf, x1, x2, id1, id2, coeffs, Q):
        coeffs = np.asarray(coeffs)
        npar = pdf.nvariations

        derivs_list = pdf.derivatives(x1, x2, id1, id2, Q)
        derivs = np.stack(derivs_list, axis=-1)  # shape (..., M)

        # Central value F(0) with all coeffs = 0
        zero_coeffs = np.zeros(npar, dtype=float)
        F0 = pdf.product_parametrizations(x1, x2, id1, id2, zero_coeffs, Q)

        # Relative expansion F/F0
        total_rel = derivs[..., 0]  # () term, should be 1

        # First-order: ('c0',)..('c_{npar-1}',)
        offset = 1
        for k in range(npar):
            total_rel = total_rel + derivs[..., offset + k] * coeffs[k]

        # Second-order: ('c0','c0'), ('c0','c1'), ..., ('c_{npar-1}','c_{npar-1}')
        idx = offset + npar
        for i in range(npar):
            for j in range(i, npar):
                w = 0.5 if i == j else 1.0
                total_rel = total_rel + w * derivs[..., idx] * coeffs[i] * coeffs[j]
                idx += 1

        return F0 * total_rel

    # ---- Nontrivial vector test ----
    x1 = np.linspace(1e-3, 0.9, 7)
    x2 = np.linspace(0.9, 1e-3, 7)
    id1 = np.array([21, 1, 21, 2, 3, 21, 4])
    id2 = np.array([1, 21, 2, 21, 21, 3, 5])
    Q   = np.full_like(x1, 100.0, dtype=float)

    rng = np.random.default_rng(1)
    coeffs = rng.uniform(-0.2, 0.2, basis.nvariations)

    F_nom = basis.product_parametrizations(x1, x2, id1, id2, coeffs, Q)
    F_taylor = taylor_reconstruct(basis, x1, x2, id1, id2, coeffs, Q)

    print("Vector test: max |F_nom - F_taylor| =",
          float(np.max(np.abs(F_nom - F_taylor))))
    assert np.allclose(F_nom, F_taylor, rtol=1e-10, atol=1e-10)

    # ---- "Scalar" tests via length-1 arrays (to avoid tuple(float) issues) ----
    xs1, xs2 = np.array([0.2]), np.array([0.5])
    Qs = np.array([100.0])

    # gg case
    ids1, ids2 = np.array([21]), np.array([21])
    F_nom_s = basis.product_parametrizations(xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    F_taylor_s = taylor_reconstruct(basis, xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    print("Scalar gg test: |F_nom - F_taylor| =",
          float(abs(F_nom_s - F_taylor_s)))
    assert np.allclose(F_nom_s, F_taylor_s, rtol=1e-10, atol=1e-10)

    # gq case (only one active leg → linear only)
    ids1, ids2 = np.array([21]), np.array([1])
    F_nom_s2 = basis.product_parametrizations(xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    F_taylor_s2 = taylor_reconstruct(basis, xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    print("Scalar gq test: |F_nom - F_taylor| =",
          float(abs(F_nom_s2 - F_taylor_s2)))
    assert np.allclose(F_nom_s2, F_taylor_s2, rtol=1e-10, atol=1e-10)

    print("All PODBasis Taylor tests passed ✅")

