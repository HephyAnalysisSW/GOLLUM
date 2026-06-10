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

## defaults from Luca's original set
# central_set   = "NNPDF31_nnlo_as_0118"
# variation_set = "250503_pod_basis_40k"
hand_picked = [1,2,4,5,10,12,16,18,23,25,31] # this is for 250503_pod_basis_40k

# New variation_sets from Elie:
#gluon_POD_nongluon_NNPDF31_hessian
#gluon_POD_nongluon_NNPDF40
#gluon_POD_nongluon_PDF4LHC21

# Computed the maximum variations for cross-normalization of PDF effects (see __main__ block)

# Note the shift in the index. We load 1...29 and index accordingly.
#b = PODBasis(variations=range(1,30), var_set="gluon_POD_nongluon_NNPDF31_hessian")
#{ i_var+1: max(map(abs, b.var_pdfs[i_var].xfxQ(21, np.linspace(0.05,0.5,100),[1.65]*100))) for i_var in range(0,29)}

max_amplitudes = {
    "250503_pod_basis_40k": 
        {1: 159.27605180060988,
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
         100: 4.777202809317785},
    "gluon_POD_nongluon_NNPDF31_hessian": 
        {1: 105.50502783611286,
         2: 75.15048827132043,
         3: 5.93392584111249,
         4: 160.73688047372028,
         5: 43.80250120224816,
         6: 143.73774958133023,
         7: 6.202079549997558,
         8: 6.110435186056835,
         9: 23.939903800583465,
         10: 23.82484199858643,
         11: 14.337452195919795,
         12: 59.29771306125364,
         13: 5.009262917088706,
         14: 5.477067247361747,
         15: 4.505135931840468,
         16: 24.99875053037051,
         17: 12.898155212598663,
         18: 8.934137591910671,
         19: 4.947814819119008,
         20: 4.5217320500944576,
         21: 4.799032496684074,
         22: 9.171535443099632,
         23: 4.327422822382489,
         24: 4.313268957675573,
         25: 6.654057541849605,
         26: 4.664104168792884,
         27: 4.768171710142807,
         28: 4.833833373606419,
         29: 4.7789162172701545},
    "gluon_POD_nongluon_NNPDF40":
        {1: 24.542320420019735,
         2: 34.67411007502919,
         3: 24.923386331141074,
         4: 10.8821608636561,
         5: 4.3334732383543795,
         6: 2.6919035254100385,
         7: 1.648520247332358,
         8: 1.570966897288628,
         9: 1.791865458638182,
         10: 1.7374097845802325,
         11: 1.824086568570185,
         12: 1.77794069118241,
         13: 1.7840713906360666,
         14: 1.7942883296306293,
         15: 1.7908158079477656,
         16: 1.7927491370759463,
         17: 1.791789520047263,
         18: 1.7921938260068964,
         19: 1.792065090817329,
         20: 1.7919936227106756,
         21: 1.792034546805632,
         22: 1.7920448280936827,
         23: 1.7920471103670683,
         24: 1.7920317120059435,
         25: 1.7920408104283414,
         26: 1.7920333119747403,
         27: 1.7920315737563104,
         28: 1.7920322464627192,
         29: 1.7920241418048137},
    "gluon_POD_nongluon_PDF4LHC21":
        {1: 25.282624838388827,
         2: 35.72003475913883,
         3: 25.675186112964493,
         4: 11.210416982244073,
         5: 4.4641897516596,
         6: 2.7730974515263216,
         7: 1.698248027698903,
         8: 2.0738268182883552,
         9: 1.7898170547155092,
         10: 1.8460667670743365,
         11: 1.81307194933828,
         12: 1.860609859984896,
         13: 1.8378863211399714,
         14: 1.8437689069879053,
         15: 1.8473462399120364,
         16: 1.8468258037526424,
         17: 1.8458372525606552,
         18: 1.8459267222905715,
         19: 1.8461213678171042,
         20: 1.8460475978133404,
         21: 1.8460896473666168,
         22: 1.8461003710634138,
         23: 1.846077712875677,
         24: 1.8460940108022865,
         25: 1.8460963718934562,
         26: 1.8460881928016524,
         27: 1.8460864966093113,
         28: 1.8460880939349986,
         29: 1.8461019016279236},
    "gluon_POD_nongluon_ATLASpdf21": # Take the same as PDF4LHC21 as the inputs are almost identical. Maybe 6th EV is a bit different.
        {1: 25.282624838388827,
         2: 35.72003475913883,
         3: 25.675186112964493,
         4: 11.210416982244073,
         5: 4.4641897516596,
         6: 2.7730974515263216,
         7: 1.698248027698903,
         8: 2.0738268182883552,
         9: 1.7898170547155092,
         10: 1.8460667670743365,
         11: 1.81307194933828,
         12: 1.860609859984896,
         13: 1.8378863211399714,
         14: 1.8437689069879053,
         15: 1.8473462399120364,
         16: 1.8468258037526424,
         17: 1.8458372525606552,
         18: 1.8459267222905715,
         19: 1.8461213678171042,
         20: 1.8460475978133404,
         21: 1.8460896473666168,
         22: 1.8461003710634138,
         23: 1.846077712875677,
         24: 1.8460940108022865,
         25: 1.8460963718934562,
         26: 1.8460881928016524,
         27: 1.8460864966093113,
         28: 1.8460880939349986,
         29: 1.8461019016279236},
    }

#max_amplitudes = {} 
#{'gluon_POD_nongluon_PDF4LHC21': {1: 93.5975626622538, 2: 249.21241221809504, 3: 339.37005603497437, 4: 264.02632535099633, 5: 225.30767182488103, 6: 168.4700626402868, 7: 99.9671081559107, 8: 63.239975107467714, 9: 40.81468668723638, 10: 1.0, 11: 20.95482103186426, 12: 10.976756475229259, 13: 5.687165558552786, 14: 3.0338760257228805, 15: 1.4341920895518476, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0}}

class PODBasis:

    all_pdg_ids = [21, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    def __init__( self, variations = hand_picked, active_pids="all", 
            #reference_pdf = None, 
            x_max = None, #0.6,
            rescale_pod_amplitudes=True, 
            gen_pdf = "NNPDF31_nnlo_as_0118", var_set = "gluon_POD_nongluon_PDF4LHC21"):

        self.original_variations = variations
        self.nvariations= len(self.original_variations)
        self.var_set = var_set
        self.x_max = x_max
        self.rescale_pod_amplitudes = rescale_pod_amplitudes
        self.var_pdfs   = [ lhapdf.mkPDF(self.var_set, var) for var in self.original_variations ]
        self.scale_c = None
        if rescale_pod_amplitudes: 
            try:
                self.scale_c = np.array([1./max_amplitudes[self.var_set][var] for var in self.original_variations])
                print("Max_amplitudes used.")
            except KeyError as e:
                print("Max_amplitudes not used.")
        else:
            print("Max_amplitudes not used.")

        #if reference_pdf:
        #    self.reference_pdf_name = reference_pdf
        #    self.reference_pdf = lhapdf.mkPDF(reference_pdf, 0)
        #else:
        self.reference_pdf_name = self.var_set
        self.reference_pdf = lhapdf.mkPDF(self.var_set, 0)

        if gen_pdf:
            self.gen_pdf_name = gen_pdf
            self.gen_pdf = lhapdf.mkPDF(gen_pdf, 0)
        else:
            self.gen_pdf_name = self.reference_pdf_name
            self.gen_pdf = self.reference_pdf
            print("Warning! No gen pdf set. Using the reference PDF")

        self.reference_equals_gen = (self.reference_pdf_name == self.gen_pdf_name)

        if active_pids == "all":
            self.active_pids = PODBasis.all_pdg_ids
        else:
            self.active_pids = active_pids

        print ("Active PIDs", self.active_pids)

        logger.debug(f"Using PODBasis with basis vectors {self.original_variations}")
        logger.debug(f"Using reference PDF {self.reference_pdf_name}")
        logger.debug(f"Using generator PDF {self.gen_pdf_name}")
        self.variables = [f"c{i}" for i in range(self.nvariations)]

    def evaluate(self, x: ArrayLike, id: ArrayLike, Q: ArrayLike,
                 coeffs: Sequence[float] = None, return_derivative=False) -> ArrayLike:

        mask = np.isin(id, self.active_pids)

        if self.x_max is not None:
            mask &= (x<self.x_max)

        mask = mask.astype(float)

        reference_vals = np.array([t.get(id_) for t, id_ in
                                   zip(self.reference_pdf.xfxQ(tuple(x), tuple(Q)), id)])
        var_vals = np.array([[t.get(id_) for t, id_ in
                              zip(self.var_pdfs[i_var].xfxQ(tuple(x), tuple(Q)), id)]
                             for i_var in range(self.nvariations)])

        if self.scale_c is not None:
            shifts = np.array(self.scale_c[:,np.newaxis]*(var_vals - reference_vals)*mask)
        else:
            shifts = np.array((var_vals - reference_vals)*mask)

        if not return_derivative:
            if coeffs is None:
                coeffs = np.zeros(self.nvariations)
            coeffs = np.array(coeffs)
            return reference_vals + coeffs @ shifts
        else:
            rel = (shifts/reference_vals)
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
        """Return [f(x1; c) / f_gen(x1)] * [f(x2; c) / f_gen(x2)] using the same coefficient vector c."""

        f1 = self.evaluate(x1, id1, Q=Q, coeffs=coeffs)
        f2 = self.evaluate(x2, id2, Q=Q, coeffs=coeffs)

        reference_vals_1 = np.array([t.get(id_) for t, id_ in
                                     zip(self.reference_pdf.xfxQ(tuple(x1), tuple(Q)), id1)])
        reference_vals_2 = np.array([t.get(id_) for t, id_ in
                                     zip(self.reference_pdf.xfxQ(tuple(x2), tuple(Q)), id2)])

        if self.reference_equals_gen:
            ref_over_gen_1 = np.ones_like(reference_vals_1)
            ref_over_gen_2 = np.ones_like(reference_vals_2)
        else:
            gen_vals_1 = np.array([t.get(id_) for t, id_ in
                                   zip(self.gen_pdf.xfxQ(tuple(x1), tuple(Q)), id1)])
            gen_vals_2 = np.array([t.get(id_) for t, id_ in
                                   zip(self.gen_pdf.xfxQ(tuple(x2), tuple(Q)), id2)])

            ref_over_gen_1 = reference_vals_1/gen_vals_1
            ref_over_gen_2 = reference_vals_2/gen_vals_2

            warning_threshold = 10.0
            if np.any(np.abs(ref_over_gen_1*ref_over_gen_2) > warning_threshold):
                logger.warning(
                    "Large reference/gen PDF ratios encountered. "
                    "max |(f_ref/f_gen)(x1) * (f_ref/f_gen)(x2)| = %s",
                    np.max(np.abs(ref_over_gen_1*ref_over_gen_2))
                )

        return (f1/reference_vals_1) * (f2/reference_vals_2) * ref_over_gen_1 * ref_over_gen_2

    __call__ = product_parametrizations  # allow pdf(x, coeffs)

    def derivatives(self, x1, x2, id1, id2, Q):
        """
        Compute all *relative* derivatives of
          F(c) = [f(x1,id1;c) / f_gen(x1,id1)] * [f(x2,id2;c) / f_gen(x2,id2)]
        at c=0, in the order:
          (), ('c0',),...,('c_n',), ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n').

        Returns a list of arrays aligned with self.combinations.
        """

        # relative derivatives r = (1/f_ref) df/dc_i, last axis = coeff index
        r1 = self.evaluate(x1, id1, Q, return_derivative=True)
        r2 = self.evaluate(x2, id2, Q, return_derivative=True)

        ones = np.ones_like(r1[..., 0])

        # first relative derivatives of the product ratio: (F'/F) = r1 + r2
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

    def print(self):
        print("PODBasis")
        print("  var_set              =", self.var_set)
        print("  reference_pdf        =", self.reference_pdf_name)
        print("  gen_pdf              =", self.gen_pdf_name)
        print("  reference_equals_gen =", self.reference_equals_gen)
        print("  nvariations          =", self.nvariations)
        print("  variations           =", list(self.original_variations))
        print("  active_pids          =", self.active_pids)
        print("  variables            =", self.variables)
        print("  scale_c is None      =", self.scale_c is None)
        if self.scale_c is not None:
            print("  scale_c.shape        =", self.scale_c.shape)

if __name__ == "__main__":
    import numpy as np

    pod = PODBasis(
        variations=[1,2,3,4,5,6,7,8,9],
        active_pids="all",
        #reference_pdf=None, #"NNPDF31_nnlo_as_0118",
        gen_pdf="NNPDF31_nnlo_as_0118",
        #var_set="gluon_POD_nongluon_PDF4LHC21"
        var_set="gluon_POD_nongluon_ATLASpdf21"
    )
    print("nvariations =", pod.nvariations)

    # --- Taylor reconstruction using relative derivatives ---
    # derivatives(...) returns a *list* of arrays, ordered as:
    # (), ('c0',)..('c_n',), ('c0','c0'), ('c0','c1'), ..., ('c_n','c_n')
    def taylor_reconstruct(pdf, x1, x2, id1, id2, coeffs, Q):
        coeffs = np.asarray(coeffs)
        npar = pdf.nvariations

        derivs_list = pdf.derivatives(x1, x2, id1, id2, Q).transpose()
        derivs = np.stack(derivs_list, axis=-1)  # shape (..., M)

        # Reference value F(0)
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
    coeffs = rng.uniform(-0.2, 0.2, pod.nvariations)

    R_nom = pod(x1, x2, id1, id2, coeffs, Q)
    R_taylor = taylor_reconstruct(pod, x1, x2, id1, id2, coeffs, Q)

    print("Vector test: max |R_nom - R_taylor| =",
          float(np.max(np.abs(R_nom - R_taylor))))
    assert np.allclose(R_nom, R_taylor, rtol=1e-10, atol=1e-10)

    # ---- "Scalar" tests via length-1 arrays (to avoid tuple(float) issues) ----
    xs1, xs2 = np.array([0.2]), np.array([0.5])
    Qs = np.array([100.0])

    # gg case
    ids1, ids2 = np.array([21]), np.array([21])
    R_nom_s = pod(xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    R_taylor_s = taylor_reconstruct(pod, xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    print("Scalar gg test: |R_nom - R_taylor| =",
          float(abs(R_nom_s - R_taylor_s)))
    assert np.allclose(R_nom_s, R_taylor_s, rtol=1e-10, atol=1e-10)

    # gq case (only one active leg → linear only)
    ids1, ids2 = np.array([21]), np.array([1])
    R_nom_s2 = pod.product_parametrizations(xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    R_taylor_s2 = taylor_reconstruct(pod, xs1, xs2, ids1, ids2, coeffs, Qs)[0]
    print("Scalar gq test: |R_nom - R_taylor| =",
          float(abs(R_nom_s2 - R_taylor_s2)))
    assert np.allclose(R_nom_s2, R_taylor_s2, rtol=1e-10, atol=1e-10)

    print("All PODBasis Taylor tests passed ✅")



    def make_max_amplitudes(var_set,
                                       nvars=29,
                                       reference_pdf=None,
                                       pid=21,
                                       x_min=1e-4,
                                       x_max=0.8,
                                       n_x_log=250,
                                       n_x_lin=250,
                                       q_values=(1.65, 5.0, 10.0, 30.0, 100.0),
                                       quantile=0.995,
                                       floor=1e-12):
        import numpy as np
        import lhapdf

        if reference_pdf is None:
            reference_pdf = var_set

        ref_pdf = lhapdf.mkPDF(reference_pdf, 0)
        var_pdfs = [lhapdf.mkPDF(var_set, i) for i in range(1, nvars + 1)]

        x_log = np.geomspace(x_min, min(1e-2, x_max), n_x_log)
        x_lin = np.linspace(max(1e-2, x_min), x_max, n_x_lin)
        xx = np.unique(np.concatenate((x_log, x_lin)))

        out = {}
        ids = np.full(len(xx), pid, dtype=int)

        for i_var, var_pdf in enumerate(var_pdfs, start=1):
            rel_all = []

            for Q0 in q_values:
                QQ = np.full(len(xx), Q0)

                ref = np.array([t.get(i) for t, i in zip(ref_pdf.xfxQ(tuple(xx), tuple(QQ)), ids)])
                var = np.array([t.get(i) for t, i in zip(var_pdf.xfxQ(tuple(xx), tuple(QQ)), ids)])

                good = np.abs(ref) > floor
                if np.any(good):
                    rel = np.abs((var[good] - ref[good]) / ref[good])
                    rel_all.append(rel)

            if len(rel_all) == 0:
                out[i_var] = 1.0
            else:
                rel_all = np.concatenate(rel_all)
                amp = float(np.quantile(rel_all, quantile))
                out[i_var] = max(amp, 1.0)

        return {var_set: out}

    max_amplitudes = make_max_amplitudes(
        var_set="gluon_POD_nongluon_PDF4LHC21",
        nvars=29,
        reference_pdf="gluon_POD_nongluon_PDF4LHC21",
        q_values=(1.65, 10.0, 30.0, 100.0),
        x_max=0.8,
        quantile=0.995,
    )
    print(max_amplitudes)
