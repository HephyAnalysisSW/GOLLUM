#!/usr/bin/env python3
"""
Verify the axis/sign convention relating the gen_top ntuples to
arXiv:2403.04371, independently of the simulated data.

Two implementations are run on the same randomly generated tt~ -> l+l-
events and compared bin by bin:

  (A) NTUPLE: a faithful Python transcription of the rotation/flip logic in
      gen_top/make_gen_top_ntuple.py (lines ~186-228), including ROOT's exact
      TLorentzVector::Boost / TVector3::RotateY / RotateZ conventions.

  (B) PAPER: the reference axes built directly from eq. (35) and Table 1 of
      arXiv:2403.04371 (page 8-9),
          p_p = (0,0,1),  k = top direction in the tt~ ZMF,
          y_p = p_p . k,  r_p = sqrt(1-y_p^2),
          r_hat = (p_p - y_p k)/r_p,   n_hat = (p_p x k)/r_p,
      with, from Table 1,
          a(n) = +sign(y_p) n_hat     b(n) = -sign(y_p) n_hat
          a(r) = +sign(y_p) r_hat     b(r) = -sign(y_p) r_hat
          a(k) = +k                   b(k) = -k
      and, from eqs. (36),(37), the starred axes carry an extra
      sign(D|y|) = sign(|y_t| - |y_tbar|) computed in the LAB frame;
      cos(theta_+) = lhat_+ . a,  cos(theta_-) = lhat_- . b   [eq. (30)]

The point at issue: (A) projects BOTH leptons on the same axis a, while (B)
projects the antitop lepton on b = -a. If the transcription is right, then

      cosThetaPlus_X   = +cos(theta_+)
      cosThetaMinus_X  = -cos(theta_-)

which, fed through eq. (29), is what fixes the signs used in
compare_BC_theory_sim.py.

Caveat this test does NOT probe: both implementations reach the t and tbar
rest frames via the tt~ ZMF (lab -> ZMF -> parent rest frame), matching the
ntuple. A direct lab -> parent boost differs by a Wigner rotation. The paper
says only "directions of flight in the t and tbar rest frames" (below
eq. (30)); the two-step route is the usual convention in this literature and
is what the ZMF-defined axes imply, but it is a convention, not a theorem.
"""
import numpy as np

MT = 172.5
rng = np.random.default_rng(20240304)


# ----------------------------------------------------------------------
# ROOT-equivalent primitives (TLorentzVector / TVector3 conventions)
# ----------------------------------------------------------------------
def boost(p4, b):
    """TLorentzVector::Boost(b). p4 = (px,py,pz,E)."""
    b2 = b @ b
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = b @ p4[:3]
    gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0
    out = np.empty(4)
    out[:3] = p4[:3] + gamma2 * bp * b + gamma * b * p4[3]
    out[3] = gamma * (p4[3] + bp)
    return out


def boost_vector(p4):
    return p4[:3] / p4[3]


def rotate_z(v, a):
    """TVector3::RotateZ(a): x' = c x - s y, y' = s x + c y."""
    s, c = np.sin(a), np.cos(a)
    out = v.copy()
    out[0] = c * v[0] - s * v[1]
    out[1] = s * v[0] + c * v[1]
    return out


def rotate_y(v, a):
    """TVector3::RotateY(a): z' = c z - s x, x' = s z + c x."""
    s, c = np.sin(a), np.cos(a)
    out = v.copy()
    out[2] = c * v[2] - s * v[0]
    out[0] = s * v[2] + c * v[0]
    return out


def rotate_z4(p4, a):
    out = p4.copy()
    out[:3] = rotate_z(p4[:3], a)
    return out


def rotate_y4(p4, a):
    out = p4.copy()
    out[:3] = rotate_y(p4[:3], a)
    return out


def phi_of(p4):
    return np.arctan2(p4[1], p4[0])


def theta_of(p4):
    return np.arctan2(np.hypot(p4[0], p4[1]), p4[2])


def rapidity(p4):
    return 0.5 * np.log((p4[3] + p4[2]) / (p4[3] - p4[2]))


# ----------------------------------------------------------------------
# (A) transcription of make_gen_top_ntuple.py
# ----------------------------------------------------------------------
def ntuple_impl(top, tbar, lp, lm):
    tt = top + tbar
    sign_star = (1.0 if abs(rapidity(top)) > abs(rapidity(tbar))
                 else -1.0 if abs(rapidity(top)) < abs(rapidity(tbar)) else 0.0)

    boostTT = boost_vector(tt)
    topcms = boost(top, -boostTT)
    tbarcms = boost(tbar, -boostTT)
    lpcms = boost(lp, -boostTT)
    lmcms = boost(lm, -boostTT)

    pz_tt = topcms[2]

    lpcms = boost(lpcms, -boost_vector(topcms))
    lmcms = boost(lmcms, -boost_vector(tbarcms))

    topphi = -phi_of(topcms)
    rottheta = -theta_of(topcms)
    lpcms = rotate_z4(lpcms, topphi)
    lmcms = rotate_z4(lmcms, topphi)
    lpcms = rotate_y4(lpcms, rottheta)
    lmcms = rotate_y4(lmcms, rottheta)

    lpcms[0] = -lpcms[0]
    lmcms[0] = -lmcms[0]
    if pz_tt < 0.0:
        lpcms[0], lpcms[1] = -lpcms[0], -lpcms[1]
        lmcms[0], lmcms[1] = -lmcms[0], -lmcms[1]

    A = lpcms[:3] / np.linalg.norm(lpcms[:3])
    B = lmcms[:3] / np.linalg.norm(lmcms[:3])

    return {
        'cosThetaPlus_r': A[0], 'cosThetaMinus_r': B[0],
        'cosThetaPlus_n': A[1], 'cosThetaMinus_n': B[1],
        'cosThetaPlus_k': A[2], 'cosThetaMinus_k': B[2],
        'cosThetaPlus_r_star': sign_star * A[0],
        'cosThetaMinus_r_star': sign_star * B[0],
        'cosThetaPlus_k_star': sign_star * A[2],
        'cosThetaMinus_k_star': sign_star * B[2],
    }


# ----------------------------------------------------------------------
# (B) the paper's definitions, eq. (35) + Table 1 + eq. (30)
# ----------------------------------------------------------------------
def paper_impl(top, tbar, lp, lm):
    tt = top + tbar
    boostTT = boost_vector(tt)
    topcms = boost(top, -boostTT)
    tbarcms = boost(tbar, -boostTT)

    # lepton directions in the t and tbar rest frames (reached via the ZMF)
    lp_t = boost(boost(lp, -boostTT), -boost_vector(topcms))
    lm_tbar = boost(boost(lm, -boostTT), -boost_vector(tbarcms))
    lhat_p = lp_t[:3] / np.linalg.norm(lp_t[:3])
    lhat_m = lm_tbar[:3] / np.linalg.norm(lm_tbar[:3])

    # eq. (35)
    k = topcms[:3] / np.linalg.norm(topcms[:3])
    p_p = np.array([0.0, 0.0, 1.0])
    y_p = p_p @ k
    r_p = np.sqrt(1.0 - y_p**2)
    r_hat = (p_p - y_p * k) / r_p
    n_hat = np.cross(p_p, k) / r_p

    sgn = np.sign(y_p)
    sign_star = np.sign(abs(rapidity(top)) - abs(rapidity(tbar)))

    # Table 1:  b = -a  for every axis
    a = {'n': sgn * n_hat, 'r': sgn * r_hat, 'k': k,
         'r_star': sign_star * sgn * r_hat, 'k_star': sign_star * k}
    b = {ax: -v for ax, v in a.items()}

    out = {}
    for ax in a:
        out[f'cosTheta+_{ax}'] = lhat_p @ a[ax]     # eq. (30)
        out[f'cosTheta-_{ax}'] = lhat_m @ b[ax]
    return out


# ----------------------------------------------------------------------
def random_event():
    """A kinematically valid tt~ -> l+l- configuration in the lab frame."""
    mtt = 2 * MT + rng.exponential(200.0)
    pcm = np.sqrt(mtt**2 / 4 - MT**2)
    ct = rng.uniform(-1, 1)
    st, ph = np.sqrt(1 - ct**2), rng.uniform(0, 2 * np.pi)
    pvec = pcm * np.array([st * np.cos(ph), st * np.sin(ph), ct])
    top = np.array([*pvec, mtt / 2])
    tbar = np.array([*(-pvec), mtt / 2])

    def lepton_from(parent):
        c = rng.uniform(-1, 1)
        s, p = np.sqrt(1 - c**2), rng.uniform(0, 2 * np.pi)
        d = np.array([s * np.cos(p), s * np.sin(p), c])
        l_rest = np.array([*(MT / 2 * d), MT / 2])       # in parent rest frame
        return boost(l_rest, boost_vector(parent))       # -> tt~ ZMF

    lp, lm = lepton_from(top), lepton_from(tbar)

    # boost the whole event from the tt~ ZMF to a random lab frame
    bz = rng.uniform(-0.85, 0.85)
    bt = rng.uniform(0.0, 0.15)
    bphi = rng.uniform(0, 2 * np.pi)
    blab = np.array([bt * np.cos(bphi), bt * np.sin(bphi), bz])
    return tuple(boost(v, blab) for v in (top, tbar, lp, lm))


def main():
    n = 20000
    axes = ['n', 'r', 'k', 'r_star', 'k_star']
    dplus = {ax: [] for ax in axes}
    dminus_same = {ax: [] for ax in axes}   # ntuple vs paper's cos(theta_-)
    dminus_flip = {ax: [] for ax in axes}   # ntuple vs -cos(theta_-)

    for _ in range(n):
        ev = random_event()
        A = ntuple_impl(*ev)
        P = paper_impl(*ev)
        for ax in axes:
            dplus[ax].append(A[f'cosThetaPlus_{ax}'] - P[f'cosTheta+_{ax}'])
            dminus_same[ax].append(A[f'cosThetaMinus_{ax}'] - P[f'cosTheta-_{ax}'])
            dminus_flip[ax].append(A[f'cosThetaMinus_{ax}'] + P[f'cosTheta-_{ax}'])

    print(f'{n} random events\n')
    print(f"{'axis':<9}{'max|ntuple(+) - paper cos0+|':>31}"
          f"{'max|ntuple(-) - paper cos0-|':>31}"
          f"{'max|ntuple(-) + paper cos0-|':>31}")
    for ax in axes:
        print(f'{ax:<9}{np.abs(dplus[ax]).max():>31.2e}'
              f'{np.abs(dminus_same[ax]).max():>31.2e}'
              f'{np.abs(dminus_flip[ax]).max():>31.2e}')

    ok_p = all(np.abs(dplus[ax]).max() < 1e-10 for ax in axes)
    ok_m = all(np.abs(dminus_flip[ax]).max() < 1e-10 for ax in axes)
    print()
    print(f'cosThetaPlus_X  == +cos(theta_+) : {"PASS" if ok_p else "FAIL"}')
    print(f'cosThetaMinus_X == -cos(theta_-) : {"PASS" if ok_m else "FAIL"}')

    if ok_p and ok_m:
        print("""
Consequences via eq. (29), 1/sigma d2sigma/dcos0+ dcos0-
                  = 1/4 (1 + B1 cos0+ + B2 cos0- - C cos0+ cos0-):

  B1 = 3<cos0+> = 3<cosThetaPlus>           -> B1_paper = +B1[axis]/W
  B2 = 3<cos0-> = -3<cosThetaMinus>         -> B2_paper = -B2[axis]/W
     => B1+B2 = (B1[axis] - B2[axis]) / W
  C  = -9<cos0+ cos0-> = +9<cosThetaPlus cosThetaMinus>
     and the ntuple stores C[a_b] = sum_w (-9 cosThetaPlus_a cosThetaMinus_b)
     => C_paper = -C[a_b] / W

which are exactly the signs used in compare_BC_theory_sim.py.""")


if __name__ == '__main__':
    main()
