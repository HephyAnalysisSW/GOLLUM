#!/usr/bin/env python3

import argparse
import json
import numpy as np

import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

# adjust this import to wherever your PODBasis lives
from pdf.PODBasis import PODBasis


PID = {
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
}


def xgrid(xmin, xmax, nx):
    return np.linspace(float(xmin), min(float(xmax), 1.0 - 1e-8), int(nx))


def xfx_derivatives(pod, xs, pid, q):
    """
    Returns d[x f_pid(x,Q)] / dc_i.

    PODBasis.evaluate(..., return_derivative=True) returns relative derivatives,
        (1 / x f_ref) d[x f] / dc_i,
    so we multiply by the reference xfx.
    """
    ids = np.full_like(xs, int(pid), dtype=int)
    Qs = np.full_like(xs, float(q), dtype=float)

    ref = pod.evaluate(xs, ids, Qs, coeffs=np.zeros(pod.nvariations))
    rel = pod.evaluate(xs, ids, Qs, return_derivative=True)

    return ref[:, None] * rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-set", default='250503_pod_basis_40k')
    ap.add_argument("--flavor", default='u', choices=sorted(PID))
    ap.add_argument("--xmin", type=float, default=0.3)
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=100.0)
    ap.add_argument("--nvars", type=int, default=100)
    ap.add_argument("--nx", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pid = PID[args.flavor]
    xs = xgrid(args.xmin, args.xmax, args.nx)

    pod = PODBasis(
        variations=range(1, args.nvars + 1),
        active_pids=[pid, -pid],
        var_set=args.pod_set,
        gen_pdf=None,
        rescale_pod_amplitudes=True,
    )

    d_q = xfx_derivatives(pod, xs, pid, args.q)
    d_qbar = xfx_derivatives(pod, xs, -pid, args.q)

    # Gradient of integral dx x(q - qbar)
    grad = np.trapz(d_q - d_qbar, xs, axis=0)

    # Normalize explicitly
    c = grad / np.linalg.norm(grad)

    bold_cut = 0.20       # bold components with |c_i| > 0.20
    top_n = 20

    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("Top c components:")
    for i in np.argsort(-np.abs(c))[:top_n]:
        line = (
            f"  member {i+1:3d}: "
            f"c[{i:3d}] = {c[i]: .8e}, "
            f"grad = {grad[i]: .8e}"
        )
        if abs(c[i]) >= bold_cut:
            line = BOLD + line + RESET
        print(line)

    print()
    print("Sparse normalized direction:")
    terms = []
    for i in np.argsort(-np.abs(c)):
        if abs(c[i]) < 1e-3:
            continue
        term = f"{c[i]:+.4f} e_{i+1}"
        if abs(c[i]) >= bold_cut:
            term = BOLD + term + RESET
        terms.append(term)

    print("  c = " + " ".join(terms))

    print(json.dumps(c.tolist()))

    if args.out:
        np.save(args.out, c)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
