
# Saved benchmark script (copy from the executed notebook cell above)
# To run: python benchmark_log1pmx.py
import numpy as np
import math
import time
import sys

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

try:
    import mpmath as mp
    MPMATH_AVAILABLE = True
except Exception:
    MPMATH_AVAILABLE = False

def weighted_sum_numpy_naive(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    tmp = np.empty_like(x)
    np.log1p(x, out=tmp)
    tmp -= x
    tmp *= w
    return np.sum(tmp, dtype=np.float64)

def _hybrid_vec(x):
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    s_small = xs*xs * (0.5 + xs*(-1/3 + xs*(1/4 + xs*(-1/5 + xs*(1/6)))))
    y[small] = -s_small
    big = ~small
    if np.any(big):
        xb = x[big]
        y[big] = np.log1p(xb) - xb
    return y

def weighted_sum_numpy_stable(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    y = _hybrid_vec(x)
    return np.sum(w * y, dtype=np.float64)

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def weighted_sum_numba(x, w):
        n = x.size
        s = 0.0
        for i in prange(n):
            xi = x[i]
            wi = w[i]
            if -1.0 < xi < 1e-4 and xi > -1.0:
                x2 = xi*xi
                t = 0.5 + xi*(-1.0/3.0 + xi*(1.0/4.0 + xi*(-1.0/5.0 + xi*(1.0/6.0))))
                y = -x2 * t
            else:
                y = math.log1p(xi) - xi
            s += wi * y
        return s

def reference_sum_per_unique_x(x, w, mp_dps=100):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    uniq, inv = np.unique(x, return_inverse=True)
    wsum = np.bincount(inv, weights=w)
    if MPMATH_AVAILABLE:
        mp.mp.dps = mp_dps
        total = mp.mpf('0')
        for u, sw in zip(uniq, wsum):
            uu = mp.mpf(str(u))
            val = mp.log1p(uu) - uu
            total += mp.mpf(str(sw)) * val
        return float(total)
    else:
        dtype_hi = np.longdouble if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps else np.float64
        total = np.longdouble(0.0) if dtype_hi is np.longdouble else 0.0
        for u, sw in zip(uniq.astype(dtype_hi), wsum.astype(dtype_hi)):
            total += sw * (np.log1p(u) - u)
        return float(total)

def make_dataset(N, seed=42):
    rng = np.random.default_rng(seed)
    x_vals = np.array([
        -0.999999, -0.99, -0.9,
        -1e-5, -1e-8, -1e-12,
        0.0, 1e-12, 1e-8, 1e-5, 1e-2, 0.1, 1.0, 10.0, 100.0
    ], dtype=np.float64)
    reps = int(np.ceil(N / len(x_vals)))
    x = np.tile(x_vals, reps)[:N].copy()
    rng.shuffle(x)
    w = rng.random(N, dtype=np.float64)
    return x, w

def benchmark(sizes=(10_000, 100_000, 1_000_000), warmup=True):
    results = []
    for N in sizes:
        x, w = make_dataset(N, seed=42)
        ref = reference_sum_per_unique_x(x, w)
        if warmup and NUMBA_AVAILABLE:
            weighted_sum_numba(x[:1000], w[:1000])
        t0 = time.perf_counter()
        res_np = weighted_sum_numpy_naive(x, w)
        t1 = time.perf_counter()
        dt_np = t1 - t0
        err_np = abs(res_np - ref) / (abs(ref) + 1e-300)
        t0 = time.perf_counter()
        res_st = weighted_sum_numpy_stable(x, w)
        t1 = time.perf_counter()
        dt_st = t1 - t0
        err_st = abs(res_st - ref) / (abs(ref) + 1e-300)
        if NUMBA_AVAILABLE:
            t0 = time.perf_counter()
            res_nb = weighted_sum_numba(x, w)
            t1 = time.perf_counter()
            dt_nb = t1 - t0
            err_nb = abs(res_nb - ref) / (abs(ref) + 1e-300)
        else:
            res_nb = float('nan')
            dt_nb = float('nan')
            err_nb = float('nan')
        results.append({
            "N": N,
            "reference": ref,
            "numpy_naive": res_np, "time_numpy_naive_s": dt_np, "rel_err_numpy_naive": err_np,
            "numpy_stable": res_st, "time_numpy_stable_s": dt_st, "rel_err_numpy_stable": err_st,
            "numba": res_nb, "time_numba_s": dt_nb, "rel_err_numba": err_nb,
        })
    return results

if __name__ == "__main__":
    sizes = (100_000, 1_000_000, 10_000_000, 100_000_000)
    results = benchmark(sizes=sizes, warmup=True)

    def fmt_time(s):
        return f"{s*1e3:7.2f} ms" if s < 2 else f"{s:7.3f} s"

    print(f"{'N':>12}  {'ref':>14}  {'numpy time':>11} {'numpy err':>10}  "
          f"{'stable time':>11} {'stable err':>10}  {'numba time':>11} {'numba err':>10}")
    print("-"*100)

    for r in results:
        n = r["N"]
        ref = r["reference"]
        t_np, e_np = r["time_numpy_naive_s"], r["rel_err_numpy_naive"]
        t_st, e_st = r["time_numpy_stable_s"], r["rel_err_numpy_stable"]
        t_nb, e_nb = r["time_numba_s"], r["rel_err_numba"]

        nb_time = "   n/a   " if np.isnan(t_nb) else fmt_time(t_nb)
        nb_err  = "   n/a   " if np.isnan(e_nb) else f"{e_nb:10.2e}"

        print(f"{n:12,d}  {ref:14.6e}  {fmt_time(t_np):>11} {e_np:10.2e}  "
              f"{fmt_time(t_st):>11} {e_st:10.2e}  {nb_time:>11} {nb_err:>10}")

