import numpy as np

def clamp_value(v, bound):
    """
    Clamp a single scalar `v` to the [low, high] interval.
    If 'low' or 'high' is None, treat it as -inf / +inf.
    """
    low, high = bound
    if low is None:
        low = -np.inf
    if high is None:
        high = np.inf
    return max(low, min(v, high))


def approximate_hessian(func, x, step, bounds):
    """
    Approximate the Hessian of `func` at point `x` using
    central finite differences, but clamp steps to stay in `bounds`.
    
    - `func(x_arr) -> float`   : objective function
    - `x`       : 1D array of parameter values
    - `step`    : nominal step size for finite differences
    - `bounds`  : list of (low, high) for each param, same order as x
    
    Returns an (n x n) NumPy array for the Hessian.
    
    Strategy:
      - For diagonal terms (i == i):
          1) Attempt full +step / -step.
          2) If out of range, reduce or zero out the step (one-sided or pinned).
      - For cross partials (i != j):
          1) Attempt all 4 corners (++, +-, -+, --).
          2) If any corner is out of bounds, fallback to 0 for that entry.
          (Simplified approach.)
    """
    n = len(x)
    hess = np.zeros((n, n), dtype=float)

    # Evaluate function at the center
    f_center = func(x)

    # --- 1) Diagonal terms ---
    for i in range(n):
        xi = x[i]
        low_i, high_i = bounds[i]

        # Step forward
        x_fwd = x.copy()
        x_fwd[i] = clamp_value(xi + step, bounds[i])
        # Step backward
        x_bwd = x.copy()
        x_bwd[i] = clamp_value(xi - step, bounds[i])

        # Effective forward/backward steps after clamping
        actual_fwd = x_fwd[i] - xi
        actual_bwd = x_bwd[i] - xi

        # If both directions are feasible (non-zero), do central difference
        if abs(actual_fwd) > 1e-14 and abs(actual_bwd) > 1e-14:
            f_fwd = func(x_fwd)
            f_bwd = func(x_bwd)
            # approximate second derivative
            # f''(x_i) ~ (f(x+step) - 2 f(x) + f(x-step)) / step^2
            # but use actual_fwd/bwd in case they differ
