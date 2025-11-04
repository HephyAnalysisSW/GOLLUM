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
            # for simplicity, average them:
            denom = 0.5 * (abs(actual_fwd) + abs(actual_bwd))
            hess[i, i] = (f_fwd - 2.0*f_center + f_bwd) / (denom**2)

        # else if forward is feasible, do two-step forward difference
        elif abs(actual_fwd) > 1e-14:
            # Evaluate at x + step and x + 2step
            x_f2 = x.copy()
            x_f2[i] = clamp_value(xi + 2*abs(actual_fwd), bounds[i])
            f_fwd1 = func(x_fwd)
            f_fwd2 = func(x_f2)
            hess[i, i] = (f_fwd2 - 2.0*f_fwd1 + f_center) / (actual_fwd**2)

        # else if backward is feasible, do two-step backward difference
        elif abs(actual_bwd) > 1e-14:
            x_b2 = x.copy()
            x_b2[i] = clamp_value(xi - 2*abs(actual_bwd), bounds[i])
            f_bwd1 = func(x_bwd)
            f_bwd2 = func(x_b2)
            hess[i, i] = (f_bwd2 - 2.0*f_bwd1 + f_center) / (actual_bwd**2)

        else:
            # pinned on both sides => second derivative effectively 0
            hess[i, i] = 0.0

    # --- 2) Off-diagonal terms ---
    for i in range(n):
        for j in range(i+1, n):
            # Attempt 4 corners for central difference: (++, +-, -+, --)
            # If any corner goes out of bounds, we skip => set 0.
            xi, xj = x[i], x[j]
            low_i, high_i = bounds[i]
            low_j, high_j = bounds[j]

            # Proposed corners
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] = clamp_value(xi + step, bounds[i])
            x_pp[j] = clamp_value(xj + step, bounds[j])
            x_pm[i] = clamp_value(xi + step, bounds[i])
            x_pm[j] = clamp_value(xj - step, bounds[j])
            x_mp[i] = clamp_value(xi - step, bounds[i])
            x_mp[j] = clamp_value(xj + step, bounds[j])
            x_mm[i] = clamp_value(xi - step, bounds[i])
            x_mm[j] = clamp_value(xj - step, bounds[j])

            # Actual steps after clamping
            dfwd_i = x_pp[i] - xi   # e.g. for x_pp
            dfwd_j = x_pp[j] - xj
            dback_i = x_mm[i] - xi
            dback_j = x_mm[j] - xj

            # Check if we truly have a "full" central difference in both i and j:
            # i.e. x_pp[i] != x[i], x_pp[j] != x[j], etc. ...
            # If *any* corner didn't move in a needed direction, let's skip -> 0.
            # (A simpler fallback.)
            corners_ok = True
            # Collect all corners:
            corners = [x_pp, x_pm, x_mp, x_mm]
            for corner in corners:
                if (corner[i] == x[i] and abs(step) > 1e-14) or \
                   (corner[j] == x[j] and abs(step) > 1e-14):
                    corners_ok = False
                    break

            if corners_ok:
                f_pp = func(x_pp)
                f_pm = func(x_pm)
                f_mp = func(x_mp)
                f_mm = func(x_mm)
                # Mixed partial via 4-point formula:
                # (f_pp - f_pm - f_mp + f_mm) / (4 * step^2)
                # but in case i and j steps differ slightly, let's approximate.
                # We'll just assume symmetrical step for simplicity:
                hess_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
                hess[i, j] = hess_ij
                hess[j, i] = hess_ij
            else:
                # We skip => set cross partial to 0
                hess[i, j] = 0.0
                hess[j, i] = 0.0

    return hess

