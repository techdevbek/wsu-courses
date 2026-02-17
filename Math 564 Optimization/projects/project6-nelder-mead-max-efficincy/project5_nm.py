# project5_nm.py: Nelder–Mead implementation (Colab Python cell)
#Implementation and usage examples.
import numpy as np
import math
from typing import Callable, Tuple, Dict, Any

def nelder_mead(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    initial_step: float = 0.05,
    max_iter: int = 1000,
    tol_f: float = 1e-4,
    tol_x: float = 1e-2,
    tol_volume: float = 1e-15,
    alpha: float = 1.0,    # reflection
    gamma: float = 2.0,    # expansion
    rho: float = 0.5,      # contraction
    sigma: float = 0.5,    # shrink
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Nelder-Mead optimizer.

    Parameters
    ----------
    func : callable
        Black-box objective taking a 1-D numpy array and returning a scalar.
    x0 : np.ndarray
        Initial guess vector (1-D, length n).
    initial_step : float
        Scale to create the initial simplex (relative to x0).
    max_iter : int
        Maximum number of iterations (simplex operations).
    tol_f : float
        Stopping tolerance on function values (std dev of simplex f-values).
    tol_x : float
        Stopping tolerance on simplex vertex spread (max distance).
    tol_volume : float
        Stopping tolerance on simplex volume L.
    alpha, gamma, rho, sigma : floats
        Standard NM coefficients (reflection, expansion, contraction, shrink).
    verbose : bool
        If True, prints progress occasionally.

    Returns
    -------
    dict
        result dictionary containing:
          - x_best : estimated minimizer
          - f_best : objective at x_best
          - nit : number of iterations performed
          - success : bool
          - message : status message
          - history : list of (x_simplex, f_values) occasionally (for debugging)
    """

    # Ensure x0 is numpy array
    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    # Build initial simplex: n+1 vertices
    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0.copy()
    # If x0 is zero vector, use absolute step; otherwise proportionally scale
    for i in range(1, n + 1):
        e = np.zeros(n, dtype=float)
        e[i - 1] = 1.0
        # create vertex offset along axis i-1
        delta = initial_step * (abs(x0[i - 1]) if abs(x0[i - 1]) > 1e-8 else 1.0)
        simplex[i] = x0 + delta * e

    # Evaluate objective at simplex vertices
    f_vals = np.array([func(v) for v in simplex], dtype=float)

    # Keep history occasionally for diagnostics (not too large)
    history = []
    iter_count = 0
    success = False
    message = "Max iterations reached."

    # Main NM loop
    while iter_count < max_iter:
        # Sort simplex by function values (ascending)
        idx = np.argsort(f_vals)
        simplex = simplex[idx]
        f_vals = f_vals[idx]

        x_best = simplex[0].copy()
        f_best = f_vals[0]
        x_worst = simplex[-1].copy()
        f_worst = f_vals[-1]
        x_second_worst = simplex[-2].copy()
        f_second_worst = f_vals[-2]

        # Diagnostics / stopping criteria
        f_std = float(np.std(f_vals))
        diff_from_first = simplex - simplex[0]
        x_spread = float(np.max(np.linalg.norm(diff_from_first, axis=1)))
        norm_volume_simplex = float(abs(np.linalg.det(diff_from_first[1:])/math.factorial(n))/x_spread) # computing the volume L

        if verbose and (iter_count % 50 == 0 or iter_count < 5):
            print(f"iter {iter_count:4d}: f_best = {f_best:.6e}, f_std = {f_std:.2e}, x_spread = {x_spread:.2e}")

        if f_std < tol_f and x_spread:
        # if volume_simplex < tol_volume: # alternatively check if volume L is shrinking
            success = True
            message = f"Converged ({f_std} < {tol_f}, {x_spread}< {tol_x}, tol_volume reached)."
            # message = f"Converged (tol_volume reached: {volume_simplex:.2e} < {tol_volume:.2e})."
            break

        # Compute centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_reflect = centroid + alpha * (centroid - x_worst)
        f_reflect = func(x_reflect)

        if f_reflect < f_best:
            # Expansion
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = func(x_expand)
            if f_expand < f_reflect:
                # accept expansion
                simplex[-1] = x_expand
                f_vals[-1] = f_expand
            else:
                # accept reflection
                simplex[-1] = x_reflect
                f_vals[-1] = f_reflect
        elif f_reflect < f_second_worst:
            # accept reflection (better than second worst)
            simplex[-1] = x_reflect
            f_vals[-1] = f_reflect
        else:
            # Contraction
            if f_reflect < f_worst:
                # outside contraction
                x_contract = centroid + rho * (x_reflect - centroid)
            else:
                # inside contraction
                x_contract = centroid + rho * (x_worst - centroid)
            f_contract = func(x_contract)

            if f_contract < f_worst:
                simplex[-1] = x_contract
                f_vals[-1] = f_contract
            else:
                # Shrink: move all points toward best
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = func(simplex[i])

        iter_count += 1

        # Save occasional snapshots to history
        if iter_count % 200 == 0:
            history.append((simplex.copy(), f_vals.copy()))

    # final sort
    idx = np.argsort(f_vals)
    simplex = simplex[idx]
    f_vals = f_vals[idx]
    x_best = simplex[0].copy()
    f_best = float(f_vals[0])

    return {
        "x_best": x_best,
        "f_best": f_best,
        "nit": iter_count,
        "success": success,
        "message": message,
        "history": history
    }

# ---------------------------------------------------------------------
# Demo: Test the Nelder–Mead implementation on a simple function (sphere)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example black-box: sphere function centered at 0: f(x)=||x||^2
    def sphere(x):
        # Ensure x is numpy array
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    # Initial guess
    n_dim = 5
    x0 = np.ones(n_dim) * 0.5  # we start away from minimum at zero just to test

    print("Running Nelder-Mead on the sphere function (should converge to x=0).")
    res = nelder_mead(sphere, x0, initial_step=0.2, max_iter=2000, tol_f=1e-9, tol_x=1e-8, verbose=True)
    print("Result summary:")
    print("  success:", res["success"])
    print("  message:", res["message"])
    print("  nit:", res["nit"])
    print("  x_best:", res["x_best"])
    print("  f_best:", res["f_best"])

# ---------------------------------------------------------------------
# We need to match this to our TF.py / pythontest.py to compute function to pass as func:
# ---------------------------------------------------------------------
# 1) If 'pythontest.py' has a function `compute_objective(params: np.ndarray) -> float`.
#    After uploading pythontest.py to Colab, we should do:
#
#    from pythontest import compute_objective
#
#    # wrapper that matches required signature
#    def black_box_objective(x):
#        x = np.asarray(x, dtype=float)
#        # pythontest.compute_objective should accept 1-D array and return scalar error
#        return float(compute_objective(x))
#
# 2) Then call Nelder-Mead:
#
#    x0 = np.array([ ... ])   # our chosen starting 36-dim vector
#    result = nelder_mead(black_box_objective, x0, initial_step=0.1, max_iter=5000, tol_f=1e-8, tol_x=1e-6)
#    print("Best params:", result["x_best"])
#    print("Best objective:", result["f_best"])
#
# 3) Notes on practical usage:
#    - TF computations might be moderately expensive. Use a larger initial_step if x0 is far from optimum.
#    - We need to adjust `max_iter` and tolerances to balance time vs accuracy.
#    - We should consider saving intermediate best solutions (e.g., every N iterations) if runs are long.
#
# If we have TF.py and pythontest.py, we will:
# - provide/verify the exact wrapper for `compute_objective` to pass our nelder_mead(...),
# - suggest good defaults for `initial_step`, `tol_f`, and `max_iter` for 36 parameters,
# - add logging or checkpointing hooks so you can resume long runs safely.
