# solve_fixed.py
# Usage: python solve_fixed.py
import numpy as np
import matlab.engine
import project5_nm_fixed
import time
import os

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates how to compute the given resonant toroidal mode
Period times (in seconds).
"""

# Import the functions necessary for the computation
# This also includes the particular modes of interest

import matlab.engine
import numpy as np
import time

eng = matlab.engine.start_matlab() # start matlab engine (it will take time at first but you do it only once)

#Add your folder to MATLAB path 
eng.addpath('d:\PhD at WSU\Math 564 Optimization\projects\project5', nargout=0)


# Example black-box: sphere function centered at 0: f(x)=||x||^2
def compute_toroidal_objective(x, verbose_on_error=True):
    # Ensure x is numpy array
    # x = np.asarray(x, dtype=float)
    # call the ToroidalPeriod calculator
    # Tc is an array of computed periods.
    # Te is an array of corresponding experimental periods

    # Tc,Te = TF.ToroidalPeriods(np.array(x))

    # MATLAB Engine needs "matlab.double" type
    # x_mat = matlab.double(x)


    # ##Call the MATLAB function
    # Tc, Te = eng.ToroidalPeriods(x_mat, nargout=2)

    # # Convert MATLAB column vectors to Python arrays
    # Tc = np.array(Tc).flatten()
    # Te = np.array(Te).flatten()
    # # print("Tc",Tc)
    # # print("Te",Te)


    # # Compute an objective function (this is an example).
    # if np.isnan(Tc[0]):
    #     f=np.inf
    # else:
    #     f=np.linalg.norm(Tc-Te)/np.linalg.norm(Te)
    #     val = np.linalg.norm(Tc - Te, ord=1) / np.linalg.norm(Te, ord=1)


    # Convert input to python list of floats (1D)
    x_list = [float(val) for val in np.asarray(x).flatten().tolist()]
    x_mat = matlab.double(x_list)  # matlab.double accepts nested lists (1D is fine)

    try:
        Tc, Te = eng.ToroidalPeriods(x_mat, nargout=2)
    except Exception as e:
        # MATLAB threw an error: log optional and return big objective
        if verbose_on_error:
            print("[MATLAB CALL ERROR] ToroidalPeriods raised exception:", e)
        return float(np.inf)

    # Convert MATLAB outputs to numpy arrays
    try:
        Tc = np.array(Tc).flatten()
        Te = np.array(Te).flatten()
    except Exception as e:
        if verbose_on_error:
            print("[CONVERSION ERROR] Could not convert Tc/Te to numpy arrays:", e)
        return float(np.inf)

    # Validate outputs: non-empty, equal length, finite numbers
    if Tc.size == 0 or Te.size == 0:
        if verbose_on_error:
            print("[BAD OUTPUT] Tc or Te is empty. Tc.size=", Tc.size, "Te.size=", Te.size)
        return float(np.inf)

    if Tc.size != Te.size:
        if verbose_on_error:
            print("[BAD OUTPUT] Tc and Te lengths differ. Tc.size=", Tc.size, "Te.size=", Te.size)
        return float(np.inf)

    if not np.isfinite(Tc).all() or not np.isfinite(Te).all():
        if verbose_on_error:
            print("[BAD OUTPUT] Non-finite values in Tc or Te.")
        return float(np.inf)

    # At this point Tc and Te look OK. Compute objective (relative L2 norm).
    denom = np.linalg.norm(Te)
    if denom == 0:
        if verbose_on_error:
            print("[BAD OUTPUT] Norm of Te is zero.")
        return float(np.inf)

    # val = np.linalg.norm(Tc - Te) / denom
    # val = np.linalg.norm(Tc - Te, ord=1) / np.linalg.norm(Te, ord=1)
    # val = np.sqrt(np.sum((w * (Tc - Te))**2)) / np.sqrt(np.sum((w * Te)**2)) # alternatively weighted L2 norm
    w = np.ones_like(Tc)
    num = np.linalg.norm(w * (Tc - Te))
    den = np.linalg.norm(w * Te)
    val = num / den


    return float(val)

    # print('Objective Value = ',f)
    # return float(f)


if __name__ == "__main__":

    # Initial guess
    # Set an initial decision variable vector
    # X set an initial decision variable vector
    x0=np.array([ 0.6 ,  2.6 ,  -3.6 ,  7.0 , -7.0 , 11.2 , -1.6 ,  5.0 ,
                -3.0 ,  5.6 ,  -6.4 ,  8.0 ,  5.6 , -1.0 , -4.4 ,  8.8 ,
                -18.6 , 22.2 ,  -4.8 , 10.0 ,  0.8 , -2.0 ,-17.2 , 22.4 ,
                -9.2 , 17.2 , -14.0 , 11.4 ,  1.0 , -2.2 ,  1.4 ,  6.4   ])
    n_dim = len(x0)
    x_init_2 = x0 * 0.5  # perturb initial guess
    x_init_3 = x0 + 0.1*np.random.randn(n_dim)  # small random perturbation

    # x0 = np.ones(n_dim) * 0.5  # we start away from minimum at zero just to test
    # print(compute_toroidal_objective(x0))

    print("Running Nelder-Mead on the compute_toroidal_objective function (should converge to x*).")
    res = project5_nm_fixed.nelder_mead(compute_toroidal_objective, x_init_3, initial_step=0.5, max_iter=5000, tol_f=1e-6, tol_x=1e-6, tol_log_vol=-250, verbose=True)
    print("Result summary:")
    print("  success:", res["success"])
    print("  message:", res["message"])
    print("  nit:", res["nit"])
    print("  x_best:", res["x_best"])
    print("x_best-np.array(x0): ", np.array(res["x_best"])-x0)
    print("  f_best:", res["f_best"])

eng.quit()
