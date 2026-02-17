import project5_nm
import numpy as np
import TF
from typing import Callable, Tuple, Dict, Any

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
def compute_toroidal_objective(x):
    # Ensure x is numpy array
    # x = np.asarray(x, dtype=float)
    # call the ToroidalPeriod calculator
    # Tc is an array of computed periods.
    # Te is an array of corresponding experimental periods

    # Tc,Te = TF.ToroidalPeriods(np.array(x))

    # MATLAB Engine needs "matlab.double" type
    x_mat = matlab.double(x)

    ##Call the MATLAB function
    Tc, Te = eng.ToroidalPeriods(x_mat, nargout=2)

    # Convert MATLAB column vectors to Python arrays
    Tc = np.array(Tc).flatten()
    Te = np.array(Te).flatten()
    # print("Tc",Tc)
    # print("Te",Te)



    # Compute an objective function (this is an example).
    if np.isnan(Tc[0]):
        f=np.inf
    else:
        f=np.linalg.norm(Tc-Te)/np.linalg.norm(Te)

    # print('Objective Value = ',f)

    return float(f)


if __name__ == "__main__":

    # Initial guess
    # Set an initial decision variable vector
    # X set an initial decision variable vector
    x0=np.array([ 0.6 ,  2.6 ,  -3.6 ,  7.0 , -7.0 , 11.2 , -1.6 ,  5.0 ,
        -3.0 ,  5.6 ,  -6.4 ,  8.0 ,  5.6 , -1.0 , -4.4 ,  8.8 ,
        -18.6 , 22.2 ,  -4.8 , 10.0 ,  0.8 , -2.0 ,-17.2 , 22.4 ,
        -9.2 , 17.2 , -14.0 , 11.4 ,  1.0 , -2.2 ,  1.4 ,  6.4   ])
    n_dim = len(x0)
    # x0 = np.ones(n_dim) * 0.5  # we start away from minimum at zero just to test
    # print(compute_toroidal_objective(x0))

    print("Running Nelder-Mead on the compute_toroidal_objective function (should converge to x*).")
    res = project5_nm.nelder_mead(compute_toroidal_objective, x0, initial_step=0.2, max_iter=2000, tol_f=1e-9, tol_x=1e-8, verbose=True)
    print("Result summary:")
    print("  success:", res["success"])
    print("  message:", res["message"])
    print("  nit:", res["nit"])
    print("  x_best:", res["x_best"])
    print("  f_best:", res["f_best"])

eng.quit()
