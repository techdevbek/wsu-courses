import numpy as np
import project5_nm

def efficiency(x):
    # Convert MATLAB-style array definition and element-wise division to NumPy
    # x = (x - np.array([0, 1, 1])) / np.array([5, 2, 3])
    # Assuming x is already a numpy array, and the values [0;1;1] and [5;2;3] are constants
    # The values in [0;1;1] and [5;2;3] are from the problem context, indicating x is a 3-element vector
    x = (x - np.array([0, 1, 1])) / np.array([5, 2, 3])

    # Set up hyperparameters
    N = 10
    a = 0.3
    w = 1.5
    normconst = 1.14

    # Read coefficients from 'pdata.csv' using numpy (assuming it's a flat list of numbers)
    A = np.loadtxt('pdata.csv', delimiter=',')
    # Reshape the 1D array A into a 3D array (N+1 x N+1 x N+1)
    # p = A.reshape(N + 1, N + 1, N + 1) # NOT right way, professor used the way below
    p = np.reshape(A, (N + 1, N + 1, N + 1), order="F") # professor used this way


    # Compute raw f
    f = 0
    for k1 in range(N + 1):
        for k2 in range(N + 1):
            for k3 in range(N + 1):
                if k1 + k2 + k3 <= N:
                    # Convert MATLAB 1-based indexing p(k1+1,k2+1,k3+1) to Python 0-based p[k1,k2,k3]
                    # Convert MATLAB x(1)^k1 to Python x[0]**k1
                    f = f + p[k1, k2, k3] * (x[0]**k1) * (x[1]**k2) * (x[2]**k3)

    # Adjust f to be 0 < f <= 100
    # Convert MATLAB f^2 to Python f**2
    f = a / (a + f**2)
    # Convert MATLAB np.linalg.norm(x)^2 to Python np.linalg.norm(x)**2
    r = np.linalg.norm(x)**2 / (w**2)
    f = f * np.exp(-r)
    f = f * normconst
    f = f * 100
    return -f # as the nelder_mead function is minimization optimizer
if __name__ == "__main__":
    # print(efficiency(np.array([3.19, 2.50, 2.44])))
    # print(efficiency(np.array([2.11, 2.10, 2.93])))
    # Initial guess, choosing best point so far
    # x0 = np.array([2.90, 1.73, 1.55])  # we start from the best point so far
    x0 = np.array([1, 1, 1])  # we start from an arbitrary point
    n_dim = len(x0)

    print("Running Nelder-Mead on the sphere function (should converge to x=0).")
    res = project5_nm.nelder_mead(efficiency, x0, initial_step=0.2, max_iter=2000, tol_f=1e-15, tol_x=1e-15, verbose=True)
    print("Result summary:")
    print("  success:", res["success"])
    print("  message:", res["message"])
    print("  nit:", res["nit"])
    print("  x_best:", res["x_best"])
    print("  f_best:", res["f_best"]) # as the nelder_mead function is minimization optimizer
    print("res['history']:", res['history'])
