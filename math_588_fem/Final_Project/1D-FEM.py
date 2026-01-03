import numpy as np
import matplotlib.pyplot as plt

def solve_tridiagonal(dim, b, diag_low, diag, diag_upper):
    w = np.zeros(dim)
    v = np.zeros(dim)
    z = np.zeros(dim)
    x = np.zeros(dim)

    for i in range(dim):
        if i == 0:
            w[i] = diag[i]
            if abs(w[i]) < 1e-12:
                raise ZeroDivisionError(f"Zero pivot at i = {i}")
            v[i] = diag_upper[i] / w[i]
            z[i] = b[i] / w[i]
        else:
            w[i] = diag[i] - diag_low[i - 1] * v[i - 1]
            if abs(w[i]) < 1e-12:
                raise ZeroDivisionError(f"Zero pivot at i = {i}")
            if i < dim - 1:
                v[i] = diag_upper[i] / w[i]
            z[i] = (b[i] - diag_low[i - 1] * z[i - 1]) / w[i]

    for i in range(dim - 1, -1, -1):
        if i == dim - 1:
            x[i] = z[i]
        else:
            x[i] = z[i] - v[i] * x[i + 1]

    return x

def main():
    nelem = 6
    nnode = nelem + 1

    x_coords = np.array([0.0, 0.1666, 0.3333, 0.4999, 0.6666, 0.8333, 1.0])
    rhs = np.ones(nnode) 
    sysm_l = np.zeros(nnode)
    sysm_d = np.zeros(nnode)
    sysm_u = np.zeros(nnode)

    pa = 0.0
    pb = 0.0

    for i in range(nelem):
        step = x_coords[i + 1] - x_coords[i]
        k = 1.0 / step
        sysm_d[i]     += k
        sysm_u[i]     += -k
        sysm_l[i]     += -k
        sysm_d[i + 1] += k

    rhs[0] = pa
    sysm_d[0] = 1.0
    sysm_u[0] = 0.0

    rhs[-1] = pb
    sysm_d[-1] = 1.0
    sysm_l[-2] = 0.0

    solution = solve_tridiagonal(nnode, rhs, sysm_l, sysm_d, sysm_u)

    with open("solution1.dat", "w") as f:
        f.write("node_id\tsolution\n")
        for i in range(nnode):
            f.write(f"{i+1}\t{solution[i]:.6f}\n")

    print("Solution has been written to solution1.dat")

    # Display contents of the solution file
    print("\nContents of solution1.dat:")
    with open("solution1.dat", "r") as f:
        print(f.read())

    # Plot the solution
    plt.plot(x_coords, solution, marker='o')
    plt.title("Solution of 1D Finite Element Problem")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.show()

