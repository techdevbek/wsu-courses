# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:00:00 2025

basic 2-D FEM solver

@author: joseph.iannelli
"""
# ------------------------------------------------------------------------------
import numpy as np
import math  # it may be needed for some differential equations

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    # ------------------------------------------------
    #     Class objects

    gov_system = Governing_System()
    fem_system = Finite_Element_System(gov_system)
    solver = System_Solve()
    view_sol = Surface_Plot()

    # ------------------------------------------------
    # ------------------------------------------------
    # calculate solution

    fem_system.system_formation()

    solver.set_data(fem_system, gov_system)
    x = solver.solve_system()

    solver.store_solution()
    solver.solution_rows_and_columns()

    # ------------------------------------------------
    # ------------------------------------------------
    # plot solution

    view_sol.set_data_labels(["Temperature Field", "X", "Y"])
    view_sol.chart_it(solver.sol_rows_cols, solver.sol_matrix)

    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Governing_System(object):

    # -----------------------------------------------------------
    def __init__(self):

        self.set_physical_parameters()

        self.q = np.zeros(1, dtype=float)

        self.f = np.zeros(1, dtype=float)
        self.df_dq = np.zeros((1, 1), dtype=float)

    # -----------------------------------------------------------
    # ------------------------------------------------
    def set_physical_parameters(self):

        # --------------------------------------------
        alpha = np.zeros((2, 2), dtype=float)
        # --------------------------------------------
        # --------------------------------------------

        alpha[0][0] = 6.7

        alpha[1][1] = 1900.0

        alpha[0][1] = 0.0

        # --------------------------------------------
        # --------------------------------------------

        alpha_12 = alpha[0][1]
        if alpha_12 > math.sqrt(alpha[0][0] * alpha[1][1]):

            alpha_12 = 0.99 * math.sqrt(alpha[0][0] * alpha[1][1])

            print("alpha_12 adjusted to 99% of feasible maximum to: ", alpha_12)

        elif alpha_12 < - math.sqrt(alpha[0][0] * alpha[1][1]):

            alpha_12 = - 0.99 * math.sqrt(alpha[0][0] * alpha[1][1])

            print("alpha_12 adjusted to 99% of feasible minimum to: ", alpha_12)

        alpha[0][1] = alpha_12
        alpha[1][0] = alpha_12

        # --------------------------------------------
        # --------------------------------------------

        self.alpha = alpha

        # --------------------------------------------

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def source_term(self, x, j_n, j_g):

        # --------------------------------------------
        # x = self.cart_coord[ j_g ]

        alpha = self.alpha
        # --------------------------------------------
        # --------------------------------------------
        alpha_11 = alpha[0][0]
        alpha_12 = alpha[0][1];
        alpha_21 = alpha_12
        alpha_22 = alpha[1][1]

        self.source = - (alpha_11 + alpha_22) * math.sin(x[0]) * math.cos(x[1])
        self.source = - (alpha_12 + alpha_21) * math.cos(x[0]) * math.sin(x[1]) + self.source

        a = 75000.0
        b = 10.0

        r_2 = (x[0] - 0.5) ** 2.0 + (x[1] - 0.5) ** 2.0

        src = a * math.exp(- b * r_2)

        self.source = src

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def equation_and_jacobians(self, opr, x, j_n, j_g):

        # --------------------------------------------
        q = self.q
        f = self.f
        df_dq = self.df_dq
        alpha = self.alpha
        # --------------------------------------------
        # --------------------------------------------
        alpha_11 = alpha[0][0]
        alpha_12 = alpha[0][1];
        alpha_21 = alpha_12
        alpha_22 = alpha[1][1]
        # --------------------------------------------
        # --------------------------------------------

        self.source_term(x, j_n, j_g)

        f[0] = alpha_11 * opr.d2_d_x1_d_x1 * q[0] + alpha_12 * opr.d2_d_x1_d_x2 * q[0]
        f[0] = alpha_21 * opr.d2_d_x2_d_x1 * q[0] + alpha_22 * opr.d2_d_x2_d_x2 * q[0] + f[0] + opr.mass * self.source

        df_dq[0][0] = alpha_11 * opr.d2_d_x1_d_x1 + alpha_12 * opr.d2_d_x1_d_x2
        df_dq[0][0] = alpha_21 * opr.d2_d_x2_d_x1 + alpha_22 * opr.d2_d_x2_d_x2 + df_dq[0][0]

        # --------------------------------------------
        # --------------------------------------------
        # self.f = f
        # self.df_dq = df_dq
        # --------------------------------------------

        return (f, df_dq)

    # ------------------------------------------------
    # ------------------------------------------------
    def surface_integral__boundary_condition_function(self, x, j_n, j_g):

        # --------------------------------------------
        # x = self.cart_coord[ j_g ]
        alpha = self.alpha
        # --------------------------------------------
        # --------------------------------------------
        alpha_11 = alpha[0][0]
        alpha_12 = alpha[0][1];
        alpha_21 = alpha_12
        alpha_22 = alpha[1][1]

        # --------------------------------------------
        # --------------------------------------------
        # this can be any desired boundary-condition function

        self.surf_int_b_cond = alpha_11 * math.cos(x[0]) * math.cos(x[1])
        self.surf_int_b_cond = alpha_12 * (- math.sin(x[0]) * math.sin(x[1])) + self.surf_int_b_cond
        self.surf_int_b_cond = 1000.0
        # --------------------------------------------

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def b_equation_and_jacobians(self, b_opr, x, j_n, j_g):

        # --------------------------------------------
        q = self.q
        f = self.f
        df_dq = self.df_dq
        # --------------------------------------------
        # --------------------------------------------
        self.surface_integral__boundary_condition_function(x, j_n, j_g)

        f[0] = b_opr.b_mass * self.surf_int_b_cond

        df_dq[0][0] = 0.0

        # --------------------------------------------
        # --------------------------------------------
        self.f = -f
        self.df_dq = -df_dq
        # --------------------------------------------

        return (-f, -df_dq)

    # ------------------------------------------------
    # ------------------------------------------------
    def Dirichlet_boundary_condition_function(self, x):

        # these can be any desired boundary-condition function

        # f = math.sin( x[ 0 ] ) * math.cos( x[ 1 ] )

        # f = x[ 0 ]**3.0 - 3.0 * x[ 0 ] * x[ 1 ]**2.0

        # f = 3.0 * x[ 1 ] * x[ 0 ]**2.0 -  x[ 1 ]**3.0

        # f = 2.0 * x[ 0 ] * x[ 0 ] - 2.0 * x[ 1 ] * x[ 1 ]

        # f = 3.0 * x[ 1 ] +  2.0 * x[ 1 ]

        T_0 = 300.0
        T_1 = 300.0

        DT = T_1 - T_0

        f = - DT * (x[0] - 1.0) ** 2.0 + T_1

        return (f)

    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Gauss_Integral(object):

    # -----------------------------------------------------------
    def __init__(self):

        self.gauss_integration_data()

        self.integration_limits_and_parameters()

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def integration_limits_and_parameters(self):

        self.xa = -1.0;
        self.xb = 1.0

        self.ya = -1.0;
        self.yb = 1.0

        self.eps = 1.0e-12
        self.max_cycle = 10

        return

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def gauss_integration_data(self):

        self.number_of_gauss_points = 7

        self.s = (-0.9491079123427585,
                  -0.7415311855993945,
                  -0.4058451513773972,
                  0.0000000000000000,
                  0.4058451513773972,
                  0.7415311855993945,
                  0.9491079123427585)

        self.w = (0.1294849661688697,
                  0.2797053914892766,
                  0.3818300505051189,
                  0.4179591836734694,
                  0.3818300505051189,
                  0.2797053914892766,
                  0.1294849661688697)

        return

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def integral_n_in(self, funct):

        self.funct = funct

        xa = self.xa
        xb = self.xb

        ya = self.ya
        yb = self.yb

        eps = self.eps

        self.n_in = 1

        previous_integral = 0.0

        diff = 1.0

        cycle = 1

        while ((diff > eps) and (cycle <= self.max_cycle)):

            dx = (xb - xa) / float(self.n_in)
            dy = (yb - ya) / float(self.n_in)

            tot_integral = 0.0

            for i in range(self.n_in):
                self.x1 = xa + float(i) * dx
                self.x2 = xa + (float(i) + 1.0) * dx

                if (i == self.n_in - 1):
                    self.x2 = xb

                for j in range(self.n_in):

                    self.y1 = ya + float(j) * dy
                    self.y2 = ya + (float(j) + 1.0) * dy

                    if (j == self.n_in - 1):
                        self.y2 = yb

                    tot_integral = self.calculate_integral() + tot_integral

            diff = abs(tot_integral - previous_integral)
            previous_integral = tot_integral

            cycle = cycle + 1
            self.n_in = 2 * self.n_in

        return (tot_integral)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def calculate_integral(self):

        funct = self.funct

        xa = self.x1
        xb = self.x2

        ya = self.y1
        yb = self.y2

        dx = (xb - xa)
        dy = (yb - ya)

        int_x_y = 0.0

        for i in range(self.number_of_gauss_points):

            x = xa + 0.5 * dx * (self.s[i] + 1.0)

            int_y = 0.0

            for j in range(self.number_of_gauss_points):
                y = ya + 0.5 * dy * (self.s[j] + 1.0)

                f_x_y = funct(x, y)

                int_y = f_x_y * self.w[j] + int_y

            int_x_y = int_y * self.w[i] + int_x_y

        int_x_y = int_x_y * 0.5 * dx * 0.5 * dy

        return (int_x_y)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def b_integral_n_in(self, funct):

        self.funct = funct

        xa = self.xa
        xb = self.xb

        eps = self.eps

        self.n_in = 1

        previous_integral = 0.0

        diff = 1.0

        cycle = 1

        while ((diff > eps) and (cycle < self.max_cycle)):

            dx = (xb - xa) / float(self.n_in)

            tot_integral = 0.0

            for i in range(self.n_in):
                self.x1 = xa + float(i) * dx
                self.x2 = xa + (float(i) + 1.0) * dx

                if (i == self.n_in - 1):
                    self.x2 = xb

                tot_integral = self.calculate_b_integral() + tot_integral

            diff = abs(tot_integral - previous_integral)
            previous_integral = tot_integral

            cycle = cycle + 1
            self.n_in = 2 * self.n_in

        return (tot_integral)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def calculate_b_integral(self):

        funct = self.funct

        xa = self.x1
        xb = self.x2

        dx = (xb - xa)

        int_x = 0.0

        for i in range(self.number_of_gauss_points):
            x = xa + 0.5 * dx * (self.s[i] + 1.0)

            f_x = funct(x)

            int_x = f_x * self.w[i] + int_x

        int_x = int_x * 0.5 * dx

        return (int_x)
    # -----------------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Shape_Functions(object):

    # ------------------------------------------------
    def __init__(self):
        self.nodes = 4  # basic version with bilinear elements
        self.b_nodes = 2  # basic version with   linear boundary elements

        self.s_functions = np.zeros(self.nodes, dtype=float)

        self.s_functions_der = np.zeros((self.nodes, 2), dtype=float)  # two-dimensional version

        self.b_s_functions = np.zeros(self.b_nodes, dtype=float)
        self.b_s_functions_der = np.zeros((self.b_nodes, 1), dtype=float)

    # ------------------------------------------------
    # ------------------------------------------------
    def sha_fun(self, eta):
        s_f = self.s_functions

        s_f[0] = (1.0 - eta[0]) * (1.0 - eta[1]) / 4.0
        s_f[1] = (1.0 + eta[0]) * (1.0 - eta[1]) / 4.0
        s_f[2] = (1.0 + eta[0]) * (1.0 + eta[1]) / 4.0
        s_f[3] = (1.0 - eta[0]) * (1.0 + eta[1]) / 4.0

        self.s_functions = s_f

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def sha_fun_der(self, eta):
        s_f_der = self.s_functions_der

        s_f_der[0][0] = (- 1.0) * (1.0 - eta[1]) / 4.0
        s_f_der[1][0] = (1.0) * (1.0 - eta[1]) / 4.0
        s_f_der[2][0] = (1.0) * (1.0 + eta[1]) / 4.0
        s_f_der[3][0] = (- 1.0) * (1.0 + eta[1]) / 4.0

        s_f_der[0][1] = (1.0 - eta[0]) * (- 1.0) / 4.0
        s_f_der[1][1] = (1.0 + eta[0]) * (- 1.0) / 4.0
        s_f_der[2][1] = (1.0 + eta[0]) * (1.0) / 4.0
        s_f_der[3][1] = (1.0 - eta[0]) * (1.0) / 4.0

        self.s_functions_der = s_f_der

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def b_sha_fun(self, eta):
        b_s_f = self.b_s_functions

        b_s_f[0] = (1.0 - eta[0]) / 2.0
        b_s_f[1] = (1.0 + eta[0]) / 2.0

        self.b_s_functions = b_s_f

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def b_sha_fun_der(self, eta):
        b_s_f_der = self.b_s_functions_der

        b_s_f_der[0][0] = (- 1.0) / 2.0
        b_s_f_der[1][0] = (1.0) / 2.0

        self.b_s_functions_der = b_s_f_der

        return
    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Finite_Element_Matrices(object):

    # ------------------------------------------------
    def __init__(self):

        self.domain_matrices()
        self.boundary_matrices()

        self.shape_fun = Shape_Functions()
        self.g_integral = Gauss_Integral()

        self.i = None;
        self.j = None

    # ------------------------------------------------
    # ------------------------------------------------
    def domain_matrices(self):

        self.nodes = 4  # basic, for bilinear elements

        self.m200 = np.zeros((self.nodes, self.nodes), dtype=float)

        self.m201 = np.zeros((self.nodes, self.nodes), dtype=float)
        self.m202 = np.zeros((self.nodes, self.nodes), dtype=float)

        self.m210 = np.zeros((self.nodes, self.nodes), dtype=float)
        self.m220 = np.zeros((self.nodes, self.nodes), dtype=float)

        self.m211 = np.zeros((self.nodes, self.nodes), dtype=float)
        self.m212 = np.zeros((self.nodes, self.nodes), dtype=float)
        self.m221 = np.zeros((self.nodes, self.nodes), dtype=float)
        self.m222 = np.zeros((self.nodes, self.nodes), dtype=float)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def boundary_matrices(self):

        self.b_nodes = 2  # basic version with   linear boundary elements

        self.b_m200 = np.zeros((self.b_nodes, self.b_nodes), dtype=float)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def form_matrices(self):

        for i in range(self.nodes):
            self.i = i

            for j in range(self.nodes):
                self.j = j

                self.matrix = "m200"
                self.m200[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m201"
                self.m201[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m202"
                self.m202[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m210"
                self.m210[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m220"
                self.m220[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m211"
                self.m211[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m212"
                self.m212[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m221"
                self.m221[i][j] = self.g_integral.integral_n_in(self.funct)

                self.matrix = "m222"
                self.m222[i][j] = self.g_integral.integral_n_in(self.funct)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def funct(self, eta_1, eta_2):

        eta = [eta_1, eta_2]

        self.shape_fun.sha_fun(eta)
        self.shape_fun.sha_fun_der(eta)

        if (self.matrix == "m200"):

            f_x_y = self.shape_fun.s_functions[self.i] * self.shape_fun.s_functions[self.j]


        elif (self.matrix == "m201"):

            f_x_y = self.shape_fun.s_functions[self.i] * self.shape_fun.s_functions_der[self.j][0]

        elif (self.matrix == "m202"):

            f_x_y = self.shape_fun.s_functions[self.i] * self.shape_fun.s_functions_der[self.j][1]


        elif (self.matrix == "m210"):

            f_x_y = self.shape_fun.s_functions[self.j] * self.shape_fun.s_functions_der[self.i][0]

        elif (self.matrix == "m220"):

            f_x_y = self.shape_fun.s_functions[self.j] * self.shape_fun.s_functions_der[self.i][1]



        elif (self.matrix == "m211"):

            f_x_y = self.shape_fun.s_functions_der[self.i][0] * self.shape_fun.s_functions_der[self.j][0]

        elif (self.matrix == "m212"):

            f_x_y = self.shape_fun.s_functions_der[self.i][0] * self.shape_fun.s_functions_der[self.j][1]

        elif (self.matrix == "m221"):

            f_x_y = self.shape_fun.s_functions_der[self.i][1] * self.shape_fun.s_functions_der[self.j][0]

        elif (self.matrix == "m222"):

            f_x_y = self.shape_fun.s_functions_der[self.i][1] * self.shape_fun.s_functions_der[self.j][1]

        return (f_x_y)

    # ------------------------------------------------
    # ------------------------------------------------
    def form_b_matrices(self):

        for i in range(self.b_nodes):
            self.i = i

            for j in range(self.b_nodes):
                self.j = j

                self.b_matrix = "b_m200"
                self.b_m200[i][j] = self.g_integral.b_integral_n_in(self.b_funct)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def b_funct(self, eta_1):

        eta = [eta_1]

        self.shape_fun.b_sha_fun(eta)
        self.shape_fun.b_sha_fun_der(eta)

        if (self.b_matrix == "b_m200"):
            f_x = self.shape_fun.b_s_functions[self.i] * self.shape_fun.b_s_functions[self.j]

        return (f_x)

    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Finite_Element_Operators(object):

    # ------------------------------------------------
    def __init__(self):

        self.shape_fun = Shape_Functions()
        self.fem_matrices = Finite_Element_Matrices()

        self.fem_matrices.form_matrices()
        self.fem_matrices.form_b_matrices()

    # ------------------------------------------------
    # ------------------------------------------------
    def metric_data(self, fem_sys, el, eta_1, eta_2):

        row = fem_sys.el_nodes[el]

        el_cart_coord = []

        array_0 = []
        array_1 = []

        for item in row:
            array_0.append(fem_sys.cart_coord[item][0])
            array_1.append(fem_sys.cart_coord[item][1])

        el_cart_coord.append(array_0)
        el_cart_coord.append(array_1)

        eta = [eta_1, eta_2]

        self.shape_fun.sha_fun_der(eta)

        s_f_der = self.shape_fun.s_functions_der

        e_11 = 0.0
        e_12 = 0.0
        e_21 = 0.0
        e_22 = 0.0

        for i in range(fem_sys.nodes_per_elmnt):
            e_11 = s_f_der[i][1] * el_cart_coord[1][i] + e_11
            e_12 = - s_f_der[i][0] * el_cart_coord[1][i] + e_12

            e_21 = - s_f_der[i][1] * el_cart_coord[0][i] + e_21
            e_22 = s_f_der[i][0] * el_cart_coord[0][i] + e_22

        det_J = e_11 * e_22 - e_12 * e_21

        self.e_11 = e_11;
        self.e_12 = e_12
        self.e_21 = e_21;
        self.e_22 = e_22
        self.det_J = det_J

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def b_metric_data(self, fem_sys, el, eta_1):

        el_cart_coord = []

        array_0 = []
        array_1 = []

        for i in range(fem_sys.nodes_per_bnd_elmnt):
            item = fem_sys.bnd_el_nodes[el][i + 1]

            array_0.append(fem_sys.cart_coord[item][0])
            array_1.append(fem_sys.cart_coord[item][1])

        el_cart_coord.append(array_0)
        el_cart_coord.append(array_1)

        eta = [eta_1]

        self.shape_fun.b_sha_fun_der(eta)

        b_s_f_der = self.shape_fun.b_s_functions_der

        e_12 = 0.0
        e_22 = 0.0

        for i in range(fem_sys.nodes_per_bnd_elmnt):
            e_12 = b_s_f_der[i][0] * el_cart_coord[1][i] + e_12

            e_22 = b_s_f_der[i][0] * el_cart_coord[0][i] + e_22

        dG_ds = e_12 * e_12 + e_22 * e_22
        dG_ds = math.sqrt(dG_ds)

        self.dG_ds = dG_ds

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def differential_operators(self, i_n, j_n):

        e_11 = self.e_11;
        e_12 = self.e_12
        e_21 = self.e_21;
        e_22 = self.e_22
        det_J = self.det_J

        self.mass = self.fem_matrices.m200[i_n][j_n] * det_J

        self.d_d_x1 = self.fem_matrices.m201[i_n][j_n] * e_11 + self.fem_matrices.m202[i_n][j_n] * e_12
        self.d_d_x2 = self.fem_matrices.m201[i_n][j_n] * e_21 + self.fem_matrices.m202[i_n][j_n] * e_22

        self.wd_d_x1 = - self.fem_matrices.m210[i_n][j_n] * e_11 - self.fem_matrices.m220[i_n][j_n] * e_12
        self.wd_d_x2 = - self.fem_matrices.m210[i_n][j_n] * e_21 - self.fem_matrices.m220[i_n][j_n] * e_22

        self.d2_d_x1_d_x1 = self.fem_matrices.m211[i_n][j_n] * e_11 * e_11 + self.fem_matrices.m212[i_n][
            j_n] * e_11 * e_12
        self.d2_d_x1_d_x1 = self.fem_matrices.m221[i_n][j_n] * e_12 * e_11 + self.fem_matrices.m222[i_n][
            j_n] * e_12 * e_12 + self.d2_d_x1_d_x1
        self.d2_d_x1_d_x1 = - self.d2_d_x1_d_x1 / det_J

        self.d2_d_x1_d_x2 = self.fem_matrices.m211[i_n][j_n] * e_11 * e_21 + self.fem_matrices.m212[i_n][
            j_n] * e_11 * e_22
        self.d2_d_x1_d_x2 = self.fem_matrices.m221[i_n][j_n] * e_12 * e_21 + self.fem_matrices.m222[i_n][
            j_n] * e_12 * e_22 + self.d2_d_x1_d_x2
        self.d2_d_x1_d_x2 = - self.d2_d_x1_d_x2 / det_J

        self.d2_d_x2_d_x1 = self.fem_matrices.m211[i_n][j_n] * e_21 * e_11 + self.fem_matrices.m212[i_n][
            j_n] * e_21 * e_12
        self.d2_d_x2_d_x1 = self.fem_matrices.m221[i_n][j_n] * e_22 * e_11 + self.fem_matrices.m222[i_n][
            j_n] * e_22 * e_12 + self.d2_d_x2_d_x1
        self.d2_d_x2_d_x1 = - self.d2_d_x2_d_x1 / det_J

        self.d2_d_x2_d_x2 = self.fem_matrices.m211[i_n][j_n] * e_21 * e_21 + self.fem_matrices.m212[i_n][
            j_n] * e_21 * e_22
        self.d2_d_x2_d_x2 = self.fem_matrices.m221[i_n][j_n] * e_22 * e_21 + self.fem_matrices.m222[i_n][
            j_n] * e_22 * e_22 + self.d2_d_x2_d_x2
        self.d2_d_x2_d_x2 = - self.d2_d_x2_d_x2 / det_J

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def boundary_operators(self, i_n, j_n):

        dG_ds = self.dG_ds

        self.b_mass = self.fem_matrices.b_m200[i_n][j_n] * dG_ds

        return
    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Finite_Element_System(object):

    # ------------------------------------------------
    def __init__(self, syst):

        self.g_syst = syst
        self.g_syst.set_physical_parameters()
        self.alpha = self.g_syst.alpha

        self.input_data = Input_Data_File_Management()

        self.set_system_parameters()

        self.set_system_arrays()

        del self.input_data

        self.fem_oprs = Finite_Element_Operators()

    # ------------------------------------------------
    # ------------------------------------------------
    def set_system_parameters(self):

        input_data = self.input_data

        input_data.retrieve_matrix("g_parameters", 1)
        self.grid_parameters = np.array(input_data.get_data(), dtype=int)

        self.n_el = self.grid_parameters[0]
        self.n_bnd_el = self.grid_parameters[1]

        self.nodes_per_elmnt = self.grid_parameters[2]
        self.nodes_per_bnd_elmnt = self.grid_parameters[3]

        self.numbr_coupled_nodes = self.grid_parameters[4]
        self.bandwidth = self.grid_parameters[5]
        self.semi_bandwidth = self.grid_parameters[6]

        self.n_nodes = self.grid_parameters[7]

        return
        # ------------------------------------------------

    # ------------------------------------------------
    def set_system_arrays(self):

        input_data = self.input_data

        input_data.retrieve_matrix("coordinates", 2)
        self.cart_coord = np.array(input_data.get_data(), dtype=float)

        input_data.retrieve_matrix("i_connectivity", 1)
        self.i_point = np.array(input_data.get_data(), dtype=int)

        input_data.retrieve_matrix("e_connectivity", 1)
        self.el_nodes = np.array(input_data.get_data(), dtype=int)

        input_data.retrieve_matrix("b_connectivity", 1)
        self.bnd_el_nodes = np.array(input_data.get_data(), dtype=int)

        input_data.retrieve_matrix("b_conditions", 1)
        self.bnd_cond = np.array(input_data.get_data(), dtype=int)

        self.a = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        self.b = np.zeros(self.n_nodes, dtype=float)

        self.a_bnd = np.zeros((self.n_nodes, self.bandwidth + 1), dtype=float)
        self.a_locl = np.zeros((self.n_nodes, self.numbr_coupled_nodes), dtype=float)

        # system solution vector
        # self.sq = np.zeros(   self.n_nodes                , dtype = float )

        # local solution vector and jacobian. For now only 1 variable per node
        self.q = np.zeros(1, dtype=float)
        self.f = np.zeros(1, dtype=float)
        self.df_dq = np.zeros((1, 1), dtype=float)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def system_formation(self):

        for el in range(self.n_el):

            self.fem_oprs.metric_data(self, el, 0.0, 0.0)

            for i_n in range(self.nodes_per_elmnt):

                i_g = self.el_nodes[el][i_n]

                for j_n in range(self.nodes_per_elmnt):
                    j_g = self.el_nodes[el][j_n]

                    self.fem_oprs.differential_operators(i_n, j_n)

                    self.nodal_variables(j_n, j_g)

                    self.f, self.df_dq = self.g_syst.equation_and_jacobians(self.fem_oprs, self.cart_coord[j_g], j_n,
                                                                            j_g)

                    self.add_to_global_system(i_g, j_g)

        self.boundary_condition_insertion()

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def boundary_condition_insertion(self):

        self.surface_integral_boundary_conditions()

        self.Dirichlet_boundary_conditions()

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def surface_integral_boundary_conditions(self):

        for el in range(self.n_bnd_el):

            cond = self.bnd_cond[el]

            if (cond == 3):

                self.fem_oprs.b_metric_data(self, el, 0.0)

                for i_n in range(self.nodes_per_bnd_elmnt):

                    i_g = self.bnd_el_nodes[el][i_n + 1]

                    for j_n in range(self.nodes_per_bnd_elmnt):
                        j_g = self.bnd_el_nodes[el][j_n + 1]

                        self.fem_oprs.boundary_operators(i_n, j_n)

                        self.nodal_variables(j_n, j_g)

                        self.f, self.df_dq = self.g_syst.b_equation_and_jacobians(self.fem_oprs, self.cart_coord[j_g],
                                                                                  j_n, j_g)

                        self.add_to_global_system(i_g, j_g)

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def nodal_variables(self, j_n, j_g):

        # self.q[ 0 ] = self.sq[ j_g ]

        self.q[0] = 0.0  # to be updated for time-dependent case

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def add_to_global_system(self, i_g, j_g):

        self.b[i_g] = self.b[i_g] - self.f[0]

        self.a[i_g][j_g] = self.a[i_g][j_g] + self.df_dq[0][0]

        j_bnd = j_g - i_g + self.semi_bandwidth

        if (j_bnd == self.bandwidth):
            pass

        self.a_bnd[i_g][j_bnd] = self.a_bnd[i_g][j_bnd] + self.df_dq[0][0]

        for j in range(self.numbr_coupled_nodes):
            jj = self.i_point[i_g][j]

            if (jj == j_g):
                self.a_locl[i_g][j] = self.a_locl[i_g][j] + self.df_dq[0][0]
                break

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def Dirichlet_boundary_conditions(self):

        x = [0.0, 0.0]

        for el in range(self.n_bnd_el):

            cond = self.bnd_cond[el]

            if (cond == 1):

                for i_n in range(self.nodes_per_bnd_elmnt):

                    i_g = self.bnd_el_nodes[el][i_n + 1]

                    for j_g in range(self.n_nodes):
                        self.a[i_g][j_g] = 0.0

                    self.a[i_g][i_g] = 1.0

                    for j_g in range(self.bandwidth + 1):
                        self.a_bnd[i_g][j_g] = 0.0

                    self.a_bnd[i_g][self.semi_bandwidth] = 1.0

                    for j_g in range(self.numbr_coupled_nodes):
                        self.a_locl[i_g][j_g] = 0.0

                    for j in range(self.numbr_coupled_nodes):
                        jj = self.i_point[i_g][j]

                        if (jj == i_g):
                            self.a_locl[i_g][j] = 1.0
                            break

                    x[0] = self.cart_coord[i_g][0]
                    x[1] = self.cart_coord[i_g][1]

                    self.b[i_g] = self.g_syst.Dirichlet_boundary_condition_function(x)
        return
    # ------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class System_Solve(object):

    # ------------------------------------------------
    def __init__(self):

        self.output_data = Output_Data_File_Management()

        pass

    # ------------------------------------------------
    # ------------------------------------------------
    def set_data(self, obj, syst):

        self.a = obj.a
        self.b = obj.b
        self.g_syst = syst

        self.semi_bandwidth = obj.semi_bandwidth
        self.a_bnd = obj.a_bnd

        self.cart_coord = obj.cart_coord
        self.boundary_condition_function = self.g_syst.Dirichlet_boundary_condition_function

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def solve_system(self):

        bb = np.linalg.solve(self.a, self.b)

        ln = len(bb)

        # bd = self.linsolve( self.a, self.b )
        # bd = self.linsolve_b( self.semi_bandwidth,  self.a_bnd, self.b )

        norm = 0.0
        for i in range(ln):
            x = self.cart_coord[i]
            f = self.boundary_condition_function(x)
            print(i, bb[i], f)
            norm = norm + (bb[i] - f) ** 2.0

        norm = norm ** 0.5
        print("norm of difference between fem and exact solution: ", norm)

        self.sol = bb

        return (bb)

    # ------------------------------------------------
    # ------------------------------------------------
    def store_solution(self):

        number_or_rows = len(self.sol)

        matrx = np.zeros((number_or_rows, 3), dtype=float)

        for i in range(number_or_rows):
            matrx[i][0] = self.cart_coord[i][0]
            matrx[i][1] = self.cart_coord[i][1]

            matrx[i][2] = self.sol[i]

        self.output_data.set_data(matrx)
        self.output_data.save_matrix()

        self.sol_matrix = matrx

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def solution_rows_and_columns(self):

        matrx = self.sol_matrix
        number_of_rows = len(matrx)

        # print( number_of_rows ) ; input()

        n_x = 1
        n_y = 0

        count_y_rows = True

        for i in range(number_of_rows - 1):

            if count_y_rows == True:
                n_y = n_y + 1
                # print( "n_y", n_y )
                # print( )

            if matrx[i + 1][1] < matrx[i][1]:

                n_x = n_x + 1
                # print( n_x )

                if count_y_rows == True:
                    count_y_rows = False

        self.sol_rows_cols = [n_x, n_y]

        # print( self.sol_rows_cols )

        return

    # ------------------------------------------------
    # ------------------------------------------------
    def linsolve(self, a, b):

        n = len(a)

        eps = 1.0e-09
        # ------------------------------------------------------------
        for i in range(n - 1):
            """
			#-------------------------------------------------------   
            pivot = abs( a[ i ][ i ] )
            ip = -1 ;

            for k in range( i + 1, n ) :

                c = abs( a[ k ][ i ] ) 					
                if ( c > pivot ) : 
                    ip = k  



            if ( ip >= 0 ) :

                for k in range( i, n ):
                    c            = a[ i  ][ k ] 
                    a[ i  ][ k ] = a[ ip ][ k ] 
                    a[ ip ][ k ] = c 

                c       = b[ i ] 
                b[ i  ] = b[ ip ]
                b[ ip ] = c 

            elif ( pivot < eps ):				
                print( "singular matrix ", i, pivot ) 
                input()  

			#-------------------------------------------------------   
            """
            pivot = 1.0 / a[i][i]

            for k in range(i + 1, n):

                c = a[k][i] * pivot
                a[k][i] = 0.0

                for j in range(i + 1, n):
                    a[k][j] = a[k][j] - a[i][j] * c;

                b[k] = b[k] - b[i] * c;

            # ------------------------------------------------------------
            # ------------------------------------------------------------
        pivot = abs(a[n - 1][n - 1]);

        if (pivot < eps):
            # print( "singular matrix ", n, pivot )
            # return ;
            pass

        b[n - 1] = b[n - 1] / a[n - 1][n - 1];

        # for ( i = n - 1 ; i >= 1 ; i = i - 1 )
        for i in range(n - 2, -1, -1):

            c = 0.0;

            for j in range(i + 1, n):
                c = c + a[i][j] * b[j]

            b[i] = (b[i] - c) / a[i][i]

        # ------------------------------------------------------------
        return (b)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    def linsolve_b(self, nband, a, b):
        # a[][] x[] = b[]. Solution x[] is returned in b[]. a[][] is n x nbandw
        n = len(a)
        nm = (n - 1) - 1;
        nmnb = (n - 1) - nband + 1;

        for i in range(nmnb):
            # ------------ Reduce the system matrix to triangular form ---------

            ip = i + 1
            ipnb = i + nband;
            minb = nband - i;

            dd = -1.0 / a[i][nband]

            for ii in range(ip, ipnb):
                jb = ipnb - ii
                d = a[ii][jb]

                if (d != 0.0):
                    d = d * dd
                    b[ii] = b[ii] + b[i] * d
                    a[ii][jb] = 0.0

                    miinb = nband - ii;

                    for j in range(ip, ipnb):
                        jbii = j + miinb
                        jbi = j + minb
                        a[ii][jbii] = a[ii][jbii] + a[i][jbi] * d
        # ----------------------------------------------------------------------
        # -------------- Complete triangularization of the system matrix -------

        for i in range(nmnb, nm + 1):

            ip = i + 1
            ipnb = i + nband
            minb = nband - i

            dd = - 1.0 / a[i][nband]

            for ii in range(ip, n):
                jb = ipnb - ii;
                d = a[ii][jb]

                if (d != 0.0):
                    d = d * dd
                    b[ii] = b[ii] + b[i] * d
                    a[ii][jb] = 0.0

                    miinb = nband - ii

                    for j in range(ip, n):
                        jbii = j + miinb
                        jbi = j + minb
                        a[ii][jbii] = a[ii][jbii] + a[i][jbi] * d

        # ----------------------------------------------------------------------
        # ------------- Compute the solution vector by back-substitution -------

        b[n - 1] = b[n - 1] / a[n - 1][nband]

        # for ( i = nm; i >  nmnb; i = i - 1 )
        for i in range(nm, nmnb, -1):

            ip = i + 1
            minb = nband - i

            d = 0.0;

            # for( j = n; j >= ip; j = j - 1 )
            for j in range(n - 1, ip - 1, -1):
                jb = j + minb
                d = d + a[i][jb] * b[j]

            b[i] = (b[i] - d) / a[i][nband]

            # for ( i = nmnb  ; i >= 1; i = i - 1 )
        for i in range(nmnb, -1, -1):

            ip = i + 1
            minb = nband - i

            d = 0.0

            # for( j = i + nband - 1; j >= ip; j = j - 1 )
            for j in range(i + nband - 1, ip - 1, -1):
                jb = j + minb
                d = d + a[i][jb] * b[j]

            b[i] = (b[i] - d) / a[i][nband]

            # ----------------------------------------------------------------------

        return (b)
    # ----------------------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Output_Data_File_Management(object):

    # ---------------------------------------------------
    def __init__(self):

        self.mtrx = []
        self.array = []

    # ---------------------------------------------------
    # ---------------------------------------------------
    def set_data(self, matrix):

        self.array = matrix

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def save_matrix(self):

        self.to_string()
        self.store_matrix()

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def to_string(self):

        self.mtrx = []

        matrix = self.array

        mtrx = self.mtrx

        for i in range(len(matrix)):

            row = matrix[i]

            try:
                string_row = str(row[0])

                for j in range(1, len(row)):
                    string_row = string_row + ' ' + str(row[j])

            except:
                string_row = str(row)

            string_row = string_row + '\n'

            mtrx.append(string_row)

        self.mtrx = mtrx

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def store_matrix(self):

        my_array = self.mtrx

        # file_name = input( "Please, enter an output file name: ")
        # print()
        file_name = "fem_solution"
        file_name = file_name + ".txt"

        external_file = open(file_name, 'wt')

        for row in my_array:
            external_file.write(row)

        external_file.close()

        return
    # ---------------------------------------------------


# --------------------------------------------------------
# ---------------------------------------------------
class Input_Data_File_Management(object):

    # ---------------------------------------------------
    def __init__(self):

        self.mtrx = []
        self.array = []

    # ---------------------------------------------------
    # ---------------------------------------------------
    def get_data(self):

        return (self.array)

    # ---------------------------------------------------
    # ---------------------------------------------------
    def retrieve_matrix(self, file_name, indx=0):

        self.read_from_file(file_name)
        self.to_matrix()

        if (indx == 1 or indx == 2):  # 1 = integers, 2 = floating point numbers
            self.to_numbers(indx)

        if (len(self.array[0]) == 1):
            self.to_vector()

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def read_from_file(self, file_name):

        # file_name = str( input( "Please, enter an input file name: ") )
        # print()

        file_name = str(file_name) + '.txt'
        external_file = open(file_name, 'r')

        mtrx = []

        row = external_file.readline()
        while (row != ''):
            row = row.rstrip('\n')
            mtrx.append(row)
            row = external_file.readline()

        external_file.close()

        self.array = mtrx
        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def to_matrix(self):

        mtrx = self.array

        for i in range(len(mtrx)):
            mtrx[i] = mtrx[i].split()

        self.array = mtrx

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def to_vector(self):

        mtrx = self.array

        for i in range(len(mtrx)):
            mtrx[i] = mtrx[i][0]

        self.array = mtrx

        return

    # ---------------------------------------------------
    # ---------------------------------------------------
    def to_numbers(self, indx):

        mtrx = self.array

        if (indx == 1):
            func = int
        else:
            func = float

        for i in range(len(mtrx)):

            for j in range(len(mtrx[0])):
                mtrx[i][j] = func(mtrx[i][j])

        self.array = mtrx

        return
    # ---------------------------------------------------


# --------------------------------------------------------
# --------------------------------------------------------
class Surface_Plot(object):

    # ----------------------------------------------------
    def __init__(self):

        self.plt = plt

        self.data = []

        self.labels = ["Title", "x-label", "y-label"]

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def set_data_labels(self, data):

        self.labels = data

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def chart_it(self, sol_rows_cols, sol_matrix):

        self.sol_rows_cols = sol_rows_cols
        self.sol_matrix = sol_matrix

        self.i_plot = True

        self.chart_it_now()

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def chart_it_now(self):

        self.form_surface_plot_data()
        self.function_to_plot()

        while (self.i_plot == True):
            self.init_plot()
            self.plot_it()
            self.function_to_plot()

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def form_surface_plot_data(self):

        sol_rows_cols = self.sol_rows_cols
        sol_matrix = self.sol_matrix

        n_x = sol_rows_cols[0];
        n_y = sol_rows_cols[1]

        x = []
        y = []
        z = []

        for i_x in range(n_x):

            data_x = []
            data_y = []
            data_z = []

            for i_y in range(n_y):
                i_g = i_y + i_x * n_y

                print(i_x, i_y, i_g)

                data_x.append(sol_matrix[i_g][0])
                data_y.append(sol_matrix[i_g][1])
                data_z.append(sol_matrix[i_g][2])

            x.append(data_x)
            y.append(data_y)
            z.append(data_z)

        # print( np.array( x ) ) ; input()
        # print( np.array( y ) ) ; input()
        # print( np.array( z ) ) ; input()

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def function_to_plot(self):

        i_msg_1 = "Enter '1' to plot the function \n"
        i_msg_2 = "Otherwise, enter '0' to exit:  "
        i_msg = i_msg_1 + i_msg_2

        print()
        self.ifunct = int(input(i_msg))
        print()

        if self.ifunct == 0 or self.ifunct != 1:

            self.i_plot = False

        else:

            self.i_plot = True

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def init_plot(self):

        x = self.x
        y = self.y

        z = self.z

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, cmap="plasma", alpha=1.0)  # linewidth=0, antialiased = False )

        cont = ax.contourf(x, y, z, zdir='z', offset=np.min(z), cmap="plasma")

        cbar = fig.colorbar(cont)
        # cont_2 = ax.contour( x, y, z, levels = 15, colors='r')

        cbar.ax.set_ylabel('Temperature')
        # cbar.add_lines( cont_2 )

        fig.suptitle(self.labels[0])
        ax.set_xlabel(self.labels[1])
        ax.set_ylabel(self.labels[2])
        ax.set_zlabel('Z')

        return

    # ----------------------------------------------------
    # ----------------------------------------------------
    def plot_it(self):

        self.plt.show()

        return
    # ----------------------------------------------------


# --------------------------------------------------------
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------


