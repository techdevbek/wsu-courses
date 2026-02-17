"""
Script to demonstrate how to solve small mixed integer programs
using the python optimize module in the scipy package
"""
"""
All decision variables are from reals:
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt

# The problem we will solve is:
#
#    max z =  x1 + x2 +  x3 + 0 y1 + 0 y2 + 0 y3
#   s.t.      x1 +     x2 +     x3 + 0 y1 + 0 y2 + 0 y3 <= 345
#         150 x1 + 200 x2 + 240 x3 + 0 y1 + 0 y2 + 0 y3 <= 50 000 - labor cost constraints
#         275 x1 + 180 x2 + 125 x3 + 0 y1 + 0 y2 + 0 y3 <= 60 000 - material cost constraints
#
#         -1 x1 + 0 x2 + 0 x3 + 150 y1 +   0 y2 +   0 y3 <= 0
#          1 x1 + 0 x2 + 0 x3 - 345 y1 +   0 y2 +   0 y3 <= 0     # 150 y1 <= x1 <= 345 y1
#          0 x1 - 1 x2 + 0 x3 +   0 y1 +  50 y2 +   0 y3 <= 0
#          0 x1 + 1 x2 + 0 x3 +   0 y1 - 345 y2 +   0 y3 <= 0     # 50 y2 <= x2 <= 345 y2
#          0 x1 + 0 x2 - 1 x3 +   0 y1 +   0 y2 +  80 y3 <= 0
#          0 x1 + 0 x2 + 1 x3 +   0 y1 +   0 y2 - 345 y3 <= 0     # 80 y3 <= x3 <= 345 y3
#
#              0 <= y1 <= 1
#              0 <= y2 <= 1
#              0 <= y3 <= 1
#              x >= 0
#              y >= 0
#           x1, x2,x3 in Z
#           y1, y2,y3 in Z


# First build the objective vector.
c = np.array([1, 1, 1, 0, 0, 0])

# Next, create the coefficient array for the inequality constraints.
# Note that the inequalities must be Ax <= b, so some sign
# changes result when converting >= into <=.
A = np.array([[1, 1, 1, 0, 0, 0], \
              [150, 200, 240, 0, 0, 0], \
              [275, 180, 125, 0, 0, 0], \
              [-1, 0, 0, 150, 0, 0], \
              [1, 0, 0, -345, 0, 0], \
              [0, -1, 0, 0, 50, 0], \
              [0, 1, 0, 0, -345, 0], \
              [0, 0, -1, 0, 0, 80], \
              [0, 0, 1, 0, 0, -345]])

# Next the right-hand-side vector for the inequalities
# Sign changes can occur here too.
b = np.array([345, 50000, 60000, 0, 0, 0, 0, 0, 0])

# The coefficient matrix for the equality constraints and
# the right hand side vector.
Ae = None  # Ae = [[1,1,1,1]]
be = None

# Next, we provide any lower and upper bound vectors, one
# value for each decision variable.  In this example all
# lower bound are zero and there are no upper bounds.
bounds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, 1), (0, 1), (0, 1))

# Lastly, we can specify which variables are required to be integer.
# If no variables are integer then isint=[];  In our example, only x2
# is integer.
isint = [1, 1, 1, 1, 1, 1]

# The call to the mixed integer solver looks like the following.
# Notice that we pass usual "c" when we have a minimization
# problem, we send "-c" when we have max problem.
# This is because the solver is expecting a minimization.

res = opt.linprog(-c, A, b, Ae, be, bounds, integrality=isint)

# The result is stored in the dictionary variable "res".
# In particular, to show the optimal objective value and the
# optimal decision variable values:
print("min z = ", np.dot(res['x'][:5], A[0][:5]))
print(res['fun'])
print(res['x'])
print(np.dot(res['x'], A[0]))  # to verify if the answer is corret
print(np.dot(res['x'], A[1]))
print(np.dot(res['x'], A[2]))

print(res)
# print(model.computeIIS())
print(35 * 4 + 26 + 21 + 15 * 3 + 4)