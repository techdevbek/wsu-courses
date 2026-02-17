#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to demonstrate how to solve small mixed integer programs
using the python optimize module in the scipy package
"""

import numpy as np
import scipy.optimize as opt

# The problem we will solve is:
#
# max z = 526 x1 + 168 x2 +  74 x3 + 102 x4
# s.t.    172 x1 +  60 x2 +  96 x3 +  48 x4 <= 2950
#         144 x1 +  36 x2 +  54 x3 +  30 x4 >= 1000
#         4.4 x1 - 5.4 x2 +11.9 x3 - 4.5 x4 <= 0
#         2.4 x1 + 9.6 x2 -21.6 x3 - 2.0 x4 >= 0
#             x1 +     x2 +     x3 +     x4  = 25
#         x >= 0
#         x1,x3,x4 in R
#         x2 in Z import numpy as np

c=np.array([526, 168, 74, 102])

# Next, create the coefficient array for the inequality constraints.
# Note that the inequalities must be Ax <= b, so some sign
# changes result when converting >= into <=.
A = np.array([[  172,   60,   96,   48],\
              [ -144,  -36,  -54,  -30],\
              [  4.4, -5.4, 11.9, -4.5],\
              [ -2.4, -9.6, 21.6,    2]] )

# First build the objective vector.
# Next the right-hand-side vector for the inequalities
# Sign changes can occur here too.  
b = np.array([2950 , -1000 , 0 , 0 ])

#The coefficient matrix for the equality constraints and
# the right hand side vector.
Ae=np.ones((1,4))     # Ae = [[1,1,1,1]]
be = np.array([25])

# Next, we provide any lower and upper bound vectors, one
# value for each decision variable.  In this example all
# lower bound are zero and there are no upper bounds.
bounds=((0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf))

# Lastly, we can specify which variables are required to be integer.
# If no variables are integer then isint=[];  In our example, only x2
# is integer.
isint=[0,1,0,0]

# The call to the mixed integer solver looks like the following.
# Notice that we pass "-c" instead of "c" when we have a maximization
# problem.  This is because the solver is expecting a minimization.
res=opt.linprog(-c,A,b,Ae,be,bounds,integrality=isint)

# The result is stored in the dictionary variable "res".
# In particular, to show the optimal objective value and the
# optimal decision variable values:
print(res['fun'])
print(res['x'])
print(res)