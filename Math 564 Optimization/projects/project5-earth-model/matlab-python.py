
import matlab.engine
import numpy as np
import time

eng = matlab.engine.start_matlab() # start matlab engine (it will take time at first but you do it only once)

#Add your folder to MATLAB path 
eng.addpath('d:\PhD at WSU\Math 564 Optimization\projects\project5', nargout=0)

# X set an initial decision variable vector
x=[ 0.6 ,  2.6 ,  -3.6 ,  7.0 , -7.0 , 11.2 , -1.6 ,  5.0 ,
   -3.0 ,  5.6 ,  -6.4 ,  8.0 ,  5.6 , -1.0 , -4.4 ,  8.8 ,
  -18.6 , 22.2 ,  -4.8 , 10.0 ,  0.8 , -2.0 ,-17.2 , 22.4 ,
   -9.2 , 17.2 , -14.0 , 11.4 ,  1.0 , -2.2 ,  1.4 ,  6.4   ]
time_now=time.time()

# MATLAB Engine needs "matlab.double" type
x_mat = matlab.double(x)

##Call the MATLAB function
Tc, Te = eng.ToroidalPeriods(x_mat, nargout=2)

# Convert MATLAB column vectors to Python arrays
Tc = np.array(Tc).flatten()
Te = np.array(Te).flatten()
# print("Tc",Tc)
# print("Te",Te)

#Compute the objective
if len(Tc) == 0:
    f = float('inf')
else:
    f = np.linalg.norm(Tc - Te) / np.linalg.norm(Te)


print("Time taken (MATLAB):", time.time() - time_now)
print("\nObjective Value =", f)

# Close the MATLAB engine
eng.quit()
