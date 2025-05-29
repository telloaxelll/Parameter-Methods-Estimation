import numpy as np
import matplotlib.pyplot as plt
from functions import scenario_1_data, rls_filter, invert_gamma
import os

time = 900 # total number of samples
t_axis = np.arange(time)

dt = 1e-1 # time step

np.random.seed(0) # for reproducibility

s_0 = 50.0 # initial space gap (60 meters)
u_0 = 33.0 # initial lead velocity (33 m/s)
v_0 = 31.0 # initial following velocity (31 m/s)

true_theta = [0.08, 0.12, 1.5]

dv_max = 3.0 # maximum acceleration/deceleration (3 m/s^2)

scenario_1_data("NON-EQ")
scenario_1_data("EQ")