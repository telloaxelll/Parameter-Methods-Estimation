# Import Needed Dependencies: 
import numpy as np
import matplotlib.pyplot as plt
from vehicles import scenario_1_data, scenario_2_data, scenario_3_data, scenario_4_data
import os

# Creates Directory for Plots:
plot_dir = os.path.join(os.path.dirname(__file__), "plots")

# Start of Script:
np.random.seed(0)

time = 900 # total number of seconds
t_axis = np.arange(time)

dt = 1e-1 # time step difference 

s_0 = 50.0 # initial space gap (60 meters)
u_0 = 33.0 # initial lead velocity (33 m/s)
v_0 = 31.0 # initial following velocity (31 m/s)

true_theta = np.array([0.08, 0.12, 1.5])

dv_max = 3.0 # maximum acceleration/deceleration (3 m/s^2)

if __name__ == "__main__":

    """'
    Case 1: Random Walk 
    """
    scenario_1_data("NON-EQ", u_0, v_0, s_0, time, dv_max, dt, true_theta)
    scenario_1_data("EQ", u_0, v_0, s_0, time, dv_max, dt, true_theta)

    """
    Case 2: Road Curvature
    """
    scenario_2_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)

    """
    Case 3: Suburban Environment
    """
    scenario_3_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)

    """
    Case 4: Aggresive Lead Vehicle
    """    
    scenario_4_data(u_0, v_0, s_0, time, dv_max, dt, true_theta)

