import numpy as np
import matplotlib.pyplot as plt
from functions import *
from vehicles import scenario_1_data
import os 

# Creates Directory for Plots:
plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Predefined Parameters: 
time = 900
t_axis = np.arange(time)
dt = 1e-1 # time step
dv_max = 3.0 # maximum acceleration/deceleration (3 m/s^2)

s_0 = 50.0 # initial space gap (50 meters)
u_0 = 33.0 # initial lead velocity (33 m/s)
v_0 = 31.0 # initial following velocity (31 m/s)

true_theta = np.array([0.08, 0.12, 1.5])

np.random.seed(0)

"""
Scenario 1: Default Case
"""
scenario_1_data("NON-EQ", time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir)
scenario_1_data("EQ", time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir)

"""
Scenario 2: Added Curve
"""

