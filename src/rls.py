"""
Recursive Least Squares (RLS) Implementation for ACC Parameter Estimation:

This script implements the Recursive Least Squares algorithm to estimate parameters 
of an Adaptive Cruise Control (ACC) model. The ACC model is a car-following model
that describes how a following vehicle adjusts its velocity based on the lead vehicle
and the space gap between them.

The model uses three parameters:
    - alpha: Sensitivity to space gap
    - beta: Sensitivity to velocity difference
    - tau: Desired time headway

The script performs the following steps:
    1. Generate synthetic data using known parameters
    2. Implement RLS to estimate parameters from simulated data
    3. Compare estimated parameters with true values
    4. Generate plots to visualize convergence and model behavior
"""

# Needed dependencies:
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from functions import * 
import os

# Create plots directory relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(plot_dir, exist_ok=True)

# Paramters from paper:
time = 900    # Total simulation steps
dt   = 0.1    # Time step in seconds

# Seed for Reproducibility:
np.random.seed(0)

"""
ACC Model Parameters

The ACC model is defined by three parameters:
    - alpha: Sensitivity to space gap (controls how strongly the vehicle responds to deviations in spacing)
    - beta: Sensitivity to velocity difference (controls how strongly the vehicle responds to speed differences)
    - tau: Desired time headway (the target time gap the vehicle tries to maintain)

The complete model is described by:
    dv/dt = alpha*(s - tau*v) + beta*(u - v)

Where:
    s = space gap between vehicles
    v = velocity of following vehicle
    u = velocity of lead vehicle
"""
# True Parameters for ACC Model:
true_theta = [0.08, 0.12, 1.5] # theta[0] = alpha, theta[1] = beta, theta[2] = tau

"""
Initial Conditions

These values set the starting state of the simulation:
    - s0: Initial space gap between vehicles (meters)
    - v0: Initial velocity of following vehicle (meters/second)
    - u0: Initial velocity of lead vehicle (meters/second)
"""
# Initial Conditions for ACC Model:
s0 = 60.0 # SI: m
v0 = 33.0 # SI: m/s (approximately 119 km/h)
u0 = 31.0 # SI: m/s (approximately 112 km/h)

"""
Physical Constraints

In real-world driving, vehicles cannot change speed instantaneously.
These constraints limit acceleration/deceleration to realistic values:
"""
# Maximum acceleration/deceleration constraint (m/s²)
dv_max = 3.0  

"""
Lead Vehicle Trajectory Generation

This section generates the velocity profile for the lead vehicle (u_t).
Two different approaches are available:

1. Curve Simulation (currently commented out):
   - Simulates a vehicle navigating a road curve
   - Sharp deceleration entering the curve (300-340 time steps)
   - Acceleration exiting the curve (340-360 time steps)
   - Random small variations elsewhere

2. Random Walk Model (currently used):
   - Adds small random increments to velocity at each time step
   - Creates more unpredictable but still realistic velocity changes
   - Uses normal distribution with mean=0, std=0.2 m/s

The velocity is preallocated as an array with dimension (time).
""" 
u_t = np.zeros(time)
u_t[0] = u0

for k in range(1, time):
    """
    # Simuulate a Curve: 
    if 300 <= k < 340:
        u_t[k] = u_t[k-1] - 2.0  # sudden deceleration entering curve
    elif 340 <= k < 360:
        u_t[k] = u_t[k-1] + 1.0  # acceleration exiting curve
    else:
        u_t[k] = u_t[k-1] + np.random.normal(0, 0.2)

    # Safety: keep velocity in physical range
    u_t[k] = np.clip(u_t[k], 0, 35)
    """
    # small random increments
    u_t[k] = u_t[k-1] + np.random.normal(loc=0, scale=0.2)

"""
Following Vehicle Trajectory Generation

This section simulates the behavior of the following vehicle according to the ACC model.
For each time step, we:

1. Calculate the change in space gap (ds):
   ds = (u_prev - v_prev) * dt
   Where u_prev and v_prev are velocities of lead and following vehicles at previous step

2. Calculate the change in velocity (dv) using the ACC model equation:
   dv = (alpha*(s_prev - tau*v_prev) + beta*(u_prev - v_prev)) * dt
   This represents the car-following behavior where the vehicle adjusts its speed based on:
     a. Deviation from desired space gap (s_prev - tau*v_prev)
     b. Velocity difference with the lead vehicle (u_prev - v_prev)

3. Apply physical constraints to velocity changes
   - Limits acceleration/deceleration to realistic values (dv_max)

The results are stored in arrays s_t (space gap) and v_t (following vehicle velocity).
"""
# Memory Allocation for s_t and v_t:
s_t = np.zeros(time)
v_t = np.zeros(time)

s_t[0] = s0
v_t[0] = v0

 # Generate sample data;
for k in range(1, time):
    s_prev = s_t[k-1]
    v_prev = v_t[k-1]
    u_prev = u_t[k-1]

    ds = (u_prev - v_prev) * dt
    dv = (true_theta[0]*(s_prev - true_theta[2]*v_prev) + true_theta[1]*(u_prev - v_prev)) * dt

    s_t[k] = s_prev + ds
    # Apply acceleration/deceleration constraint
    v_t[k] = v_prev + np.clip(dv, -dv_max * dt, dv_max * dt)


"""
Recursive Least Squares (RLS) Implementation

RLS is an adaptive filter algorithm that recursively updates parameter estimates
as new data arrives. The algorithm:

1. Uses a different parameterization (gamma) for estimation:
   - The ACC model equation dv = (alpha*(s - tau*v) + beta*(u - v))*dt can be rewritten as:
   - v[k] = gamma1*v[k-1] + gamma2*s[k-1] + gamma3*u[k-1]
   - Where gamma parameters relate to theta parameters (alpha, beta, tau) through:
      gamma1 = 1 - (alpha*tau + beta)*dt
      gamma2 = alpha*dt
      gamma3 = beta*dt

2. Later, we'll convert gamma back to theta using the invert_gamma function

The algorithm consists of:
    1. Initialize gamma_est (gamma1, gamma2, gamma3) to some initial guess
    2. Initialize P (covariance matrix) to a large value for fast initial convergence
    3. Initialize arrays to track parameter estimates over time
    4. For each time step:
       a. Compute Kalman gain K
       b. Update parameter estimates based on prediction error
       c. Update covariance matrix P
"""
gamma_est = np.array([0.9, 0.01, 0.01])  # Some initial guess for [gamma1, gamma2, gamma3]

P = np.eye(3)*1000.0 # Initial covariance matrix 

gamma_history = np.zeros((time, 3))
theta_history = np.zeros((time, 3))  # [alpha, beta, tau] at each step - 3 x 900 matrix storing all values of theta

# Initialize:
gamma_history[0] = gamma_est
theta_history[0] = invert_gamma(gamma_est, dt)

"""
RLS Algorithm Core Loop

This is the main RLS algorithm implementation that processes each time step:

1. Define measurement model: v_t[k] = gamma1*v_t[k-1] + gamma2*s_t[k-1] + gamma3*u_t[k-1]
   - Y: Observed following vehicle velocity at current time step
   - X: Input vector containing previous velocities and space gap

2. Compute Kalman gain (K):
   - Represents how much we should adjust parameters based on new measurement
   - K = (P*X) / (1 + X'*P*X)

3. Update parameter estimates:
   - gamma_est = gamma_est + K*(Y - X'*gamma_est)
   - Y - X'*gamma_est is the prediction error

4. Update covariance matrix:
   - P = P - K*(X'*P)
   - Reduces uncertainty in directions where we gain information

5. Store parameter history:
   - Convert gamma parameters back to original theta (alpha, beta, tau) using invert_gamma
"""
# RLS Algorithm:
for k in range(1, time):
    # Y = v_t[k], X = [v_t[k-1], s_t[k-1], u_t[k-1]]
    Y = v_t[k]
    X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])

    # Compute Kalman gain
    denominator = 1.0 + X @ P @ X
    K = (P @ X) / denominator

    # RLS update for gamma parameters
    gamma_est = gamma_est + K*(Y - X.dot(gamma_est))
    P = P - np.outer(K, X.dot(P))

    # Store history of parameter estimates
    gamma_history[k] = gamma_est
    theta_history[k] = invert_gamma(gamma_est, dt)


alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]
print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, true_theta[0]))
print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final, true_theta[1]))
print("Final estimated tau   = %.3f (true=%.3f)"   % (tau_est_final, true_theta[2]))


# Plot - Alpha, Beta, Tau Convergence
t_axis = np.arange(time)
fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)

params  = ["alpha", "beta", "tau"]
trueval = [true_theta[0], true_theta[1], true_theta[2]]
colors  = ["r", "g", "b"]

for i, ax in enumerate(axes):
    ax.plot(t_axis, theta_history[:, i], label=f"Estimated {params[i]}", color=colors[i])
    ax.axhline(y=trueval[i], color=colors[i], linestyle="--", label=f"True {params[i]}")
    ax.legend()
    ax.grid()
axes[-1].set_xlabel("Time step (k)")
plt.suptitle("RLS Parameter Convergence")
plt.savefig(os.path.join(plot_dir, "rls_parameter_convergence.png"))
plt.close()

# Compute MAE and MSE
errors = np.abs(theta_history - np.array([true_theta[0], true_theta[1], true_theta[2]]))
mae = np.mean(errors, axis=1)  # mean absolute error at each time step
mse = np.mean(errors**2, axis=1)  # mean squared error at each time step

"""
This portion of the code will proceed to produce the plots for:
    1. MAE and MSE 
    2. Absolute error in the theta parameters (alpha, beta, tau)
    3. All data (u_t, v_t, s_t) in one plot
    4. Velocities (lead and following) in one plot
    5. Space gap in one plot. 
"""

# Plot 1: Plot MAE and MSE over time
plt.figure(figsize=(10,4))
plt.plot(t_axis, mae, label="MAE", color="green")
plt.plot(t_axis, mse, label="MSE", color="purple")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Error Value")
plt.title("MAE and MSE of Parameter Estimates")
plt.legend()
plt.savefig(os.path.join(plot_dir, "rls_MAE_MSE.png"))
plt.close()

# Plot 2: Absolute Error in alpha, beta, tau
plt.figure(figsize=(10,4))
for i, param in enumerate(params):
    plt.semilogy(t_axis, errors[:, i], label=f"Error in {param}")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Absolute Error (log scale)")
plt.title("Convergence of RLS Parameter Estimates")
plt.legend()
plt.savefig(os.path.join(plot_dir, "rls_error_convergence.png"))
plt.close()

# Plot 3: All three plots in one figure:
plt.figure(figsize=(12,5))
plt.plot(u_t, label="Lead Vel (u)", linestyle="--", color="g")
plt.plot(v_t, label="ACC Vel (v)", color="r")
plt.plot(s_t, label="Space Gap (s)", color="b")
plt.xlabel("Time step (k)")
plt.ylabel("Value (m or m/s)")
plt.title("ACC Model Data (Synthetic)")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "ACC_model_data.png"))
plt.close()

# Plot 4: Velocities (Lead and Following)
plt.figure(figsize=(12,5))
plt.plot(u_t, label="Lead Vehicle Velocity (u_t)", linestyle="--", color="green")
plt.plot(v_t, label="Following Vehicle Velocity (v_t)", color="red")
plt.xlabel("Time step (k)")
plt.ylabel("Velocity (m/s)")
plt.title("Lead vs Following Vehicle Velocity")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "lead_following_velocity.png"))
plt.close()

# Plot 5: Space Gap
plt.figure(figsize=(12,5))
plt.plot(s_t, label="Space Gap (s_t)", color="blue")
plt.xlabel("Time step (k)")
plt.ylabel("Gap (m)")
plt.title("Space Gap Between Vehicles")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "space_gap.png"))
plt.close()
