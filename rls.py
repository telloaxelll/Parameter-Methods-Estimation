# Necessary Dependencies:
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from functions import * 
import os

# Create directory for plots
if not os.path.exists("plots"):
    os.makedirs("plots", exist_ok=True)

# Parameters Taken from Paper:
time = 900
dt   = 0.1

# Set Seed for Reproducibility:
np.random.seed(0)

# True parameters for ACC Model:
theta = [0.08, 0.12, 1.5] # theta[0] = alpha, theta[1] = beta, theta[2] = tau

# Initial Conditions for ACC Model:
s0 = 60.0 # SI: m
v0 = 33.0 # SI: m/s
u0 = 31.0 # SI: m/s

"""
This portion of the code will preallocate arrays of dimension (time) for 
all of the vectors that we will be working with in the ACC model.
    1. Allocates vector for u_t (lead) and generates the rest of the trajectories of u_t (mean = 0, variance = 0.2)
        1.5. Will simulate a curve in the velocity of the leading vehicle
    2. 
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
Generate data for Space Gap and Velocity of Following vehicle:
   1. Allocate vector for s_t, v_t
   2. Generate and append values into array using the ACC model equations for s_t and v_t
"""
# Memory allocation for s_t and v_t:
s_t = np.zeros(time)
v_t = np.zeros(time)

s_t[0] = s0
v_t[0] = v0


for k in range(1, time):
    s_prev = s_t[k-1]
    v_prev = v_t[k-1]
    u_prev = u_t[k-1]

    ds = (u_prev - v_prev) * dt
    dv = (theta[0]*(s_prev - theta[2]*v_prev) + theta[1]*(u_prev - v_prev)) * dt

    s_t[k] = s_prev + ds
    v_t[k] = v_prev + dv


"""
RLS Initialization & Implementation: 
    1. Initialize gamma_est (gamma1, gamma2, gamma3) to some initial guess
    2. Initialize P (covariance matrix) to a large value
    3. Initialize gamma_history and theta_history arrays to store values at each time step
"""
gamma_est = np.array([0.9, 0.01, 0.01])  # Some initial guess for [gamma1, gamma2, gamma3]
P = np.eye(3)*1000.0 # Initial covariance matrix 

gamma_history = np.zeros((time, 3))
theta_history = np.zeros((time, 3))  # [alpha, beta, tau] at each step - 3 x 900 matrix storing all values of theta

# Initialize:
gamma_history[0] = gamma_est
theta_history[0] = invert_gamma(gamma_est, dt)

# RLS Algorithm:
for k in range(1, time):
    # Y = v_t[k], X = [v_t[k-1], s_t[k-1], u_t[k-1]]
    Y = v_t[k]
    X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])

    denominator = 1.0 + X @ P @ X
    K = (P @ X) / denominator

    # RLS update for gamma
    gamma_est = gamma_est + K*(Y - X.dot(gamma_est))
    P = P - np.outer(K, X.dot(P))

    gamma_history[k] = gamma_est
    theta_history[k] = invert_gamma(gamma_est, dt)


# Comparison between final alpha,beta,tau with ground truth
# Compute absolute errors for each parameter over time
errors = np.abs(theta_history - np.array([theta[0], theta[1], theta[2]]))
alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]
print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, theta[0]))
print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final, theta[1]))
print("Final estimated tau   = %.3f (true=%.3f)"   % (tau_est_final, theta[2]))


# Plot - Alpha, Beta, Tau Convergence
t_axis = np.arange(time)
fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)

params  = ["alpha", "beta", "tau"]
trueval = [theta[0], theta[1], theta[2]]
colors  = ["r", "g", "b"]

for i, ax in enumerate(axes):
    ax.plot(t_axis, theta_history[:, i], label=f"Estimated {params[i]}", color=colors[i])
    ax.axhline(y=trueval[i], color=colors[i], linestyle="--", label=f"True {params[i]}")
    ax.legend()
    ax.grid()
axes[-1].set_xlabel("Time step (k)")
plt.suptitle("RLS Parameter Convergence")
plt.savefig("plots/rls_estimation_convergence_gamma.png")

# Compute MAE and MSE
mae = np.mean(errors, axis=1)  # mean absolute error at each time step
mse = np.mean(errors**2, axis=1)  # mean squared error at each time step

# Plot 1: Plot MAE and MSE over time
plt.figure(figsize=(10,4))
plt.plot(t_axis, mae, label="MAE", color="green")
plt.plot(t_axis, mse, label="MSE", color="purple")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Error Value")
plt.title("MAE and MSE of Parameter Estimates")
plt.legend()
plt.savefig("plots/rls_mae_mse_gamma.png")

# Plot 2: Absolute Error in alpha, beta, tau
errors = np.abs(theta_history - np.array([theta[0], theta[1], theta[2]]))
plt.figure(figsize=(10,4))
for i, param in enumerate(params):
    plt.semilogy(t_axis, errors[:, i], label=f"Error in {param}")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Absolute Error (log scale)")
plt.title("Convergence of RLS Parameter Estimates")
plt.legend()
plt.savefig("plots/rls_estimation_error_gamma.png")

"""
Plot - Plots lead vehicle velocity (u_t), following vehicle velocity (v_t),
       and space gap (s_t) over time.
"""
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
plt.savefig("plots/acc_model_data.png")


# Plot 4: Velocities (Lead and Following)
plt.figure(figsize=(12,5))
plt.plot(u_t, label="Lead Vehicle Velocity (u_t)", linestyle="--", color="green")
plt.plot(v_t, label="Following Vehicle Velocity (v_t)", color="red")
plt.xlabel("Time step (k)")
plt.ylabel("Velocity (m/s)")
plt.title("Lead vs Following Vehicle Velocity")
plt.grid()
plt.legend()
plt.savefig("plots/velocity_comparison.png")

# Plot 5: Space Gap
plt.figure(figsize=(12,5))
plt.plot(s_t, label="Space Gap (s_t)", color="blue")
plt.xlabel("Time step (k)")
plt.ylabel("Gap (m)")
plt.title("Space Gap Between Vehicles")
plt.grid()
plt.legend()
plt.savefig("plots/space_gap.png")