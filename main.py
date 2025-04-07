# Author: Axel Muniz Tello 
# Date: 03/26/2025
# Description: This script is used to simulate the Recursive Least Squares (RLS) algorithm for online parameter estimation.
#              The algorithm is used to estimate the parameters of the ACC model, which are alpha, beta, and tau.
#              The script also plots the convergence of the estimated parameters and the error in the estimates.

# Necessary Modules: 
import matplotlib.pyplot as plt
import numpy as np 
import os

# Directory for saving plots
if not os.path.exists("plots"):
    os.makedirs("plots", exist_ok=True)

"""
Total number of samples that will be used to sample the space gap, 
velocity v(t) (speed of the trailing car), and velocity u(t) (which is the leading car).
"""
time = 900 # Total number of samples
delta_T = 1e-1 # Time step constant from paper
gap_noise = np.random.normal(0, 0.5, time) # Noise for the space gap

# Initial conditions for space gap, velocity v(t), and velocity u(t): 
s_0 = 30 # Measured in meters (m)
u_0 = 30 # Measured in meters per second (m/s)
v_0 = 30 # Measured in meters per second (m/s)

# True Theta Parameter Values:
true_theta = np.array([0.08, 0.12, 1.5]) # true_theta[0] = alpha, true_theta[1] = beta, true_theta[2] = tau

""" 
Arrays for space gap s(t), velocity v(t), and velocity u(t). Uses "np.cumsum()" to ensure 
that it's overall accumulation of previous s, v, and u. Data is more realistic, and the 
changes aren't as drastic and makes the data more smooth.
"""

# Generate lead vehicle velocity
u_t = u_0 + np.cumsum(np.random.normal(0, 0.2, time))  # Lead Vehicle Velocity
v_t = v_0 + np.cumsum(np.random.normal(0, 0.2, time))  # ACC Vehicle Velocity
delta_v = u_t - v_t  # Relative Velocity Between ACC and Lead Vehicle
s_t = s_0 + np.cumsum(delta_v * delta_T + gap_noise)  # Space Gap Between ACC and Lead Vehicle

# Recursive Least Squares (RLS) Initialization
P = np.eye(3) * 1000  # Large initial covariance
theta_hat = np.array([0.1, 0.1, 1.0])  # Initial guess for [alpha, beta, tau] arbitrary values

# Storage for estimates
theta_estimates = np.zeros((time, 3))

# Recursive Least Squares (RLS) Algorithm: 
for k in range(1, time):
    # X matrix from paper
    X = np.array([
        v_t[k - 1],  # v_k-1
        s_t[k - 1],  # s_k-1
        u_t[k - 1]   # u_k-1
    ])
    # Y matrix from paper
    Y = v_t[k]  # Observed velocity at step k
    
    # Gain computation
    K = P @ X / (1 + X.T @ P @ X)
    
    # Parameter update
    theta_hat = theta_hat + K * (Y - X.T @ theta_hat)
    
    # Covariance update
    P = P - np.outer(K, X.T @ P)
    
    # Store estimates
    theta_estimates[k] = theta_hat

# Plot estimated parameters with subplots
time_axis = np.arange(time)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
params = ['alpha', 'beta', 'tau']
true_values = [true_theta[0], true_theta[1], true_theta[2]]
colors = ['r', 'g', 'b']

for i, ax in enumerate(axes):
    ax.plot(time_axis, theta_estimates[:, i], label=f"Estimated {params[i]}", color=colors[i])
    ax.axhline(y=true_values[i], color=colors[i], linestyle="dashed", label=f"True {params[i]}")
    ax.fill_between(time_axis, theta_estimates[:, i] - 0.02, theta_estimates[:, i] + 0.02, color=colors[i], alpha=0.2)
    ax.set_ylabel(f"{params[i]} Estimate")
    ax.legend()
    ax.grid()
    if i == 2:
        ax.set_xlabel("Time Steps")

plt.suptitle("RLS Online Parameter Estimation Convergence")
plt.savefig("plots/rls_estimation_convergence.png")
plt.show()


# Log-scale error plot for convergence tracking
errors = np.abs(theta_estimates - np.array([true_theta[0], true_theta[1], true_theta(2)]))
plt.figure(figsize=(12, 5))
plt.semilogy(time_axis, errors, label=["Error in alpha", "Error in beta", "Error in tau"])
plt.xlabel("Time Steps")
plt.ylabel("Log Absolute Error")
plt.title("Convergence of RLS Parameter Estimates")
plt.legend()
plt.grid()
plt.savefig("plots/rls_error_plot.png")
plt.show()

# Plot: Velocity Comparison
plt.figure(figsize=(15, 5))
plt.plot(u_t, label="Lead Vehicle Velocity u_k", color="g", linestyle="dashed")
plt.plot(v_t, label="ACC Vehicle Velocity v_k", color="r")
plt.plot(s_t, label="Space Gap", color="b")
plt.xlabel("Time Step")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity of Lead and ACC Vehicles")
plt.legend()
plt.grid()
plt.savefig("plots/velocity_comparison.png")
plt.show()