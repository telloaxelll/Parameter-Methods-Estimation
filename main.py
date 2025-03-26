# Relevant modules 
import numpy as np 
import matplotlib.pyplot as plt
import os

# Directory for saving plots
os.makedirs("plots", exist_ok=True)

"""
Total number of samples that will be used to sample 
the space gap, velocity v(t) (speed of the trailing car),
and velocity u(t) (which is the leading car).
"""
time = 1e3 # Number of samples & Sample = Time (s)
delta_T = 1e-1 # Time step constant
gap_noise = np.random.normal(0, 0.5, time) # Noise for space gap

# Initial velocities and gap
s_0 = 30 
u_0 = 30
v_0 = 30

# est. Theta vector parameters:
alpha_estimate = 0.7
beta_estimate = 0.25
tau_estimate = 2

""" 
Arrays for space gap s(t), velocity v(t), and velocity u(t). 
Uses "np.cumsum()" to ensure that it's overall accumulation of previous s, v, and u. 
Data is more realistic, and the changes aren't as drastic and makes the data more smooth.
"""
# Generate lead vehicle velocity
u_t = u_0 + np.cumsum(np.random.normal(0, 0.2, time))  # Lead Vehicle Velocity
v_t = v_0 + np.cumsum(np.random.normal(0, 0.2, time))  # ACC Vehicle Velocity
delta_v = u_t - v_t  # Velocity difference between leader and follower
s_t = s_0 + np.cumsum(delta_v * delta_T + gap_noise)  # Space Gap

# Recursive Least Squares (RLS) Initialization
P = np.eye(3) * 1000  # Large initial covariance
theta_hat = np.array([0.1, 0.1, 1.0])  # Initial guess for [alpha, beta, tau]

# Storage for estimates
theta_estimates = np.zeros((time, 3))

# RLS Algorithm
for k in range(1, time):
    phi_k = np.array([
        v_t[k - 1],  # v_k-1
        s_t[k - 1],  # s_k-1
        u_t[k - 1]   # u_k-1
    ])
    y_k = v_t[k]  # Observed velocity at step k
    # Gain computation
    K = P @ phi_k / (1 + phi_k.T @ P @ phi_k)
    # Parameter update
    theta_hat = theta_hat + K * (y_k - phi_k.T @ theta_hat)
    # Covariance update
    P = P - np.outer(K, phi_k.T @ P)
    # Store estimates
    theta_estimates[k] = theta_hat

# Plot estimated parameters with subplots
time_axis = np.arange(time)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
params = ['alpha', 'beta', 'tau']
true_values = [alpha_estimate, beta_estimate, tau_estimate]
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
errors = np.abs(theta_estimates - np.array([alpha_estimate, beta_estimate, tau_estimate]))
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
