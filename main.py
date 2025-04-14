# Needed dependencies:
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os

# Create directory for plots 
if not os.path.exists("plots"):
    os.makedirs("plots", exist_ok=True)

# Parameters from paper 
time = 900     
dt   = 0.1  
   
np.random.seed(0)  

# True parameters for ACC model
alpha_true = 0.08
beta_true  = 0.12
tau_true   = 1.5

# Intial conditions
s0 = 30.0 
v0 = 30.0
u0 = 30.0 

"""
Generate data for ACC model:
   1. Allocate array for u_t
   2. Generate and append values into array with mean 0 and std 0.2
"""

u_t = np.zeros(time)
u_t[0] = u0
for k in range(1, time):
    # small random increments
    u_t[k] = u_t[k-1] + np.random.normal(loc=0, scale=0.2)

"""
Generate data for Space Gap and Velocity of Following vehicle:
   1. Allocate array for s_t, v_t
   2. Generate and append values into array using the ACC model equations
"""
s_t = np.zeros(time)
v_t = np.zeros(time)
s_t[0] = s0
v_t[0] = v0

for k in range(1, time):
    s_prev = s_t[k-1]
    v_prev = v_t[k-1]
    u_prev = u_t[k-1]
    
    ds = (u_prev - v_prev) * dt
    dv = (alpha_true*(s_prev - tau_true*v_prev) + beta_true*(u_prev - v_prev)) * dt

    s_t[k] = s_prev + ds
    v_t[k] = v_prev + dv

gamma_est = np.array([0.9, 0.01, 0.01])  # some initial guess for [gamma1, gamma2, gamma3]
P = np.eye(3)*1000.0

gamma_history = np.zeros((time, 3))
theta_history = np.zeros((time, 3))  # [alpha, beta, tau] at each step

def invert_gamma(gamma, dt):
    """Given gamma1, gamma2, gamma3 => alpha, beta, tau."""
    gamma1, gamma2, gamma3 = gamma
    alpha = gamma2/dt
    beta  = gamma3/dt
    if abs(alpha) < 1e-8:
        tau = 0.0  
    else:
        tau = ((1 - gamma1)/dt - beta) / alpha
    return alpha, beta, tau

# Initialize
gamma_history[0] = gamma_est
theta_history[0] = invert_gamma(gamma_est, dt)


for k in range(1, time):
    # Y = v_t[k], X = [v_t[k-1], s_t[k-1], u_t[k-1]]
    Y = v_t[k]
    X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])
    
    denom = 1.0 + X @ P @ X
    K = (P @ X) / denom
    
    # RLS update for gamma
    gamma_est = gamma_est + K*(Y - X.dot(gamma_est))
    P = P - np.outer(K, X.dot(P))
    
    gamma_history[k] = gamma_est
    theta_history[k] = invert_gamma(gamma_est, dt)

# Comparison between final alpha,beta,tau with ground truth
# Compute absolute errors for each parameter over time
errors = np.abs(theta_history - np.array([alpha_true, beta_true, tau_true]))
alpha_est_final, beta_est_final, tau_est_final = theta_history[-1]
print("Final estimated alpha = %.3f (true=%.3f)" % (alpha_est_final, alpha_true))
print("Final estimated beta  = %.3f (true=%.3f)"  % (beta_est_final,  beta_true))
print("Final estimated tau   = %.3f (true=%.3f)"   % (tau_est_final,   tau_true))


# Plot - Alpha, Beta, Tau Convergence
t_axis = np.arange(time)
fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)

params  = ["alpha", "beta", "tau"]
trueval = [alpha_true, beta_true, tau_true]
colors  = ["r", "g", "b"]

for i, ax in enumerate(axes):
    ax.plot(t_axis, theta_history[:, i], label=f"Estimated {params[i]}", color=colors[i])
    ax.axhline(y=trueval[i], color=colors[i], linestyle="--", label=f"True {params[i]}")
    ax.legend()
    ax.grid()
axes[-1].set_xlabel("Time step (k)")
plt.suptitle("RLS Parameter Convergence via Gamma-Linearization")
plt.savefig("plots/rls_estimation_convergence_gamma.png")
plt.show()

# Compute MAE and MSE
mae = np.mean(errors, axis=1)  # mean absolute error at each time step
mse = np.mean(errors**2, axis=1)  # mean squared error at each time step

# Plot MAE and MSE over time
plt.figure(figsize=(10,4))
plt.plot(t_axis, mae, label="MAE", color="orange")
plt.plot(t_axis, mse, label="MSE", color="purple")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Error Value")
plt.title("MAE and MSE of Parameter Estimates")
plt.legend()
plt.savefig("plots/rls_mae_mse_gamma.png")
plt.show()

# Plot - Absolute Error in alpha, beta, tau
errors = np.abs(theta_history - np.array([alpha_true, beta_true, tau_true]))
plt.figure(figsize=(10,4))
for i, param in enumerate(params):
    plt.semilogy(t_axis, errors[:, i], label=f"Error in {param}")
plt.grid()
plt.xlabel("Time step (k)")
plt.ylabel("Absolute Error (log scale)")
plt.title("Convergence of RLS Parameter Estimates")
plt.legend()
plt.savefig("plots/rls_estimation_error_gamma.png")
plt.show()

"""
Plot - Plots lead vehicle velocity (u_t), following vehicle velocity (v_t),
       and space gap (s_t) over time.
"""
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
plt.show()