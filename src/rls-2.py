# Needed Dependencies:
import numpy as np
import matplotlib.pyplot as plt
from functions import * 
import os

# Makes Directory for Plots:
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(plot_dir, exist_ok=True)

time = 900
dt = 0.1

# For Reproducibility:
np.random.seed(0)

# True Theta Paramaters Vector:
true_theta = [0.08, 0.12, 1.5]

# Initial Conditions
s0 = 60.0
v0 = 31.0 
u0 = 33.0

dv_max = 3.0  


u_t = np.zeros(time)
u_t[0] = u0

scenario_key = int(input("Enter scenario key (1-4): "))

if scenario_key == 1:
    # Random Walk Simulation:
    for i in range(1, 900):
        u_t[i] = u_t[i-1] + np.random.normal(loc=0, scale=0.2) # random increments

elif scenario_key == 2:
    # Road Curve Simulation:
    for i in range(1, 900):
        if 300 <= i < 340:
            u_t[i] = u_t[i-1] - 2.0  # sudden deceleration entering curve
        elif 340 <= i < 360:
            u_t[i] = u_t[i-1] + 1.0  # acceleration exiting curve
        else:
            u_t[i] = u_t[i-1] + np.random.normal(0, 0.2)
        # Safety: keep velocity in physical range
        u_t[i] = np.clip(u_t[i], 0, 35)

elif scenario_key == 3:
    # Suburbs Driving Simulation:
    vehicle_cruise  = 11.176      # target speed (m/s)
    accel_step      = 1.0        # m/s gained per timestep when accelerating
    decel_step      = -2.0       # m/s lost per timestep when braking
    stop_duration   = 3          # timesteps to stay stopped at a stop sign

    # generate event‐indices (avoid too close to start/end)
    stop_signs    = np.random.choice(range(50, time-50), size=3, replace=False)
    pedestrians   = np.random.choice(range(50, time-50), size=5, replace=False)
    random_brakes = np.random.choice(range(50, time-50), size=7, replace=False)

    stop_timer = 0
    for i in range(1, time):
        u_prev = u_t[i-1]

        if stop_timer > 0:
            # currently stopped at a stop sign
            u_t[i] = 0.0
            stop_timer -= 1

        elif i in stop_signs:
            # hit a stop sign → brake then start stop_timer
            u_t[i] = max(0.0, u_prev + decel_step)
            stop_timer = stop_duration - 1

        elif i in pedestrians:
            # pedestrian crossing: one‐step hard brake
            u_t[i] = max(0.0, u_prev + decel_step)

        elif i in random_brakes:
            # mild random brake (other car slows)
            u_t[i] = max(0.0, u_prev + decel_step * 0.5)

        else:
            # normal acceleration up to cruise
            u_t[i] = min(vehicle_cruise, u_prev + accel_step)

    # ensure no “overshoot” of cruise speed
    u_t = np.clip(u_t, 0.0, vehicle_cruise)


elif scenario_key == 4: # Scenario 4 implicitly
    for i in range(1, time):
        u_t[i] = u_t[i - 1] + np.random.normal(0, 1.0)  # Aggressive, erratic driving
        u_t[i] = np.clip(u_t[i], 0, 40)

else:
    raise ValueError("Invalid scenario_key. Must be 1-4.")


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
    raw_dv = true_theta[0]*(s_prev - true_theta[2]*v_prev) + true_theta[1]*(u_prev - v_prev)
    clipped_dv = np.clip(raw_dv, -dv_max, dv_max)
    v_t[k] = v_prev + clipped_dv * dt


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
print("-------------------------")
print(f"Alpha Error: {true_theta[0] - alpha_est_final:.3f}")
print(f"Beta Error: {true_theta[1] - beta_est_final:.3f}")
print(f"Tau Error: {true_theta[2] - tau_est_final:.3f}")


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