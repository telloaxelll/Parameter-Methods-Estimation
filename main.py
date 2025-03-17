# Relevant modules 
import numpy as np 
import matplotlib.pyplot as plt

# Total number of samples & sample time
time = 1_000  # Number of samples
delta_T = 1e-1  # Time step constant
gap_noise = np.random.normal(0, 0.5, time)  # Noise for space gap

# Initial velocities and gap
s_0 = 30 
u_0 = 30
v_0 = 30

# Theta vector parameters (true): 
alpha = 0.08
beta = 0.12
tau = 1.5

# Generate velocity profiles
u_t = u_0 + np.cumsum(np.random.normal(0, 0.2, time))  # Lead Vehicle Velocity
v_t = v_0 + np.cumsum(np.random.normal(0, 0.2, time))  # ACC Vehicle Velocity
delta_v = u_t - v_t  # Velocity difference between leader and follower

# Compute space gap
s_t = s_0 + np.cumsum(delta_v * delta_T + gap_noise)  # Space Gap

# Plot Lead vs ACC Vehicle Velocities
plt.figure(figsize=(12, 5))
plt.plot(u_t, label="Lead Vehicle Velocity (u_t)", linestyle="dashed", color="g")
plt.plot(v_t, label="ACC Vehicle Velocity (v_t)", color="r")
plt.xlabel("Time Step")
plt.ylabel("Velocity (m/s)")
plt.title("Lead vs. ACC Vehicle Velocity")
plt.legend()
plt.grid()
plt.show()

# Plot Space Gap Evolution
plt.figure(figsize=(12, 5))
plt.plot(s_t, label="Space Gap (s_t)", color="b")
plt.xlabel("Time Step")
plt.ylabel("Space Gap (m)")
plt.title("Space Gap Evolution Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Velocity Difference
plt.figure(figsize=(12, 5))
plt.plot(delta_v, label="Velocity Difference (delta_v = u_t - v_t)", color="purple")
plt.axhline(y=0, color='black', linestyle='dashed')  # Reference line at zero
plt.xlabel("Time Step")
plt.ylabel("Velocity Difference (m/s)")
plt.title("Velocity Difference Between Lead and ACC Vehicles")
plt.legend()
plt.grid()
plt.show()