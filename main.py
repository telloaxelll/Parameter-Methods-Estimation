# Relevant modules 
import numpy as np 
import matplotlib.pyplot as plt

"""
Total number of samples that will be used to sample 
the space gap, velocity v(t) (speed of the trailing car),
and velocity u(t) (which is the leading car).
"""
time = 500 # Number of samples & Sample = Time (s)
delta_T = 1e-1 # Time step constant
gap_noise = np.random.normal(0, 0.5, time) # Noise for space gap

# Initial velocities and gap
s_0 = 30 
u_0 = 30
v_0 = 30

# Theta vector parameters (true): 
alpha = 0.08
beta = 0.12
tau = 1.5

""" 
Arrays for space gap s(t), velocity v(t), and velocity u(t). 
Uses "np.cumsum()" to ensure that it's overall accumulation of previous s, v, and u. 
Data is more realistic, and the changes aren't as drastic and makes the data more smooth.
"""
#s = np.cumsum(np.random.normal(0, 0.5, time)) + 30 # s -> data generated from N(0, 1) distribution | 20 => starting s(t) | METHOD #1 SAMPLING
#v = np.cumsum(np.random.normal(0, 0.2, time)) + 20 # v -> data generated from N(0, 1) distribution (m/s) | 20 => starting v(t) | ** ACC Vehicle**
# Generate lead vehicle velocity
u_t = u_0 + np.cumsum(np.random.normal(0, 0.2, time))  # Lead Vehicle Velocity
v_t = v_0 + np.cumsum(np.random.normal(0, 0.2, time))  # ACC Vehicle Velocity
delta_v = u_t - v_t  # Velocity difference between leader and follower
s_t = s_0 + np.cumsum(delta_v * delta_T + gap_noise)  # Space Gap

'''
# Plot results: 
# Plot for s(k)
plt.figure(figsize=(15, 5))
plt.plot(s_t, label = "Space Gap (s_k)")
plt.xlabel("Time Steps")
plt.ylabel("Space Gap (m)")
plt.legend()
plt.grid()
plt.show()
'''
plt.figure(figsize=(15, 5))
plt.plot(u_t, label="Lead Vehicle Velocity u_k", color="g", linestyle="dashed")
plt.plot(v_t, label="ACC Vehicle Velocity v_k", color="r")
plt.xlabel("Time Step")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity of Lead and ACC Vehicles")
plt.legend()
plt.grid()
plt.show()
