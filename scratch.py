"""
This portion of the script will generate the vehicle data for 
the overall simulation. Given a key, the script will generate vehicle u(t) data
based on the parameters and instructions of that specific profile.

key = #
for i in range(1, time): 
    if i == 1:
        # Scenario 1: 

"""
import numpy as np
import matplotlib.pyplot as plt

time = 900
u0 = 30.0
u_t = np.zeros(time)
u_t[0] = u0

# Key to decide which scenario to use:
# Change this key to switch between scenarios
# 1 = Random Walk Model
# 2 = Curve Simulation 
# 3 = Suburban Driving Simulation
# 4 - Aggresive Driving Simulation
scenario_key = 4

if scenario_key == 1:
    for i in range(1, 900):
        u_t[i] = u_t[i-1] + np.random.normal(loc=0, scale=0.2) # random increments
elif scenario_key == 2:
    # Simulated Curve:
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
    for i in range(1, time):
        if i % 100 < 20:
            u_t[i] = max(u_t[i - 1] - 3.0, 0)  # Stop zone
        else:
            u_t[i] = min(u_t[i - 1] + 0.5, 30) + np.random.normal(0, 0.2)

elif scenario_key == 4: # Scenario 4 implicitly
    for i in range(1, time):
        u_t[i] = u_t[i - 1] + np.random.normal(0, 1.0)  # Aggressive, erratic driving
        u_t[i] = np.clip(u_t[i], 0, 40)

else:
    raise ValueError("Invalid scenario_key. Must be 1-4.")


plt.plot(u_t, label="Lead Velocity (u)", linestyle="--", color="r")
plt.xlabel("Time step (k)")
plt.ylabel("Speed (m/s)")
plt.title("ACC Lead Vehicle Speed Profile")
plt.grid(True)
plt.legend()
plt.show()