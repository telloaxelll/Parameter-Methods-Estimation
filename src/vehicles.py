"""
This file contains all possible vehicle profiles for both the lead vehicle u(t), and 
and the following vehicle v(t). 
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import *

def scenario_1_data (scenario, time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir):
    if scenario == "NON-EQ":
        # Allocate Arrays for u_t, v_t, and s_t:
        u_t = np.zeros(time)
        v_t = np.zeros(time)
        s_t = np.zeros(time)

        u_t[0] = u_0
        v_t[0] = v_0
        s_t[0] = s_0

        """
        Note: This section generates samples for the non-equillibrium case. 
        """
        for i in range(1, time):
            u_t[i] = u_t[i - 1] + np.random.normal(loc=0, scale=0.25) # mean 0, std 0.5

            # Compute Current Gap and Velocity: 
            s_prev = s_t[i - 1]
            v_prev = v_t[i - 1]
            u_prev = u_t[i - 1]

            # CTH-RV Update (Following Vehicle):
            acc = true_theta[0] * (s_prev - true_theta[2] * v_prev)  + true_theta[1] * (u_prev - v_prev) 
            acc = np.clip(acc, -dv_max, dv_max)  # apply physical constraint

            # Update Follower Velocity and Space Gap
            v_t[i] = v_prev + acc * dt
            s_t[i] = s_prev + (u_prev - v_prev) * dt 

        final_u_t = u_t
        final_v_t = v_t
        final_s_t = s_t

        #final_data = [u_t, v_t, s_t]
        rls_filter(final_u_t, final_v_t, final_s_t, time, dt, true_theta)
        
        # Plot Velocities:
        plt.figure(figsize=(12, 5))
        plt.plot(u_t, label="u(t) - Lead Vehicle", color="red")
        plt.plot(v_t, label="v(t) - Follower", color="blue")
        plt.title("Velocity Samples Under Non-Equilibrium")
        plt.xlabel("Time step (k)")
        plt.ylabel("m/s")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir, "Velocities Between u(t) and v(t)")
        plt.close()

        # Plot Space Gap: 
        plt.figure(figsize=(12, 5))
        plt.plot(s_t, label="s(t) - Space Gap", color="black")
        plt.title("Space Gap Between u(t) and v(t)")
        plt.xlabel("Time Step (k)")
        plt.ylabel("meter")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir)
        plt.close()


    elif scenario == "EQ":
        # Allocate Arrays for u_t, v_t, and s_t:
        u_t = np.zeros(time)
        v_t = np.zeros(time)
        s_t = np.zeros(time)

        # Set initial conditions for equilibrium
        equilibrium_speed = 30.0
        u_t[0] = equilibrium_speed
        v_t[0] = equilibrium_speed
        s_t[0] = true_theta[2] * equilibrium_speed  # s = τv

        # Generate equilibrium samples
        for i in range(1, time):
            u_t[i] = u_t[i - 1] + np.random.normal(0, 0.01)  # small noise for realism

            s_prev = s_t[i - 1]
            v_prev = v_t[i - 1]
            u_prev = u_t[i - 1]

            acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
            acc = np.clip(acc, -dv_max, dv_max)

            v_t[i] = v_prev + acc * dt
            s_t[i] = s_prev + (u_prev - v_prev) * dt

        final_u_t = u_t
        final_v_t = v_t
        final_s_t = s_t

        #final_data = [u_t, v_t, s_t]
        rls_filter(final_u_t, final_v_t, final_s_t, time, dt, true_theta)
        
        # Plot velocities
        plt.figure(figsize=(12, 5))
        plt.plot(u_t, label="u(t) - Lead Vehicle", color="red")
        plt.plot(v_t, label="v(t) - Follower", color="blue")
        plt.title("Velocity Samples Under Equilibrium")
        plt.xlabel("Time step (k)")
        plt.ylabel("m/s")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir)
        plt.close()

    else: None

#def scenario_2_data(scenario, time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir)


#def scenario_3_data(scenario, time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir)


#def scenario_4_data(scenario, time, u_0, v_0, s_0, true_theta, dv_max, dt, plot_dir)
