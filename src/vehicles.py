# Import Needed Dependencies: 
import numpy as np
import matplotlib.pyplot as plt
from functions import rls_filter
import os

plot_dir = os.path.join(os.path.dirname(__file__), "plots")

# Case 1: Random Walk Scenario
def scenario_1_data(scenario, u_0, v_0, s_0, time, dv_max, dt, true_theta):
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
            # Velocity Sample Generation for u(t)
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

        #final_data = [u_t, v_t, s_t]
        rls_filter(u_t, v_t, s_t, time, dt, true_theta)
        
        # Plot Velocities:
        plt.figure(figsize=(12, 5))
        plt.plot(u_t, label="u(t) - Lead Vehicle", color="red")
        plt.plot(v_t, label="v(t) - Follower", color="blue")
        plt.title("Velocity Samples Under Non-Equilibrium | Scenario 1")
        plt.xlabel("Time step (k)")
        plt.ylabel("m/s")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,"1_Velocities_Non_Equilibrium.png"))
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
        plt.savefig(os.path.join(plot_dir, "1_Space_Gap_Equilibrium.png"))
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

        #final_data = [u_t, v_t, s_t]
        rls_filter(u_t, v_t, s_t, time, dt, true_theta)
        
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

        plt.savefig(os.path.join(plot_dir,"1_Velocities_Equilibrium.png"))
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
        plt.savefig(os.path.join(plot_dir, "1_Space_Gap_Equilibrium.png"))
        plt.close()

    else: None

# Case 2: Induced Curved Road
def scenario_2_data(u_0, v_0, s_0, time, dv_max, dt, true_theta):
    """
    Case Details: 
    - Since realistically only one scenario will be simulated which is the non-equilibrium case since
      if it was under equilibrium assuming that v_k = u_k therefore it would mathematically break main 
      principle of matrix X and cause it to be a rank deficient matrix where, rank(X) = 1 << 3
    - Therefore there would be infite solutions with an ill-posed estimation

    - We can do this proof by using the condition number and taking the singular values from SVD and determining whether 
      matrix X will be ill-conditioned.
    """

    u_t = np.zeros(time)
    v_t = np.zeros(time)
    s_t = np.zeros(time)

    u_t[0] = u_0
    v_t[0] = v_0
    s_t[0] = s_0

    center = time // 2         # midpoint of time
    curve_width = 100          # controls tightness of curve
    min_speed = 20             # slowest point on the curve (m/s)

    for i in range(1, time):
        u_t[i] = u_0 - (u_0 - min_speed) * np.exp(-((i - center) ** 2) / (2 * curve_width ** 2))

        s_prev = s_t[i - 1]
        v_prev = v_t[i - 1]
        u_prev = u_t[i - 1]

        # Compute acceleration with CTH-RV model
        acc = true_theta[0] * (s_prev - true_theta[2] * v_prev) + true_theta[1] * (u_prev - v_prev)
        acc = np.clip(acc, -dv_max, dv_max)

        v_t[i] = v_prev + acc * dt
        s_t[i] = s_prev + (u_prev - v_prev) * dt

    rls_filter(u_t, v_t, s_t, time, dt, true_theta, scenario="Scenario 2 - Curved Road")

    # Plot Velocities:
    plt.figure(figsize=(12, 5))
    plt.plot(u_t, label="u(t) - Lead Vehicle", color="red")
    plt.plot(v_t, label="v(t) - Follower", color="blue")
    plt.title("Velocity Samples on Curved Road | Scenario 2")
    plt.xlabel("Time step (k)")
    plt.ylabel("m/s")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "2_Velocities_Curved_Road.png"))
    plt.close()

    # Plot Space Gap:
    plt.figure(figsize=(12, 5))
    plt.plot(s_t, label="s(t) - Space Gap", color="black")
    plt.title("Space Gap Between u(t) and v(t) on Curve")
    plt.xlabel("Time Step (k)")
    plt.ylabel("meters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "2_Space_Gap_Curved_Road.png"))
    plt.close()