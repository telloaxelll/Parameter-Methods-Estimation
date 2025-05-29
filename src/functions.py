import numpy as np
import matplotlib.pyplot as plt

# Invert Gamma Function:  
def invert_gamma(gamma, dt):
    """
    Given gamma1, gamma2, gamma3 = alpha, beta, tau, we are able to directly
    solve for alpha, beta, tau by using the following equations algebraically:
    - alpha = gamma2 / dt
    - beta = gamma3 / dt
    - tau = ((1 - gamma1 - gamma3) / gamma2)
    """
    gamma1, gamma2, gamma3 = gamma

    alpha = gamma2/dt
    beta  = gamma3/dt

    if abs(alpha) < 1e-8: # Added tolerance for numerical stability
        tau = 0.0
    else:
        tau = ((1 - gamma1 - gamma3) / gamma2)
    return alpha, beta, tau

# RLS Filter Function: 
"""
Given u_t, v_t, s_t, time, and dt:
    - We will set initial guess for theta vector
    - Initialize covariance matrix
    - Allocate estimation tracking arrays for recursive updates
RLS: 
    - Concatenate all of u_t, v_t, s_t into one matrix X
    - Initialize v_t into Y as the output vector
    - Compute Kalman gain
    - Update parameters at each step
    - Update covariance matrix
"""

def rls_filter(u_t, v_t, s_t, time, dt):
    gamma_est = np.array([0.9, 0.01, 0.01])  # some initial guess
    P = np.eye(3)*1000.0 # covariance matrix 

    gamma_history = np.zeros((time, 3))
    theta_history = np.zeros((time, 3))  # [alpha, beta, tau] at each step - 3 x 900 matrix storing all values of theta

    # Initialize RLS:
    gamma_history[0] = gamma_est
    theta_history[0] = invert_gamma(gamma_est, dt)

    # RLS Algorithm:
    for k in range(1, time):
        X = np.array([v_t[k-1], s_t[k-1], u_t[k-1]])
        y = v_t[k]

        # Compute Kalman gain
        denominator = 1.0 + X.T.dot(P).dot(X)
        K = P.dot(X) / denominator

        # RLS Update for Gamma Parameters:
        gamma_est = gamma_est + K * (y - X.T.dot(gamma_est))
        P = (np.eye(3) - np.outer(K, X)).dot(P)

        # Store History of Parameter Estimates:
        gamma_history[k] = gamma_est
        theta_history[k] = invert_gamma(gamma_est, dt)

    # Print Results
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
    plt.show()
    plt.close()


def scenario_1_data (scenario):
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
        rls_filter(final_u_t, final_v_t, final_s_t, time, dt)
        
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
        plt.show()

        # Plot Space Gap: 
        plt.figure(figsize=(12, 5))
        plt.plot(s_t, label="s(t) - Space Gap", color="black")
        plt.title("Space Gap Between u(t) and v(t)")
        plt.xlabel("Time Step (k)")
        plt.ylabel("meter")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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
        rls_filter(final_u_t, final_v_t, final_s_t, time, dt)
        
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
        plt.show()

    else: None