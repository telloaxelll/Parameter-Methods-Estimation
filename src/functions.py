"""
This functions.py file contains a list of all the necessary or used functions
within this project. 

Top half of the file contains the functions that are used in the Recursive Least Squares (RLS) algorithm.
The bottom half of the file contains the functions that are used in Stochastic Gradient Descent (SGD) algorithm.
"""

# RLS Function(s): 
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