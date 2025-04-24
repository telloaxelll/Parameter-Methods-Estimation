# Parameter-Methods-Estimation

## Introduction
This repository implements **Recursive Least Squares (RLS)** for online parameter estimation in Adaptive Cruise Control (ACC), based on Wang et al., “Online Parameters Estimation Methods for Adaptive Cruise Control Systems.”  
[Research Paper](https://inria.hal.science/hal-03011790/file/wang-gunter-nice-dellemonache-work-2020.pdf)

## Methodology
We focus on **RLS** to estimate the model
\[
\frac{dv}{dt} = \alpha\,(s - \tau v) + \beta\,(u - v)
\]
- \(s\): space gap  
- \(u\): leader speed  
- \(v\): follower speed  

## File Structure
. ├── functions.py # invert_gamma: γ → (α, β, τ) ├── rls.py # data generation, RLS estimation loop, plotting ├── notebook.ipynb # Jupyter notebook for RLS and Particle Filter experiments └── plots/ # saved output figures (PNG)

## Usage: 

## Usage
1. Clone the repo and navigate into it:
    `` bash
    git clone <repo-url>
    cd Parameter-Methods-Estimation
2. Run the RLS script
    `` bash 
    python rls.py

## Key Function (functions.py)
def invert_gamma(gamma, dt):
    """
    Convert RLS-estimated gamma vector into model parameters (α, β, τ).
    """
    γ1, γ2, γ3 = gamma
    α = γ2 / dt
    β = γ3 / dt
    τ = (1 - γ1 - γ3) / γ2 if abs(γ2) > 1e-8 else 0.0
    return α, β, τ

## RLS Workflow: 
1. Generate synthetic data for leader speed u_t with Gaussian noise, simulate follower speed v_t and gap s_t using ground truth parameters.
2. Initialize RLS: set covariance matrix P=1000 IP=1000I.
3. Iterate over each time step to update parameter vector γ=[γ1,γ2,γ3]γ=[γ1​,γ2​,γ3​].
4. Convert γγ to α,β,τα,β,τ via invert_gamma.
5. Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE) against true parameters.
6. Plot convergence of estimates and error metric