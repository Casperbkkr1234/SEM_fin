import torch
import numpy as np

from Process import GBM
from Lower_bound import L_hat

torch.set_default_dtype(torch.float64)

# Parameters
dt = 0.01
mu = 0.6
sigma = 0.7
T = 1
n_steps = int(T/dt)
n_paths = 100
S0 = 1

# Create paths
gbm = GBM(dt, mu, sigma, n_steps, years=T, n_paths=n_paths, S0=S0)
paths = gbm.GBM_analytic()

# Network parameters
widths = [51,51]
dimension = 1
# Create instance of bound class
bound = L_hat(1, widths, n_steps)
# Create instance of payoff class
payoff = 1 # Todo add payoff function/class
# Calculate stopping times and train networks
s_times = bound.Stopping_times(paths, n_steps, payoff)
# Calculate earliest stopping times
tau = bound.Tau(payoff, paths, n_steps)



asdasd=1


