import torch
import numpy as np

from Process import GBM
from Lower_bound import L_hat
from Approximator import C_theta
torch.set_default_dtype(torch.float64)
from Options import Options
# Parameters
dt = 0.01
mu = 0.6
sigma = 0.7
T = 1
n_steps = int(T/dt)
n_paths = 127
S0 = 1

# Create paths
gbm = GBM(dt, mu, sigma, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm.GBM_analytic()

# Network parameters
widths = [51,51]
dimension = 1
strike = 2

a = C_theta(1, widths)
t = torch.Tensor(1)
x = a.forward(t)
# Create instance of bound class
bound = L_hat(1, widths, n_steps)
# Calculate stopping times and train networks
s_times = bound.Stopping_times(paths1, n_steps)#, payoff)
# Calculate earliest stopping times
min = bound.Tau(paths1, n_steps)
# calculate lower bound
gbm._Random_walk(dt, n_steps, n_paths=n_paths)
paths2 = gbm.GBM_analytic()
l_bound = bound.Bound(paths2, min)
print(l_bound)

asdasd=1


