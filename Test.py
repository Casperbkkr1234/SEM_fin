import torch
import numpy as np

from Process import GBM
from Approximator import C_theta

torch.set_default_dtype(torch.float64)

# Parameters
dt = 0.001
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

# Create network
network = C_theta(dimension, widths)
optimizer =  torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)



asdasd=1


