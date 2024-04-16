import torch

from Process import GBM
from Lower_bound import L_hat
from Approximator import C_theta
from Options import Options

torch.set_default_dtype(torch.float64)
# Parameters
dt = 0.01
mu = 0.05
sigma = 0.3
T = 1
n_steps = int(T/dt)
n_paths = 1000
S0 = 1

# Create paths
gbm = GBM(dt, mu, sigma, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm.GBM_analytic()
#gbm.show_paths()
# Network parameters
widths = [51,51]
dimension = 1
strike = 1

# Create instance of bound class
bound = L_hat(1, widths, n_steps)
# Calculate stopping times and train networks
g = Options.American(paths1, strike, 0.05)
s_times = bound.Stopping_times(paths1, n_steps, g)
# Calculate earliest stopping times
min = bound.Tau(paths1, n_steps, g)
# Calculate lower bound
gbm._Random_walk(dt, n_steps, n_paths=n_paths)
paths2 = gbm.GBM_analytic()
f = Options.American(paths1, strike, 0.05)

l_bound = bound.Bound(f, min)
print(l_bound)

asdasd=1


