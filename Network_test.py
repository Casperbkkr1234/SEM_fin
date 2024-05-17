import torch

from Network.Process import GBM
from Network.Lower_bound import L_hat
from Network.Options import Options
from Network.Portfolio import Max_Portfolio
torch.set_default_dtype(torch.float64)

# Parameters
dt = 0.01
r = 0.05
mu = r  # r
sigma = 0.2
delta = 0.1
T = 1
n_steps = int(T / dt)
n_paths = 100
S0 = 1
n_exercises = n_steps
exercise_dates = [i * T / n_exercises for i in range(1, n_exercises + 1)]

# Create paths
gbm = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm.GBM_analytic()

# Network parameters
widths = [51, 51]
dimension = 1
strike = 1.1

# Create instance of bound class
bound = L_hat(dimension, widths, n_exercises)
# Calculate stopping times and train networks
g = Options.American(paths1, strike, r)
s_times = bound.Stopping_times(paths1, n_exercises, g)

networks = bound.c_thetas
out = torch.zeros(paths1.shape)


gbm2 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths2 = gbm2.GBM_analytic()

for i in range(1, len(networks)):
	ni = networks[i]
	out[:, :, i] = ni.forward(paths2[:, :, i])

gbm.show_paths(out)
gbm.show_paths(g)
asd = 1