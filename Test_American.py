import numpy as np
import torch
import matplotlib.pyplot as plt
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
n_paths = 1000
S0 = 90
n_exercises = n_steps
exercise_dates = [i * T / n_exercises for i in range(1, n_exercises + 1)]

# Create paths
gbm1 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm1.GBM_analytic()
paths2 = paths1.squeeze(1)
paths3 = paths2.numpy()
avg = np.mean(paths3, axis=0)
var = np.var(paths3, axis=0)
t = np.linspace(0, T, n_steps)
an_var = np.exp((sigma**2)*t)-1
#plt.plot(avg)
#plt.plot(var)
#plt.plot(an_var)
#plt.show()
#gbm.show_paths(1P)
# Network parameters
widths = [51, 51]
dimension = 1
strike = 90

# Create instance of bound class
bound = L_hat(dimension, widths, n_exercises)
# Calculate stopping times and train networks
g1 = Options.American(paths1, strike, r)
s_times = bound.Stopping_times(paths1, n_exercises, g1)

# Calculate lower bound
gbm2 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths2 = gbm2.GBM_analytic()

g2 = Options.American(paths2, strike, r)
# Calculate earliest stopping times
min = bound.Tau(paths2, n_exercises, exercise_dates, g2, dt)

l_bound = bound.Bound(g2, min)
print(l_bound)

asdasd = 1
