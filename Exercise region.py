import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from Network.Process import GBM
from Network.Lower_bound import L_hat
from Network.Options import Options
from Network.Portfolio import Max_Portfolio



torch.set_default_dtype(torch.float64)

#  Parameters
dt = 0.01
r = 0.05
mu = r  # r
sigma = 0.2
delta = 0.1
T = 3

n_steps = int(T / dt)

n_paths = 1000
portfolio_size = 2
n_portfolios = int(n_paths / portfolio_size)

S0 = 90
n_exercises = 3

# Network parameters
dimension = portfolio_size
widths = [50 + dimension, 50 + dimension]

# Strike price of option
strike = 100




#exercise_dates = [i * T / n_exercises for i in range(0, n_exercises)]
exercise_dates = np.linspace(0, T, n_exercises)
ex_times = [int(date / dt)-1 if date > 0 else int(date / dt) for date in exercise_dates]
# ex_times = ex_times[:1]
t_ex_times = torch.tensor(ex_times)
t_ex_times = t_ex_times.repeat(n_portfolios, dimension+1, 1)

# Create paths
gbm1 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm1.GBM_analytic()

MP = Max_Portfolio(n_portfolios, portfolio_size)
P1 = MP.Create_portfolios(paths1, strike, r)
P1_stopped = torch.gather(P1, 2, t_ex_times)
#gbm1.show_paths(P1[1, :, :])
B = P1_stopped.numpy()
ga1 = P1_stopped[:, -1, :]
ga1 = ga1.unsqueeze(1)
#P1_stopped = P1_stopped.permute(0, 2, 1)
# Create instance of bound class
bound = L_hat(dimension, widths, n_exercises)
# Calculate stopping times and train networks
s_times = bound.Stopping_times(P1_stopped, n_exercises, ga1)

# Create paths
gbm2 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths2 = gbm2.GBM_analytic()
MP = Max_Portfolio(n_portfolios, portfolio_size)
P2 = MP.Create_portfolios(paths2, strike, r)
P2_stopped = torch.gather(P2, 2, t_ex_times)

#P2_stopped = P2_stopped.permute(0, 2, 1)
A = P2_stopped.numpy()
ga2 = P2_stopped[:, -1, :]
ga2 = ga2.unsqueeze(1)

# Calculate earliest stopping times
min = bound.Tau(P2_stopped, n_exercises, exercise_dates, ga2, dt)



l_bound = bound.Bound(ga2, min)
print(l_bound*1.1)

asdasd = 1


