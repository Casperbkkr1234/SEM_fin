import torch

from Network.Process import GBM
from Network.Lower_bound import L_hat
from Network.Options import Options
from Network.Portfolio import Max_Portfolio
torch.set_default_dtype(torch.float64)

# Parameters
dt = 0.01
r = 0#.05
mu = r  # r
sigma = 0.2
delta = 0.1
T = 1
n_steps = int(T / dt)
n_paths = 1000
portfolio_size = 10
n_portfolios = int(n_paths/portfolio_size)
S0 = 90
n_exercises = 100


# Network parameters
dimension = portfolio_size
widths = [50 + dimension, 50 + dimension]

strike = 100


exercise_dates = [i * T / n_exercises for i in range(1, n_exercises + 1)]
ex_times = [int(date / dt)-1 for date in exercise_dates]
t_ex_times = torch.tensor(ex_times)
t_ex_times = t_ex_times.repeat(n_portfolios, 1, 1)

# Create paths
gbm = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
paths1 = gbm.GBM_analytic()

P1 = Max_Portfolio(n_portfolios, portfolio_size, paths1).portfolio_paths
P1_g = torch.gather(P1, 2, t_ex_times)
#gbm.show_paths(P1)


# Create instance of bound class
bound = L_hat(dimension, widths, n_exercises)
# Calculate stopping times and train networks
g = Options.Ber_max(P1, strike, r, exercise_dates, dt)
s_times = bound.Stopping_times(P1_g, n_exercises, g)

gbm._Random_walk(dt, n_steps, n_paths=n_paths)
paths2 = gbm.GBM_analytic()
P2 = Max_Portfolio(n_portfolios, portfolio_size, paths2).portfolio_paths
P2_g = torch.gather(P2, 2, t_ex_times)


f = Options.Ber_max(P2, strike, r, exercise_dates, dt)
# Calculate earliest stopping times
min = bound.Tau(P2_g, n_exercises, exercise_dates, f, dt)

gbm._Random_walk(dt, n_steps, n_paths=n_paths)
paths3 = gbm.GBM_analytic()
P3 = Max_Portfolio(n_portfolios, portfolio_size, paths2).portfolio_paths
#P3_g = torch.gather(P3, 2, t_ex_times)

h = Options.Ber_max(P3, strike, r, exercise_dates, dt)
l_bound = bound.Bound(h, min)
print(l_bound)

asdasd = 1
