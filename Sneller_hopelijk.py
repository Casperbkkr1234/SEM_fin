import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from Network.Process import GBM
from Network.Lower_bound import L_hat
from Network.Options import Options
from Network.Portfolio import Max_Portfolio

torch.set_default_dtype(torch.float64)

# initialize cuda operations
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
dt = 0.01
r = 0.05
mu = r  # r
sigma = 0.2
delta = 0.1
T = 3

n_steps = int(T / dt)

n_paths = 1000
portfolio_size = 5
n_portfolios = int(n_paths / portfolio_size)

S0 = 90
n_exercises = 9

# Network parameters
dimension = portfolio_size
widths = [50 + dimension, 50 + dimension]

# Strike price of option
strike = 100

df = pd.DataFrame(index=["Samples"], columns=["Option", "No option", "Option dev", "No option dev"])
n_portfolio = [2 ** (8 + 2 * i) for i in range(5)]
a = 1

for n_portfolios in n_portfolio:
	n_paths = portfolio_size * n_portfolios
	res1 = []
	res2 = []
	for _ in range(100):
		print(_ / 100, "%")

		exercise_dates = np.linspace(0, T, n_exercises)
		ex_times = [int(date / dt) - 1 if date > 0 else int(date / dt) for date in exercise_dates]
		# ex_times = ex_times[:1]
		t_ex_times_temp = torch.tensor(ex_times)
		t_ex_times = t_ex_times_temp.repeat(n_portfolios, dimension + 1, 1)
		t_ex_times_m = t_ex_times_temp.repeat(n_portfolios, dimension, 1)

		# Create paths
		gbm1 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
		paths1 = gbm1.GBM_analytic()
		paths1 = paths1.to(dev)

		MP = Max_Portfolio(n_portfolios, portfolio_size)
		P1 = MP.Create_portfolios(paths1, strike, r)
		P1_M = MP.Create_portfolios_no_max(paths1, strike, r)
		#  npro = P1_M.numpy()

		P1_stopped = torch.gather(P1, 2, t_ex_times)
		P1_stopped_M = torch.gather(P1_M, 2, t_ex_times_m)
		# gbm1.show_paths(P1[1, :, :])
		#B = P1_stopped.numpy()
		ga1 = P1_stopped[:, -1, :]
		ga1 = ga1.unsqueeze(1)

		ga1_M = ga1

		# P1_stopped = P1_stopped.permute(0, 2, 1)
		# Create instance of bound class
		bound1 = L_hat(dimension, widths, n_exercises, dev)
		bound2 = L_hat(dimension - 1, widths, n_exercises, dev)
		# Calculate stopping times and train networks
		s_times = bound1.Stopping_times(P1_stopped, n_exercises, ga1)
		s_times_M = bound2.Stopping_times(P1_stopped_M, n_exercises, ga1_M)

		# Create paths
		gbm2 = GBM(dt, mu, sigma, delta, n_steps, years=T, n_paths=n_paths, S0=S0)
		paths2 = gbm2.GBM_analytic()
		paths2 = paths2.to(dev)

		MP2 = Max_Portfolio(n_portfolios, portfolio_size)

		P2 = MP2.Create_portfolios(paths2, strike, r)
		P2_M = MP2.Create_portfolios_no_max(paths2, strike, r)

		P2_stopped = torch.gather(P2, 2, t_ex_times)
		P2_stopped_M = torch.gather(P2_M, 2, t_ex_times_m)
		# P2_stopped = P2_stopped.permute(0, 2, 1)
		A = P2_stopped.numpy()
		ga2 = P2_stopped[:, -1, :]
		ga2 = ga2.unsqueeze(1)

		ga2_M = ga2

		# Calculate earliest stopping times
		min = bound1.Tau(P2_stopped, n_exercises, exercise_dates, ga2, dt)
		min_M = bound2.Tau(P2_stopped_M, n_exercises, exercise_dates, ga2_M, dt)
		tg = min.numpy()
		tf = min_M.numpy()
		l_bound = bound1.Bound(ga2, min)
		l_bound_M = bound2.Bound(ga2_M, min_M)

		print(1.1 * l_bound)
		print(1.1 * l_bound_M)

		res1.append(1.1 * l_bound.item())
		res2.append(1.1 * l_bound_M.item())

	all1 = np.array(res1)
	all2 = np.array(res2)

	conf1 = np.std(all1)
	conf2 = np.std(all2)

	avg = np.mean(all1)
	avg_M = np.mean(all2)

	df.loc[n_portfolios] = [avg, avg_M, conf1, conf2]
	df.to_csv("adf.csv")
	# df = df.append({"Option": avg, "No option": avg_M})
	#print(avg, avg_M)

#print(df)
