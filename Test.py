import numpy as np

from Process import GBM

#Parameters
dt = 0.001
mu = 0.6
sigma = 0.7
T = 1
n_steps = int(T/dt)
n_paths = 100
S0 = 1


gbm = GBM(dt, mu, sigma, n_steps, years=T, n_paths=n_paths, S0=S0)
gbm.GBM_analytic()
gbm.show_paths()

