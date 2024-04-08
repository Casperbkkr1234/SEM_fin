import torch

from Approximator import C_theta
import numpy as np
from Payoff import Payoff

class L_hat():
	def __init__(self, dimension, widths, n_steps):
		self.dimension = dimension
		self.widths = widths
		self.n_steps = n_steps
		self.c_thetas = self.Make_networks(n_steps)

	def Make_networks(self, n_steps):
		networks = []
		for _ in range(n_steps):
			c_theta_n = C_theta(self.dimension, self.widths)
			networks.append(c_theta_n)
		return networks

	def Stopping_times(self, paths, n_steps):#, payoff):
		stopping_times = np.arange(0, self.n_steps)
		P = Payoff()
		g = P.American(paths, 2, 1)
		g = torch.Tensor(g)
		paths = torch.Tensor(paths)
		g = g.unsqueeze(1)
		paths = paths.unsqueeze(1)
		for i in range(1, n_steps):
			j = n_steps - i
			network = self.c_thetas[j]

			network.Train(paths[j,:,:], g[j,:,:])
			# TODO make work
			if g[j,:,:] >= network.forward(paths[j,:,:]):
				stopping_times[j-1] = j
			else:
				stopping_times[j-1] = stopping_times[j]

		stopping_times[0] = np.mean(g(paths[0,:])) # TODO make this work

		return stopping_times

	def C(self, paths, n_steps):
		out = []
		for t in range(n_steps):
			a = self.c_thetas[t](paths[t,:])
			out.append(a)
		return np.array(out)

	def Tau(self, g, paths, n_steps):
		c = self.C(paths, n_steps)
		index = np.arange(0, n_steps, paths.size)
		A = np.where(g>=c, index, -100)

		return

	def Bound(self, paths):

		return






