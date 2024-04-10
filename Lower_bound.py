import torch

from Approximator import C_theta
import numpy as np
from Options import Options

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
		stopping_times = torch.arange(0, self.n_steps)
		stopping_times = stopping_times.repeat(paths.shape[0], 1, 1)
		g = Options.American(paths, 1.2, 0.05)
		for i in range(2, n_steps+1):
			print(i)
			j = n_steps - i
			network = self.c_thetas[j]
			stop_time = stopping_times[:,:,j+1]
			g = g.squeeze(1)
			g_s = torch.take_along_dim(g, stop_time, 1)

			# Train c function on option value at stopping time n+1
			x_n = paths[:,:,j+1]
			network.Train(x_n, g_s)
			# Set stopping time n equal n if value at n greater than c at n
			forward = network.forward(paths[:,:,j+1])

			stopping_times[:,:,j] = torch.where(g_s >= forward, j, stopping_times[:,:,j+1])

		s1 = stopping_times[:, :, 0]

		s2 = s1.to(torch.float64)
		theta_0 = s2.mean()
		# TODO return all c functions including c_theta_0
		return

	def C(self, paths, n_steps):
		out = []
		for t in range(n_steps):
			a = self.c_thetas[t](paths[t,:])
			out.append(a)
		return np.array(out)

	def Tau(self, g, paths, n_steps):
		c = self.C(paths, n_steps)
		index = np.arange(0, n_steps, paths.size)
		A = np.where(g >= c, index, -100)

		return

	def Bound(self, paths):

		return






