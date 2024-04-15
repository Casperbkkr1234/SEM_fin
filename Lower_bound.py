import torch

from Approximator import C_theta
import numpy as np
from Options import Options

class L_hat():
	def __init__(self, dimension: int, widths: list, n_steps: int):
		self.dimension = dimension
		self.widths = widths
		self.n_steps = n_steps
		self.c_thetas = self.Make_networks(n_steps)

	def Make_networks(self, n_steps: int) -> list:
		networks = []
		for _ in range(n_steps):
			c_theta_n = C_theta(self.dimension, self.widths)
			networks.append(c_theta_n)
		return networks

	def Stopping_times(self, paths: torch.tensor,
	                         n_steps: int,
	                         g: torch.Tensor):
		stopping_times = torch.arange(0, self.n_steps)
		stopping_times = stopping_times.repeat(paths.shape[0], 1, 1)

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
		self.c_thetas[0] = theta_0

		return



	def Tau(self, paths: torch.tensor,
	              n_steps: int,
	              g: torch.Tensor):
		stopping_times = torch.arange(0, self.n_steps)
		stopping_times = stopping_times.repeat(paths.shape[0], 1, 1)
		networks = self.c_thetas
		for i in range(n_steps):
			net = networks[i]
			if i != 0:
				forw = net.forward(paths[:,:,i])
				stopping_times[:,:,i] = torch.where(g[:,:,i] >= forw, i, 1000) #TODO fix 1000
			else:
				stopping_times[:,:,0] = torch.where(g[:,:,i] >= networks[0], i, 1000) #TODO fix 1000


		s1 = stopping_times.squeeze(1)
		s2 = s1.numpy()

		min = torch.min(stopping_times, dim=2).indices

		min2 = torch.min(min, dim=0).values.item()
		out = torch.full(min.shape, min2)
		return out

	def Bound(self, paths: torch.tensor,
	                min: torch.tensor):

		paths = paths.squeeze(1)
		g_tau = torch.take_along_dim(paths, min, 1)
		mean = torch.mean(g_tau)

		return mean






