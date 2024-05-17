import torch

from Network.Approximator import C_theta

import matplotlib.pyplot as plt


class L_hat():
	"""

	"""
	def __init__(self, dimension: int, widths: list, n_steps: int, dev):
		self.dimension = dimension
		self.widths = widths
		self.n_steps = n_steps
		self.dev = dev
		self.c_thetas = self.Make_networks(n_steps)


	def Make_networks(self, n_steps: int) -> list:
		networks = []
		for _ in range(n_steps):
			c_theta_n = C_theta(self.dimension, self.widths, self.dev)
			networks.append(c_theta_n)
		return networks


	def Stopping_times(self, paths: torch.tensor,
	                   n_exercise_times: int,
	                   g: torch.Tensor):
		# stopping_times = torch.arange(0, self.n_steps)
		stopping_times = torch.arange(0, g.shape[2])
		stopping_times = stopping_times.repeat(paths.shape[0], 1, 1)
		stopping_times.to(self.dev)

		# for i in range(2, n_steps + 1):
		for i in range(2, n_exercise_times):

			# reverse direction
			j = n_exercise_times - i
			#print(j)
			# select network corresponding to time step
			network = self.c_thetas[j]

			stop_time = stopping_times[:, :, j+1]
			stop_t = stop_time.numpy()
			g1 = g.squeeze(1)
			a = g1.numpy()
			g_stop = torch.take_along_dim(g1, stop_time, 1)
			a_stop = g_stop.numpy()
			# Train c function on option value at stopping time n+1
			x_n = paths[:, :, j]
			b = x_n.numpy()
			#forward1 = network.forward(x_n)
			network.Train_batch(x_n, g_stop)
			# Set stopping time n equal n if value at n greater than c at n
			forward = network.forward(x_n)
			c = forward.detach().numpy()
			n_time = torch.full(stop_time.shape, j)
			g_n = torch.take_along_dim(g1, n_time, 1)
			stopping_times[:, :, j] = torch.where(g_n >= forward, j, stop_time)

		#s1 = stopping_times.squeeze(1)
		#s2 = s1.numpy()
		#plt.plot([i for i in range(0, 100)], s2.T)
		#plt.show()
		#h2 = g.squeeze(1)
		#h3 = h2.numpy()
		g_0 = torch.take_along_dim(g.squeeze(1), stopping_times[:, :, 1], 1)
		#h1 = g_0.numpy()
		m1 = g_0[:, 0]
		m2 = m1.type(torch.DoubleTensor)
		theta_0 = m2.mean()

		# TODO return all c functions including c_theta_0
		self.c_thetas[0] = theta_0

		return #stopping_times


	def Tau(self, paths: torch.tensor,
	        n_exercises: int,
	        exercise_times: list,
	        g: torch.Tensor,
	        dt):

		stopping_times = torch.arange(0, n_exercises)
		stopping_times = stopping_times.repeat(paths.shape[0], 1, 1)
		stopping_times = stopping_times.to(self.dev)
		networks = self.c_thetas


		for i in range(n_exercises):
			net = networks[i]

			if i != 0:
				forw = net.forward(paths[:, :, i])
				stopping_times[:, :, i] = torch.where(g[:, :, i] >= forw, i, g.shape[-1]-1)
				a = g[:, :, i]
				k1 = a.numpy()
				b = forw  # TODO check negative values in continuation
				k9 = b.detach().numpy()
				c = 1
			else:
				c = g[:, :, i]
				k1 = c.numpy()
				d = networks[i]  # TODO check negative values in continuation
				k9 = d.detach().numpy()
				reqw = 1
				stopping_times[:, :, i] = torch.where(g[:, :, i] >= networks[i], 0, g.shape[-1]-1)  # TODO fix 10**10
				ar = 3

		#a1 = stopping_times.squeeze(1)
		#a2 = a1.numpy()
		min = torch.min(stopping_times, dim=2).values


		return min


	def Bound(self, option: torch.tensor,
	          min: torch.tensor):

		option = option.squeeze(1)
		g_tau = torch.take_along_dim(option, min, 1)
		mean = torch.mean(g_tau)

		return mean
