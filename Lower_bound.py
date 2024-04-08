from Approximator import C_theta
import numpy as np

class L_hat():
	def __init__(self, dimension, widths):
		self.dimension = dimension
		self.widths = widths
		self.c_thetas = self.make_networks

	def Make_networks(self):
		networks = []
		for _ in range(self.n_steps):
			c_theta_n = C_theta(self.dimension, self.widths)
			networks.append(c_theta_n)
		return networks

	def Stopping_times(self, paths, n_steps):
		stopping_times = np.arrange(0, self.n_steps)
		for i in range(1, n_steps):
			j = n_steps - i
			network = self.c_thetas[j,:]
			g = None # TODO add call function
			network.Train(paths[j,:], g)

			if g >= network.forward(paths[j,:]):
				stopping_times[j-1] = j
			else:
				stopping_times[j-1] = stopping_times[j]

		stopping_times[0] = np.mean(g(paths[0,:])) # TODO make this work
		return stopping_times





