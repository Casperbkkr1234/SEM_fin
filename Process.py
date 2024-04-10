import numpy as np
import matplotlib.pyplot as plt
import torch


class GBM:
	"""

	"""
	def __init__(self, dt: float,
	             mu: float,
	             sigma: float,
	             n_steps: int,
	             *, d_sigma: np.ndarray = None,
	             years: int = 1,
	             n_paths: int = 1,
	             seed=None,
	             S0: float = 100) -> None:
		"""
	      This class numerically computes paths random walks, Wiener processes and, Geometric brownian motions.

		:param dt: size of timesteps
		:param mu: drift parameter
		:param sigma: volatility parameter, can be constant or a vector with respect to time
		:param years: number of years to simulate
		:param n_paths: number of paths to simulate
		:param seed: Random seed, is false by default
		:param S0: Starting point of GBM
		:param n_steps: Number of steps to take between 0 and T (T=years)
		"""
		self.dt = dt
		self.mu = mu
		self.sigma = sigma
		self.n_steps = n_steps
		self.n_paths = n_paths
		self.seed = seed
		self.S0 = S0
		self.years = years
		self.Random_walk_paths = self._Random_walk(dt, n_steps, n_paths=n_paths, seed=seed)

		# Empty paths if paths not simulated yet
		self.analytic_paths = None
		self.euler_paths = None
		self.milstein_paths = None
		self.Wiener = None

	def _Random_walk(self, dt: float, n_steps: float, *, n_paths: int = 1, seed=None):# -> np.array:
		"""Returns Random walk paths"""
		# set seed for numpy if seed is given
		self.seed = torch.random.manual_seed(seed) if seed is not None else None

		#torch.set_default_dtype(torch.float64)
		means = torch.zeros(n_paths, 1, n_steps-1)
		stds = torch.full((n_paths, 1, n_steps-1), fill_value=np.sqrt(dt))
		# returns numpy array with random samples from N(0,sqrt(dt))
		return torch.normal(mean=means, std=stds)

	def Make_wiener(self) -> np.ndarray:
		"""Returns Wiener paths"""
		n_paths = self.n_paths
		dW = self.Random_walk_paths

		# make array with zeros to prepend to all paths
		zeros = torch.zeros(n_paths, 1,1)

		# prepend the zeros array to cumulative sum of the random walk to get the Wiener paths
		#Wiener_paths = np.concatenate((zeros, dW.cumsum(axis=1)), axis=1)
		C_sum = torch.cumsum(dW, 2)
		Wiener_paths = torch.cat((zeros, dW), 2)

		self.Wiener = Wiener_paths

		return Wiener_paths

	def GBM_analytic(self) -> np.ndarray:
		"""Returns Geometric brownian motion paths"""
		dt = self.dt
		mu = self.mu
		sigma = self.sigma
		S0 = self.S0
		dW = self.Make_wiener()


		t = torch.arange(start=0, end=self.years , step=dt)
		t = t.unsqueeze(0)
		t = t.unsqueeze(1)
		t = t.repeat(self.n_paths,1,1)

		C = (mu - ((sigma ** 2) / 2)) * t
		St = C + sigma * dW
		# take exponent of S(t) and transpose array
		expSt = torch.exp(St)
		# start gbm at S(0)
		gbm_paths = S0 * expSt

		self.analytic_paths = gbm_paths
		# return array with geometric brownian motion paths
		return gbm_paths


	def show_paths(self, paths=None) -> None:
		"""Plots the GBM paths"""
		paths = self.analytic_paths if paths is None else paths
		paths = paths.squeeze(1)
		paths = paths.numpy()
		#plt.style.use('seaborn')
		# Define time interval correctly
		time = np.linspace(0, self.years, self.n_steps)# + 1)
		# Require numpy array that is the same shape as St
		tt = np.full(shape=(self.n_paths, self.n_steps), fill_value=time)

		plt.plot(tt.T, paths.T)
		plt.xlabel("Years $(t)$")
		plt.ylabel("Stock Price $(S_t)$")
		plt.title(
			"$dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(self.S0,
			                                                                                      self.mu,
			                                                                                      self.sigma)
			)
		plt.show()
