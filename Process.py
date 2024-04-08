import numpy as np
import matplotlib.pyplot as plt

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

	def _Random_walk(self, dt: float, n_steps: float, *, n_paths: int = 1, seed=None) -> np.array:
		"""Returns Random walk paths"""
		# set seed for numpy if seed is given
		self.seed = np.random.seed(seed) if seed is not None else None

		return np.random.normal(0.0,
		                        np.sqrt(dt),
		                        [n_paths, n_steps])  # returns numpy array with random samples from N(0,sqrt(dt))


	def Make_wiener(self) -> np.ndarray:
		"""Returns Wiener paths"""
		n_paths = self.n_paths
		dW = self.Random_walk_paths

		# make array with zeros to prepend to all paths
		zeros = np.zeros((n_paths, 1))
		# prepend the zeros array to cumulative sum of the random walk to get the Wiener paths
		Wiener_paths = np.concatenate((zeros, dW.cumsum(axis=1)), axis=1)

		self.Wiener = Wiener_paths

		return Wiener_paths


	def GBM_analytic(self) -> np.ndarray:
		"""Returns Geometric brownian motion paths"""
		dt = self.dt
		mu = self.mu
		sigma = self.sigma
		S0 = self.S0
		dW = self.Make_wiener()[:,:-1]


		t = np.arange(0, self.years, self.n_steps)

		C = (mu - ((sigma ** 2) / 2)) * t
		St = C + sigma * dW
		# take exponent of S(t) and transpose array
		expSt = (np.exp(St)).T
		# start gbm at one

		# multiply by S(0)
		gbm_paths = S0 * expSt

		self.analytic_paths = gbm_paths
		# return array with geometric brownian motion paths
		return gbm_paths



	def Euler(self):
		dt: float = self.dt
		mu: float = self.mu
		sigma: float or np.ndarray = self.sigma
		S0: float = self.S0
		dW: np.ndarray = self.Random_walk_paths

		increment = 1 + mu*dt + sigma*dW
		Y0 = S0 * np.ones(dW.shape[0])
		steps = np.vstack([Y0, increment.T])

		#save euler paths to object
		self.euler_paths = np.cumprod(steps, axis=0)

		return self.euler_paths

	def Milstein(self) -> np.ndarray:
		dt: float = self.dt
		mu: float = self.mu
		sigma: float or np.ndarray = self.sigma
		S0: float = self.S0
		dW: np.ndarray = self.Random_walk_paths


		a = mu * dt
		b = sigma * dW
		c = 0.5 * (sigma**2)*(dW**2 - dt)
		d = 1 + a + b + c


		Y0 = S0 * np.ones(dW.shape[0])
		steps = np.vstack([Y0, d.T])

		self.milstein_paths = np.cumprod(steps, axis=0)

		return self.milstein_paths


	def show_paths(self, paths=None) -> None:
		"""Plots the GBM paths"""
		paths = self.analytic_paths if paths is None else paths

		#plt.style.use('seaborn')
		# Define time interval correctly
		time = np.linspace(0, self.years, self.n_steps + 1)
		# Require numpy array that is the same shape as St
		tt = np.full(shape=(self.n_paths, self.n_steps + 1), fill_value=time).T

		plt.plot(tt, paths)
		plt.xlabel("Years $(t)$")
		plt.ylabel("Stock Price $(S_t)$")
		plt.title(
			"$dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(self.S0,
			                                                                                      self.mu,
			                                                                                      self.sigma)
			)
		plt.show()
