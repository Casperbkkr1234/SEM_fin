import torch

from Network.Options import Options


class Max_Portfolio:
	def __init__(self, n_portfolios, portfolio_size):
		self.n_portfolios = n_portfolios
		self.portfolio_size = portfolio_size
		#self.portfolio_paths = self.Create_portfolios(paths)

	def Create_portfolios_no_max(self, paths, strike, r):
		portfolio_paths = torch.zeros(self.n_portfolios, self.portfolio_size, paths.shape[-1])
		paths = paths.squeeze(1)

		for i in range(self.n_portfolios):
			start = i*self.portfolio_size
			end = (i+1)*self.portfolio_size
			portfolio_paths[i, :, :] = paths[start:end, :]

		return portfolio_paths

	def Create_portfolios(self, paths, strike, r):
		portfolio_paths = torch.zeros(self.n_portfolios, self.portfolio_size + 1, paths.shape[-1])
		CM = self.Create_max(paths, strike, r)
		paths = paths.squeeze(1)
		CM.squeeze(1)
		for i in range(self.n_portfolios):
			start = i*self.portfolio_size
			end = (i+1)*self.portfolio_size
			#paths = paths.squeeze(1)
			portfolio_paths[i, :-1, :] = paths[start:end, :]

			portfolio_paths[i, -1, :] = CM[i, :]


		return portfolio_paths

	def Create_max(self, paths, strike, r):
		portfolio_paths = torch.zeros(self.n_portfolios, 1, paths.shape[-1])
		# paths = paths.squeeze(1)
		for i in range(self.n_portfolios):
			start = i * self.portfolio_size
			end = (i + 1) * self.portfolio_size
			portfolio_paths[i, :, :] = torch.max(paths[start:end, :, :], 0).values

		BM = Options.Ber_max(portfolio_paths, strike, r)
		a = BM.squeeze(1)
		b = a.numpy()
		return BM