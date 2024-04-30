import torch

class Max_Portfolio:
	def __init__(self, n_portfolios, portfolio_size, paths):
		self.n_portfolios = n_portfolios
		self.portfolio_size = portfolio_size
		self.portfolio_paths = self.Create_portfolios(paths)

	def Create_portfolios(self, paths):
		out = torch.zeros(self. n_portfolios, 1, paths.shape[-1])
		for i in range(self.n_portfolios):
			start = i*self.portfolio_size
			end = (i+1)*self.portfolio_size

			paths = paths.squeeze(1)
			out[i, :, :] = torch.max(paths[start:end, :], dim=0).values
			a = out.squeeze(1)
			f = a.numpy()
			s = 1
		return out