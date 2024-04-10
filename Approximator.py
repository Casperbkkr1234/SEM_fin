import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

class C_theta(nn.Module):
	"""
	Linear approximation function for Longstaff-Schwartz approximator of payoff
	"""
	def __init__(self, d: int, widths: list):
		super().__init__()
		self.dimension = d
		self.widths = widths
		self.depth= len(widths) + 2
		self.loss = nn.MSELoss()
		self.activation = nn.Tanh
		self.network = self.Create_network()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

	def Create_network(self) -> torch.nn.Sequential:
		"""
		Creates a linear network with Tanh activation function.

		:return: Linear sequential network
		"""
		layers = [nn.Linear(1, self.widths[0]), self.activation()]

		for i in range(1, len(self.widths)):
			width_in = self.widths[i - 1]
			width_out = self.widths[i]

			layers.append(nn.Linear(width_in, width_out))
			layers.append(self.activation())

		layers.append(nn.Linear(self.widths[-1], 1))
		layers.append(self.activation())

		return nn.Sequential(*layers)

	def forward(self, X_t: np.ndarray):
		return self.network(X_t)

	def Train(self, X_t: np.ndarray, target: np.ndarray) -> list:
		running_loss = []
		for (i, idx) in enumerate(X_t):
			self.optimizer.zero_grad()

			output = self.forward(idx)
			#a = target[:,i]
			loss = self.loss(output, target[i,:])
			loss.backward()
			self.optimizer.step()

			running_loss.append(loss.item())


		return running_loss

