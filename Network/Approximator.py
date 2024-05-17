import torch.nn as nn
import torch

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


class C_theta(nn.Module):
	"""
	Linear approximation function for Longstaff-Schwartz approximator of payoff
	"""


	def __init__(self, d: int, widths: list, dev):
		super().__init__()
		self.dimension = d
		self.widths = widths
		self.dev = dev
		self.depth = len(widths) + 2
		self.loss = nn.MSELoss()#reduction='sum')
		self.activation = nn.Tanh
		self.network = self.Create_network()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)



	def Create_network(self) -> torch.nn.Sequential:
		"""
		Creates a linear network with Tanh activation function.

		:return: Linear sequential network
		"""
		layers = [nn.Linear(self.dimension + 1, self.widths[0], device=self.dev), self.activation()]

		for i in range(1, len(self.widths)):
			width_in = self.widths[i - 1]
			width_out = self.widths[i]

			layers.append(nn.Linear(width_in, width_out, device=self.dev))
			layers.append(self.activation())

		layers.append(nn.Linear(self.widths[-1], 1, device=self.dev))

		return nn.Sequential(*layers)


	def forward(self, X_t: torch.Tensor) -> torch.Tensor:
		"""
		Perform forward pass of network
		"""
		x_t = 1
		return self.network(X_t)


	def Train(self, X_t: torch.Tensor, target: torch.Tensor) -> list:
		running_loss = []
		for (i, idx) in enumerate(X_t):
			self.optimizer.zero_grad()

			output = self.forward(idx)
			loss = self.loss(output, target[i, :])
			loss.backward()
			self.optimizer.step()

			running_loss.append(loss.item())

		return running_loss

	def criterion(self, target, loss):

		diff = target - loss
		diff_2 = diff**2
		som = torch.sum(diff_2)

		return som

	def Train_batch(self, X_t: torch.Tensor, target: torch.Tensor, batch_size=1000) -> list:
		running_loss = []
		n_samples = X_t.shape[0]
		batches = int(n_samples / batch_size)
		for _ in range(100):
			#print(_)
			for i in range(1, batches):
				batch = X_t[(i - 1) * batch_size:i * batch_size, :]
				#self.optimizer.zero_grad()
				m = nn.BatchNorm1d(X_t.shape[1], affine=False)
				batch = m(batch)
				output = self.forward(batch)
				t = target[(i - 1) * batch_size:i * batch_size, :]
				#loss = self.criterion(t, batch)
				loss = self.loss(output, target[(i - 1) * batch_size:i * batch_size, :])#, reduction="sum")
				loss.backward()
				if i == 1:
					running_loss.append(loss.item())
				self.optimizer.step()
				self.optimizer.zero_grad()

		#plt.plot(running_loss)
		#plt.show()
		return running_loss
