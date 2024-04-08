import torch.nn as nn
import torch

torch.set_default_dtype(torch.float64)
class C_theta():
	def __init__(self, d, widths):
		self.dimension = d
		self.widths = widths
		self.depth= len(widths) + 2

		self.activation = nn.Tanh
		self.network = self.Create_network()

	def Create_network(self):
		layers = [nn.Linear(1, self.widths[0]), self.activation()]

		for i in range(1, len(self.widths)):
			width_in = self.widths[i - 1]
			width_out = self.widths[i]

			layers.append(nn.Linear(width_in, width_out))
			layers.append(self.activation())

		layers.append(nn.Linear(self.widths[-1], 1))
		layers.append(self.activation())

		return nn.Sequential(*layers)

	def Train(self):

		return
