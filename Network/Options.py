import numpy as np
import torch



class Options:

	def __Vanilla_European_stat(self, spot_price: torch.Tensor,
	                            strike_price: float,
	                            r: float,
	                            discount: float) -> torch.Tensor:
		zeros = torch.zeros(spot_price.shape)
		m = torch.maximum(spot_price - strike_price,                                          zeros)
		return np.exp(-discount * r) * torch.maximum(spot_price - strike_price,
		                                             zeros)  # Call option payoff function


	# discounted payoff of american call option


	@staticmethod
	def Vanilla_European(spot_price, strike_price, r, discount) -> torch.Tensor:
		return Options.__Vanilla_European_stat(Options, spot_price, strike_price, r, discount)


	@staticmethod
	def American(spot_price, strike_price, r) -> torch.Tensor:

		D1 = torch.arange(0, spot_price.shape[-1])  # 1,2,3,4,5..
		D2 = D1.repeat(spot_price.shape[0], 1, 1)  # stacks
		discount = D2 / spot_price.shape[-1]

		return Options.__Vanilla_European_stat(Options, spot_price, strike_price, r, discount)


	# discounted payoff Asian option
	@staticmethod
	def Asian(spot_price, strike, r) -> torch.Tensor:
		D1 = torch.arange(0, spot_price.shape[-1])
		D2 = D1.repeat(spot_price.shape[0], 1, 1)
		discount = D2 / spot_price.shape[-1]

		out = torch.zeros(spot_price.shape)
		zeros = torch.zeros(spot_price.shape[0], 1)
		for i in range(1, spot_price.shape[-1]):
			avg_value = torch.mean(spot_price[:, :, :i], dim=2) - strike

			out[:, :, i] = torch.exp(-discount[:, :, i] * r) * torch.maximum(avg_value, zeros)
		return out


	def __Ber_max_t(self, spot_price_t, strike_price, r):

		return


	# discounted payoff bermuda max call option
	@staticmethod
	def Ber_max(spot_price, strike_price, r, ex_times, dt):
		D1 = torch.arange(0, spot_price.shape[-1])
		D2 = D1.repeat(spot_price.shape[0], 1, 1)
		discount = D2 / spot_price.shape[-1]
		exercise_steps = [int(date / dt) for date in ex_times]

		out = torch.zeros(spot_price.shape[0], 1, len(ex_times))

		zeros = torch.zeros(spot_price.shape[0], 1)

		for k in range(len(exercise_steps)):
			i = exercise_steps[k]
			out[:, :, k] = np.exp(-discount[:, :, i-1] * r) * torch.maximum(spot_price[:, :, i-1] - strike_price, zeros)



		out1 = out.squeeze(1)
		a = out1.numpy()
		return out
