# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:12 2024

@author: Arco
"""
import numpy as np



# To compare Monte Carlo simulation with a Neural Network we use this class since we
# do not want to spend computational budget on transforming the data to tensors

class Options:





	# def __init__(self, payoff_function):
	#     self.payoff_function = payoff_function

	# def calculate_payoff(self, spot_price, strike_price,tau):

	#     return self.payoff_function(spot_price, strike_price,tau)
	# tau is time to maturity i.e. T-t.
	# discounted payoff of vanilla european call option

	@staticmethod
	def Vanilla_European(spot_price, strike_price, r, discount):
		return np.exp(-discount * r) * np.maximum(spot_price - strike_price, 0)  # Call option payoff function


	# discounted payoff of american call option
	@staticmethod
	def American(self, S, strike_price, r, T):
		p = len(S[0, :])
		V = np.zeros([len(S[:, 0]), p])
		for i in range(p):
			discount = i / p
			V[:, i] = self.Vanilla_European(S[:, i], strike_price, r, discount)

		return V


	# discounted payoff Asian option
	@staticmethod
	def Asian(S, strike, r, T):
		p = len(S[0, :])
		V = np.zeros([len(S[:, 0]), p])
		for i in range(1, p):
			discount = i / p
			V[:, i] = np.exp(-discount * r) * np.maximum(np.mean(S[:, :i], axis=1) - strike, 0)
		return V


	# discounted payoff bermuda max call option not yet correct
	@staticmethod
	def Ber_max(S, strike_price, r, T, exercise_dates):
		# S should be 3 dimensional.: S1 = np.zeros([NoOfPaths,len(X_0),NoOfSteps+1])
		p = len(S[0, 0, :])
		if len(exercise_dates) > p:
			raise ValueError("More exercise dates than time points")

		prices = np.zeros(len(S[:, 0, 0]))
		for n in range(len(S[:, 0, 0])):
			dt = T / (p - 1)
			payoff = np.maximum(S[n, :, :] - strike_price, 0)
			# Discounted payoff at each exercise date
			discount_factors = np.exp(-r * np.array(exercise_dates))
			exercise_steps = [int(date / dt) for date in exercise_dates]  # Map exercise dates to time steps

			discounted_payoff = payoff[:, exercise_steps] * discount_factors

			prices[n] = np.mean(np.max(discounted_payoff, axis=1))

		return np.mean(prices)
