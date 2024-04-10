# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:12 2024

@author: Arco
"""
import numpy as np
import torch
class Options:
    def __Vanilla_European_stat(self, spot_price, strike_price, r, discount):
        zeros = torch.zeros(spot_price.shape)
        return np.exp(-discount*r)*torch.maximum(spot_price - strike_price, zeros)  # Call option payoff function
        #discounted payoff of american call option

    @staticmethod
    def  Vanilla_European(spot_price, strike_price, r, discount):
        return Options.__Vanilla_European_stat(Options, spot_price, strike_price, r, discount)

    @staticmethod
    def American(Spot_price, strike_price, r):

        D1 = torch.arange(0, Spot_price.shape[-1])
        D2 = D1.repeat(Spot_price.shape[0], 1, 1)
        discount = D2 / Spot_price.shape[-1]

        return Options.__Vanilla_European_stat(Options, Spot_price, strike_price, r, discount)
    
    #discounted payoff Asian option
    @staticmethod
    def Asian(Spot_price, strike, r):
        D1 = torch.arange(0, Spot_price.shape[-1])
        D2 = D1.repeat(Spot_price.shape[0], 1, 1)
        discount = D2 / Spot_price.shape[-1]

        out = torch.zeros(Spot_price.shape)
        zeros =  torch.zeros(Spot_price.shape[0], 1)
        for i in range(1, Spot_price.shape[-1]):
            avg_value = torch.mean(Spot_price[:,:,:i], dim=2) - strike

            out[:,:,i] = torch.exp(-discount[:,:,i] * r) *torch.maximum(avg_value, zeros)
        return out
        
    #discounted payoff bermuda max call option
    @staticmethod
    def Ber_max(Spot_price, strike_price, r):
        # TODO make this work with tensors
        p=len(Spot_price[0, :])
        V = np.zeros(p)
        for i in range(1,p):
            discount = i/p
            V[i] = np.exp(-discount*r)*np.maximum(np.max(Spot_price[:, i]) - strike_price, 0)
        return V