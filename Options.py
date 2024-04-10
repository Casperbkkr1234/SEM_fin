# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:12 2024

@author: Arco
"""
import numpy as np
import torch
class Options:
    # def __init__(self, payoff_function):
    #     self.payoff_function = payoff_function

    # def calculate_payoff(self, spot_price, strike_price,tau):

    #     return self.payoff_function(spot_price, strike_price,tau)
    #tau is time to maturity i.e. T-t. 
    #discounted payoff of vanilla european call option

    def __Vanilla_European_stat(self, spot_price, strike_price, r, discount):
        zeros = torch.zeros(spot_price.shape)
        return np.exp(-discount*r)*torch.maximum(spot_price - strike_price, zeros)  # Call option payoff function
        #discounted payoff of american call option

    @staticmethod
    def  Vanilla_European(spot_price, strike_price, r, discount):
        return Options.__Vanilla_European_stat(Options, spot_price, strike_price, r, discount)

    @staticmethod
    def American(S, strike_price, r, T):

        D1 = torch.arange(0, S.shape[-1])
        D2 = D1.repeat(S.shape[0],1,1)
        D3 = D2/S.shape[-1]

        return Options.__Vanilla_European_stat(Options, S, strike_price, r, D3)
    
    #discounted payoff Asian option
    @staticmethod
    def Asian(self,S,strike,r,T):
        p=len(S[0,:])
        V = np.zeros([len(S[:,0]),p])
        for i in range(1,p):
            discount=i/p
            V[:,i] = np.exp(-discount*r)*np.maximum(np.mean(S[:,:i],axis=1)-strike,0)
        return V
        
    #discounted payoff bermuda max call option
    def Ber_max(self,S,strike_price,r,T):
        p=len(S[0,:])
        V = np.zeros(p)
        for i in range(1,p):
            discount = i/p
            V[i] = np.exp(-discount*r)*np.maximum(np.max(S[:,i])-strike_price,0)
        return V