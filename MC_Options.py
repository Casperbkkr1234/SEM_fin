# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:12 2024

@author: Arco
"""
import numpy as np

#To compare Monte Carlo simulation with a Neural Network we use this class since we
#do not want to spend computational budget on transforming the data to tensors

class Options:
    # def __init__(self, payoff_function):
    #     self.payoff_function = payoff_function

    # def calculate_payoff(self, spot_price, strike_price,tau):

    #     return self.payoff_function(spot_price, strike_price,tau)
    #tau is time to maturity i.e. T-t. 
    #discounted payoff of vanilla european call option
    @staticmethod
    def Vanilla_European(spot_price, strike_price,r,discount):
        return np.exp(-discount*r)*np.maximum(spot_price - strike_price, 0)  # Call option payoff function
    #discounted payoff of american call option
    @staticmethod
    def American(self,S,strike_price,r,T):
        p=len(S[0,:])
        V = np.zeros([len(S[:,0]),p])
        for i in range(p):
            discount=i/p
            V[:,i] = self.Vanilla_European(S[:,i],strike_price,r,discount)
            
        return V
    
    #discounted payoff Asian option
    @staticmethod
    def Asian(self,S,strike,r,T):
        p=len(S[0,:])
        V = np.zeros([len(S[:,0]),p])
        for i in range(1,p):
            discount=i/p
            V[:,i] = np.exp(-discount*r)*np.maximum(np.mean(S[:,:i],axis=1)-strike,0)
        return V
        
    #discounted payoff bermuda max call option not yet correct
    @staticmethod
    def Ber_max(self,S,strike_price,r,T):
        p=len(S[0,:])
        V = np.zeros(p)
        for i in range(1,p):
            discount = i/p
            V[i] = np.exp(-discount*r)*np.maximum(np.max(S[:,i])-strike_price,0)
        return V