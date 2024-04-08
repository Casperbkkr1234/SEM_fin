# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:12 2024

@author: Arco
"""
import numpy as np

class calloptionpayoff:
    # def __init__(self, payoff_function):
    #     self.payoff_function = payoff_function

    # def calculate_payoff(self, spot_price, strike_price,tau):

    #     return self.payoff_function(spot_price, strike_price,tau)
    #tau is time to maturity i.e. T-t. 
    #discounted payoff of vanilla european call option
    def Vanilla_European(self,spot_price, strike_price,r,discount):
        return np.exp(-discount*r)*np.maximum(spot_price - strike_price, 0)  # Call option payoff function
    #discounted payoff of american call option
    def American(self,S,strike_price,r,T):
        p=len(S[0,:])
        V = np.zeros([len(S[:,0]),p])
        for i in range(p):
            discount=i/p
            V[:,i] = self.Vanilla_European(S[:,i],strike_price,r,discount)
            
        return V
    
    #discounted payoff Asian option
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