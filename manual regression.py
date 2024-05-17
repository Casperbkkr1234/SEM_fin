# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:46:08 2024

@author: Arco
"""
import numpy as np
r=0.06
Y = np.exp(-r)*np.array([0.18,0.2,0,0.07])
X = np.array([0.97,0.77,1.08,1.07])
ones=np.ones_like(X)
X2=X*X
Xs = np.column_stack((ones,X,X2))
coefficients = np.linalg.inv(Xs.T@Xs)@Xs.T@Y
print("beta=",coefficients)
fit = Xs@coefficients
g = np.exp(-r)*np.array([0.13,0.33,0.02,0.03])
print("realized payoff",g)
print("fit=",fit)

Y = np.exp(-r)*np.array([0.13,0.33,0.02])
X = np.exp(-r)*np.array([0.93,0.76,1.09])
ones =np.ones_like(X)
X2 = X*X
Xs = np.column_stack((ones,X,X2))
coefficients = np.linalg.inv(Xs.T@Xs)@Xs.T@Y
print("beta=",coefficients)
fit = Xs@coefficients
g=np.exp(-r)*np.array([0.17,0.34,0.01])
print("realized payoff",g)
print("fit=",fit)


