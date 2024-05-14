
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:19 2024

@author: Arco
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial
import scipy.stats as stats
import scipy
import time
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



np.random.seed(1)
#decide whether sampling the GBM paths should be included in the computational time. 
# Record the start time
start_time = time.time()

# Parameters
r = 0.06
# mu=0.2
sigma = 0.2
T = 1
NoOfSteps = 1000
NoOfPaths = 1000
S_0 = 90
K = 100
delta = 0
N_exercises = 50
exercise_dates = [np.round(i*T/N_exercises,2) for i in range(1,N_exercises+1)]
# payoff = lambda s: np.maximum(s-K,0)
payoff = lambda s: np.maximum(K-s,0)
print(exercise_dates)
def GBM(NoOfPaths,NoOfSteps,T,r,delta,sigma,S_0):
    S = np.zeros([NoOfPaths,NoOfSteps+1])
    Z = np.random.normal(0,1,[NoOfPaths,NoOfSteps])
    
    S[:,0] = S_0
    dt = T / NoOfSteps
    for i in range(NoOfSteps):
        S[:,i+1] = S[:,i] + r*S[:,i]*dt + sigma*S[:,i]*np.sqrt(dt)*Z[:,i]
    return S

paths = GBM(2*NoOfPaths, NoOfSteps, T, r, delta, sigma, S_0)
# S = np.load(f"GBM_sample_{NoOfPaths}.npy")
S = paths[:NoOfPaths,:]
S2 = paths[NoOfPaths:,:]
dt = T / NoOfSteps
cf = np.zeros_like(S)
exercise_times = [int(date /dt) -1  for date in exercise_dates]
for i in range(len(S[0,:])):
    cf[:,i] = payoff(S[:,i])
    
cf = cf[:,exercise_times]
print("cf=",cf)
dcf = np.zeros_like(cf)
dcf[:,-1] = np.exp(-r*exercise_dates[-1]) * cf[:,-1]
coefficients = []
print('dcf',dcf)
for k in range(len(cf[0,:])-2,-1,-1):
    print(k)
    i = exercise_times[k]
    spot_price = S[:,i]
    print("spot_price",spot_price)
    itm = cf[:,k] > 0

    if all(not x for x in itm):
       cf[:,k+1] = 0
       ee = [False for _ in range(len(cf[:,k]))]
    else:
        
        X = spot_price[itm]
        # X = payoff(spot_price)
        ones = np.ones_like(X)
        X2 = X*X
        Xs = np.column_stack([ones,X,X2])
        Y  = np.exp(-r*(exercise_dates[k+1]-exercise_dates[k]))*cf[itm,k+1]
        beta = np.linalg.inv(Xs.T@Xs)@Xs.T @ Y
        coefficients.append(beta)

S = S2
for k in range(len(coefficients)):
    print(k)
    i = exercise_times[k]
    spot_price = S[:,i]
    print("spot_price",spot_price)
    itm = cf[:,k] > 0

    if all(not x for x in itm):
       cf[:,k+1] = 0
       ee = [False for _ in range(len(cf[:,k]))]
    else:
        
        X = spot_price[itm]
        ones = np.ones_like(X)
        X2 = X*X
        Xs = np.column_stack([ones,X,X2])
        beta = coefficients[k]
        ce = Xs @ beta
        continuation = np.zeros_like(cf[:,k])
        continuation[itm] = ce
        cf[:,k] = np.where(cf[:,k]<continuation,0,cf[:,k])
        ee = cf[:,k] > continuation
    # cf[ee,k+1:] = 0
    # dcf[ee,k+1:] =0
    print("cf",cf)
    dcf[:,k] = np.exp(-r*exercise_dates[k])*cf[:,k]
    print("dcf",dcf)

for row in dcf:
    i = np.argmax(row > 0)
    row[:i] = 0
    row[i+1:] = 0
    
sup_dcf = [np.max(dcf,1)]
price = np.mean(sup_dcf)
print("price",price)
strategy = np.where(cf > 0, 1,0)
print("exercise strategy", strategy)
for k in range(len(strategy[0,:])):
    print(f"amount of exercises in period {k}:",np.sum(strategy[:,k]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    