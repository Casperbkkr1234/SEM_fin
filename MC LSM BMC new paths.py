# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:15:45 2024

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
r = 0.05
# mu=0.2
sigma = 0.2
T = 3
NoOfSteps = 100
NoOfPaths = 10000
n_assets =5
#assume all have same initial stock price.
S_0 = 90
S_0 = [S_0 for k in range(n_assets)]
K = 100
delta = 0.1
N_exercises = 9
exercise_dates = [np.round(i*T/N_exercises,2) for i in range(1,N_exercises+1)]
payoff = lambda s: np.maximum(s-K,0)
# payoff = lambda s: np.maximum(K-s,0)
print(exercise_dates)
def GBM(NoOfPaths,NoOfSteps,T,r,delta,sigma,X_0):    
    #Define an array for len(X_0) dimensional asset process.
    #k th simulation is = S[k,:,:], which contains len(X_0) assets and has NoOfSteps timesteps. The number of
    #simulations is running from 0 to NoOfPaths, ie 0<k<NoOfPaths
    #i th asset in k th simulation is S[k,i,:]
    #value at time t of ith asset in kth simulation is S[k,i,t]
    S1 = np.zeros([NoOfPaths,len(X_0),NoOfSteps+1])
    print(S1)
    for n in range(NoOfPaths):
        # np.random.seed(n)
        Z = np.random.normal(0.0,1.0,[len(X_0),NoOfSteps])
        
        X = np.zeros([len(X_0), NoOfSteps+1])
        

        X[:,0] = np.log(X_0)

        dt = T / float(NoOfSteps)
        for i in range(0,NoOfSteps):
    
           # Making sure that samples from a normal have mean 0 and variance 1
           
           # if len(X_0) > 1:
           #     Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
    
           X[:,i+1] = X[:,i] + (r-delta-(sigma**2 )/2) * dt + sigma *np.power(dt, 0.5)*Z[:,i]

     
        # Compute exponent of ABM

        S1[n,:,:] = np.array(np.exp(X))
        # print(np.exp(X))

            
        # plt.plot(np.exp(X[0]))
    return S1

paths = GBM(2*NoOfPaths, NoOfSteps, T, r, delta, sigma, S_0)
S = paths[:NoOfPaths]
S2 = paths[NoOfPaths:]
# S = np.load(f"GBM_sample_{NoOfPaths}.npy")
dt = T / NoOfSteps
cf = np.zeros([len(S[:,0,0]),len(S[0,0,:])])
exercise_times = [int(date /dt) -1  for date in exercise_dates]
for t in range(len(S[:,0,0])):
    for i in range(1,len(S[0,0,:])):
        cf[t,i] = payoff(np.max(S[t,:,i]))

    
cf = cf[:,exercise_times]
print("cf=",cf)
dcf = np.zeros_like(cf)
dcf[:,-1] = np.exp(-r*exercise_dates[-1]) * cf[:,-1]
coefficients = []
print('dcf',dcf)
for k in range(len(cf[0,:])-2,-1,-1):
    print(k)
    i = exercise_times[k]
    
    itm = cf[:,k] > 0
    spot_price = S[itm,:,i]
    print("spot_price",spot_price)
    if all(not x for x in itm):
       cf[:,k+1] = 0
       ee = [False for _ in range(len(cf[:,k]))]
    else:
        #idea split up for each stock
        beta = []
        for idx in range(n_assets):
            X = spot_price[:,idx]
            ones = np.ones_like(X)
            X2 = X*X
            Xs = np.column_stack([ones,X,X2])
            Y = np.exp(-r*(exercise_dates[k+1]-exercise_dates[k]))*cf[itm,k+1]
            beta.append(np.linalg.inv(Xs.T@Xs)@Xs.T @ Y)
        
        
        # X = np.mean(spot_price,1)
        # ones = np.ones_like(X)
        # X2 = X*X
        # Xs = np.column_stack([ones,X,X2])
        # Y  = np.exp(-r*(exercise_dates[k+1]-exercise_dates[k]))*cf[itm,k+1]
        # beta = np.linalg.inv(Xs.T@Xs)@Xs.T @ Y
        coefficients.append(beta)

S = S2
for k in range(len(coefficients)):
    print(k)
    i = exercise_times[k]

    itm = cf[:,k] > 0
    spot_price = S[itm,:,i]
    print("spot_price",spot_price)
    if all(not x for x in itm):
       cf[:,k+1] = 0
       ee = [False for _ in range(len(cf[:,k]))]
    else:
        beta = coefficients[k]
        continuation = np.zeros_like(S[:,:,k])
        for idx in range(n_assets):
            X = spot_price[:,idx]
            ones = np.ones_like(X)
            X2 = X*X
            Xs = np.column_stack([ones,X,X2])
            ce = Xs @ beta[idx]
            continuation[itm,idx] = ce
        continuation_payoff = np.zeros_like(continuation[:,0])
        for idx in range(len(continuation_payoff)):
            continuation_payoff[idx] = payoff(np.max(continuation[idx]))
        
        
        
        cf[:,k] = np.where(cf[:,k] < continuation_payoff,0,cf[:,k])
        
        
        
        
        
        
        
        
        # X = np.mean(spot_price,1)
        # ones = np.ones_like(X)
        # X2 = X*X
        # Xs = np.column_stack([ones,X,X2])
        # beta = coefficients[k]
        # ce = Xs @ beta
        # continuation = np.zeros_like(cf[:,k])
        # continuation[itm] = ce
        # cf[:,k] = np.where(cf[:,k]<continuation,0,cf[:,k])
        ee = cf[:,k] > continuation_payoff
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
strategy = np.where(dcf > 0, 1,0)
print("exercise strategy", strategy)
for k in range(len(strategy[0,:])):
    print(f"amount of exercises in period {k}:",np.sum(strategy[:,k]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    