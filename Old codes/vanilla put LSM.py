# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:26:47 2024

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



# np.random.seed(1)
#decide whether sampling the GBM paths should be included in the computational time. 
# Record the start time
start_time = time.time()

# Parameters
r = 0.06
# mu=0.2
sigma = 0.2
T = 1
NoOfSteps = 1000
NoOfPaths = 50000
S_0 = 36
K = 40
delta = 0
N_exercises = 50
exercise_dates = [i*T/N_exercises for i in range(1,N_exercises+1)]
# exercise_dates = [1,2,3]

def GBM(NoOfPaths,NoOfSteps,T,r,delta,sigma,S_0):
    S = np.zeros([NoOfPaths,NoOfSteps+1])
    Z = np.random.normal(0,1,[NoOfPaths,NoOfSteps])
    
    S[:,0] = S_0
    dt = T / NoOfSteps
    for i in range(NoOfSteps):
        S[:,i+1] = S[:,i] + r*S[:,i]*dt + sigma*S[:,i]*np.sqrt(dt)*Z[:,i]
    return S
#Stock price paths as defined in the LSM article
# S = np.array([[1,1.09,1.08,1.34],
#               [1,1.16,1.26,1.54],
#               [1,1.22,1.07,1.03],
#               [1,.93,.97,.92],
#               [1,1.11,1.56,1.52],
#               [1,.76,.77,.90],
#               [1,.92,.84,1.01],
#                 [1,.88,1.22,1.34]])




S = GBM(NoOfPaths, NoOfSteps, T, r, delta, sigma, S_0)

dt = T / len(S[0,:])
if int(exercise_dates[0]/dt) ==1:
    raise ValueError("Too many exercise dates selected")
cf = np.zeros_like(S)
exercise_times = [int(round(date /dt,0)) -1  for date in exercise_dates]
# exercise_times = [1,2,3]
for i in range(len(S[0,:])):
    cf[:,i] = np.maximum(K-S[:,i],0)

cf = cf[:,exercise_times]
print("cf=",cf)
dcf = np.zeros_like(cf)
dcf[:,-1] = np.exp(-r*exercise_dates[-1]) * cf[:,-1]
print('dcf',dcf)
for k in range(len(cf[0,:])-2,-1,-1):
    print(k)
    i = exercise_times[k]
    spot_price = S[:,i]
    print("spot_price",spot_price)
    itm = spot_price < K
    print(itm)
    if all(not x for x in itm):
       cf[:,k+1] = 0
       ee = [False for _ in range(len(cf[:,k]))]
    else:
        
        X = spot_price[itm]
        
        ones = np.ones_like(X)
        X2 = X*X
        # X3 = X*X*X
        Xs = np.column_stack([ones,X,X2])
        Y  = np.exp(-r*(exercise_dates[k+1]-exercise_dates[k]))*cf[itm,k+1]
        print("Xs",Xs)
        print("Y",Y)
        coef = np.linalg.inv(Xs.T@Xs)@Xs.T @ Y
        print(coef)
        ce = Xs @ coef
        print("ce",ce)
        continuation = np.zeros_like(cf[:,k])
        continuation[itm] = ce
        print("cont",continuation)
        cf[:,k] = np.where(cf[:,k]<continuation,0,cf[:,k])
        ee = cf[:,k] > continuation
        
    print("ee", ee)
    cf[ee,k+1:] = 0
    dcf[ee,k+1:] = 0
    print("cf",cf)
    dcf[:,k] = np.exp(-r*exercise_dates[k])*cf[:,k]
    print("dcf",dcf)

sup_dcf = [np.max(dcf,1)]
price = np.mean(sup_dcf)
print("price",price)
strategy = np.where(cf > 0, 1,0)
print("exercise strategy", strategy)
for k in range(len(strategy[0,:])):
    print(f"amount of exercises in period {k}:",np.sum(strategy[:,k]))

