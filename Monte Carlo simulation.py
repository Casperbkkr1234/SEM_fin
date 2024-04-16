
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:37:53 2024

@author: Arco
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from Process import GBM
from MC_Options import Options
import scipy.stats as stats
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
mu=0.2
sigma = 0.3
T = 1
n_steps = 100
n_paths = 5
S0 = 100
K = 100
N_exercises = 10
exercise_dates = [i*T/N_exercises for i in range(1,N_exercises+1)]

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,X_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])

    X[:,0] = np.log(X_0)

    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        X[:,i+1] = X[:,i] + (r-(sigma**2 )/2) * dt + sigma *np.power(dt, 0.5)*Z[:,i]

    # Compute exponent of ABM

    S1 = np.exp(X)

    return S1

def Final_price_GBM(NoOfPaths,NoOfSteps,T,r,sigma,S0):
    Z = np.random.normal(0,1,NoOfPaths)
    #for vanilla options one only needs last price hence we can use solution of GBM
    
    if NoOfPaths > 1: 
        Z = (Z-np.mean(Z))/np.std(Z)
    
    S = S0*np.exp((r-(sigma**2)/2)*T + sigma*np.sqrt(T)*Z)
    return S
    

Sample_size = np.array([1e3,1e4,1e5,1e6,1e7],dtype=int)
K = np.array([70,100,130])
def MC_call():
    running_times = np.zeros([len(K),len(Sample_size)])
    price = np.zeros([len(K),len(Sample_size)])
    error = np.zeros([len(K),len(Sample_size)])
    iid_runs = 100
    
    for idx,s in enumerate(Sample_size):
        print(s)
        for j in range(len(K)):
            times = np.zeros(iid_runs)
            errors = np.zeros(iid_runs)
            payoffs = np.zeros(iid_runs)
            for _ in range(iid_runs):
                t = time.time()
                k = K[j]
                spot_price = Final_price_GBM(s, n_steps, T, r, sigma, S0)
                payoff = np.mean(Options.Vanilla_European(spot_price, k, r, T))
                payoffs[_] = payoff
                errors[_] = abs(black_scholes_call(S0, k, T, r, sigma)-payoff)
                et = time.time()
                times[_] = round(et-t,5)
            running_times[j,idx] = np.mean(times)
            error[j,idx] = np.mean(errors)
            price[j,idx] = np.mean(payoffs)
    
    
    names=["running_times", "price", "error"]
    data = [running_times,price,error]
    for k in range(len(data)):
        df = pd.DataFrame(data[k],columns=Sample_size, index=K)
        name = names[k]
        df.to_excel(f"Vanilla_call_{name}.xlsx")
    
S = GeneratePathsGBM(n_paths, n_steps, T, r, sigma, S0)
BM_price = Options.Ber_max(S, 100, r, T, exercise_dates)
print("BM price:",BM_price)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")
