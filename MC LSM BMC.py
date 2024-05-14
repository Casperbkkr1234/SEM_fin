
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



# np.random.seed(1)
#decide whether sampling the GBM paths should be included in the computational time. 
# Record the start time
start_time = time.time()

# Parameters
r = 0.05
# mu=0.2
sigma = 0.2
T = 3
NoOfSteps = 100
NoOfPaths = 100000
n_assets =5 #Setting this to 1 results in regular one dimensional options
#assume all have same initial stock price.
S_0 = 100
S_0 = [S_0 for k in range(n_assets)]
K = 100
delta = 0.1#dividend parameter
N_exercises = 9
N_simulations = 25
confidence_level = 0.95
exercise_dates = [np.round(i*T/N_exercises,2) for i in range(1,N_exercises+1)]


#choose payoff function
payoff = lambda s: np.maximum(s-K,0)
# payoff = lambda s: np.maximum(K-s,0)
print(exercise_dates)


#Function for simulating multidimensional GBM processes
def GBM(NoOfPaths,NoOfSteps,T,r,delta,sigma,X_0):    
    #Define an array for len(X_0) dimensional asset process.
    #k th simulation is S[k,:,:], which contains len(X_0) assets and has NoOfSteps timesteps. The number of
    #simulations is running from 0 to NoOfPaths, ie 0<k<NoOfPaths
    #i th asset in k th simulation is S[k,i,:]
    #value at time t of ith asset in kth simulation is S[k,i,t]
    S1 = np.zeros([NoOfPaths,len(X_0),NoOfSteps+1])#Initialize array of processes
    # print(S1)
    for n in range(NoOfPaths):
        # np.random.seed(n)
        Z = np.random.normal(0.0,1.0,[len(X_0),NoOfSteps])
        
        X = np.zeros([len(X_0), NoOfSteps+1])
        

        X[:,0] = np.log(X_0)

        dt = T / float(NoOfSteps)
        for i in range(0,NoOfSteps):
           X[:,i+1] = X[:,i] + (r-delta-(sigma**2 )/2) * dt + sigma *np.power(dt, 0.5)*Z[:,i]
        # Compute exponent of ABM

        #add n-th simulation to S1
        S1[n,:,:] = np.array(np.exp(X))
        # print(np.exp(X))

            
        # plt.plot(np.exp(X[0]))
    return S1

def LSM():
    #Generate 2*NoOfPaths of simulations where we use the first half for estimating continuation values
    #and the second half for pricing.
    S = GBM(NoOfPaths, NoOfSteps, T, r, delta, sigma, S_0)
    # S = np.load(f"GBM_sample_{NoOfPaths}.npy")
    # S = paths[NoOfPaths:]
    # S2 = paths[:NoOfPaths]
    dt = T / NoOfSteps
    
    #Initialize cashflow array
    cf = np.zeros([len(S[:,0,0]),len(S[0,0,:])])
    exercise_times = [int(date /dt) -1  for date in exercise_dates] #Compute exercise date indices
    for t in range(len(S[:,0,0])):
        for i in range(1,len(S[0,0,:])):
            cf[t,i] = payoff(np.max(S[t,:,i]))
    
    #Only consider cashflows at exercise times.
    cf = cf[:,exercise_times]
    # print("cf=",cf)
    
    #Initialize Discounted Cash Flow array where elements are present value of future cashflow 
    dcf = np.zeros_like(cf)
    dcf[:,-1] = np.exp(-r*exercise_dates[-1]) * cf[:,-1] 
    coefficients = [] #Initialize list of coefficients of conditional expectation regression
    print('dcf',dcf)
    #Loop backwards over the cashflows
    for k in range(len(cf[0,:])-2,-1,-1):
        print(k)
        i = exercise_times[k]
        itm = cf[:,k] > 0 #check whether stock is in the money
        spot_price = S[itm,:,i] #only consider stock prices that are in the money
        print("spot_price",spot_price)
        # if all(not x for x in itm):
        if any(itm):#check if there is any stock in the money
            #When there is a stock itm regress the discounted cash flow in the next period on a constant,
            #on the stock price and squared stock price.
            # X = np.mean(spot_price,1)
            X = spot_price
            ones = np.ones_like(X[:,0])
    
            X2 = X*X
            # X3=X2*X
            Xs = np.column_stack((ones,X,X2))
            # Xs=Xs.reshape(-1, Xs.shape[-1])
    
            #Define the discounted cash flow of 
            Y = np.exp(-r*(exercise_dates[k+1]-exercise_dates[k]))*cf[itm,k+1]
    
            # beta = np.linalg.inv(Xs.T@Xs)@Xs.T @ Y
            beta= np.linalg.lstsq(Xs,Y,rcond=None)[0]
            coefficients.append(beta)
    
        else:
            cf[:,k+1] = 0
            ee = [False for _ in range(len(cf[:,k]))]
            
            
    #Do a forward pass to determine the first stopping time
    for k in range(len(coefficients)):
        print(k)
        i = exercise_times[k]
    
        itm = cf[:,k] > 0
        spot_price = S[itm,:,i]
        print("spot_price",spot_price)
        # if all(not x for x in itm):
        if any(itm):
            X = spot_price
            ones = np.ones_like(X[:,0])
    
            X2 = X*X
            # X3 = X2*X
            Xs = np.column_stack((ones,X,X2))
            Xs=Xs.reshape(-1, Xs.shape[-1])
            beta = coefficients[k]
            ce = Xs @ beta
            continuation = np.zeros_like(cf[:,k])
            continuation[itm] = ce
            cf[:,k] = np.where(cf[:,k]<continuation,0,cf[:,k])
            ee = cf[:,k] > continuation
    
        else:
            cf[:,k+1] = 0
            ee = [False for _ in range(len(cf[:,k]))]
    
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
    
    return price
    

prices = np.zeros(N_simulations)
for idx in range(N_simulations):
    prices[idx] = LSM()
    print("Simulation", idx)

#Compute mean price and confidence intervals
mean_price = np.mean(prices)
std_prices = np.std(prices)
margin_of_error = std_prices / np.sqrt(N_simulations) * \
                  stats.t.ppf((1 + confidence_level) / 2, N_simulations - 1)
CI = [mean_price - margin_of_error, mean_price + margin_of_error]
end_time = time.time() 
   
    
run_time = end_time - start_time
print("runtime:",round(run_time),"seconds")    
print("average runtime:", round(run_time/N_simulations,2),"seconds")
print("mean payoff",mean_price)
print("CI size", 2*margin_of_error)
print("CI",CI)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    