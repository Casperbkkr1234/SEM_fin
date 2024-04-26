
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:37:53 2024

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
T = 1
n_steps = 100
n_paths = 5
# S0 = 100
# K = 100
delta = 0
N_exercises = 4
exercise_dates = [i*T/N_exercises for i in range(1,N_exercises+1)]

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,delta,sigma,X_0):    
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


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price
# Sample_size = np.array([1e3,1e4,1e5,1e6,1e7],dtype=int)
# K = np.array([70,100,130])

n_assets =1
S_0 = 100
S_0 = [S_0 for k in range(n_assets)]
# print(S_0)
S = GeneratePathsGBM(n_paths, n_steps, T, r, delta,sigma, S_0)
# #Check wheter GBM is correct
# plt.plot(np.mean(S,axis=0)[0])
# plt.plot(np.var(S,axis=0)[0])
# t = np.linspace(0,T,n_steps)
# plt.plot(S_0[0]**2*(np.exp((sigma**2)*t)-1))
K=100
def longstaff_schwartz(S, strike, r,exercise_dates):
    #LS Algo for Bermuda Max call option
    #Portfolio simulations = S[:,0,0]
    #i th asset in k th simulation is S[k,i,:]
    #value at time t of ith asset in kth simulation is S[k,i,t]
    dt = exercise_dates[-1]/len(S[0,0,:])
    print("dt",dt)
    cash_flows = np.zeros([len(S[:,0,0]),len(S[0,0,:])])
    # print(cash_flows,len(cash_flows))

    
    for t in range(len(S[:,0,0])):
        for i in range(1,len(S[0,0,:])):
            # print("S[t,:,i]",S[t,:,i])
            # print(S[t,:,i],np.maximum(S[t,:,i]-strike,0))
            # cash_flows[t,i] = np.maximum(np.max(S[t,:,i])-strike,0)
            cash_flows[t,i] = np.maximum((S[t,:,i]-strike),0)
    
    #compute indices of exercise dates
    exercise_steps = [int(date /dt) -1  for date in exercise_dates]
    print(exercise_steps)
    #consider only the cash flows at exercise steps
    cash_flows = cash_flows[:,exercise_steps]
    print("paths", S[:,:,exercise_steps])
    print("first cash flow",cash_flows)
    #initialize an array of discounted cashflows to today given option not exercised until time t
    discounted_cash_flows = np.zeros_like(cash_flows)
    
    
    for i in range(len(exercise_steps)-1,-1,-1):
        # print(i)
        #Compute the payoff at final time T given not exerecised until T.
        if i == len(exercise_steps)-1:
            discounted_cash_flows[:,i] = np.exp(-r * exercise_dates[-1])*cash_flows[:,i]
            # print(discounted_cash_flows,cash_flows)
        else:
            k = exercise_steps[i]
            #Decide wheter option is in the money
            in_the_money=np.max(S[:,:,k],1) > strike
            print(all(not x for x in in_the_money))
            
            #If there do not exist any path in the money then regression on in the money paths is impossible
            #We do not exercise hence we can skip regression part. 
            if all(not x for x in in_the_money):
               cash_flows[:,i+1] = 0
               exercised_early = [False for k in range(len(cash_flows[:,i]))]
            else:
                # print(S[:,:,:k+1])
                # print("sim",np.max(S[:,:,k],1))
                #Initialize payoff regressors for conditional expectation
                # X=np.max(S[in_the_money,:,k],1)
                X=S[in_the_money,:,k]
                # X1=np.exp(-X/2)
                # X2=X1*(1-X1)
                # X3=np.exp(-X1/2)*(1-2*X1+(X1**2)/2)
                # print(X)
                X2 = X*X
                ones = np.ones_like(X)
                
    
                print("discount time",exercise_dates[i+1]-exercise_dates[i])
                #Initialize dependent variable the discounted cashflow from next exercise date to exercise date k
                Y=np.exp(-r*(exercise_dates[i+1]-exercise_dates[i]))*cash_flows[in_the_money,i+1]
                print("X",X,"X^2",X*X,"Y",Y)
                Xs = np.column_stack([ones,X,X2])
                #Compute coefficients using linear regression
                # fitted = Polynomial.fit(X,Y,10)
                # conditional_exp= fitted(X)
                # model_sklearn = LinearRegression()
                # model = model_sklearn.fit(Xs, Y)
                # print(Xs,"coef",model.coef_,"intercept",model.intercept_)
                # conditional_exp = model.predict(Xs)
                print("XS",Xs)
                #manual OLS
                coef = np.linalg.inv(Xs.T@Xs)@Xs.T@Y
                print("Coef",coef)
                conditional_exp = Xs @ coef
                print("ce",conditional_exp)
                #Continuation array to decide whether to exercise or not
                continuation = np.zeros_like(cash_flows[:,i])
                continuation[in_the_money] = conditional_exp
                # continuation = conditional_exp
                print("cont",continuation)
                print("pre cont cf",cash_flows[:,i])
                # print("which 0",continuation > cash_flows[:,i])
                #If continuation value > immidiate exercise then not exercise hence cashflow 0
                cash_flows[:,i] = np.where(continuation> cash_flows[:,i], 0, cash_flows[:,i])
                print("cf",cash_flows)
                #Decide whether we exercised or not
                exercised_early=cash_flows[:,i] > continuation
            print("ee",exercised_early)
            #If we exercise at time t then all future cashflows are zero since option ceases to exist
            cash_flows[exercised_early,i+1:] =0
            # discounted_cash_flows[exercised_early,i+1:]=0
            print("cf",cash_flows)
            # print(k)
            #Discount the cashflow to t_0
            print(i,exercise_dates[i])
            discounted_cash_flows[:,i] = np.exp(-r*exercise_dates[i])*cash_flows[:,i]
            print("dcf",discounted_cash_flows)
    # print(discounted_cash_flows)        
    #For each path harvest the supremum of discounted cash flow
    sim_price = [np.max(discounted_cash_flows[i,:]) for i in range(len(discounted_cash_flows[:,0]))]
    print(sim_price)
    #Compute option value by averaging over all simulations
    option_price = np.mean(sim_price)

    return option_price,discounted_cash_flows,cash_flows
        
# plt.figure(figsize=(8,5))
# for i in range(len(S[:,0,0])):
#     print(S[i])
#     plt.plot(S[i,0,:])
#     plt.plot(S[i,1,:])

price,dcf,cf=longstaff_schwartz(S,K, r,exercise_dates)
print("price=",price)
print("dcf",dcf)
print("cf",cf)
# BS_price=black_scholes_call(S_0, K, T, r, sigma)
# print("BS price", BS_price)



# plt.show()
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")
