import numpy as np
a=1
np.random.seed(1)

class Payoff:
    def __init__(self):
        self.a=1

    # def calculate_payoff(self, spot_price, strike_price,tau):

    #     return self.payoff_function(spot_price, strike_price,tau)
    #tau is time to maturity i.e. T-t. 

    def Vanilla_European(self, spot_price, strike_price,discount):
        return np.exp(-discount)*np.maximum(spot_price - strike_price, 0)  # Call option payoff function
    
    def American(self, S, strike_price, T):
        n_steps=S.shape[0]
        V = np.zeros(S.shape)
        for i in range(n_steps):
            discount=i/n_steps
            V[i,:] = self.Vanilla_European(S[i,:], strike_price, discount)
            
        return V

    def Asian(self, S, strike, T):
        p=len(S[0,:])
        V = np.zeros([len(S[:,0]),p])
        for i in range(1,p):
            discount=i/p
            V[:,i] = np.exp(-discount)*np.maximum(np.mean(S[:,:i],axis=1)-strike,0)
        return V
        
    
    def Ber_max(self,portfolio_spot,strike_price,T):
        return np.exp(-tau)*np.maximum(max(portfolio_spot)-strike_price,0)
"""
European_option = calloptionpayoff()

r = 0.05   # Drift
sigma = 0.2 # Volatility
n_steps = 100  # Number of steps
n_paths = 2  # Number of paths to simulate
S0 = 100  # Initial stock price
strike_price = 90  # Strike price
T=1
def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,X_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(X_0)

    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        X[:,i+1] = X[:,i] - r * dt + sigma *np.power(dt, 0.5)*Z[:,i]

        
        time[i+1] = time[i] +dt
        
    # Compute exponent of ABM

    S1 = np.exp(X)

    return S1
S=GeneratePathsGBM(n_paths,n_steps, T, r, sigma, S0)
spot_prices = S[:,-1]
tau=1


bermuda_max_call = European_option.Ber_max(spot_prices, strike_price, tau)
American_call = European_option.American(S, strike_price, T)
Asian_call = European_option.Asian(S, strike_price, T)
print(S,American_call)
print("asian payoffs",Asian_call)
"""