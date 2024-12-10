import numpy as np
from scipy.stats import norm

def black_scholes_call(S0, K, T, r, sigma):
    """
    Black-Scholes formula for a European call option price.
    
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    return: Call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def option_greeks(S0, K, T, r, sigma):
    """
    Calculate the Greeks for a European call option using the Black-Scholes formula.
    
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    return: Dictionary containing delta, gamma, vega, theta, and rho
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta = - (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
