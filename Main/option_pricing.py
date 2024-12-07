import numpy as np

def black_scholes_call(S0, K, T, r, sigma):
    """
    Black-Scholes formula for a European call option price.
    return: Call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
