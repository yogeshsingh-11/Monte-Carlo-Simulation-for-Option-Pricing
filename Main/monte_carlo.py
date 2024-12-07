import numpy as np
import pandas as pd

def monte_carlo_simulation(S0, K, T, r, sigma, num_simulations, num_steps):
    """
    Perform Monte Carlo simulation to estimate the price of a European call option.
    
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    num_simulations: Number of simulation paths
    num_steps: Number of time steps in each simulation
    return: Estimated option price
    """
    dt = T / num_steps
    option_payoffs = []
    
    for _ in range(num_simulations):
        # Simulate the price path
        prices = [S0]
        for _ in range(num_steps):
            # Geometric Brownian Motion (GBM) model for stock price evolution
            dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion term
            S = prices[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
            prices.append(S)
        
        # Calculate the payoff at maturity
        payoff = max(prices[-1] - K, 0)  # European Call Option payoff
        option_payoffs.append(payoff)
    
    # Discount the payoffs to present value
    option_price = np.exp(-r * T) * np.mean(option_payoffs)
    return option_price

