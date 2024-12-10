import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.monte_carlo import monte_carlo_simulation
from src.option_pricing import black_scholes_call, option_greeks
from src.stock_data import get_stock_data

st.title("Monte Carlo Simulation for Option Pricing")

# Streamlit UI components for user input
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

# Fetch stock data
st.write(f"Fetching data for {symbol} from {start_date} to {end_date}")
stock_data = get_stock_data(symbol, start_date, end_date)

st.write("Historical Stock Data", stock_data.tail())

# Get latest stock price (current close price)
S0 = stock_data['Close'].iloc[-1]

# Inputs for option pricing
K = st.number_input("Strike Price", min_value=1, value=int(S0), step=1)
T = st.number_input("Time to Maturity (in years)", min_value=0.01, value=1.0, step=0.01)
r = st.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.05, step=0.001)
sigma = st.number_input("Volatility", min_value=0.01, value=0.2, step=0.01)
num_simulations = st.number_input("Number of Simulations", min_value=100, value=10000, step=100)
num_steps = st.number_input("Number of Time Steps", min_value=10, value=252, step=1)

# Run Monte Carlo simulation
if st.button("Run Simulation"):
    option_price_monte_carlo = monte_carlo_simulation(S0, K, T, r, sigma, num_simulations, num_steps)
    option_price_bs = black_scholes_call(S0, K, T, r, sigma)
    greeks = option_greeks(S0, K, T, r, sigma)
    
    st.write(f"Monte Carlo Estimated Option Price: ${option_price_monte_carlo:.2f}")
    st.write(f"Black-Scholes Estimated Option Price: ${option_price_bs:.2f}")
    st.write("Option Greeks", greeks)

    # Plotting the Monte Carlo simulation results
    option_prices = [monte_carlo_simulation(S0, K, T, r, sigma, 1, num_steps) for _ in range(num_simulations)]
    plt.hist(option_prices, bins=50, alpha=0.75, color='blue', label='Simulated Option Prices')
    plt.axvline(option_price_monte_carlo, color='red', linestyle='dashed', linewidth=2, label="Monte Carlo Price")
    plt.axvline(option_price_bs, color='green', linestyle='dashed', linewidth=2, label="Black-Scholes Price")
    plt.legend()
    st.pyplot()

