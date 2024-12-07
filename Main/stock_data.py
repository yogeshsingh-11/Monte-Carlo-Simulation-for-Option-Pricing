import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    :param symbol: Stock symbol (e.g., 'AAPL')
    :param start_date: Start date for data retrieval (e.g., '2020-01-01')
    :param end_date: End date for data retrieval (e.g., '2024-01-01')
    :return: DataFrame containing historical stock prices
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    stock_data = stock_data[['Date', 'Close']]  # We only need the Date and Close price
    return stock_data

