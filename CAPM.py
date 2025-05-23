import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm


tickers = ['MSFT', 'DE', 'COST', 'BYDDY', 'AMD', 'GLD']
data = yf.download(tickers, start="2019-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

print(prices.head())