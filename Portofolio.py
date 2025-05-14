import yfinance as yf
import numpy as np
import pandas as pd
# Minor edit to trigger Git change detection

tickers = ['AAPL', 'LMT', 'ALHC', 'MC.PA', 'PBR']

data = yf.download(tickers, start="2020-01-01", auto_adjust=False)['Adj Close']
log_returns = np.log(data / data.shift(1)).dropna()

print(data.head())
print(log_returns.head())
print("Este es un cambio de prueba")