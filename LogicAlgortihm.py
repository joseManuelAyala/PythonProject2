import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AAPL','JNJ','XOM','HD','TSLA']
data = yf.download(tickers, start="2020-01-01", end = "2025-05-15",auto_adjust = False)
prices = data['Adj Close']
log_return = np.log(prices / prices.shift())
log_return = log_return.dropna()

#Caracterizacion Investment Oportunity Set
mean_return = log_return.mean()
Cov= log_return.cov()
Corr= log_return.corr()

#pesos aleatorios
num_portfolios = 10000
weights = np.random.random((num_portfolios, len(tickers)))
weights /= np.sum(weights, axis=1, keepdims=True)



print(weights)
ret = np.dot(weights, mean_return)
vol = np.array([np.sqrt(np.dot(np.dot(w, Cov), w.T)) for w in weights])

plt.figure(figsize = (10,6))
plt.scatter(vol, ret,c=ret/vol, cmap='viridis')
plt.title("Investment opportunity set")
plt.xlabel("Volatility")
plt.ylabel("Expected return")
plt.colorbar(label="Sharpe Ratio")
plt.show()