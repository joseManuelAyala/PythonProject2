import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Descargar datos
tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA', '^GSPC']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

# Calcular retornos diarios
ret = (prices - prices.shift()) / prices.shift()
ret = ret.dropna()

# Calcular retornos anuales
mu = ret.mean() * 252

# Parámetros para el CAPM
ex_returnMarket = '^GSPC'
risk_free_rate = 0.03

# Calcular Beta de cada Asset
market_return = ret[ex_returnMarket]
betas = {}
for ticker in tickers:
    if ticker == ex_returnMarket:
        continue
    cov = np.cov(ret[ticker], market_return)
    beta = cov[0, 1] / cov[1, 1]
    betas[ticker] = beta

# Expected return del mercado
market_avg_return = mu[ex_returnMarket]  # ← funciona porque mu es un Series

# Calcular retorno esperado vía CAPM
expected_returns = {}
for ticker in tickers:
    if ticker == ex_returnMarket:
        continue
    beta = betas[ticker]
    expected = risk_free_rate + beta * (market_avg_return - risk_free_rate)
    expected_returns[ticker] = expected

# Plot SML
plt.figure(figsize=(10, 10))
beta_range = np.linspace(0, 2, 100)
sml_y = risk_free_rate + beta_range * (market_avg_return - risk_free_rate)
plt.plot(beta_range, sml_y, label='SML', color='blue')

for ticker in betas:
    plt.scatter(betas[ticker], mu[ticker], label=ticker)
    plt.text(betas[ticker], mu[ticker], ticker)
plt.xlabel('Beta')
plt.ylabel('Expected Return (Annual)')
plt.title('Security Market Line (SML) - CAPM')
plt.legend()
plt.grid(True)
plt.show()