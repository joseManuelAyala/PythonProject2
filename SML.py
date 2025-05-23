import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Download data
tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA', '^GSPC']
data = yf.download(tickers, start="2018-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

# Calculate daily returns
returns = (prices - prices.shift()) / prices.shift()
returns = returns.dropna()

# Calculate annual returns
mean_returns = returns.mean() * 252  # Annualized mean returns

# Parameters for CAPM
ex_returnMarket = '^GSPC'
risk_free_rate = 0.03  # 3% risk-free rate

# Calculate Beta for each Asset
market_return = returns[ex_returnMarket]
betas = {}
for ticker in tickers:
    if ticker == ex_returnMarket:
        continue
    cov = np.cov(returns[ticker], market_return)
    beta = cov[0, 1] / cov[1, 1]
    betas[ticker] = beta

# Expected return of the market
market_avg_return = mean_returns[ex_returnMarket]  # This works because mean_returns is a pandas Series

# Calculate expected returns via CAPM
expected_returns = {}
for ticker in tickers:
    if ticker == ex_returnMarket:
        continue
    beta = betas[ticker]
    expected = risk_free_rate + beta * (market_avg_return - risk_free_rate)
    expected_returns[ticker] = expected

# Plot the Security Market Line (SML)
plt.figure(figsize=(10, 10))
beta_range = np.linspace(0, 2, 100)
sml_y = risk_free_rate + beta_range * (market_avg_return - risk_free_rate)
plt.plot(beta_range, sml_y, label='SML', color='blue')

# Scatter plot of each asset with its beta and expected return
for ticker in betas:
    plt.scatter(betas[ticker], mean_returns[ticker], label=ticker)
    plt.text(betas[ticker], mean_returns[ticker], ticker)
plt.xlabel('Beta')
plt.ylabel('Expected Return (Annual)')
plt.title('Security Market Line (SML) - CAPM')
plt.legend()
plt.grid(True)
plt.show()