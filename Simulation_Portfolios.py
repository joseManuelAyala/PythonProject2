from scipy.ndimage import standard_deviation
from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *

# Download data
prices = load_prices(investable_tickers)

# Calculate daily returns
returns = compute_returns(prices)

# Calculate annualized returns and standard deviation
mean_returns = compute_anual_returns(returns)
std_dev = compute_anual_volatility(returns)

# Calculate correlation and covariance matrices
correlation_matrix = compute_correlation_matrix(returns)
covariance_matrix = np.outer(std_dev, std_dev) * correlation_matrix

# Number of portfolios to simulate
n_portfolios = 100000
results = np.zeros((3, n_portfolios))
weights_record = []

# Simulate portfolios
for i in range(n_portfolios):
    weights = np.random.random(len(investable_tickers))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    weights_record.append(weights)

    # Portfolio return and risk
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])

# Plot Efficient Frontier
plt.figure(figsize=(12, 7))
scatter = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'],
                      cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.xlim(0,0.6)
plt.ylabel('Expected Return')
plt.title('Portfolio Simulation')
plt.grid(True)
plt.show()

