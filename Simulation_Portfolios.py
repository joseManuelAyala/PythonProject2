from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Download data
tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

# Calculate daily returns
returns = (prices - prices.shift()) / prices.shift()
returns = returns.dropna()

# Calculate annualized returns and standard deviation
mean_returns = returns.mean().values * 252  # Annualized mean returns
std_dev = returns.std().values * np.sqrt(252)  # Annualized standard deviation

# Calculate correlation and covariance matrices
correlation_matrix = returns.corr().values
cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Number of portfolios to simulate
n_portfolios = 100000
results = np.zeros((3, n_portfolios))
weights_record = []

# Risk-free rate
rf = 0.02  # Risk-free rate

# Simulate portfolios
for i in range(n_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    weights_record.append(weights)

    # Portfolio return and risk
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - rf) / port_volatility

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
plt.ylabel('Expected Return')
plt.title('Portfolio Simulation')
plt.grid(True)
plt.show()

