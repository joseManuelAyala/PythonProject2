from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define tickers and download data from Yahoo Finance
tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

# Calculate daily returns
returns = (prices - prices.shift()) / prices.shift()
returns = returns.dropna()

# Calculate annual returns
mean_returns = returns.mean() * 252  # Annualized mean returns

# Calculate the standard deviation (volatility)
std_dev = returns.std() * np.sqrt(252)  # Annualized standard deviation

# Calculate the correlation matrix
correlation_matrix = returns.corr()

# Calculate the covariance matrix
cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix

# Inverse of the covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Risk-free rate (assumed)
rf = 0.02

# Define the negative Sharpe ratio function
def negative_sharpe_ratio(w, mu, cov_matrix, rf):
    port_return = np.dot(w, mu)
    port_volatility = np.sqrt(w.T @ cov_matrix @ w)
    return - (port_return - rf) / port_volatility

# Constraints: sum of weights = 1 (full investment)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Optional: no short-selling, i.e., weights should be between 0 and 1
bounds = [(0, 1) for _ in range(len(mean_returns))]

# Initialize with equal weights
w0 = np.ones(len(mean_returns)) / len(mean_returns)

# Optimize the portfolio using the negative Sharpe ratio
opt_result = minimize(negative_sharpe_ratio, w0, args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=constraints)

# Get the optimal weights and the maximum Sharpe ratio
w_opt = opt_result.x
sharpe_opt = -opt_result.fun  # Remove the negative sign

# Print the optimal weights and the Sharpe ratio
print("Optimal weights:", w_opt)
print("Maximum Sharpe ratio:", sharpe_opt)

# Calculate the return and volatility of the optimal portfolio
ret_opt = np.dot(w_opt, mean_returns)
vol_opt = np.sqrt(w_opt.T @ cov_matrix @ w_opt)

# Calculate the optimal complete portfolio (including risk-free asset)
A = 4
y_star = (ret_opt - rf) / (A * vol_opt ** 2)

w_complete = y_star * w_opt
w_rf = 1 - y_star
ret_complete = rf + y_star * (ret_opt - rf)
vol_complete = y_star * vol_opt

# CAPM Analysis (Capital Asset Pricing Model)
benchmark = yf.download("^GSPC", start="2020-01-01", end="2025-05-15", auto_adjust=True)
benchmark_ret = benchmark['Adj Close'].pct_change().dropna()

# Calculate portfolio returns using the optimal weights
portfolio_returns = (returns @ w_opt)

# Align the portfolio returns with the benchmark returns
common_index = benchmark_ret.index.intersection(returns.index)
portfolio_returns = portfolio_returns.loc[common_index]
benchmark_ret = benchmark_ret.loc[common_index]

# Perform regression to calculate alpha and beta using statsmodels
import statsmodels.api as sm

X = sm.add_constant(benchmark_ret)
model = sm.OLS(portfolio_returns, X).fit()
print(model.summary())

# Extract beta and alpha from the model
beta = model.params[1]
alpha = model.params[0]

# Plot CAPM regression
plt.figure(figsize=(8, 5))
plt.scatter(benchmark_ret, portfolio_returns, alpha=0.5)
plt.plot(benchmark_ret, model.predict(X), color='red', label='Security Market Line')
plt.xlabel("Benchmark Return")
plt.ylabel("Portfolio Return")
plt.title("CAPM Regression")
plt.legend()
plt.grid(True)
plt.show()

# Replot everything including the tangency portfolio
if __name__ == "__main__":
    plt.figure(figsize=(10, 6))

    # Plot the Tangency Portfolio
    plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')

    # Plot the optimal complete portfolio
    plt.scatter(vol_complete, ret_complete, color='purple', marker='o', s=150, label='Optimal Complete Portfolio')

    # Plot the Capital Market Line (CML)
    plt.plot([0, vol_opt], [rf, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Expected Return')
    plt.title('Efficient Frontier with Tangency Portfolio')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, None)
    plt.show()