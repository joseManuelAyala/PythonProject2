from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

ret = (prices - prices.shift()) / prices.shift()
ret = ret.dropna()

mu = ret.mean().values*252

std_dev = ret.std().values * np.sqrt(252)
correlation_matrix = ret.corr().values

Cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix
Inv_cov_matrix = np.linalg.inv(Cov_matrix)

rf = 0.02

def negative_sharpe_ratio(w, mu, cov_matrix, rf):
    port_return = np.dot(w, mu)
    port_volatility = np.sqrt(w.T @ cov_matrix @ w)
    return - (port_return - rf) / port_volatility

# Restricción: suma de los pesos = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Opcional: no short-selling
bounds = [(0, 1) for _ in range(len(mu))]

# Inicialización
w0 = np.ones(len(mu)) / len(mu)

# Optimización
opt_result = minimize(negative_sharpe_ratio, w0,
                      args=(mu, Cov_matrix, rf),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

# Resultao
w_opt = opt_result.x
sharpe_opt = -opt_result.fun  # Le sacamos el signo negativo

print("Pesos óptimos:", w_opt)
print("Sharpe ratio máximo:", sharpe_opt)
ret_opt = np.dot(w_opt, mu)
vol_opt = np.sqrt(w_opt.T @ Cov_matrix @ w_opt)
#Optimal portfolio
A = 4
y_star = (ret_opt - rf)/(A*vol_opt**2)

w_complete =  y_star * w_opt
w_rf = 1- y_star
ret_complete = rf + y_star * (ret_opt - rf)
vol_complete = y_star * vol_opt



# CAPM Analysis
benchmark = yf.download("^GSPC", start="2020-01-01", end="2025-05-15", auto_adjust=True)
benchmark_ret = benchmark['Adj Close'].pct_change().dropna()
portfolio_returns = (ret@w_opt)
common_index = benchmark_ret.index.intersection(ret.index)
portfolio_returns = portfolio_returns.loc[common_index]
benchmark_ret = benchmark_ret.loc[common_index]

import statsmodels.api as sm

X = sm.add_constant(benchmark_ret)
model = sm.OLS(portfolio_returns, X).fit()
print(model.summary())

beta = model.params[1]
alpha = model.params[0]

plt.figure(figsize=(8,5))
plt.scatter(benchmark_ret, portfolio_returns, alpha=0.5)
plt.plot(benchmark_ret, model.predict(X), color='red', label='Security Market Line')
plt.xlabel("Benchmark Return")
plt.ylabel("Portfolio Return")
plt.title("CAPM Regression")
plt.legend()
plt.grid(True)
plt.show()









if __name__ == "__main__":
# Replotear todo incluyendo el tangency portfolio
    plt.figure(figsize=(10, 6))
#Tangency Portfolio
    plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')
# Plot del portafolio completo óptimo
    plt.scatter(vol_complete, ret_complete, color='purple', marker='o', s=150, label='Optimal Complete Portfolio')

    plt.plot([0, vol_opt], [rf, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Expected Return')
    plt.title('Efficient Frontier with Tangency Portfolio')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, None)
    plt.show()

#asdasda