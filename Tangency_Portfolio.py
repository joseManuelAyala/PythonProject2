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

# Resultado
w_opt = opt_result.x
sharpe_opt = -opt_result.fun  # Le sacamos el signo negativo

print("Pesos óptimos:", w_opt)
print("Sharpe ratio máximo:", sharpe_opt)
ret_opt = np.dot(w_opt, mu)
vol_opt = np.sqrt(w_opt.T @ Cov_matrix @ w_opt)

# Replotear todo incluyendo el tangency portfolio
plt.figure(figsize=(10, 6))

# Tangency Portfolio
plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')
plt.plot([0, vol_opt], [rf, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected Return')
plt.title('Efficient Frontier with Tangency Portfolio')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.show()