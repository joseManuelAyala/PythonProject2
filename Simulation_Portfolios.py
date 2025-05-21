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

n_portfolios = 100000
results = np.zeros((3,n_portfolios))
weights_record = []

rf = 0.02  # tasa libre de riesgo

for i in range(n_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)

    # retorno y riesgo del portafolio
    port_return = np.dot(weights, mu)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(Cov_matrix, weights)))
    sharpe_ratio = (port_return - rf) / port_volatility

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio

# 5. Convertir a DataFrame para análisis
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])

# 6. Graficar Frontera Eficiente
plt.figure(figsize=(12, 7))
scatter = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'],
                      cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Exptected Return')
plt.title('Portfolio Simulation')
plt.grid(True)
plt.show()

