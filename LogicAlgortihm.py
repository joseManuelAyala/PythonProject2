import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

ret = (prices - prices.shift()) / prices.shift()
ret = ret.dropna()
mu = ret.mean().values.reshape(-1, 1)
Cov_matrix = ret.cov().values
Inv_cov_matrix = np.linalg.inv(Cov_matrix)
ones = np.ones((len(ret.columns), 1))

# VL Formula
Z = ones.T @ Inv_cov_matrix @ ones
X = mu.T @ Inv_cov_matrix @ mu
Y = mu.T @ Inv_cov_matrix @ ones
D = (X * Z) - (Y ** 2)

g = (1 / D) * (X * (Inv_cov_matrix @ ones) - Y * (Inv_cov_matrix @ mu))
h = (1 / D) * (Z * (Inv_cov_matrix @ mu) - Y * (Inv_cov_matrix @ ones))

# Efficient Frontier
n = 400
mu_min = mu.min() * 0.8
mu_max = mu.max() * 1.2
incr = (mu_max - mu_min) / (n - 1)
w_MV = np.zeros((n, len(tickers)))

mu_MV = np.zeros((n, 1))
sigma_MV = np.zeros((n, 1))

for i in range(n):
    mu_i = mu_min + i * incr
    w_i = g + h * mu_i
    # Asegurarse de que los pesos sean positivos y normalizados
    w_i = np.clip(w_i, 0, 1)
    w_i /= np.sum(w_i)
    w_MV[i, :] = w_i.T
    mu_MV[i] = mu.T @ w_i
    # Corrigiendo el cálculo de sigma
    sigma_value = (w_i.T @ Cov_matrix @ w_i)[0, 0]
    sigma_MV[i] = np.sqrt(sigma_value)

# Plot
plt.figure(figsize=(12, 7))
plt.plot(sigma_MV, mu_MV, color='blue', label='Minimum Variance Frontier')
plt.scatter(np.sqrt(np.diag(Cov_matrix)), mu[:, 0], s=50, color='black', label='Individual Assets')
plt.xlabel('Desviación Estándar (Riesgo)')
plt.ylabel('Retorno Esperado')
plt.title("Minimum Variance Frontier (Corregida)")
plt.legend()
plt.show()

