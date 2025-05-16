import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']


ret = (prices - prices.shift()) / prices.shift()
ret = ret.dropna()


mu = ret.mean().values.reshape(-1, 1)*252

std_dev = ret.std().values * np.sqrt(252)
correlation_matrix = ret.corr().values


Cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix
Inv_cov_matrix = np.linalg.inv(Cov_matrix)
ones = np.ones((len(ret.columns), 1))

# Lecture Formula
Z = ones.T @ Inv_cov_matrix @ ones
X = mu.T @ Inv_cov_matrix @ mu
Y = mu.T @ Inv_cov_matrix @ ones
D = (X * Z) - (Y ** 2)


g = (1 / D) * (X * (Inv_cov_matrix @ ones) - Y * (Inv_cov_matrix @ mu))
h = (1 / D) * (Z * (Inv_cov_matrix @ mu) - Y * (Inv_cov_matrix @ ones))

# Efficient Frontier
n = 1000
mu_min = mu.min() * 0.8
mu_max = mu.max() * 1.2
incr = (mu_max - mu_min) / (n - 1)
w_MV = np.zeros((n, len(tickers)))

mu_MV = np.zeros((n, 1))
sigma_MV = np.zeros((n, 1))

for i in range(n):
    mu_i = mu_min + i * incr
    w_i = g + h * mu_i
    w_i = np.clip(w_i, 0, 1)
    w_i /= np.sum(w_i)
    w_MV[i, :] = w_i.T
    mu_MV[i] = mu.T @ w_i
    # Calcular la desviación estándar directamente ya que la covarianza está anualizada
    sigma_value = (w_i.T @ Cov_matrix @ w_i)[0, 0]
    sigma_MV[i] = np.sqrt(sigma_value)


rf = 0.02
SR_MV = (mu_MV - rf)/sigma_MV

TP_index = np.argmax(SR_MV)

w_TP = w_MV[TP_index,:]
mu_TP = mu_MV[TP_index,0]
sigma_TP = sigma_MV[TP_index,0]
SR_TP = (mu_TP - rf)/sigma_TP

# Plot
sigma_cml = np.linspace(0, max(sigma_MV), 100)
mu_cml = rf + SR_TP * sigma_cml

individual_sigma = np.sqrt(np.diag(Cov_matrix))
individual_mu = mu[:, 0]

plt.figure(figsize=(12, 7))
plt.plot(sigma_MV, mu_MV, color='blue', label='Minimum Variance Frontier ')
plt.plot(sigma_cml, mu_cml, color='gray', linestyle='--', label='Capital Market Line ')
plt.scatter(sigma_TP, mu_TP, color='red', s=100, label='Tangency Portfolio (TP)', marker='*')
plt.scatter(0, rf, color='green', s=100, label='Risk-Free Rate ($R_f$)', marker='o')
for i, ticker in enumerate(tickers):
    plt.scatter(individual_sigma[i], individual_mu[i], s=50, label=ticker, marker='o', color='black')
    plt.text(individual_sigma[i], individual_mu[i], ticker, fontsize=9, ha='right', va='bottom')

plt.xlabel('Standard Deviation (Annual)')
plt.ylabel('Expected Return (Annual)')
plt.title("Minimum Variance Frontier")
plt.legend()
plt.show()