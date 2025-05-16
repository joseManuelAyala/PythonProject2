import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AAPL','JNJ','XOM','HD','TSLA']
data = yf.download(tickers, start="2020-01-01", end = "2025-05-15",auto_adjust = False)
prices = data['Adj Close']

ret = (prices - prices.shift())/prices.shift()
ret = ret.dropna()
mu = ret.mean()
mu = mu.values.reshape(-1,1)
Cov_matrix = ret.cov()
Inv_cov_matrix = np.linalg.inv(Cov_matrix)

#Calculation of the Unconstrained MV Frontier

ones = np.ones((len(ret.columns),1))

Z=ones.T@Inv_cov_matrix@ones
X=mu.T@Inv_cov_matrix@mu
Y= mu.T@Inv_cov_matrix@ones
D = (X*Z) - (Y**2)

g = (1/D)*(X*(Inv_cov_matrix@ones) - Y*(Inv_cov_matrix@mu))
h = (1/D)*(Z*(Inv_cov_matrix@mu) - Y*(Inv_cov_matrix@ones))

w = g+ h*mu
w_normalized = w / np.sum(w)
print(w_normalized)

#plot
n = 400

mu_min = 0
mu_max = 0.2
incr = (mu_max - mu_min)/(n-1)
w_MV= np.zeros((n,mu.shape[0]))

mu_MV = np.zeros((n,1))
sigma_MV = np.zeros((n,1))

for i in range(n):
    mu_i = i*incr
    w_i = g + h *mu_i
    w_MV[i,:] = w_i.T
    mu_MV[i] = mu.T@w_i
    sigma_MV[i] = np.sqrt(float(w_i.T @ Cov_matrix @ w_i))

plt.figure(1, figsize=(12,7))
plt.plot(sigma_MV, mu_MV, color = 'blue', label = 'Minimum Varience Frontier')
plt.scatter(np.diag(Cov_matrix)**0.5, mu[:,0], s = 20, color ='black', label = 'Individual assets')
plt.xlabel('$\sigma(r)$')
plt.ylabel('$\E(r)$')
plt.title("Minimum Variance Forntier (unconstrained", fontweight='bold')
plt.legend()
plt.show()