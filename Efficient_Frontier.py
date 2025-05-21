from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tickers = ['AAPL', 'JNJ', 'XOM', 'HD', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']


ret = (np.log(prices/prices.shift(1)))
ret = ret.dropna()

mu = ret.mean().values*252

std_dev = ret.std().values * np.sqrt(252)
correlation_matrix = ret.corr().values

Cov_matrix = ret.cov().values * 252
Inv_cov_matrix = np.linalg.inv(Cov_matrix)
n = len(mu)
# Tangency portfolio in the plot
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

print(ret_opt, vol_opt, opt_result)

# Replotear todo incluyendo el tangency portfolio
plt.figure(figsize=(10, 6))

# Tangency Portfolio

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def solve_unconstrained_frontier(mu, Cov_matrix, n_points = 100):
    mu_targets = np.linspace(min(mu)*0.8, max(mu)*1.2, n_points)
    mu_vals, sigma_vals = [], []

    for mu_target in mu_targets:
        constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) -  1},    #Sum of weights must be 1
        {'type': 'eq', 'fun': lambda w: w@mu - mu_target }  #Expected return must be equal to objective value
        ]
        result = minimize(portfolio_variance, x0 = np.ones(n)/n,
                          args=(Cov_matrix,), method = 'SLSQP',
                          constraints = constraints)
        if result.success:
            w = result.x
            mu_vals.append(np.dot(w,mu))
            sigma_vals.append(np.sqrt(w@Cov_matrix@w))
        else:
            mu_vals.append(np.nan)
            sigma_vals.append(np.nan)
    return np.array(mu_vals), np.array(sigma_vals)

def solve_constrained_frontier(mu, Cov_matrix, n_points=100):
    mu_targets = np.linspace(min(mu)*0.8, max(mu)*1.2, n_points)
    mu_vals, sigma_vals = [], []
    bounds = [(0, 0.5) for _ in range(n)]

    for mu_target in mu_targets:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ mu - mu_target}
        ]
        result = minimize(portfolio_variance, x0=np.ones(n)/n,
                          args=(Cov_matrix,), method='SLSQP',
                          bounds=bounds, constraints=constraints)
        if result.success:
            w = result.x
            mu_vals.append(np.dot(w,mu))
            sigma_vals.append(np.sqrt(w @ Cov_matrix @ w))
        else:
            mu_vals.append(np.nan)
            sigma_vals.append(np.nan)

    return np.array(mu_vals), np.array(sigma_vals)
mu_uncon, sigma_uncon = solve_unconstrained_frontier(mu, Cov_matrix)
mu_con, sigma_con = solve_constrained_frontier(mu, Cov_matrix)


#Simulation of Diverse Portfolios

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


results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])
print("Retornos esperados anuales:")
for ticker, r in zip(tickers, mu):
    print(f"{ticker}: {r:.2%}")

print("\nVolatilidades anuales:")
for ticker, s in zip(tickers, std_dev):
    print(f"{ticker}: {s:.2%}")

#Optimal portfolio
A = 4
y_star = (ret_opt - rf)/(A*vol_opt**2)

w_complete =  y_star * w_opt
w_rf = 1- y_star
ret_complete = rf + y_star * (ret_opt - rf)
vol_complete = y_star * vol_opt

# Plot
print(ret)
plt.figure(figsize=(10, 6))

plt.plot(sigma_uncon, mu_uncon, label='Unconstrained Frontier', lw=2)
plt.plot(sigma_con, mu_con, label='Constrained Frontier', lw=2)
scatter = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'],
                      cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
# Plot del portafolio completo óptimo
plt.scatter(vol_complete, ret_complete, color='purple', marker='o', s=150, label='Optimal Complete Portfolio')

plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')
plt.plot([0, vol_opt], [rf, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected return')
plt.title(' Efficient Frontier')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.show()

