from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define tickers and download data from Yahoo Finance
tickers = ['MSFT', 'DE', 'COST', 'BYDDY', 'AMD', 'GLD']
data = yf.download(tickers, start="2018-01-01", end="2025-05-15", auto_adjust=False)
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
print(correlation_matrix)

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

# Initialize with equal weights
w0 = np.ones(len(mean_returns)) / len(mean_returns)

# Optimize the portfolio using the negative Sharpe ratio
opt_result = minimize(negative_sharpe_ratio, w0,
                      args=(mean_returns, cov_matrix, rf),
                      method='SLSQP',
                      constraints=constraints)

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

########## CAPM Analysis (Capital Asset Pricing Model)###############
benchmark = yf.download("^GSPC", start="2018-01-01", end="2025-05-15", auto_adjust=True)
benchmark_ret = benchmark['Close'].pct_change().dropna()

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

#Robust Regression
model_rlm = sm.RLM(portfolio_returns, X).fit()
print(model_rlm.summary())



plt.figure(figsize=(8, 5))
plt.scatter(benchmark_ret, portfolio_returns, alpha=0.4, label='Datos')
plt.plot(benchmark_ret, model_rlm.predict(), color='orange', label='CAPM (RLM robusto)')
plt.title('CAPM robusto (RLM)')
plt.xlabel('Rendimiento mercado')
plt.ylabel('Rendimiento portafolio')
plt.legend()
plt.grid(True)
plt.show()


#################FF Model##############
import pandas as pd
ff = pd.read_csv("F-F_Research_Data_Factors_daily.CSV", names=["Date", "Mkt-RF", "SMB", "HML", "RF"], skiprows=1)
ff["Date"] = pd.to_datetime(ff['Date'],format = '%Y%m%d')
ff.set_index('Date', inplace = True)
ff = ff.loc["2020-01-01":"2025-05-15"]

common_dates = ff.index.intersection(portfolio_returns.index)
rp = portfolio_returns.loc[common_dates]
ff = ff.loc[common_dates]

ff["Mkt-RF"] = ff["Mkt-RF"]/100
ff["SMB"] = ff["SMB"]/100
ff["HML"] = ff["HML"]/100
ff["RF"] = ff["RF"]/100

rp_excess = rp - ff["RF"]
X = ff[["Mkt-RF"]]
import pandas as pd
import statsmodels.api as sm

def run_capm_rolling(Y, X, window_size=252):
    alphas, betas, t_alphas, t_betas = [], [], [], []
    dates = []

    for i in range(window_size, len(Y)):
        y_subset = Y.iloc[i - window_size:i]
        X_subset = X.iloc[i - window_size:i]

        model = sm.OLS(y_subset, sm.add_constant(X_subset)).fit()
        robust = model.get_robustcov_results(cov_type='HAC', maxlags=30)

        alphas.append(model.params[0])
        betas.append(model.params[1])
        t_alphas.append(robust.tvalues[0])
        t_betas.append(robust.tvalues[1])
        dates.append(Y.index[i])

    return pd.DataFrame({
        'alpha': alphas,
        'beta': betas,
        't_alpha': t_alphas,
        't_beta': t_betas
    }, index=dates)
rolling_capm = run_capm_rolling(rp_excess, X, window_size=252)

# Beta

fig, zx1 = plt.subplots(figsize=(10, 5))

# Eje izquierdo: coeficiente beta
ax1.plot(rolling_capm.index, rolling_capm['beta'], color='blue', label='Beta')
ax1.set_ylabel('Beta', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Eje derecho: t-stat de beta
ax2 = ax1.twinx()
ax2.plot(rolling_capm.index, rolling_capm['t_beta'], color='orange', label='t-Beta')
ax2.set_ylabel('t-Beta', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.axhline(2, color='red', linestyle='--', linewidth=0.8)
ax2.axhline(-2, color='red', linestyle='--', linewidth=0.8)

# Título y leyenda
fig.suptitle('Rolling Beta y su t-Statistic (CAPM)', fontsize=14)
fig.tight_layout()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 5))

# Eje izquierdo: Alpha
ax1.plot(rolling_capm.index, rolling_capm['alpha'], color='red', label='Alpha')
ax1.set_ylabel('Alpha', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Eje derecho: t-Alpha
ax2 = ax1.twinx()
ax2.plot(rolling_capm.index, rolling_capm['t_alpha'], color='purple', label='t-Alpha')
ax2.set_ylabel('t-Alpha', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.axhline(2, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(-2, color='black', linestyle='--', linewidth=0.8)

# Título y ajuste
fig.suptitle('Rolling Alpha y su t-Statistic (CAPM)', fontsize=14)
fig.tight_layout()
plt.grid(True)
plt.show()

# Replot everything including the tangency portfolio

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