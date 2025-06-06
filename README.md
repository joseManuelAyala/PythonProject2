# 📊 Strategic Portfolio Optimization

This project implements an end-to-end system for portfolio analysis and optimization. It is structured as part of a Financial Data Science assignment and integrates statistical analysis, simulation, and CAPM-based decomposition to evaluate a resilient investment strategy.

## 📁 Project Structure

```
📦 StrategicPortfolioOptimization
 ┣ 📜 CAPM.py                  # Performs CAPM regression and rolling analysis
 ┣ 📜 Efficient_Frontier.py    # Builds constrained and unconstrained efficient frontiers
 ┣ 📜 JB_Statistic.py          # Applies Jarque-Bera test for normality on asset returns
 ┣ 📜 Simulation_Portfolios.py # Generates 10,000 simulated portfolios
 ┣ 📜 SML.py                   # Security Market Line plot and Sharpe ratio validation
 ┣ 📜 Tangency_Portfolio.py    # Computes and visualizes the tangency portfolio
 ┣ 📜 data_utils.py            # Loads and processes return data
 ┣ 📄 Report_FDS.pdf           # Final report detailing methodology and results
 ┣ 📂 /data
 ┃ ┗ 📄 F-F_Research_Data_Factors_daily.CSV  # CAPM factor dataset
 ┗ 📄 requirements.txt         # Python packages to install
```

## ✅ Main Features

- Jarque-Bera normality testing on return distributions
- Correlation matrix and diversification analysis
- Efficient frontier simulation (with and without constraints)
- Tangency portfolio optimization using Sharpe Ratio
- CAPM regression and rolling alpha/beta analysis
- Realized vs. expected performance comparison
- Visualization of Capital Market Line and SML

## ▶️ How to Run

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Run scripts individually** depending on your goal:
   - `python JB_Statistic.py` → Normality analysis
   - `python Simulation_Portfolios.py` → Simulate and plot portfolios
   - `python Efficient_Frontier.py` → Constrained/unconstrained frontier
   - `python Tangency_Portfolio.py` → Compute optimal Sharpe ratio portfolio
   - `python CAPM.py` → CAPM & rolling regressions
   - `python SML.py` → Security Market Line & model check

## 📈 Data

All data used is included in the `/data` folder, specifically:
- `F-F_Research_Data_Factors_daily.CSV`: Market factors used for CAPM regression

## 📚 Reference

This repository was built as part of the Financial Data Science course at KIT, under the supervision of Prof. Maxim Ulrich.

