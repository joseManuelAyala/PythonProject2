import yfinance as yf
import numpy as np
import pandas as pd
# Minor edit to trigger Git change detection

tickers = ['AAPL', 'LMT', 'ALHC', 'MC.PA', 'PBR']

data = yf.download(tickers, start="2020-01-01", end = "2025-05-14")

simple_returns = data.pct_change().dropna()

def jarque_bera_testStatistic(sample):
    n = len(sample)
    mean = np.mean(sample)
    standard_deviation = np.std(sample)

    skewness = np.mean(((sample - mean) / standard_deviation)**3)
    kurtosis = np.mean(((sample - mean) / standard_deviation)**4)

    jb_value = (n / 6) * (skewness**2 + ((kurtosis - 3)**2) / 4)

    return jb_value, skewness, kurtosis

for ticker in simple_returns.columns:
    serie = simple_returns[ticker].dropna()
    jb_value, skewness, kurtosis = jarque_bera_testStatistic(serie)
    print(f"{ticker}: JB_Value = {jb_value:.2f}, skewness = {skewness:.2f}, kurtosis = {kurtosis:.2f}")