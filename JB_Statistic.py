import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Jarque-Bera test statistic function
def jarque_bera_testStatistic(sample):
    n = len(sample)
    mean = np.mean(sample)
    standard_deviation = np.std(sample)
    skewness = np.mean(((sample - mean) / standard_deviation) ** 3)
    kurtosis = np.mean(((sample - mean) / standard_deviation) ** 4)
    jb_value = (n / 6) * (skewness ** 2 + ((kurtosis - 3) ** 2) / 4)
    return jb_value, skewness, kurtosis

# Define tickers and download data from Yahoo Finance
tickers = ['MSFT', 'DE', 'COST', 'BYDDY', 'AMD', 'GLD']
data = yf.download(tickers, start="2018-01-01", end="2025-05-15", auto_adjust=False)
prices = data['Adj Close']

# Calculate simple returns
simple_returns = (prices - prices.shift()) / prices.shift()
simple_returns = simple_returns.dropna()

# Set up the plot
plt.figure(figsize=(15, 10))

# Loop through each ticker and plot the histogram with Jarque-Bera statistics
for i, ticker in enumerate(simple_returns.columns):
    serie = simple_returns[ticker].dropna()
    jb_value, skewness, kurtosis = jarque_bera_testStatistic(serie)

    # Plot histogram for each ticker
    plt.subplot(3, 2, i + 1)
    plt.hist(serie, bins=30, color='lightblue', edgecolor='black', density=True)
    plt.title(f"Return Histogram: {ticker}")
    plt.xlabel('Return')
    plt.ylabel('Frequency')

    # Display Jarque-Bera statistics on the plot
    plt.text(0.05, 0.95, f"JB = {jb_value:.2f}\nSkew = {skewness:.2f}\nKurt = {kurtosis:.2f}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()