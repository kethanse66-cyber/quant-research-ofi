# Day 9 - DataLoader Class
# First Python class in the project
# Wraps all Phase 1 functions into one reusable object

import yfinance as yf
from scipy import stats
import numpy as np

class DataLoader:
    def __init__(self, ticker):
        self.ticker = ticker

    def load(self):
        self.data = yf.download(self.ticker, start='2020-01-01', end='2024-12-31')
        return self.data

    def compute_returns(self):
        self.returns = self.data['Close'].pct_change().dropna()
        return self.returns

    def get_stats(self):
        print("Mean:", self.returns.mean())
        print("Std:", self.returns.std())
        print("Skew:", stats.skew(np.array(self.returns).flatten()))
        print("Kurtosis:", stats.kurtosis(np.array(self.returns).flatten()))
        # Skewness = -0.54 means SPY crashes harder than it rallies
        # Kurtosis = 11.44 means extreme days happen far more than normal predicts


# Usage
loader = DataLoader("SPY")
loader.load()
loader.compute_returns()
loader.get_stats()
