import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#plt.style.use('seaborn-darkgrid')

class FinancialDataAnalysis:
    def __init__(self, tickers, start_date='2010-01-01', end_date='2023-12-31'):
        """
        Initializes the FinancialDataAnalysis object.
        
        Parameters:
        tickers (list): List of ticker symbols to analyze.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.cleaned_data = None
        self.daily_pct_change = None

    def load_historical_data(self):
        """
        Load historical data for specified tickers using yfinance.
        """
        print(f"Loading data for: {self.tickers}")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        self.data = data['Adj Close']
        print("Data loaded successfully.")

    def clean_data(self):
        """
        Cleans the loaded data by handling missing values using forward and backward fill.
        """
        if self.data is None:
            raise ValueError("No data to clean. Please load data first.")
        print("Cleaning data...")
        print("Initial missing values in data:\n", self.data.isnull().sum())
        self.cleaned_data = self.data.fillna(method='ffill').fillna(method='bfill')
        print("Missing values after cleaning:\n", self.cleaned_data.isnull().sum())

    def check_statistics(self):
        """
        Prints basic statistics to understand the distribution of the data.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        print("\nBasic statistics of the data:")
        print(self.cleaned_data.describe())

    def plot_closing_prices(self):
        """
        Plots the closing prices for the tickers.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        print("Plotting closing prices...")
        self.cleaned_data.plot(title="Closing Prices Over Time", figsize=(12, 6))
        plt.xlabel("Date")
        plt.ylabel("Adjusted Closing Price (USD)")
        plt.legend(self.tickers)
        plt.show()

    def calculate_daily_percentage_change(self):
        """
        Calculates and plots the daily percentage change for each stock.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        print("Calculating daily percentage change...")
        self.daily_pct_change = self.cleaned_data.pct_change().dropna()
        self.daily_pct_change.plot(figsize=(12, 8), title="Daily Percentage Change")
        plt.xlabel("Date")
        plt.ylabel("Percentage Change")
        plt.show()

    def calculate_rolling_statistics(self, window=30):
        """
        Calculates and plots the rolling mean and standard deviation for volatility analysis.
        
        Parameters:
        window (int): Window size for calculating rolling statistics.
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        print(f"Calculating rolling statistics with window size = {window}...")
        rolling_mean = self.cleaned_data.rolling(window=window).mean()
        rolling_std = self.cleaned_data.rolling(window=window).std()

        plt.figure(figsize=(14, 7))
        plt.plot(self.cleaned_data, label="Actual Closing Price", alpha=0.5)
        plt.plot(rolling_mean, label=f"{window}-Day Rolling Mean", linestyle='--')
        plt.plot(rolling_std, label=f"{window}-Day Rolling Std Dev", linestyle='-.')
        plt.title(f"Rolling Mean and Standard Deviation ({window}-Day)")
        plt.xlabel("Date")
        plt.ylabel("Price / Volatility")
        plt.legend()
        plt.show()

    def decompose_time_series(self, ticker, period=365):
        """
        Decomposes the time series data into trend, seasonality, and residuals.
        
        Parameters:
        ticker (str): Ticker symbol to decompose.
        period (int): Period for decomposition (default is 365 days).
        """
        if self.cleaned_data is None or ticker not in self.cleaned_data.columns:
            raise ValueError(f"Invalid ticker '{ticker}' or no cleaned data available.")
        print(f"Decomposing time series for {ticker}...")
        decomposition = seasonal_decompose(self.cleaned_data[ticker].dropna(), model='multiplicative', period=period)
        decomposition.plot()
        plt.suptitle(f"Seasonal Decomposition of {ticker}", fontsize=14)
        plt.show()

    def calculate_financial_metrics(self):
        """
        Calculates Value at Risk (VaR) and Sharpe Ratio for the daily percentage change data.
        """
        if self.daily_pct_change is None:
            raise ValueError("No daily percentage change data available. Please calculate it first.")
        print("Calculating financial metrics (VaR and Sharpe Ratio)...")
        # VaR Calculation (95% confidence interval)
        var_95 = self.daily_pct_change.quantile(0.05)
        print(f"Value at Risk (95% Confidence Level):\n{var_95}\n")

        # Sharpe Ratio Calculation (assuming risk-free rate = 0)
        mean_return = self.daily_pct_change.mean()
        std_dev = self.daily_pct_change.std()
        sharpe_ratio = mean_return / std_dev
        print(f"Sharpe Ratios:\n{sharpe_ratio}\n")
