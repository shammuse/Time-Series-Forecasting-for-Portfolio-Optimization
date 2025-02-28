import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

# Configure logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:    
    def __init__(self):
        """Initialize the DataProcessor with an empty data dictionary."""
        self.data = {}
        
    def get_data(self, symbols, start_date, end_date):
        """
        Fetch historical data for the given symbols within the specified date range.
        Uses Yahoo Finance as the data source.
        """
        for symbol in symbols:
            try:
                logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                df = yf.download(symbol, start=start_date, end=end_date)
                df.reset_index(inplace=True)
                self.data[symbol] = df
                logging.info(f"Data fetched successfully for {symbol}")
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
        return self.data

    def clean_data(self):
        """
        Clean data by handling missing values using forward fill and dropping rows 
        with remaining null values.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Cleaning data for {symbol}")
                df['Date'] = pd.to_datetime(df['Date'])
                df.fillna(method='ffill', inplace=True)
                df.dropna(inplace=True)
                logging.info(f"Data cleaned successfully for {symbol}")
            except Exception as e:
                logging.error(f"Error cleaning data for {symbol}: {e}")
        return self.data

    def basic_statistics(self):
        """
        Compute and return basic statistical summaries (mean, std, etc.) for each symbol.
        """
        stats = {}
        for symbol, df in self.data.items():
            try:
                logging.info(f"Calculating basic statistics for {symbol}")
                stats[symbol] = df.describe()
            except Exception as e:
                logging.error(f"Error calculating statistics for {symbol}: {e}")
        return stats
    
    def plot_closing_prices(self):
        """
        Plot the closing prices over time for each symbol.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Plotting closing prices for {symbol}")
                plt.figure(figsize=(14, 6))
                plt.plot(df['Date'], df['Close'], label=f'{symbol} Close Price')
                plt.title(f'{symbol} Closing Price Over Time')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.legend()
                plt.show()
            except Exception as e:
                logging.error(f"Error plotting closing prices for {symbol}: {e}")
    
    def calculate_daily_returns(self):
        """
        Calculate daily percentage changes (returns) and add them as a new column 
        to each dataframe.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Calculating daily returns for {symbol}")
                df['Daily Return'] = df['Close'].pct_change()
            except Exception as e:
                logging.error(f"Error calculating daily returns for {symbol}: {e}")
        return self.data
    
    def plot_daily_returns(self):
        """
        Plot daily returns over time to observe volatility for each symbol.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Plotting daily returns for {symbol}")
                plt.figure(figsize=(14, 6))
                plt.plot(df['Date'], df['Daily Return'], label=f'{symbol} Daily Return')
                plt.title(f'{symbol} Daily Percentage Change')
                plt.xlabel('Date')
                plt.ylabel('Daily Return')
                plt.legend()
                plt.show()
            except Exception as e:
                logging.error(f"Error plotting daily returns for {symbol}: {e}")
    
    def calculate_rolling_stats(self, window=20):
        """
        Compute rolling mean and standard deviation for the closing price of each symbol.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Calculating rolling statistics for {symbol}")
                df['Rolling Mean'] = df['Close'].rolling(window=window).mean()
                df['Rolling Std'] = df['Close'].rolling(window=window).std()
            except Exception as e:
                logging.error(f"Error calculating rolling statistics for {symbol}: {e}")
        return self.data
    
    def plot_rolling_stats(self):
        """
        Plot the closing price along with its rolling mean and standard deviation.
        """
        for symbol, df in self.data.items():
            try:
                logging.info(f"Plotting rolling statistics for {symbol}")
                plt.figure(figsize=(14, 6))
                plt.plot(df['Date'], df['Close'], label=f'{symbol} Close')
                plt.plot(df['Date'], df['Rolling Mean'], label='20-Day Rolling Mean')
                plt.fill_between(df['Date'], df['Rolling Mean'] - df['Rolling Std'], 
                                 df['Rolling Mean'] + df['Rolling Std'], color='lightgray')
                plt.title(f'{symbol} Price and Rolling Statistics')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.show()
            except Exception as e:
                logging.error(f"Error plotting rolling statistics for {symbol}: {e}")
    
    def detect_outliers(self, threshold=3):
        """
        Detect outliers in daily returns based on a standard deviation threshold.
        """
        outliers = {}
        for symbol, df in self.data.items():
            try:
                logging.info(f"Detecting outliers for {symbol}")
                mean_return = df['Daily Return'].mean()
                std_dev_return = df['Daily Return'].std()
                outliers[symbol] = df[np.abs(df['Daily Return'] - mean_return) > threshold * std_dev_return]
            except Exception as e:
                logging.error(f"Error detecting outliers for {symbol}: {e}")
        return outliers
    
    def plot_outliers(self):
        """
        Plot daily returns and highlight outliers detected.
        """
        for symbol, df in self.data.items():
            try:
                outliers = self.detect_outliers()
                outlier_data = outliers.get(symbol)
                if outlier_data is not None and not outlier_data.empty:
                    logging.info(f"Plotting outliers for {symbol}")
                    plt.figure(figsize=(14, 6))
                    plt.plot(df['Date'], df['Daily Return'], label=f'{symbol} Daily Return')
                    plt.scatter(outlier_data['Date'], outlier_data['Daily Return'], color='red', 
                                label='Outliers', zorder=5)
                    plt.title(f'{symbol} Daily Returns with Outliers')
                    plt.xlabel('Date')
                    plt.ylabel('Daily Return')
                    plt.legend()
                    plt.show()
                else:
                    logging.info(f"No outliers found for {symbol}")
            except Exception as e:
                logging.error(f"Error plotting outliers for {symbol}: {e}")
    
    def decompose_time_series(self):
        """
        Decompose the closing price time series into trend, seasonal, and residual components.
        """
        decomposition_results = {}
        for symbol, df in self.data.items():
            try:
                logging.info(f"Decomposing time series for {symbol}")
                df.set_index('Date', inplace=True)
                decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)
                decomposition.plot()
                plt.suptitle(f'{symbol} Seasonal Decomposition')
                plt.show()
                decomposition_results[symbol] = decomposition
                df.reset_index(inplace=True)  # Reset index for further operations
            except Exception as e:
                logging.error(f"Error decomposing time series for {symbol}: {e}")
        return decomposition_results

    def calculate_risk_metrics(self, confidence_level=0.05):
        """
        Calculate Value at Risk (VaR) and Sharpe Ratio for each symbol's daily returns.
        """
        metrics = {}
        for symbol, df in self.data.items():
            try:
                logging.info(f"Calculating risk metrics for {symbol}")
                daily_returns = df['Daily Return'].dropna()
                VaR = daily_returns.quantile(confidence_level)
                Sharpe_Ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                metrics[symbol] = {'VaR': VaR, 'Sharpe Ratio': Sharpe_Ratio}
            except Exception as e:
                logging.error(f"Error calculating risk metrics for {symbol}: {e}")
        return metrics