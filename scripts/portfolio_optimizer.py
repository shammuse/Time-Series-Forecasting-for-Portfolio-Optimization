import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

class PortfolioOptimization:
    def __init__(self, tsla_data, bnd_data, spy_data):
        """
        Initializes the PortfolioOptimization class with data for each asset.
        
        :param tsla_data: DataFrame for Tesla stock.
        :param bnd_data: DataFrame for Vanguard Total Bond Market ETF.
        :param spy_data: DataFrame for S&P 500 ETF.
        """
        tsla_data['Close'] = pd.to_numeric(tsla_data['Close'], errors='coerce')
        bnd_data['Close'] = pd.to_numeric(bnd_data['Close'], errors='coerce')
        spy_data['Close'] = pd.to_numeric(spy_data['Close'], errors='coerce')

        # Calculate daily returns for each asset
        self.tsla_returns = tsla_data['Close'].pct_change().dropna()
        self.bnd_returns = bnd_data['Close'].pct_change().dropna()
        self.spy_returns = spy_data['Close'].pct_change().dropna()

        # Calculate annualized returns for each asset
        self.tsla_annualized_return = self.tsla_returns.mean() * 252
        self.bnd_annualized_return = self.bnd_returns.mean() * 252
        self.spy_annualized_return = self.spy_returns.mean() * 252

        # Calculate the covariance matrix between returns (only necessary for optimization)
        self.cov_matrix = np.cov([self.tsla_returns, self.bnd_returns, self.spy_returns]) * 252

    def optimize_portfolio(self):
        """
        Optimizes the portfolio weights to maximize the Sharpe ratio.
        
        :return: The optimal portfolio weights.
        """
        # Initial guess for the portfolio weights (equal weights)
        initial_weights = np.array([1/3, 1/3, 1/3])

        # Constraints: the sum of weights should be 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Bounds for the weights: each weight should be between 0 and 1
        bounds = [(0, 1) for _ in range(3)]

        # Optimize portfolio
        optimal_weights = minimize(self.negative_sharpe_ratio, initial_weights, args=(self.tsla_annualized_return, 
                                                                                      self.bnd_annualized_return, 
                                                                                      self.spy_annualized_return, 
                                                                                      self.cov_matrix),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal_weights.x

    def negative_sharpe_ratio(self, weights, tsla_annualized_return, bnd_annualized_return, spy_annualized_return, cov_matrix):
        """
        The objective function for portfolio optimization, which minimizes the negative Sharpe ratio.
        
        :param weights: Portfolio weights for each asset.
        :param tsla_annualized_return: Annualized return of Tesla.
        :param bnd_annualized_return: Annualized return of Bond ETF.
        :param spy_annualized_return: Annualized return of S&P 500 ETF.
        :param cov_matrix: Covariance matrix of the asset returns.
        :return: Negative Sharpe ratio (we minimize this to maximize the Sharpe ratio).
        """
        portfolio_return = np.dot(weights, [tsla_annualized_return, bnd_annualized_return, spy_annualized_return])
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # We negate the Sharpe Ratio to minimize

    def calculate_portfolio_performance(self, weights):
        """
        Calculates the expected return, volatility (risk), and Sharpe ratio of the portfolio.
        
        :param weights: The weights for each asset in the portfolio.
        :return: A tuple containing the portfolio's expected return, risk, and Sharpe ratio.
        """
        portfolio_return = np.dot(weights, [self.tsla_annualized_return, self.bnd_annualized_return, self.spy_annualized_return])
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def efficient_frontier(self):
        """
        Generates the Efficient Frontier by plotting portfolio performance for different weight combinations.
        
        :return: Arrays of returns, volatilities, and Sharpe ratios for portfolios.
        """
        results = np.zeros((3, 1000))
        for i in range(1000):
            weights = np.random.random(3)
            weights /= np.sum(weights)
            portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(weights)
            results[0,i] = portfolio_return
            results[1,i] = portfolio_volatility
            results[2,i] = sharpe_ratio
        return results

    def plot_results(self, optimal_weights):
        """
        Plot results for portfolio optimization including:
        - Efficient Frontier
        - Portfolio Weights
        - Correlation Heatmap
        """
        # Efficient Frontier
        results = self.efficient_frontier()
        
        # Plot Efficient Frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.scatter(self.calculate_portfolio_performance(optimal_weights)[1], 
                    self.calculate_portfolio_performance(optimal_weights)[0], 
                    color='red', marker='*', s=200, label='Optimized Portfolio')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Portfolio Weights
        plt.figure(figsize=(8, 5))
        plt.bar(['TSLA', 'BND', 'SPY'], optimal_weights, color='lightblue')
        plt.title('Optimized Portfolio Weights')
        plt.ylabel('Weight')
        plt.show()

        # Correlation Heatmap
        correlation_matrix = np.corrcoef([self.tsla_returns, self.bnd_returns, self.spy_returns])
        sns.heatmap(correlation_matrix, annot=True, xticklabels=['TSLA', 'BND', 'SPY'], yticklabels=['TSLA', 'BND', 'SPY'], cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Asset Returns')
        plt.show()